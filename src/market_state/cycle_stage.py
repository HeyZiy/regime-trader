# -*- coding: utf-8 -*-
"""
===================================
Cycle 吸收 — 指数循环定位（A1 仓位档位 + A2 恐慌底快速通道）
===================================

证据存档：research/cycle_absorption/（设计、回测、消融全链路）——11 年全成本回测
基准 CAGR −55.4% → 含五组件 −5.4%，MDD −62.8% → −43.7%，三重稳健性检验通过。

组件语义（全部只消费日线收盘后可得字段）：
  A1  指数顶部延伸（bias_MA10 ≥ ext_idx_thr 进入，≤ ext_idx_thr×ext_idx_reset 解除，滞回）
      → stage="top"，新开仓受组合仓位上限 cap_top 约束（主升期 cap=1.0 不约束）
  A2  恐慌底登记（bias_MA10 ≤ panic_thr + 量 ≥ panic_vol_mult×前20日均量 + 反转K）
      后 fp_window 个交易日内收复 MA10 且收阳 → stage="bottom"：当日 gate 禁开仓也放行
      （allow_override，仅当日有效），组合上限 cap_bottom；窗口内创新低则恐慌低点跟随
      下移（登记保持，收复判定以窗口真实最低收盘为准）
  gate 禁开仓（trending_down/sideways/chaos）→ cap=0（非快速通道日）

五组件（A1/A2/B1/C1/D1）作为整体使用：单独停用任一组件属未验证组合，
消融显示停用 A1/B1 有害（research/cycle_absorption/output/ablation.csv）。
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.market_state.market_gate import REGIME_CAN_OPEN, SIDEWAYS_BIAS

logger = logging.getLogger(__name__)

CYCLE_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "cycle_state.json"

# 日报展示用的阶段文案
STAGE_LABELS = {
    "main": "主升",
    "top": "顶部延伸",
    "bottom": "恐慌底回升",
    "down": "下行",
    "range": "禁开（非快速通道日）",
}


@dataclass
class CycleConfig:
    """Cycle 吸收五组件的阈值参数（回测验证过的默认值）。

    改默认值即改策略，须同步 strategy/trend_strategy.md 并重跑证据；
    组件须作为整体使用——消融显示单独停用 A1/B1 有害
    （research/cycle_absorption/output/ablation.csv）。
    """

    # A1 指数循环定位 → 组合仓位档位
    ext_idx_thr: float = 4.5        # 指数顶部延伸阈值：bias vs MA10（%）
    ext_idx_reset: float = 0.5      # 延伸滞回：回落到 thr×0.5 以下解除
    cap_top: float = 0.30           # 顶部循环组合仓位上限
    cap_bottom: float = 0.40        # 底部循环（快速通道日）组合仓位上限
    # A2 恐慌底登记 + 右侧快速通道
    panic_thr: float = -4.0         # 恐慌偏离阈值：bias vs MA10（%）
    panic_vol_mult: float = 1.5     # 恐慌量能：量 ≥ 1.5×前20日均量
    fp_window: int = 10             # 恐慌登记后的快速通道窗口（交易日）
    # B1 个股延伸计数分批止盈（组件在 src/trend/cycle_overlay.py，阈值同源于此）
    ext_stk_thr: float = 10.0       # 个股延伸阈值：bias vs MA10（%）
    ext_stk_reset: float = 4.0      # 延伸事件重置：bias 回落 < 4% 后再延伸才计新事件
    # C1 ATR 扩张过滤 / D1 MA5 方向门（组件在 src/trend/cycle_overlay.py）
    atr_exp_thr: float = 1.3        # ATR5/ATR20 > 1.3 视为扩张，剔除信号


# 进程级唯一参数实例（阈值单一来源；组件模块直接引用）
CYCLE_PARAMS = CycleConfig()


def build_idx_features(index_df: Optional[pd.DataFrame]) -> Optional[Dict]:
    """把指数日线（fetch_index_df 口径）组装成 CycleStage 消费的特征数组。

    指标口径与探索包 trend_core.load_index 一致：MA10、bias_MA10（%），
    前 20 日均量（shift(1) 不含当日）、当日涨跌幅（由收盘价推导）。
    不足 21 根（MA10 + 前 20 日均量的最小需求）返回 None。
    """
    if index_df is None or len(index_df) < 21 or "close" not in index_df.columns:
        return None
    df = index_df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    ma10 = close.rolling(10).mean()
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    return {
        "dates": [str(pd.to_datetime(d).date()) for d in df["date"]],
        "open": df["open"].astype(float).to_numpy() if "open" in df.columns else close.to_numpy(),
        "high": df["high"].astype(float).to_numpy() if "high" in df.columns else close.to_numpy(),
        "low": df["low"].astype(float).to_numpy() if "low" in df.columns else close.to_numpy(),
        "close": close.to_numpy(),
        "volume": volume.to_numpy(),
        "ma10": ma10.to_numpy(),
        "bias_ma10": ((close - ma10) / ma10 * 100).to_numpy(),
        "vol20p": volume.shift(1).rolling(20).mean().to_numpy(),
        "pct_chg": (close.pct_change() * 100).to_numpy(),
    }


def regime_series(index_df: pd.DataFrame) -> List[str]:
    """逐日复刻 market_gate.diagnose_regime 的五态判定（同优先级、同阈值）。

    供 CycleStage 历史回放使用：A2 快速通道是否触发取决于"当日 gate 是否禁开"，
    回放需要每一天的状态。与 diagnose_regime 是单一事实来源的两份实现——
    改 diagnose_regime 必须同步改这里（tests/test_cycle_stage.py 有逐日对拍测试兜底）。
    """
    df = index_df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    ma5 = close.rolling(5).mean().to_numpy()
    ma10 = close.rolling(10).mean().to_numpy()
    ma20 = close.rolling(20).mean().to_numpy()
    closes = close.to_numpy()

    regimes: List[str] = []
    for i in range(len(closes)):
        if i < 19 or any(np.isnan(x) for x in (ma5[i], ma10[i], ma20[i])):
            regimes.append("chaos")
            continue
        c, m5, m10, m20 = closes[i], ma5[i], ma10[i], ma20[i]
        if m5 < m10 < m20 and c < m10:
            regimes.append("trending_down")
        elif m5 > m10 > m20 and c > m10:
            regimes.append("trending_up")
        elif abs(c - m20) / m20 < SIDEWAYS_BIAS:
            regimes.append("sideways")
        elif c > m20:
            regimes.append("weak_up")
        else:
            regimes.append("chaos")
    return regimes


class CycleStage:
    """A1+A2 指数级循环定位：逐日 update(i, gate) 后可读 stage/cap/allow_override。"""

    def __init__(self, cfg: CycleConfig, idx: Dict):
        self.cfg = cfg
        self.idx = idx
        self.stage = "main"
        self.cap = 1.0
        self.allow_override = False   # True：当日 gate 禁开仓但快速通道放行（仅当日）
        # A2 状态
        self._panic_pending_until = -1
        self._panic_low = np.inf
        # A1 顶部延伸（滞回）
        self._idx_ext_active = False
        self.history: List[tuple] = []  # (date, stage, cap, fastpath, panic_event, idx_ext)

    def _gate_allows(self, gate: str) -> bool:
        return gate in REGIME_CAN_OPEN

    def update(self, i: int, gate: str) -> None:
        cfg, x = self.cfg, self.idx
        date = x["dates"][i]
        panic_event = False
        fastpath = False
        bias10 = x["bias_ma10"][i]

        # ---- A2 恐慌底登记：深偏离 + 放量 + 反转K（阳线或长下影）----
        if i >= 20 and not np.isnan(bias10):
            vol, vol20 = x["volume"][i], x["vol20p"][i]
            o, c, l = x["open"][i], x["close"][i], x["low"][i]
            body = abs(c - o)
            lower_wick = min(o, c) - l
            reversal_bar = (c > o) or (body > 0 and lower_wick >= 2 * body)
            if (bias10 <= cfg.panic_thr and not np.isnan(vol20) and vol20 > 0
                    and vol >= cfg.panic_vol_mult * vol20 and reversal_bar):
                panic_event = True
                self._panic_pending_until = i + cfg.fp_window
                self._panic_low = c
            elif self._panic_pending_until >= i:
                self._panic_low = min(self._panic_low, c)  # 窗口内更新恐慌低点

        # ---- A2 快速通道：窗口内收复 MA10 + 收阳 + 未破恐慌低点 ----
        if (self._panic_pending_until >= i and not np.isnan(bias10)
                and x["close"][i] >= x["ma10"][i] and x["pct_chg"][i] > 0
                and x["close"][i] > self._panic_low):
            if not self._gate_allows(gate):
                fastpath = True
            self._panic_pending_until = -1  # 一次性消费（gate 已放行的收复日同样消耗窗口）

        # 窗口内创新低：上方 min 分支已把 panic_low 跟随下移（登记保持），
        # 故此失效分支实际不可达——保留与探索包实现一致的顺序（行为对拍过）
        if self._panic_pending_until >= i and x["close"][i] < self._panic_low:
            self._panic_pending_until = -1
            self._panic_low = np.inf

        # ---- A1 顶部延伸（滞回）----
        if not np.isnan(bias10):
            if not self._idx_ext_active and bias10 >= cfg.ext_idx_thr:
                self._idx_ext_active = True
            elif self._idx_ext_active and bias10 <= cfg.ext_idx_thr * cfg.ext_idx_reset:
                self._idx_ext_active = False

        # ---- 汇总 stage / cap ----
        if fastpath:
            self.stage, self.cap = "bottom", cfg.cap_bottom
            self.allow_override = True
        elif not self._gate_allows(gate):
            self.stage, self.cap = ("down" if gate == "trending_down" else "range"), 0.0
            self.allow_override = False
        elif self._idx_ext_active:
            self.stage, self.cap = "top", cfg.cap_top
            self.allow_override = False
        else:
            self.stage, self.cap = "main", 1.0
            self.allow_override = False
        self.history.append((date, self.stage, self.cap, fastpath, panic_event,
                             self._idx_ext_active))

    def snapshot(self) -> Dict:
        """当日快照（落盘 / 日报 / 开仓裁决共用口径）。"""
        i = len(self.idx["dates"]) - 1
        return {
            "stage": self.stage,
            "cap": self.cap,
            "allow_override": self.allow_override,
            "panic_pending": self._panic_pending_until >= i,
            "panic_low": None if np.isinf(self._panic_low) else float(self._panic_low),
            "data_date": self.idx["dates"][i],
        }


def run_cycle_stage(index_df: Optional[pd.DataFrame]) -> Optional[Dict]:
    """对整段指数日线回放 CycleStage，返回当日快照；数据不足返回 None。"""
    feats = build_idx_features(index_df)
    if feats is None:
        return None
    gates = regime_series(index_df)
    stage = CycleStage(CYCLE_PARAMS, feats)
    for i, gate in enumerate(gates):
        stage.update(i, gate)
    return stage.snapshot()


def save_cycle_state(snap: Dict) -> None:
    """当日快照落盘 data/cycle_state.json（fail-soft，写入失败仅告警）。"""
    try:
        CYCLE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CYCLE_STATE_FILE.write_text(
            json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"cycle_state 写入失败: {e}")


def load_cycle_state() -> Optional[Dict]:
    """读取 data/cycle_state.json（fail-soft：缺失/损坏返回 None）。"""
    try:
        if CYCLE_STATE_FILE.exists():
            return json.loads(CYCLE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"cycle_state 读取失败: {e}")
    return None


def evaluate_open_gate(can_trade: bool, snap: Optional[Dict],
                       invested: float, equity: float) -> Tuple[bool, float, str]:
    """A1/A2 开仓放行裁决：gate 结论之上叠加快速通道放行与组合档位截断。

    Args:
        can_trade: 既有 gate 放行结论（语义不动，这里只做旁路叠加）
        snap: run_cycle_stage 的当日快照（None = 数据不足，旁路不生效）
        invested/equity: 组合持仓市值与总资产（元；equity≤0 视为取不到，跳过截断）

    Returns:
        (allow, cap, note) — 是否放行新开仓、组合仓位上限（0~1）、人读裁决说明
    """
    if snap is None:
        return can_trade, 1.0, ""
    cap = float(snap.get("cap", 1.0) or 1.0)
    allow = can_trade
    notes: List[str] = []

    # A2 快速通道：gate 禁开仓但恐慌底收复 MA10 → 当日放行，cap 已折叠为 cap_bottom
    if not can_trade and snap.get("allow_override"):
        allow = True
        notes.append("A2 快速通道放行（恐慌底登记后收复MA10，仅当日有效）")

    # A1/A2 组合敞口截断：invested 已达档位上限 → 不再放行新开仓
    if allow and equity > 0 and invested >= cap * equity:
        allow = False
        notes.append(
            f"组合敞口 {invested:.0f} 元已达档位上限（{cap:.0%} × 权益 {equity:.0f} 元），截断新开仓"
        )

    note = "；".join(notes)
    return allow, cap, note
