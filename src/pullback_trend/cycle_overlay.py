# -*- coding: utf-8 -*-
"""
===================================
趋势回踩的 Cycle 风险叠加（B1 延伸计数 + C1 ATR 扩张过滤 + D1 MA5 方向门）
===================================

证据存档：research/cycle_absorption/。指数级组件（A1 仓位档位 + A2 快速通道）
在 src/market_state/cycle_stage.py；五组件作为整体使用，勿单独启停。

组件语义（全部只消费日线收盘后可得字段）：
  B1  持仓级延伸计数：bias_MA10 ≥ ext_stk_thr(10%) 计一次延伸事件——第 2 次减半、
      第 3 次清仓；事件间 bias 回落 < ext_stk_reset(4%) 才重置。卖出侧的主力止盈。
  C1  ATR5/ATR20 > atr_exp_thr(1.3) 视为波动扩张日，剔除当日信号。
      TR 取简单滚动均值口径。
  D1  MA5 较昨日向下 → 剔除当日信号（零参数）。

状态与生命周期：
- B1 不自持状态单例，直接寄生 data/position_exit_state.json 的每仓 dict
  （与 peak 同生命周期）：新仓由 trend_sell 的 setdefault 创建，
  清仓后由既有 gone-codes 清理移除，不引入新的状态文件。
- C1/D1 为当日 bar 的纯函数判定，无状态。
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from src.indicators import atr
from src.market_state.cycle_stage import CYCLE_PARAMS


class ExhaustionTracker:
    """B1 个股延伸计数。

    状态寄生在调用方（trend_sell）持有的每仓 dict 上，就地追加
    ext_episodes / ext_in_episode / ext_day 字段——与 peak/entry_date 等
    既有字段共存，只增不改。
    """

    def update(self, st: Dict, bias10: Optional[float], date_str: str
               ) -> Optional[Tuple[str, str]]:
        """对单仓更新延伸计数，返回 (action, reason) 或 None。

        Args:
            st: position_exit_state.json 中该仓的 dict（就地变更并随后续 save 落盘）
            bias10: 当日收盘对 MA10 的乖离（%）；NaN/None 跳过（停牌、数据缺失）
            date_str: 当日交易日（YYYY-MM-DD），同一日重复运行不重复计数
        """
        if bias10 is None or pd.isna(bias10):
            return None
        if st.get("ext_day") == date_str:
            return None  # 今日已计（14:45 任务当日重跑不重复触发）

        episodes = int(st.get("ext_episodes", 0) or 0)
        in_episode = bool(st.get("ext_in_episode", False))

        if in_episode:
            # 事件进行中：bias 回落 < reset 才解除，之后再次延伸才计新事件
            if bias10 < CYCLE_PARAMS.ext_stk_reset:
                st["ext_in_episode"] = False
        elif bias10 >= CYCLE_PARAMS.ext_stk_thr:
            episodes += 1
            st["ext_in_episode"] = True
            st["ext_episodes"] = episodes
            st["ext_day"] = date_str
            if episodes >= 3:
                return ("clear",
                        f"B1 延伸计数：第{episodes}次偏离MA10≥{CYCLE_PARAMS.ext_stk_thr:g}%，清仓")
            if episodes == 2:
                return ("reduce_half",
                        f"B1 延伸计数：第2次偏离MA10≥{CYCLE_PARAMS.ext_stk_thr:g}%，减仓50%")
        return None


def evaluate_c1(df: pd.DataFrame) -> Optional[str]:
    """C1 条件判定：ATR 扩张返回原因，否则 None。

    NaN（数据不足）或 ATR20=0 时放行（fail-open，与探索包一致）。
    """
    if df is None or len(df) < 21:
        return None  # ATR20 需 20 根 TR → 21 根 bar
    a5 = atr(df, 5).iloc[-1]
    a20 = atr(df, 20).iloc[-1]
    if pd.isna(a5) or pd.isna(a20) or a20 == 0:
        return None
    if a5 / a20 > CYCLE_PARAMS.atr_exp_thr:
        return f"ATR5/ATR20={a5 / a20:.2f}>{CYCLE_PARAMS.atr_exp_thr:g}（波动扩张日）"
    return None


def evaluate_d1(df: pd.DataFrame) -> Optional[str]:
    """D1 条件判定：MA5 较昨日向下返回原因，否则 None。

    MA5 或前值缺失（NaN）时放行（fail-open）。
    """
    if df is None or len(df) < 2 or "ma5" not in df.columns:
        return None
    m5 = df["ma5"].iloc[-1]
    m5_prev = df["ma5"].iloc[-2]
    if pd.isna(m5) or pd.isna(m5_prev):
        return None
    if float(m5) < float(m5_prev):
        return f"MA5 向下（{m5_prev:.2f}→{m5:.2f}）"
    return None


def signal_filter_reason(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """C1/D1 后置过滤判定。

    返回首个命中 (rule_id, reason)；未命中返回 None。信号 1/3 与信号 2 共用
    同一根 bar，一次判定两分支同口径（信号 2 的"次日弱转强确认"是次日重新检测，
    走同一判定）。
    """
    reason = evaluate_c1(df)
    if reason:
        return ("C1", reason)
    reason = evaluate_d1(df)
    if reason:
        return ("D1", reason)
    return None
