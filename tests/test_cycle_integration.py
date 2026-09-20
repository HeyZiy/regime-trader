# -*- coding: utf-8 -*-
"""Cycle 吸收接线层集成测试。

覆盖三类语义：
1. 基线：常规行情下信号/卖出判定照常产出（C1/D1/B1 在平静 bar 上不误伤）；
2. B1 ext_action 并入卖出动作池后取最强（clear > reduce_half）；
3. C1 开启态（即默认态）命中 ATR 扩张的当日信号被剔除；
4. 日报展示：市场环境节渲染 Cycle 阶段/上限/截断说明。
"""
import pandas as pd
import pytest

import src.pullback_trend.signal_detector as sd
from src.pullback_trend.cycle_overlay import ExhaustionTracker
from src.pullback_trend.report import generate_technical_report
from src.pullback_trend.sell_rules import detect_sell_signals


def _stock_df(closes, *, amplitudes=None, volumes=None):
    """合成个股日线：amplitudes 为每日 (high-low)/close 比例（默认 1%）。

    初始 open=昨收，保证 pct_change 只由 close 序列决定。
    """
    n = len(closes)
    amplitudes = amplitudes if amplitudes is not None else [0.01] * n
    volumes = volumes if volumes is not None else [1e6] * n
    opens = [closes[0]] + list(closes[:-1])
    highs = [max(o, c) * (1 + a) for o, c, a in zip(opens, closes, amplitudes)]
    lows = [min(o, c) * (1 - a) for o, c, a in zip(opens, closes, amplitudes)]
    df = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=n),
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes,
    })
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    return df


def _signal_df():
    """能触发缩量回踩 MA5 信号且不被 C1/D1 误伤的合成日线。

    26 根 bar：25 天缓涨（+0.4%/日）+ 末日 −0.4% 小幅收跌贴近 MA5。
    末日 MA5 仍高于昨日（一次小幅回踩不逆转 5 日均线），D1 放行；
    振幅恒定 1%，C1 放行。
    """
    closes = [round(100 * 1.004 ** i, 3) for i in range(25)]
    closes.append(round(closes[-1] * 0.996, 3))   # −0.4%：回踩形态，bias 仍在带内
    return _stock_df(closes)


def _wild_signal_df():
    """末日能出信号、但近 4 日大振幅（C1 命中）的合成日线。"""
    closes = [round(100 * 1.004 ** i, 3) for i in range(25)]
    all_closes = closes[:-4] + closes[-4:] + [round(closes[-1] * 0.996, 3)]
    amps = [0.01] * 25 + [0.05, 0.05, 0.05, 0.05, 0.01]
    return _stock_df(all_closes, amplitudes=amps)


def _position():
    return {"code": "600519", "name": "贵州茅台", "count": 200,
            "avail_count": 200, "cost_price": 100.0, "current_price": 105.0,
            "profit_pct": 5.0}


# ==================== 基线：平静 bar 上组件不误伤 ====================

def test_signal_detection_baseline():
    """常规缓涨+回踩 → 信号照常产出（C1/D1 在平静 bar 上放行）。"""
    signals = sd.detect_pullback_signals("600519", "贵州茅台", _signal_df())
    assert len(signals) == 1 and signals[0].signal_type == "pullback_ma5"


def test_sell_detection_baseline_no_ext_action():
    """无规则触发且 B1 无动作 → 无卖出信号（与原签名行为一致）。"""
    df = _stock_df([100 * 1.004 ** i for i in range(25)] + [100 * 1.004 ** 25 * 1.002])
    assert detect_sell_signals("600519", "贵州茅台", df, _position()) is None


# ==================== B1 并入卖出动作池（取最强）====================

def test_b1_ext_action_produces_reduce_half():
    """标准规则全不触发 + B1 减半动作 → reduce_half 信号，理由带 B1 前缀。"""
    df = _stock_df([100 * 1.004 ** i for i in range(25)] + [100 * 1.004 ** 25 * 1.002])
    sig = detect_sell_signals("600519", "贵州茅台", df, _position(),
                              ext_action=("reduce_half", "B1 延伸计数：第2次偏离MA10≥10%，减仓50%"))
    assert sig is not None and sig.action == "reduce_half"
    assert any(r.startswith("B1") for r in sig.reasons)


def test_b1_clear_wins_over_reduce_reasons():
    """B1 clear 与既有 reduce 触发并存 → clear 取最强（既有优先级语义不变）。"""
    # 构造触发"阶段高点回撤≥5%"（reduce）的 df：新高后回落 6%
    closes = [100 * 1.01 ** i for i in range(20)] + [100 * 1.01 ** 19 * 0.94]
    df = _stock_df(closes)
    position = {**_position(), "current_price": 118.0, "profit_pct": 18.0}
    sig = detect_sell_signals("600519", "贵州茅台", df, position,
                              ext_action=("clear", "B1 延伸计数：第3次偏离MA10≥10%，清仓"))
    assert sig is not None and sig.action == "clear"
    assert any(r.startswith("B1") for r in sig.reasons)
    # 既有"取最强"语义：clear 胜出时只返回清仓池理由，reduce 池理由不并列展示
    assert not any("回撤" in r for r in sig.reasons)


def test_b1_tracker_via_trend_sell_state_shape():
    """B1 状态寄生 position_exit_state 每仓 dict：只增 ext_* 字段、保留 peak 等旧字段。"""
    st = {"peak": 110.0, "entry_date": "2026-08-01", "entry_price": 100.0}
    trk = ExhaustionTracker()
    trk.update(st, 11.0, "2026-09-19")
    trk.update(st, 3.0, "2026-09-22")
    act = trk.update(st, 11.5, "2026-09-23")
    assert act is not None and act[0] == "reduce_half"
    assert st["peak"] == 110.0 and st["entry_date"] == "2026-08-01"   # 旧字段原样
    assert st["ext_episodes"] == 2 and st["ext_in_episode"] is True


# ==================== C1 剔除信号（默认即生效）====================

def test_c1_removes_signal_by_default():
    """近 4 日大振幅（ATR 扩张）→ 末日信号被 C1 剔除（无任何开关操作）。"""
    assert sd.detect_pullback_signals("600519", "贵州茅台", _wild_signal_df()) == []


# ==================== 日报展示 ====================

def _report_env(can_trade=True):
    return (can_trade, "summary", "weak_up")


def test_report_without_cycle_info_has_no_cycle_line():
    """cycle_info=None（指数数据不足未产出快照）→ 日报不含 Cycle 行。"""
    text = generate_technical_report([], market_env=_report_env())
    assert "Cycle" not in text


def test_report_renders_stage_and_cap():
    """cycle_info 展示：阶段 + 组合仓位上限。"""
    info = {"stage": "top", "cap": 0.30, "allow_override": False,
            "cap_note": "", "data_date": "2026-09-19"}
    text = generate_technical_report([], market_env=_report_env(), cycle_info=info)
    assert "**Cycle 循环阶段**：顶部延伸" in text
    assert "组合仓位上限 30%" in text
    assert "影子" not in text


def test_report_cycle_fastpath_override_note():
    """A2 快速通道日：展示放行标注与截断说明。"""
    info = {"stage": "bottom", "cap": 0.40, "allow_override": True,
            "cap_note": "组合敞口 4000 元已达档位上限（40% × 权益 10000 元），截断新开仓",
            "data_date": "2026-09-19"}
    text = generate_technical_report([], market_env=_report_env(False), cycle_info=info)
    assert "A2 快速通道放行" in text
    assert "截断新开仓" in text


def test_report_cycle_missing_snapshot_note():
    """指数数据不足（stage 缺失）→ 显式标注快照缺失，不静默。"""
    info = {"cap_note": "", "data_date": "2026-09-19"}
    text = generate_technical_report([], market_env=_report_env(), cycle_info=info)
    assert "快照缺失" in text
