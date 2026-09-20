# -*- coding: utf-8 -*-
"""cycle_overlay（B1 延伸计数 + C1 ATR 扩张过滤 + D1 MA5 方向门）组件单测。

B1 用 docs/07 §6.2 的标准序列：bias 12%→3%→11%→3%→11% → episodes=3 第三次返回 clear。
C1/D1 用合成 bar 验证阈值边界与 NaN 放行（fail-open）。
"""
import pandas as pd
import pytest

from src.market_state.cycle_stage import CYCLE_PARAMS
from src.pullback_trend.cycle_overlay import (
    ExhaustionTracker, evaluate_c1, evaluate_d1, signal_filter_reason,
)


# ==================== B1 延伸计数 ====================

def test_b1_third_extension_clears():
    """标准序列：12%→3%→11%→3%→11% → 第2次减半、第3次清仓。"""
    trk = ExhaustionTracker()
    st = {}
    seq = [12.0, 3.0, 11.0, 3.0, 11.0]
    results = [trk.update(st, b, f"2026-09-{i + 1:02d}") for i, b in enumerate(seq)]
    assert st["ext_episodes"] == 3
    assert results[0] is None                       # 第1次：只计数不出动作
    assert results[1] is None
    assert results[2] is not None and results[2][0] == "reduce_half"   # 第2次：减半
    assert results[3] is None
    assert results[4] is not None and results[4][0] == "clear"         # 第3次：清仓
    assert "第3次" in results[4][1]


def test_b1_first_extension_no_action_but_counts():
    """第 1 次延伸只登记不出动作（与探索包口径一致）。"""
    st = {}
    assert ExhaustionTracker().update(st, 12.0, "2026-09-01") is None
    assert st["ext_episodes"] == 1 and st["ext_in_episode"] is True


def test_b1_reset_requires_fall_below_4pct():
    """bias 在 4~10% 区间徘徊不解除事件，再上 10% 不计新次数。"""
    trk = ExhaustionTracker()
    st = {}
    trk.update(st, 12.0, "d1")          # 第1次
    trk.update(st, 6.0, "d2")           # 未回落 <4%：事件未解除
    act = trk.update(st, 11.0, "d3")    # 仍在进行中 → 不计第2次
    assert act is None and st["ext_episodes"] == 1


def test_b1_bad_data_is_noop():
    """bias 缺失（停牌/数据缺口）→ 跳过且不写字段。"""
    st = {}
    assert ExhaustionTracker().update(st, None, "d1") is None
    assert ExhaustionTracker().update(st, float("nan"), "d1") is None
    assert "ext_episodes" not in st


def test_b1_same_day_idempotent():
    """同一交易日重复运行（当日重跑）不重复计数。"""
    trk = ExhaustionTracker()
    st = {}
    trk.update(st, 12.0, "2026-09-19")
    assert trk.update(st, 12.0, "2026-09-19") is None
    assert st["ext_episodes"] == 1


def test_b1_second_extension_then_partial_reset():
    """减半后回落 <4% 再延伸 → 第3次清仓（跨周期完整链路）。"""
    trk = ExhaustionTracker()
    st = {}
    trk.update(st, 10.5, "d1")                       # 第1次
    trk.update(st, 3.0, "d2")                        # 解除
    act2 = trk.update(st, 10.5, "d3")                # 第2次 → 减半
    assert act2[0] == "reduce_half"
    trk.update(st, 2.0, "d4")                        # 解除
    act3 = trk.update(st, 10.5, "d5")                # 第3次 → 清仓
    assert act3[0] == "clear"


# ==================== C1 ATR 扩张过滤 ====================

def _make_stock_df(closes, *, volumes=None):
    n = len(closes)
    volumes = volumes if volumes is not None else [1e6] * n
    return pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=n),
        "open": [c for c in closes],
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": volumes,
    })


def test_c1_expansion_rejects():
    """末 5 日振幅放大（ATR5/ATR20 > 1.3）→ 命中。"""
    closes = [100.0] * 25 + [101.0, 99.0, 103.0, 97.0, 105.0]
    df = _make_stock_df(closes)
    hit = evaluate_c1(df)
    assert hit is not None and hit.startswith("ATR5/ATR20")
    assert signal_filter_reason(df)[0] == "C1"


def test_c1_calm_passes():
    """振幅恒定 → ATR5/ATR20 ≈ 1.0 → 放行。"""
    df = _make_stock_df([100.0 + (i % 2) * 0.2 for i in range(30)])
    assert evaluate_c1(df) is None


def test_c1_insufficient_bars_and_nan_pass():
    """数据不足（<21 根）放行；不足 20 根 TR 的 NaN 场景放行（fail-open）。"""
    assert evaluate_c1(_make_stock_df([100.0] * 10)) is None
    assert evaluate_c1(None) is None


# ==================== D1 MA5 方向门 ====================

def test_d1_ma5_falling_rejects():
    closes = [100.0 + i for i in range(15)] + [114.0, 111.0, 108.0]  # 冲高回落 → MA5 拐头
    df = _make_stock_df(closes)
    df["ma5"] = df["close"].rolling(5).mean()
    hit = evaluate_d1(df)
    assert hit is not None and "MA5 向下" in hit
    assert signal_filter_reason(df)[0] == "D1"


def test_d1_ma5_rising_passes():
    closes = [100.0 + i * 0.5 for i in range(20)]
    df = _make_stock_df(closes)
    df["ma5"] = df["close"].rolling(5).mean()
    assert evaluate_d1(df) is None


def test_d1_nan_passes():
    df = _make_stock_df([100.0] * 5)
    df["ma5"] = float("nan")
    assert evaluate_d1(df) is None
    assert evaluate_d1(None) is None


# ==================== C1/D1 组合判定 ====================

def test_signal_filter_c1_first_then_d1():
    """两条件同时命中时报 C1（先 ATR 后方向，与探索包过滤顺序一致）。"""
    closes = [100.0] * 25 + [101.0, 99.0, 103.0, 97.0, 95.0]  # 扩张 + 尾部落
    df = _make_stock_df(closes)
    df["ma5"] = df["close"].rolling(5).mean()
    hit = signal_filter_reason(df)
    assert hit is not None and hit[0] == "C1"
    # 阈值来源一致性：过滤阈值取自 CYCLE_PARAMS
    assert f">{CYCLE_PARAMS.atr_exp_thr:g}" in hit[1]
