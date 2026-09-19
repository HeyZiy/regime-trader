# -*- coding: utf-8 -*-
"""CycleStage（A1 仓位档位 + A2 恐慌底快速通道）组件单测。

合成 bar 依据 docs/07_integration_patch.md §6.2 回归清单；
判定语义对拍对象：探索包 cycle_overlay.CycleStage（research/cycle_absorption/）。
"""
import numpy as np
import pandas as pd
import pytest

from src.market_state.cycle_stage import (
    CYCLE_PARAMS, CycleConfig, CycleStage, build_idx_features,
    evaluate_open_gate, regime_series, run_cycle_stage,
)
from src.market_state.market_gate import diagnose_regime


def make_index_df(closes, volumes=None, opens=None, lows=None, highs=None,
                  start="2024-01-01"):
    """合成指数日线：默认 open=前收、high/low=开盘收盘外扩 0.1、量恒定，逐项可覆盖。"""
    n = len(closes)
    volumes = volumes if volumes is not None else [1_000_000.0] * n
    opens = opens if opens is not None else [closes[0]] + list(closes[:-1])
    lows = lows if lows is not None else [min(o, c) - 0.1 for o, c in zip(opens, closes)]
    highs = highs if highs is not None else [max(o, c) + 0.1 for o, c in zip(opens, closes)]
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n),
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


def fake_idx(bias_seq, closes=None, pct_chg=None):
    """直接构造 CycleStage 消费的特征 dict（绕开真实行情，单测状态机用）。"""
    n = len(bias_seq)
    closes = closes if closes is not None else [100.0] * n
    return {
        "dates": [f"d{i:03d}" for i in range(n)],
        "open": [c - 0.1 for c in closes],
        "high": [c + 0.2 for c in closes],
        "low": [c - 0.2 for c in closes],
        "close": closes,
        "volume": [1e6] * n,
        "ma10": [c / (1 + b / 100) for c, b in zip(closes, bias_seq)],
        "bias_ma10": list(bias_seq),
        "vol20p": [1e6] * n,
        "pct_chg": pct_chg if pct_chg is not None else [1.0] * n,
    }


# ==================== 阈值单一来源（防参数漂移）====================

def test_validated_thresholds_pinned():
    """探索包回测验证过的默认阈值——改动须同步 strategy 文档与证据重跑。"""
    assert CYCLE_PARAMS.ext_idx_thr == 4.5 and CYCLE_PARAMS.ext_idx_reset == 0.5
    assert CYCLE_PARAMS.cap_top == 0.30 and CYCLE_PARAMS.cap_bottom == 0.40
    assert CYCLE_PARAMS.panic_thr == -4.0 and CYCLE_PARAMS.panic_vol_mult == 1.5
    assert CYCLE_PARAMS.fp_window == 10
    assert CYCLE_PARAMS.ext_stk_thr == 10.0 and CYCLE_PARAMS.ext_stk_reset == 4.0
    assert CYCLE_PARAMS.atr_exp_thr == 1.3


# ==================== A1 顶部延伸（滞回）====================

def test_a1_top_extension_end_to_end():
    """25 天平盘 + 连续上拉 → bias_MA10 ≥ 4.5% → stage=top、cap=cap_top。"""
    closes = [100.0] * 25 + [round(100 * 1.02 ** k, 2) for k in range(1, 16)]
    snap = run_cycle_stage(make_index_df(closes))
    assert snap["stage"] == "top"
    assert snap["cap"] == pytest.approx(CYCLE_PARAMS.cap_top)


def test_a1_hysteresis_state_machine():
    """滞回：bias 5.0 进入(≥4.5) → 3.0 维持 → 2.0(≤2.25) 解除，最终回主升。"""
    bias = [0.0] * 30 + [5.0] * 5 + [3.0] * 3 + [2.0] * 5
    stage = CycleStage(CYCLE_PARAMS, fake_idx(bias))
    for i in range(len(bias)):
        stage.update(i, "trending_up")
    assert stage._idx_ext_active is False
    assert stage.stage == "main"
    assert stage.cap == 1.0
    assert any(row[1] == "top" for row in stage.history)  # 中途确曾进入顶部档


# ==================== A2 恐慌底登记 + 快速通道 ====================

def _panic_setup():
    """25 天平盘 → 恐慌日（gap-down 阳线，bias −5.4%、量 2x）→ 收复日（99.5）。"""
    closes = [100.0] * 25 + [94.0, 99.5]
    opens = [100.0] * 25 + [93.5, 94.0]
    lows = [99.9] * 25 + [93.2, 93.9]
    highs = [100.1] * 25 + [94.5, 99.8]
    volumes = [1_000_000.0] * 25 + [2_000_000.0, 1_000_000.0]
    return closes, opens, lows, highs, volumes


def test_a2_panic_fastpath_one_day():
    """恐慌底登记后次日收复 MA10 且收阳 → 放行恰一日（gate=sideways 禁开）。"""
    closes, opens, lows, highs, volumes = _panic_setup()
    df = make_index_df(closes, volumes=volumes, opens=opens, lows=lows, highs=highs)
    snap = run_cycle_stage(df)
    # 最后一根 bar 是收复日：快速通道放行
    assert snap["allow_override"] is True
    assert snap["stage"] == "bottom"
    assert snap["cap"] == pytest.approx(CYCLE_PARAMS.cap_bottom)
    assert snap["panic_low"] == pytest.approx(94.0)

    # 快速通道一次性消费：再走一天平盘 → 回到禁开常态，不再放行
    closes3 = closes + [99.5]
    opens3 = opens + [99.5]
    lows3 = lows + [99.4]
    highs3 = highs + [99.6]
    df3 = make_index_df(closes3, volumes=volumes + [1_000_000.0],
                        opens=opens3, lows=lows3, highs=highs3)
    snap3 = run_cycle_stage(df3)
    assert snap3["allow_override"] is False
    assert snap3["stage"] == "range"


def test_a2_new_low_updates_panic_low():
    """窗口内创新低：恐慌低点跟随下移、登记保持（对齐探索包已验证语义）。

    探索包实现中窗口内每日以 min 更新 panic_low——新低不销毁登记，只是抬高
    "收复"的判定基准（收复收盘必须高于窗口真实最低收盘）。
    """
    closes = [100.0] * 25 + [94.0, 93.9, 99.5]
    opens = [100.0] * 25 + [93.5, 94.0, 93.9]
    lows = [99.9] * 25 + [93.2, 93.5, 93.8]
    highs = [100.1] * 25 + [94.5, 94.2, 99.8]
    volumes = [1_000_000.0] * 25 + [2_000_000.0, 1_000_000.0, 1_000_000.0]
    df = make_index_df(closes, volumes=volumes, opens=opens, lows=lows, highs=highs)
    snap = run_cycle_stage(df)
    # 末日收复 MA10（99.5 ≥ 98.74）+ 收阳 + 高于窗口真实最低收盘（93.9）→ 快速通道放行
    assert snap["allow_override"] is True
    assert snap["stage"] == "bottom"
    assert snap["panic_low"] == pytest.approx(93.9)   # 低点已跟随下移，而非停留在首日 94.0


def test_a2_requires_reversal_and_volume():
    """深偏离但缩量/无反转K → 不登记。恐慌日改阴线且缩量。"""
    closes = [100.0] * 25 + [94.0, 99.5]
    opens = [100.0] * 25 + [94.5, 94.0]   # 阴线（c<o）
    volumes = [1_000_000.0] * 25 + [1_000_000.0, 1_000_000.0]  # 量不足 1.5x
    df = make_index_df(closes, volumes=volumes, opens=opens)
    snap = run_cycle_stage(df)
    assert snap["allow_override"] is False


# ==================== regime_series 与 diagnose_regime 对拍 ====================

def test_regime_series_matches_diagnose_regime():
    """逐日对拍：regime_series 必须与 market_gate.diagnose_regime 逐 bar 一致。

    diagnose_regime 是市场状态判定的单一事实来源；此处漂移会导致 A2 快速通道
    在错误的日期触发（单一来源约定见 cycle_stage.regime_series docstring）。
    """
    rng = np.random.default_rng(7)
    closes = [round(float(c), 2) for c in 100 * np.cumprod(1 + rng.normal(0, 0.02, 60))]
    df = make_index_df(closes)
    series = regime_series(df)
    assert len(series) == len(df)
    for i in range(20, len(df)):
        assert series[i] == diagnose_regime(df.iloc[: i + 1]).regime


# ==================== 特征构建与开仓裁决 ====================

def test_build_idx_features_insufficient_data():
    assert build_idx_features(None) is None
    assert build_idx_features(pd.DataFrame({"close": [1.0] * 10})) is None


def test_run_cycle_stage_insufficient_data_returns_none():
    """指数数据不足 → 快照 None（入口据此跳过档位旁路，gate 行为不受影响）。"""
    assert run_cycle_stage(None) is None
    assert run_cycle_stage(make_index_df([100.0] * 10)) is None


def test_evaluate_open_gate():
    snap_top = {"stage": "top", "cap": 0.30, "allow_override": False,
                "data_date": "2026-09-19"}
    snap_bottom = {"stage": "bottom", "cap": 0.40, "allow_override": True,
                   "data_date": "2026-09-19"}

    # 顶部档位：敞口已达标位上限 → 截断新开仓
    allow, cap, note = evaluate_open_gate(True, snap_top, 3500.0, 10000.0)
    assert allow is False and cap == pytest.approx(0.30) and "截断" in note
    # 未达上限 → 放行
    allow, cap, note = evaluate_open_gate(True, snap_top, 2000.0, 10000.0)
    assert allow is True and cap == pytest.approx(0.30) and note == ""
    # A2 快速通道：gate 禁开但 override 放行
    allow, cap, note = evaluate_open_gate(False, snap_bottom, 1000.0, 10000.0)
    assert allow is True and cap == pytest.approx(0.40) and "快速通道" in note
    # 权益取不到（0）→ 不截断（fail-open）
    allow, _, _ = evaluate_open_gate(True, snap_top, 999_999.0, 0.0)
    assert allow is True
    # 快照缺失（指数数据不足）→ 旁路不生效
    assert evaluate_open_gate(False, None, 5000.0, 10000.0) == (False, 1.0, "")
