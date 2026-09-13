# -*- coding: utf-8 -*-
"""市场状态 5 态分解：weak_up / sideways 到底有没有超额（裁决"完全不做 vs 收紧做"）

按线上 market_gate.diagnose_regime 的判定结构对指数日线逐日定状态：
  trending_down — 空头排列 + 收盘 < MA10
  trending_up   — 多头排列 + 收盘 > MA10（线上还要求 met_count≥2，此处近似忽略）
  sideways      — 收盘偏离 MA20 < 1.5%
  weak_up       — 收盘 > MA20，非以上（线上还要求 met_count≥2，近似忽略）
  chaos         — 其余（收盘 < MA20 且非空头排列）

信号口径与 bt_entry_bs 相同（detect_signal1，含多头排列），T 收盘买死拿 T+16。
各状态报：n / 均值 / 中位 / 胜率 / 超额（vs 上证同期）/ 位置分层（0-30% vs 30%+，
对应线上收紧档"位置 <50%/65%"是否有数据支撑）。

用法：python bt_regime_states.py
"""
import os
import statistics
import sys

import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "research", "trend_bt", "bs")
os.chdir(BASE)
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))

from bt_entry_bs import load_frames, load_bench, detect_signal1, SIG_START, SIG_END, DEDUP_DAYS

MAX_HOLD = 16
SIDEWAYS_BIAS = 0.015  # 与 market_gate.SIDEWAYS_BIAS 同值


def state_map(df):
    """按 market_gate 结构逐日定 5 态（met_count 不可得，trending_up/weak_up 忽略该项）。"""
    ma5 = df["close"].rolling(5).mean()
    ma10 = df["close"].rolling(10).mean()
    ma20 = df["close"].rolling(20).mean()
    out = {}
    for i in range(len(df)):
        c, m5, m10, m20 = df["close"].iloc[i], ma5.iloc[i], ma10.iloc[i], ma20.iloc[i]
        d = str(df["date"].iloc[i])[:10]
        if pd.isna(m20):
            out[d] = "chaos"
        elif m5 < m10 < m20 and c < m10:
            out[d] = "trending_down"
        elif m5 > m10 > m20 and c > m10:
            out[d] = "trending_up"
        elif abs(c - m20) / m20 < SIDEWAYS_BIAS:
            out[d] = "sideways"
        elif c > m20:
            out[d] = "weak_up"
        else:
            out[d] = "chaos"
    return out


def blk(rets, exs):
    if not rets:
        return "n=0"
    s = sorted(rets)
    n = len(rets)
    line = (f"n={n:<5} 均值 {sum(rets)/n*100:+.2f}%  中位 {s[n//2]*100:+.2f}%  "
            f"胜率 {sum(1 for v in rets if v > 0)/n*100:.0f}%")
    if exs:
        line += f"  超额均值 {sum(exs)/len(exs)*100:+.2f}%"
    return line


def main():
    frames = load_frames()
    bench = load_bench()
    bmap = bench["上证"]
    smap = state_map(bench and pd.read_csv("index_sh.000001.csv"))
    print(f"股票 {len(frames)} 只；指数状态构成：")
    for st, n in pd.Series(list(smap.values())).value_counts().items():
        print(f"  {st:<14} {n}")

    by_state = {}
    for code, df in frames.items():
        last_trig = -99
        for i in range(60, len(df)):
            d = str(df.iloc[i]["date"])[:10]
            if not (SIG_START <= d <= SIG_END) or not detect_signal1(df, i):
                continue
            if i - last_trig < DEDUP_DAYS:
                continue
            last_trig = i
            if i + 1 >= len(df) or str(df.iloc[i + 1]["tradestatus"]) != "1":
                continue
            end = min(i + MAX_HOLD, len(df) - 1)
            r = df.iloc[i]
            ret = df.iloc[end]["close"] / r["close"] - 1
            ex = None
            if d in bmap and str(df.iloc[end]["date"])[:10] in bmap:
                ex = ret - (bmap[str(df.iloc[end]["date"])[:10]] / bmap[d] - 1)
            low60 = df["close"].iloc[max(0, i - 59):i + 1].min()
            pos = (r["close"] - low60) / low60 * 100 if low60 > 0 else 99.0
            st = smap.get(d, "chaos")
            cell = by_state.setdefault(st, {"ret": [], "ex": [], "lo": [], "lo_ex": [], "hi": []})
            cell["ret"].append(ret)
            if ex is not None:
                cell["ex"].append(ex)
            (cell["lo"] if pos < 30 else cell["hi"]).append(ret)
            if ex is not None and pos < 30:
                cell["lo_ex"].append(ex)

    print(f"\n=== 市场状态 × 信号1（T收盘死拿T+16，含排列口径）===")
    for st in ("trending_up", "weak_up", "sideways", "trending_down", "chaos"):
        c = by_state.get(st)
        if not c:
            continue
        print(f"\n{st}:  {blk(c['ret'], c['ex'])}")
        print(f"  位置<30%:  {blk(c['lo'], c['lo_ex'])}")
        print(f"  位置≥30%:  {blk(c['hi'], [])}")


if __name__ == "__main__":
    main()
