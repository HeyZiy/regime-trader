# -*- coding: utf-8 -*-
"""veto 负面清单开箱：V2/V3/V4/V6/V8/V9 有没有选别力（裁决"左尾保护放哪层"）

口径：信号 1 触发（与 bt_entry_bs 同口径，V7 的 80% 上限已含在其中），
仅 trending_up 门控，T 收盘买死拿 T+16（退出规则无关化，纯检验准入选择力）。
 veto 判定与 src/trend/veto_rules.py 同阈值，逐日在信号日收盘数据上计算（无前视）：
  V2 近60日累计涨幅 > 100%          V6 近20日 |涨跌幅|≥9.5% 天数 ≥ 3
  V3 近20日换手均值 > 12%           V8 反弹逼近前高（≥3日前高点，距 <10%，曾深跌 ≥15%）
  V4 近20日 ≥2 次单日跌幅 > 7%      V9 近20日日收益率 std > 5%
  V1（公告）/ V5（主力资金）依赖妙想，不可测；V7 已含在基线，另做阈值收紧扫描。

各规则报：否决率 / 被否决组 vs 通过组的死拿均值·中位·胜率 / 剔除该组后的整体均值。
判据：被否决组显著更差 → 该规则有选别力，左尾保护放准入端成立；
      两组相当 → 该规则与门控/位置重复，无独立存在资格。

用法：python bt_veto_scan.py
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
# 阈值与 veto_rules.py 一致
V2_GAIN_60D_MAX = 100.0
V3_TURNOVER_20D_MAX = 12.0
V4_BIG_DROP_PCT = -7.0
V4_MIN_COUNT = 2
V6_LIMIT_DAYS = 3
V6_THRESHOLD = 9.5  # 主板非 ST
V7_FROM_LOW_MAX = 80.0
V8_NEAR_HIGH_PCT = 10.0
V8_DEPTH_RATIO = 0.85
V8_HIGH_NOT_RECENT = 3
V9_VOL_20D_MAX = 5.0


def veto_flags(df, i):
    """在信号日 i 的收盘数据上计算各 veto（与 veto_rules.py 同口径），返回 dict[bool]。"""
    n = i + 1
    closes = df["close"].iloc[:n].astype(float)
    pct = closes.pct_change() * 100
    flags = {}

    # V3 近20日换手均值
    tr = pd.to_numeric(df["turn"].iloc[max(0, i - 19):i + 1], errors="coerce")
    tr_mean = tr.mean()
    flags["V3"] = pd.notna(tr_mean) and tr_mean > V3_TURNOVER_20D_MAX

    # V4 近20日 ≥2 次单日跌幅 > 7%
    if n >= 20:
        flags["V4"] = int((pct.iloc[-20:] <= V4_BIG_DROP_PCT).sum()) >= V4_MIN_COUNT
    else:
        flags["V4"] = False

    # V6 近20日涨跌停 ≥ 3 天
    if n >= 21:
        flags["V6"] = int((pct.iloc[-20:].abs() >= V6_THRESHOLD).sum()) >= V6_LIMIT_DAYS
    else:
        flags["V6"] = False

    # V9 近20日收益率 std
    if n >= 20:
        vol = float(pct.iloc[-20:].std())
        flags["V9"] = pd.notna(vol) and vol > V9_VOL_20D_MAX
    else:
        flags["V9"] = False

    if n >= 61:
        window = closes.iloc[-60:]
        last = float(closes.iloc[-1])
        high60, low60 = float(window.max()), float(window.min())
        base = float(closes.iloc[-61])

        # V2 近60日累计涨幅
        flags["V2"] = base > 0 and (last - base) / base * 100 > V2_GAIN_60D_MAX

        # V8 反弹逼近前高
        bars_from_high = int(len(window) - 1 - int(window.values.argmax()))
        dist = (high60 - last) / high60 * 100
        flags["V8"] = (bars_from_high >= V8_HIGH_NOT_RECENT
                       and 0 <= dist < V8_NEAR_HIGH_PCT
                       and low60 <= high60 * V8_DEPTH_RATIO)

        # V7 距 60 日低点涨幅（基线已含 80% 否决，此处记录数值供收紧扫描）
        flags["_pos"] = (last - low60) / low60 * 100 if low60 > 0 else 999.0
    else:
        flags["V2"] = flags["V8"] = False
        flags["_pos"] = 999.0

    return flags


def stats(rets):
    if not rets:
        return "n=0"
    s = sorted(rets)
    n = len(rets)
    return (f"n={n:<5} 均值 {sum(rets)/n*100:+.2f}%  中位 {s[n//2]*100:+.2f}%  "
            f"胜率 {sum(1 for v in rets if v > 0)/n*100:.0f}%")


def main():
    frames = load_frames()
    regs = load_bench()["regime"]

    trades = []  # (ret16, flags)
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
            if regs.get(d) != "trending_up":
                continue
            end = min(i + MAX_HOLD, len(df) - 1)
            ret = df.iloc[end]["close"] / df.iloc[i]["close"] - 1
            trades.append((ret, veto_flags(df, i)))

    n = len(trades)
    all_rets = [t[0] for t in trades]
    base_mean = statistics.mean(all_rets) * 100
    print(f"门控内交易 {n} 笔（死拿 T+16，基线均值 {base_mean:+.2f}%）\n")

    print("── 单规则选别力（被否决组差 → 有选别力）──")
    print(f"  {'规则':<6}{'否决率':>7}   被否决组{'':<26} 通过组{'':<28} 剔除后整体")
    for rule in ("V2", "V3", "V4", "V6", "V8", "V9"):
        hit = [t[0] for t in trades if t[1][rule]]
        passed = [t[0] for t in trades if not t[1][rule]]
        after = statistics.mean(passed) * 100 if passed else 0
        print(f"  {rule:<6}{len(hit)/n*100:>6.1f}%   {stats(hit):<34} {stats(passed):<34} {after:+.2f}% ({after-base_mean:+.2f}pp)")

    # 组合：任一触发即否决
    any_hit = [t for t in trades if any(t[1][r] for r in ("V2", "V3", "V4", "V6", "V8", "V9"))]
    passed = [t for t in trades if not any(t[1][r] for r in ("V2", "V3", "V4", "V6", "V8", "V9"))]
    print(f"\n── 组合（6 条任一触发即否决）──")
    print(f"  被否决组 : {stats([t[0] for t in any_hit])}")
    print(f"  通过组   : {stats([t[0] for t in passed])}")

    # V7 阈值收紧扫描（基线已含 80%）
    print("\n── V7 位置上限收紧扫描（基线已含 >80% 否决）──")
    for cap in (80, 60, 50, 40, 30):
        kept = [t[0] for t in trades if t[1]["_pos"] <= cap]
        print(f"  上限 {cap:>3}%: {stats(kept)}")


if __name__ == "__main__":
    main()
