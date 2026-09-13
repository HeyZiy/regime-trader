# -*- coding: utf-8 -*-
"""裁决性实验：入场形态 × regime 门控 × MA10 止损

四臂（信号 1 触发，T 收盘基准）：
  A_hold     : T 收盘买，死拿 T+16（参照）
  A_ma10     : T 收盘买，期间收盘 < MA10 即当日收盘卖出（近似 14:45 执行），否则 T+16 到期
  B_ma10     : 现行纪律——T+1 确认通过才按 T+1 收盘买，之后同 A_ma10（自 T+2 起判 MA10）
  C_ma10     : T 收盘买 + T+1 不确认即 T+1 收盘卖，确认则转 A_ma10
门控：all = 全部触发；gate = 仅触发日上证 regime == trending_up。
各臂报：每笔均值/中位/胜率、平均持有天数、合计收益（每笔 1 单位不复利）、
以及趋势段内 A_ma10 vs B_ma10 的配对差（同信号同 gate 的终极对比）。

用法：python bt_exit_regime.py
"""
import json
import os
import statistics
import sys

import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "research", "trend_bt", "bs")
os.chdir(BASE)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)))))

from bt_entry_bs import load_frames, load_bench, detect_signal1, HORIZONS, CONFIRM_VOL_RATIO, SIG_START, SIG_END, DEDUP_DAYS, stat_block

MAX_HOLD = 16


def sim_ma10(df, i_entry, i0):
    """自 i0+1 起判"连续 2 天收盘 < MA10"（第二卖点口径），第 2 天收盘出；
    否则 MAX_HOLD 到期。返回 (收益率, 持有交易日数, 是否止损出)。"""
    end = min(i_entry + MAX_HOLD, len(df) - 1)
    below_streak = 0
    for j in range(i0 + 1, end + 1):
        below = pd.notna(df.iloc[j]["ma10"]) and df.iloc[j]["close"] < df.iloc[j]["ma10"]
        below_streak = below_streak + 1 if below else 0
        if below_streak >= 2:
            return df.iloc[j]["close"] / df.iloc[i_entry]["close"] - 1, j - i_entry, True
    return df.iloc[end]["close"] / df.iloc[i_entry]["close"] - 1, end - i_entry, False


def main():
    frames = load_frames()
    bench = load_bench()
    regs = bench["regime"]

    arms = {}
    pair_ab = []  # 趋势段内 B_ma10 - A_ma10 配对差

    for code, df in frames.items():
        last_trig = -99
        for i in range(60, len(df)):
            d = str(df.iloc[i]["date"])[:10]
            if not (SIG_START <= d <= SIG_END) or not detect_signal1(df, i):
                continue
            if i - last_trig < DEDUP_DAYS:
                continue
            last_trig = i
            if i + 1 >= len(df):
                continue
            r0, r1 = df.iloc[i], df.iloc[i + 1]
            if str(r1["tradestatus"]) != "1":
                continue
            reg = regs.get(d, "chaos")
            vol_ratio = r1.volume / r1.vol_ma5 if pd.notna(r1.vol_ma5) and r1.vol_ma5 > 0 else 99
            confirmed = (r1.close >= r1.ma5 * 0.995) and (vol_ratio < CONFIRM_VOL_RATIO)

            ret_hold, _, _ = sim_ma10(df, i, i + MAX_HOLD + 1)  # 不判止损的持满版
            # A_hold：持满（重算，绕开止损）
            end = min(i + MAX_HOLD, len(df) - 1)
            ret_hold = df.iloc[end]["close"] / r0["close"] - 1
            ret_a, hold_a, stop_a = sim_ma10(df, i, i)

            for gate in ("all", "gate"):
                if gate == "gate" and reg != "trending_up":
                    continue
                arms.setdefault((gate, "A_hold"), []).append((ret_hold, MAX_HOLD, reg, False))
                arms.setdefault((gate, "A_ma10"), []).append((ret_a, hold_a, reg, stop_a))

            if confirmed:
                ret_b, hold_b, stop_b = sim_ma10(df, i + 1, i + 1)
                ret_c = ret_a  # 确认通过时 C 与 A 同路径
                for gate in ("all", "gate"):
                    if gate == "gate" and reg != "trending_up":
                        continue
                    arms.setdefault((gate, "B_ma10"), []).append((ret_b, hold_b, reg, stop_b))
                    arms.setdefault((gate, "C_ma10"), []).append((ret_c, hold_a, reg, stop_a))
                if reg == "trending_up":
                    pair_ab.append(ret_b - ret_a)
            else:
                # C 不确认：T+1 收盘离场
                ret_c = r1.close / r0["close"] - 1
                for gate in ("all", "gate"):
                    if gate == "gate" and reg != "trending_up":
                        continue
                    arms.setdefault((gate, "C_ma10"), []).append((ret_c, 1, reg, True))

    def report(gate):
        print(f"\n══ 门控 = {gate} ══")
        rows = {}
        for name in ("A_hold", "A_ma10", "B_ma10", "C_ma10"):
            v = arms.get((gate, name), [])
            if not v:
                continue
            rets = [x[0] for x in v]
            holds = [x[1] for x in v]
            stops = sum(1 for x in v if x[3]) / len(v) * 100
            sb = stat_block(rets)
            rows[name] = sb
            print(f"  {name:<8} n={len(v):<6} 均值{sb.get('mean', 0):+.2f}% 中位{sb.get('median', 0):+.2f}% "
                  f"胜率{sb.get('win', 0)}% P10{sb.get('p10', 0):+.1f} 合计{sum(rets)*100:+.0f}pp "
                  f"均持{statistics.mean(holds):.1f}天 止损率{stops:.0f}%")
        return rows

    rows_all = report("all")
    rows_gate = report("gate")

    n_all = len(arms.get(("all", "A_ma10"), []))
    n_gate = len(arms.get(("gate", "A_ma10"), []))
    print(f"\n门控利用率：{n_gate}/{n_all} = {n_gate/max(1,n_all)*100:.0f}% 的触发落在 trending_up")
    if pair_ab:
        print(f"趋势段内 B_ma10 − A_ma10 配对差（同信号同门控的终极对比）: {stat_block(pair_ab)}")

    json.dump({f"{g}_{n}": v for (g, n), v in arms.items()},
              open("bt_exit_regime_results.json", "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
