# -*- coding: utf-8 -*-
"""退出规则诊断 + 参数扫描（裁决"MA10 两日止损摧毁价值"）

样本：信号 1 触发（与 bt_entry_bs 同口径），仅 trending_up 门控（与 +0.97% 基准一致），
T 收盘买，持有上限 16 个交易日。

诊断（为什么现行止损砍掉价值）：
  D1 入场价距 MA10/MA20 的距离分布——止损带是否落在正常回踩噪声内
  D2 现行止损（MA10 连续2天）的止损率、被止损交易的 T+16 死拿收益 vs 未被止损交易
     ——止损信号有没有信息含量（提前离场是否避开了更差的后续）
  D3 被止损交易的"离场后漂移" = 死拿T+16收益 − 实际止损收益 > 0 即止损付出了机会成本

扫描（怎么改）：网格 均线(MA10/MA20) × 连续天数(1-5) × 缓冲带(0/1%/2%/3%)，
  外加 移动止盈（距入场后最高收盘回撤 z% 离场）、"亏损才止损"（2天破MA10且亏损）。
  各变体报：均值 / 中位 / 胜率 / 均持 / 每持有一天收益 / 止损率 / vs 死拿差值。

用法：python bt_exit_scan.py
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
BUFFERS = (0.0, 0.01, 0.02, 0.03)
STREAKS = (1, 2, 3, 4, 5)
TRAIL = (0.05, 0.08, 0.10, 0.15)


def collect_trades():
    """门控内（trending_up）全部触发 → 每笔的 T+1..T+16 价格路径。"""
    frames = load_frames()
    regs = load_bench()["regime"]
    trades = []
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
            entry = float(df.iloc[i]["close"])
            cs, m10s, m20s, hs, ls, os_ = [], [], [], [], [], []
            for j in range(i + 1, end + 1):
                cs.append(float(df.iloc[j]["close"]))
                m10s.append(float(df.iloc[j]["ma10"]) if pd.notna(df.iloc[j]["ma10"]) else None)
                m20s.append(float(df.iloc[j]["ma20"]) if pd.notna(df.iloc[j]["ma20"]) else None)
                hs.append(float(df.iloc[j]["high"]))
                ls.append(float(df.iloc[j]["low"]))
                os_.append(float(df.iloc[j]["open"]))
            if not cs:
                continue
            trades.append({
                "entry": entry, "cs": cs, "m10s": m10s, "m20s": m20s, "hs": hs, "ls": ls, "os": os_,
                "d_ma10": (entry / float(df.iloc[i]["ma10"]) - 1) * 100,
                "d_ma20": (entry / float(df.iloc[i]["ma20"]) - 1) * 100,
                "r_hold": cs[-1] / entry - 1,
            })
    return trades


def sim_streak(t, line_key, streak, buf):
    """收盘 < line*(1-buf) 连续 streak 天，第 streak 天收盘离场；否则持满。"""
    cs, mas = t["cs"], t[line_key]
    run = 0
    for j, c in enumerate(cs):
        ma = mas[j]
        if ma is None or ma <= 0:
            run = 0
            continue
        run = run + 1 if c < ma * (1 - buf) else 0
        if run >= streak:
            return c / t["entry"] - 1, j + 1, True
    return t["r_hold"], len(cs), False


def sim_trail(t, z):
    """距入场后最高收盘回撤 z% 离场（当日收盘判定、当日收盘价出）。"""
    cs = t["cs"]
    peak = t["entry"]
    for j, c in enumerate(cs):
        peak = max(peak, c)
        if c < peak * (1 - z):
            return c / t["entry"] - 1, j + 1, True
    return t["r_hold"], len(cs), False


def sim_trail_touch(t, z):
    """盘中触价版移动止盈：peak 取入场价与此前各日日内最高；当日 low ≤ peak*(1-z)
    即成交——开盘已低于触发价按开盘价成交（跳空），否则按触发价。
    当日自身的 high 在判定之后才计入 peak。"""
    hs, ls, os_ = t["hs"], t["ls"], t["os"]
    entry = t["entry"]
    peak = entry
    for j in range(len(hs)):
        trigger = peak * (1 - z)
        if ls[j] <= trigger:
            price = os_[j] if os_[j] <= trigger else trigger
            return price / entry - 1, j + 1, True
        peak = max(peak, hs[j])
    return t["r_hold"], len(hs), False


def sim_trail_touch_closepeak(t, z):
    """收盘 peak + 盘中触价：每日收盘后把触发价更新为 max(历史收盘,入场价)*(1-z)，
    次日盘中触及即成交（跳空按开盘）。对应"14:45 任务每日更新条件单触发价"的实现。"""
    cs, hs, ls, os_ = t["cs"], t["hs"], t["ls"], t["os"]
    entry = t["entry"]
    peak = entry
    for j in range(len(cs)):
        trigger = peak * (1 - z)
        if ls[j] <= trigger:
            price = os_[j] if os_[j] <= trigger else trigger
            return price / entry - 1, j + 1, True
        peak = max(peak, cs[j])
    return t["r_hold"], len(cs), False


def sim_loss_only(t):
    """连续2天收盘<MA10 且第2天收盘低于成本价才走（盈利中的破位不砍）。"""
    cs, m10s = t["cs"], t["m10s"]
    run = 0
    for j, c in enumerate(cs):
        ma = m10s[j]
        if ma is None or ma <= 0:
            run = 0
            continue
        run = run + 1 if c < ma else 0
        if run >= 2 and c < t["entry"]:
            return c / t["entry"] - 1, j + 1, True
    return t["r_hold"], len(cs), False


def report(name, rets, holds, stops):
    n = len(rets)
    mean = statistics.mean(rets) * 100
    s = sorted(rets)
    hold = statistics.mean(holds)
    print(f"  {name:<34} 均值 {mean:+.2f}%  中位 {s[n//2]*100:+.2f}%  胜率 "
          f"{sum(1 for v in rets if v > 0)/n*100:.0f}%  均持 {hold:4.1f}天 "
          f"每日 {mean/hold:+.3f}%/天  止损率 {sum(1 for v in stops if v)/n*100:.0f}%")


def main():
    trades = collect_trades()
    print(f"门控内交易 {len(trades)} 笔\n")

    # ── 诊断 ──
    d10 = sorted(t["d_ma10"] for t in trades)
    d20 = sorted(t["d_ma20"] for t in trades)
    n = len(trades)
    print("── D1 入场价距均线（止损带宽度，%）──")
    print(f"  距MA10  P10 {d10[int(n*0.1)]:+.1f}  P25 {d10[int(n*0.25)]:+.1f}  "
          f"P50 {d10[n//2]:+.1f}  P75 {d10[int(n*0.75)]:+.1f}")
    print(f"  距MA20  P10 {d20[int(n*0.1)]:+.1f}  P25 {d20[int(n*0.25)]:+.1f}  "
          f"P50 {d20[n//2]:+.1f}  P75 {d20[int(n*0.75)]:+.1f}")

    print("\n── D2 现行止损（MA10 连续2天）信息含量检验 ──")
    stopped, kept = [], []
    for t in trades:
        ret, hold, stop = sim_streak(t, "m10s", 2, 0.0)
        (stopped if stop else kept).append(t["r_hold"])  # 死拿口径的 T+16
    for name, grp in (("被止损交易", stopped), ("未被止损交易", kept)):
        if grp:
            g = sorted(grp)
            print(f"  {name:<8} n={len(grp):<5} 死拿T+16均值 {sum(grp)/len(grp)*100:+.2f}%  "
                  f"中位 {g[len(grp)//2]*100:+.2f}%  胜率 {sum(1 for v in grp if v > 0)/len(grp)*100:.0f}%")

    print("\n── D3 被止损交易的离场后漂移（死拿T+16 − 止损实际所得）──")
    drift = []
    stop_ret_win = []
    for t in trades:
        ret, hold, stop = sim_streak(t, "m10s", 2, 0.0)
        if stop:
            drift.append(t["r_hold"] - ret)
            if ret > 0:
                stop_ret_win.append(ret)
    if drift:
        print(f"  漂移均值 {statistics.mean(drift)*100:+.2f}%（>0 = 止损付出了机会成本），"
              f"中位 {sorted(drift)[len(drift)//2]*100:+.2f}%；止损离场时仍在盈利的比例 "
              f"{len(stop_ret_win)/len(drift)*100:.0f}%")

    # ── 扫描 ──
    print("\n── 扫描：均线 × 连续天数 × 缓冲带 ──")
    report("死拿（基准）", [t["r_hold"] for t in trades], [len(t["cs"]) for t in trades], [False] * n)
    results = []
    for line_key, line_name in (("m10s", "MA10"), ("m20s", "MA20")):
        for streak in STREAKS:
            for buf in BUFFERS:
                rets, holds, stops = [], [], []
                for t in trades:
                    ret, hold, stop = sim_streak(t, line_key, streak, buf)
                    rets.append(ret); holds.append(hold); stops.append(stop)
                results.append((f"{line_name}×{streak}天×buf{buf*100:.0f}%", rets, holds, stops))
    for z in TRAIL:
        rets, holds, stops = [], [], []
        for t in trades:
            ret, hold, stop = sim_trail(t, z)
            rets.append(ret); holds.append(hold); stops.append(stop)
        results.append((f"移动止盈 回撤{z*100:.0f}%（收盘判定）", rets, holds, stops))
    for z in (0.05, 0.06, 0.08, 0.10, 0.12):
        rets, holds, stops = [], [], []
        for t in trades:
            ret, hold, stop = sim_trail_touch(t, z)
            rets.append(ret); holds.append(hold); stops.append(stop)
        results.append((f"移动止盈 回撤{z*100:.0f}%（盘中触价）", rets, holds, stops))
    for z in (0.06, 0.08, 0.10, 0.12):
        rets, holds, stops = [], [], []
        for t in trades:
            ret, hold, stop = sim_trail_touch_closepeak(t, z)
            rets.append(ret); holds.append(hold); stops.append(stop)
        results.append((f"移动止盈 回撤{z*100:.0f}%（收盘peak+盘中触发）", rets, holds, stops))
    rets, holds, stops = [], [], []
    for t in trades:
        ret, hold, stop = sim_loss_only(t)
        rets.append(ret); holds.append(hold); stops.append(stop)
    results.append(("MA10×2天×亏损才走", rets, holds, stops))

    for name, rets, holds, stops in results:
        report(name, rets, holds, stops)


if __name__ == "__main__":
    main()
