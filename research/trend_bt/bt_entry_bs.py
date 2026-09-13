# -*- coding: utf-8 -*-
"""信号 1 买点回测（baostock 全市场版，一年半窗口）

- 数据：data/bs/kl/*.csv（前复权，含 turn/tradestatus/isST），宇宙含窗口内退市股
- 信号窗口：2025-03-13 ~ 2026-09-11（一年半，前置数据用于 60 日指标暖机）
- 过滤：信号日与 T+1 必须 tradestatus=1；信号日 isST=0；同股 10 个交易日内去重
- 三臂：A=T收盘持有；B=T+1确认后 T+1 收盘买；C=T收盘买+T+1不确认即T+1收盘卖
- 超额基准：上证（分 regime 报告）

用法：python bt_entry_bs.py
"""
import csv
import json
import os
import statistics
import sys

import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "research", "trend_bt", "bs")
os.chdir(BASE)

HORIZONS = [3, 5, 10, 16]
CONFIRM_VOL_RATIO = 1.2
SIG_START, SIG_END = "2025-03-13", "2026-09-11"
DEDUP_DAYS = 10  # 同股触发去重间隔（交易日）


def load_frames():
    frames = {}
    for fn in os.listdir("kl"):
        code = fn[:-4]
        df = pd.read_csv(os.path.join("kl", fn), dtype={"tradestatus": str, "isST": str})
        if len(df) < 80:
            continue
        for c in ("open", "high", "low", "close", "volume", "turn"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        for n in (5, 10, 20):
            df[f"ma{n}"] = df["close"].rolling(n).mean()
        df["vol_ma5"] = df["volume"].rolling(5).mean()
        frames[code] = df
    return frames


def load_bench():
    out = {}
    fn = "index_sh.000001.csv"
    if os.path.exists(fn):
        df = pd.read_csv(fn)
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        out["上证"] = {str(d)[:10]: float(c) for d, c in zip(df["date"], df["close"])}
        regs = {}
        for _, r in df.iterrows():
            d = str(r["date"])[:10]
            if pd.isna(r["ma20"]):
                regs[d] = "chaos"
            elif r["close"] > r["ma20"] and r["ma5"] > r["ma10"] > r["ma20"]:
                regs[d] = "trending_up"
            elif r["close"] < r["ma20"] and r["ma5"] < r["ma10"] < r["ma20"]:
                regs[d] = "down"
            else:
                regs[d] = "sideways"
        out["regime"] = regs
    return out


def detect_signal1(df, i, require_alignment=True):
    """signal_detector 同口径，附加：T 日须正常交易、非 ST。

    require_alignment=False 时关闭条件①（均线多头排列），供实验对比：
    排列条件在全样本上压低均值（消融见 README），是否保留/改为仅趋势段生效
    是实验变量而非定案，故做成开关而非焊死。
    """
    r = df.iloc[i]
    if str(r["tradestatus"]) != "1" or str(r["isST"]) == "1":
        return False
    if i < 20 or pd.isna(r.ma20) or pd.isna(r.vol_ma5) or r.vol_ma5 <= 0:
        return False
    # 条件① 均线多头排列（ma20 非 NaN 时 ma5/ma10 必非 NaN）
    if require_alignment and not (r.ma5 > r.ma10 > r.ma20):
        return False
    prev = df.iloc[i - 1]
    if str(prev["tradestatus"]) != "1":
        return False
    pct = (r.close / prev.close - 1) * 100 if prev.close > 0 else 0.0
    if not (r.close >= r.ma5 * 0.995):
        return False
    if not (r.volume < r.vol_ma5 * 1.1):
        return False
    bias5 = (r.close - r.ma5) / r.ma5 * 100
    if not (-1.5 < bias5 < 3.5):
        return False
    rng = r.high - r.low
    if rng > 0:
        close_pos = (r.close - r.low) / rng
        lower_shadow = (min(r.open, r.close) - r.low) / rng
        if not (close_pos > 0.4 and lower_shadow > 0.1):
            return False
    tr = r["turn"]
    if pd.isna(tr) or tr <= 3.0:
        return False
    g3 = (r.close / df.iloc[i - 3].close - 1) * 100 if i >= 3 and df.iloc[i - 3].close > 0 else 0
    g5 = (r.close / df.iloc[i - 5].close - 1) * 100 if i >= 5 and df.iloc[i - 5].close > 0 else 0
    biases5 = [(df.iloc[j].close - df.iloc[j].ma5) / df.iloc[j].ma5 * 100
               for j in range(max(0, i - 4), i + 1) if pd.notna(df.iloc[j].ma5)]
    amps = [(df.iloc[j].high - df.iloc[j].low) / df.iloc[j - 1].close * 100
            for j in range(max(1, i - 2), i + 1) if df.iloc[j - 1].close > 0]
    bias20 = (r.close - r.ma20) / r.ma20 * 100
    if (g3 >= 18 or g5 >= 30 or (biases5 and max(biases5) >= 12)
            or (amps and max(amps) >= 15) or pct > 7 or bias20 > 15):
        return False
    low60 = df["close"].iloc[max(0, i - 59):i + 1].min()
    if low60 <= 0 or (r.close - low60) / low60 * 100 >= 80:
        return False
    if sum(1 for j in range(max(0, i - 5), i) if df.iloc[j].close > df.iloc[j].ma5) < 3:
        return False
    if not (-5 < pct < 7):
        return False
    if not (pct < 0 or r.low <= r.ma5 * 1.005):
        return False
    return True


def stat_block(vals):
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    n = len(vals)
    return {"n": n,
            "median": round(statistics.median(vals) * 100, 2),
            "mean": round(statistics.mean(vals) * 100, 2),
            "win": round(sum(1 for v in vals if v > 0) / n * 100),
            "p10": round(s[int(n * 0.1)] * 100, 1),
            "p90": round(s[min(int(n * 0.9), n - 1)] * 100, 1)}


def main():
    frames = load_frames()
    bench = load_bench()
    bmap, regs = bench["上证"], bench["regime"]
    print(f"股票 {len(frames)} 只，基准 {len(bmap)} 根")

    A, B, C = ({k: [] for k in HORIZONS} for _ in range(3))
    A_ex, C_ex = ({k: [] for k in HORIZONS} for _ in range(2))
    CA, CB = ({k: [] for k in HORIZONS} for _ in range(2))
    by_reg = {}
    n_raw = n_uniq = confirm_skip = 0

    for code, df in frames.items():
        last_trig = -99
        for i in range(60, len(df)):
            d = str(df.iloc[i]["date"])[:10]
            if not (SIG_START <= d <= SIG_END):
                continue
            if not detect_signal1(df, i):
                continue
            if i - last_trig < DEDUP_DAYS:
                continue
            last_trig = i
            n_raw += 1
            if i + 1 >= len(df):
                continue
            r0, r1 = df.iloc[i], df.iloc[i + 1]
            if str(r1["tradestatus"]) != "1":
                continue  # T+1 停牌无法操作
            n_uniq += 1
            vol_ratio = r1.volume / r1.vol_ma5 if pd.notna(r1.vol_ma5) and r1.vol_ma5 > 0 else 99
            confirmed = (r1.close >= r1.ma5 * 0.995) and (vol_ratio < CONFIRM_VOL_RATIO)
            if not confirmed:
                confirm_skip += 1
            reg = regs.get(d, "chaos")

            for k in HORIZONS:
                a_ret = df.iloc[i + k].close / r0.close - 1 if i + k < len(df) else None
                c_ret = a_ret if confirmed else r1.close / r0.close - 1
                if a_ret is not None:
                    A[k].append(a_ret)
                    if d in bmap and dTk_ok(bmap, d, df, i, k):
                        ex = a_ret - (bmap[date_at(df, i, k)] / bmap[d] - 1)
                        A_ex[k].append(ex)
                        c_ex = ex if confirmed else (
                            (c_ret - (bmap[str(r1["date"])[:10]] / bmap[d] - 1))
                            if str(r1["date"])[:10] in bmap else None)
                        if c_ex is not None:
                            C_ex[k].append(c_ex)
                if c_ret is not None:
                    C[k].append(c_ret)
                    if a_ret is not None:
                        CA[k].append(c_ret - a_ret)
                if confirmed and i + 1 + k < len(df):
                    b_ret = df.iloc[i + 1 + k].close / r1.close - 1
                    B[k].append(b_ret)
                    if a_ret is not None:
                        CB[k].append(c_ret - b_ret)

                bucket = by_reg.setdefault((reg, k), {"A": [], "A_ex": [], "C_ex": [], "n": 0})
                bucket["n"] += 1
                if a_ret is not None and d in bmap and dTk_ok(bmap, d, df, i, k):
                    ex = a_ret - (bmap[date_at(df, i, k)] / bmap[d] - 1)
                    bucket["A_ex"].append(ex)
                    c_ex = ex if confirmed else (
                        (c_ret - (bmap[str(r1["date"])[:10]] / bmap[d] - 1))
                        if str(r1["date"])[:10] in bmap else None)
                    if c_ex is not None:
                        bucket["C_ex"].append(c_ex)

    print(f"\n原始触发 {n_raw}，去重后可操作 {n_uniq}，确认通过 {n_uniq - confirm_skip} "
          f"({(n_uniq - confirm_skip) / max(1, n_uniq) * 100:.0f}%)")
    for k in HORIZONS:
        print(f"\n── T+{k} ──")
        print(f"  A T收盘持有       : {stat_block(A[k])}")
        print(f"  B T+1确认后买     : {stat_block(B[k])}")
        print(f"  C T收盘买+不确认卖 : {stat_block(C[k])}")
        print(f"  A 超额(上证)      : {stat_block(A_ex[k])}")
        print(f"  C 超额(上证)      : {stat_block(C_ex[k])}")
        print(f"  配对 C-A / C-B    : {stat_block(CA[k])} | {stat_block(CB[k])}")

    print("\n── regime 分桶（A 超额 / C 超额，vs 上证）──")
    for (reg, k), b in sorted(by_reg.items()):
        if k != 16:
            continue
        sa, sc = stat_block(b["A_ex"]), stat_block(b["C_ex"])
        print(f"  {reg:<12} n={b['n']:<5} A超额 {sa.get('median', 0):+.2f}/{sa.get('mean', 0):+.2f} "
              f"(胜率{sa.get('win', 0)}%)   C超额 {sc.get('median', 0):+.2f}/{sc.get('mean', 0):+.2f}")

    json.dump({"A": A, "B": B, "C": C, "A_ex": A_ex, "C_ex": C_ex,
               "CA": CA, "CB": CB, "by_reg": {f"{r}|{k}": v for (r, k), v in by_reg.items()}},
              open("bt_entry_bs_results.json", "w", encoding="utf-8"))
    print("\n已写出 bt_entry_bs_results.json")


def date_at(df, i, k):
    return str(df.iloc[i + k]["date"])[:10]


def dTk_ok(bmap, d, df, i, k):
    return date_at(df, i, k) in bmap


if __name__ == "__main__":
    main()
