# -*- coding: utf-8 -*-
"""baostock 全市场日线拉取（无幸存者偏差版）

- 宇宙：query_stock_basic 全列表 → 沪深主板（sh.60/sz.00）→ 窗口内存在过的全部股票，
  **含窗口中途退市的票**（outDate 落在窗口内也拉，拉到退市日为止）
  注意：宇宙层**不做** ST/退市名称过滤——名称是"当前终态"，按它筛股会把
  "窗口内曾是正常股、后来才戴帽/退市"的历史整段抹掉（幸存者偏差）。
  非 ST / 正常交易由信号层逐日 isST / tradestatus 判定。
- K 线：前复权日线（adjustflag=2），字段含 turn（换手）、tradestatus（停牌）、isST（逐日 ST）
- 基准指数：上证 / 国证2000 / 中证500 / 中证1000（能取到哪个算哪个）

用法：python fetch_bs.py            # 断点续传：已有 CSV 的股票跳过
输出：data/bs/kl/<code>.csv，data/bs/universe.json，data/bs/index_<code>.csv
"""
import json
import os
import sys
import time

import baostock as bs

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "research", "trend_bt", "bs")
KL = os.path.join(DATA, "kl")
os.makedirs(KL, exist_ok=True)

FETCH_START, FETCH_END = "2024-10-01", "2026-09-11"
FIELDS = "date,open,high,low,close,volume,amount,turn,tradestatus,pctChg,isST"
INDICES = {"sh.000001": "上证", "sz.399303": "国证2000", "sh.000905": "中证500", "sh.000852": "中证1000"}


def build_universe():
    rs = bs.query_stock_basic()
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    # fields: code, code_name, ipoDate, outDate, type, status
    uni = []
    for r in rows:
        code, name, ipo, out, typ, status = r[0], r[1], r[2], r[3], r[4], r[5]
        if typ != "1":  # 1=股票
            continue
        if not (code.startswith("sh.60") or code.startswith("sz.00")):
            continue  # 沪深主板
        listed = ipo <= FETCH_END
        delisted_in_window = out and (FETCH_START <= out <= FETCH_END)
        alive = status == "1" or delisted_in_window
        if listed and alive:
            uni.append({"code": code, "name": name, "ipoDate": ipo, "outDate": out})
    json.dump(uni, open(os.path.join(DATA, "universe.json"), "w", encoding="utf-8"), ensure_ascii=False)
    n_out = sum(1 for u in uni if u["outDate"])
    print(f"宇宙：沪深主板 {len(uni)} 只（其中窗口内退市 {n_out} 只——幸存者偏差修正项）")
    return uni


def fetch_stock(code):
    rs = bs.query_history_k_data_plus(code, FIELDS, start_date=FETCH_START, end_date=FETCH_END,
                                      frequency="d", adjustflag="2")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return rows


def main():
    lg = bs.login()
    if lg.error_code != "0":
        sys.exit(f"登录失败: {lg.error_msg}")

    uni_path = os.path.join(DATA, "universe.json")
    if os.path.exists(uni_path):
        uni = json.load(open(uni_path, encoding="utf-8"))
        print(f"universe.json 已存在：{len(uni)} 只")
    else:
        uni = build_universe()

    for code, name in INDICES.items():
        out = os.path.join(DATA, f"index_{code}.csv")
        if os.path.exists(out):
            continue
        rows = fetch_stock(code)
        with open(out, "w", encoding="utf-8") as f:
            f.write("date,open,high,low,close,volume,amount\n")
            for r in rows:
                f.write(",".join(r[:7]) + "\n")
        print(f"指数 {name}({code}): {len(rows)} 根")

    t0 = time.time()
    done = fail = 0
    for i, u in enumerate(uni, 1):
        out = os.path.join(KL, f"{u['code']}.csv")
        if os.path.exists(out):
            continue
        try:
            rows = fetch_stock(u["code"])
            with open(out, "w", encoding="utf-8") as f:
                f.write(FIELDS + "\n")
                for r in rows:
                    f.write(",".join(r) + "\n")
        except Exception as e:
            fail += 1
            print(f"  失败 {u['code']}: {type(e).__name__}")
        done += 1
        if done % 100 == 0:
            print(f"  进度 {i}/{len(uni)} 新取{done} 失败{fail} {time.strftime('%H:%M:%S')}")
    print(f"完成：新取 {done}，失败 {fail}，耗时 {(time.time()-t0)/60:.1f} 分钟")
    bs.logout()


if __name__ == "__main__":
    main()
