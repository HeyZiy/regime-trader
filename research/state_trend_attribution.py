# -*- coding: utf-8 -*-
"""
==================================
状态 × 趋势归因（v0，研究脚本）
==================================

回答 style_state 的验证问题在**个股趋势打法**上的版本：
不同风格周期下，趋势代理组合的周收益分布是否不同？
（state_attribution.py 已回答"状态 × 行业动量代理"，本脚本是股票维度的补齐。）

代理口径（简化重放，折扣必须记住）：
- 标的池：沪深300 现任成分股（幸存者偏差，与核心列同折扣）；日线走 AkshareFetcher 三源备援（qfq）
- 代理买入（日线，信号日收盘确认、按当日收盘建仓）：
  多头排列（收盘 > MA20 > MA60）且 回踩触碰 MA10（最低价 ≤ MA10×1.01 且
  收盘 ≥ MA10×0.99）且 缩量（成交量 < 5 日均量）——对应趋势策略 S1/S2
  "缩量回踩 MA5/MA10"买点的合并简化
- 代理退出（满足其一，收盘执行）：收盘 < MA20（趋势破坏）或 距持有期最高
  收盘回撤 > 8%（"阶段回撤减仓 → 破位清仓"两段式的简化）
- 组合：全部在持标的等权；组合空仓的交易日记 0（现金）

已知折扣：
- 不重放松筛/评分/位置扣分/负面清单；信号全收、无"评分择优"
- 卖出的"减仓 50% → 止损线上移 → 清仓"两段式简化为一次性退出
- 成分股用现任名单；无交易成本；回答的是"趋势股票池收益是否状态相关"，
  不是完整趋势策略评估

用法：
  python research/state_trend_attribution.py --start 2021-01-01
  python research/state_trend_attribution.py --start 2021-01-01 --max-stocks 30   # 快速试跑
"""

import argparse
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import setup_env  # noqa: E402

setup_env()

from src.logging_config import setup_logging  # noqa: E402

setup_logging()

from src.market_state.style_state import backtest_states  # noqa: E402

logger = __import__("logging").getLogger(__name__)

HS300 = "000300"
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "cache", "trend_attr_hs300.pkl")
FETCH_START_DEFAULT = "20200601"   # 预留 MA60/均量预热期
EXIT_DRAWDOWN = 0.08               # 持有期最高收盘回撤退出阈值
TOUCH_TOL = 0.01                   # 回踩触碰 MA10 容差


def _universe_codes() -> List[str]:
    """沪深300 现任成分股代码（现任名单，幸存者偏差已知）。"""
    import akshare as ak

    cons = ak.index_stock_cons_csindex(symbol=HS300)
    code_col = next((c for c in cons.columns if "成分券代码" in c or "成分股代码" in c), None)
    if code_col is None:
        raise ValueError(f"成分股代码列未找到，现有列：{list(cons.columns)}")
    return cons[code_col].astype(str).str.zfill(6).tolist()


def _load_universe(codes: List[str], fetch_start: str, end_date: str,
                   max_stocks: int = 0, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """加载个股日线（qfq），本地 pickle 缓存，缺失的逐只拉取。

    走项目 AkshareFetcher 三源备援（新浪→腾讯→东财）；返回
    {code: df(date, close, low, volume)}。
    """
    from data_provider.fetchers.akshare_fetcher import AkshareFetcher

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cache: Dict[str, pd.DataFrame] = {}
    if os.path.exists(CACHE_PATH) and not refresh:
        try:
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            logger.info(f"缓存加载：{len(cache)} 只")
        except Exception as e:
            logger.warning(f"缓存读取失败，重新拉取: {e}")
            cache = {}

    if max_stocks:
        codes = codes[:max_stocks]

    fetcher = AkshareFetcher(sleep_min=0.3, sleep_max=0.8)
    start_str = f"{fetch_start[:4]}-{fetch_start[4:6]}-{fetch_start[6:]}"
    end_str = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    out: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []
    for i, code in enumerate(codes):
        if code in cache:
            out[code] = cache[code]
            continue
        df = None
        try:
            raw = fetcher.get_daily_data(code, start_str, end_str)
            if raw is not None and hasattr(raw, "empty") and not raw.empty:
                df = raw.rename(columns={"日期": "date", "收盘": "close",
                                         "最低": "low", "成交量": "volume"})
                df = df[["date", "close", "low", "volume"]].copy()
                df["date"] = df["date"].astype(str)
        except Exception as e:
            logger.info(f"{code} 日线拉取失败: {e}")
        if df is not None and len(df) > 100:
            out[code] = df
            cache[code] = df
        else:
            failed.append(code)
        if (i + 1) % 50 == 0:
            logger.info(f"拉取进度 {i + 1}/{len(codes)}，累计失败 {len(failed)}")

    if failed:
        logger.warning(f"趋势归因标的池拉取失败 {len(failed)}/{len(codes)} 只"
                       f"（{', '.join(failed[:5])}{'...' if len(failed) > 5 else ''}），已跳过")
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
        logger.info(f"缓存已写入：{len(cache)} 只 → {CACHE_PATH}")
    except Exception as e:
        logger.warning(f"缓存写入失败: {e}")
    return out


def _simulate_stock(df: pd.DataFrame) -> List[Tuple[str, float]]:
    """单标的代理回放：返回在持交易日的 (date, 日收益) 序列。

    信号日收盘确认、当日收盘建仓（当日收益不计）；退出在收盘执行（当日收益计入）。
    """
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)
    ma5v = vol.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    dates = df["date"].tolist()
    n = len(df)
    out: List[Tuple[str, float]] = []
    holding = False
    peak = 0.0

    for i in range(1, n):
        px = float(close.iloc[i])
        prev = float(close.iloc[i - 1])
        if holding:
            peak = max(peak, px)
            out.append((dates[i], px / prev - 1))
            if px < float(ma20.iloc[i]) or px < peak * (1 - EXIT_DRAWDOWN):
                holding = False
        else:
            if (pd.notna(ma60.iloc[i])
                    and px > float(ma20.iloc[i]) > float(ma60.iloc[i])
                    and float(low.iloc[i]) <= float(ma10.iloc[i]) * (1 + TOUCH_TOL)
                    and px >= float(ma10.iloc[i]) * (1 - TOUCH_TOL)
                    and float(vol.iloc[i]) < float(ma5v.iloc[i])):
                holding = True
                peak = px
    return out


def run(start: str, max_stocks: int = 0, refresh: bool = False) -> str:
    from data_provider import get_index_daily

    states = backtest_states(start)
    if len(states) < 10:
        return f"回放周数不足（{len(states)}）"
    fridays = [r["date"] for r in states]
    labels = [r["state"] for r in states]

    hs300 = get_index_daily(HS300, days=2500)
    hs300_close = hs300.set_index("date")["close"].astype(float)
    hs300_close.index = hs300_close.index.astype(str)

    codes = _universe_codes()
    fetch_start = (pd.Timestamp(fridays[0]) - pd.Timedelta(days=230)).strftime("%Y%m%d")
    end_date = pd.Timestamp.today().strftime("%Y%m%d")
    universe = _load_universe(codes, fetch_start, end_date, max_stocks=max_stocks, refresh=refresh)
    logger.info(f"标的池：{len(universe)}/{len(codes)} 只有数据")

    # 组合日收益：在持标的等权
    daily: Dict[str, List[float]] = {}
    for code, df in universe.items():
        for d, r in _simulate_stock(df):
            daily.setdefault(d, []).append(r)
    port_daily = {d: float(np.mean(rs)) for d, rs in daily.items()}

    # 逐周：状态（期初）× 组合收益 × 相对沪深300 超额
    per_state: Dict[str, List[Tuple[float, float]]] = {}
    for i in range(len(fridays) - 1):
        a, b = fridays[i], fridays[i + 1]
        window = [d for d in hs300_close.index if a < d <= b]
        if not window:
            continue
        tr = 1.0
        for d in window:
            tr *= 1 + port_daily.get(d, 0.0)
        trend_ret = (tr - 1) * 100
        pa, pb = hs300_close.reindex([a, b]).tolist()
        if pd.isna(pa) or pd.isna(pb) or pa <= 0:
            continue
        bench_ret = (float(pb) / float(pa) - 1) * 100
        per_state.setdefault(labels[i], []).append((trend_ret, trend_ret - bench_ret))

    lines = [f"# 📊 状态 × 趋势归因（{start} 起，标的池 沪深300 成分 ×{len(universe)}）", ""]
    lines.append("| 状态 | 周数 | 趋势(均值) | 胜率 | 超额(均值) | 超额胜率 |")
    lines.append("|---|---|---|---|---|---|")
    order = ["strong", "forming", "vacuum", "fading"]
    from src.market_state.style_state import STATE_LABELS

    all_rows = [r for rs in per_state.values() for r in rs]
    for st in order + [s for s in per_state if s not in order]:
        rows = per_state.get(st)
        if not rows:
            continue
        t = [r[0] for r in rows]
        ex = [r[1] for r in rows]
        name = STATE_LABELS.get(st, st) + ("（全部）" if st == "全部" else "")
        lines.append(
            f"| {name} | {len(rows)} | {np.mean(t):+.2f}% "
            f"| {sum(x > 0 for x in t) / len(t):.0%} "
            f"| {np.mean(ex):+.2f}% "
            f"| {sum(x > 0 for x in ex) / len(ex):.0%} |"
        )
    t_all = [r[0] for r in all_rows]
    ex_all = [r[1] for r in all_rows]
    lines.append(
        f"| **全部** | {len(all_rows)} | {np.mean(t_all):+.2f}% "
        f"| {sum(x > 0 for x in t_all) / len(t_all):.0%} "
        f"| {np.mean(ex_all):+.2f}% "
        f"| {sum(x > 0 for x in ex_all) / len(ex_all):.0%} |"
    )
    lines.append("")
    lines.append("**口径**：趋势代理 = 多头排列内缩量回踩 MA10 建仓、破 MA20 或回撤 8% 退出，"
                 "标的池 沪深300 现任成分，等权、无成本；状态标签为期初（周五）回放值，无未来函数。")
    lines.append("**解读门槛**：与 state_attribution 相同——各状态收益/超额分布无差异 → 标签对趋势"
                 "打法无区分度，不接入；有差异且方向与 ADVICE 一致（强势期趋势占优）→ 可讨论接入。"
                 "折扣见脚本头。")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="状态 × 趋势归因")
    parser.add_argument("--start", default="2021-01-01", help="回放起始日")
    parser.add_argument("--max-stocks", type=int, default=0, help="限制标的数（0=全部 300 只）")
    parser.add_argument("--refresh", action="store_true", help="忽略缓存重新拉取")
    args = parser.parse_args()
    print(run(args.start, max_stocks=args.max_stocks, refresh=args.refresh))
