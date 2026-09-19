# -*- coding: utf-8 -*-
"""
===================================
行业动量轮动 — 卫星仓引擎
===================================

主账户卫星仓的战术策略：动量策略族的行业粒度子策略，
赚行业相对市场超额延续的钱。三层结构：

  1. 选行业（周频截面排名，momentum_score）—— 价能 60（ETF 自身收盘价相对沪深300
     的 20/60 日超额截面排名，2026-09-16 口径切换 v2.1）+ 量能 40（资金迁移 20 行业
     口径 + 交易热度 20 自身分位），前 3 名进关注池
  2. 入场时机（突破触发）—— 关注池内 ETF 出现放量突破（60日新高/平台突破 +
     涨幅区间 + 量比门槛）才建仓；排名只说明该关注，突破才说明资金来了
  3. 持仓管理（动量退出）—— 拿住直到动量证伪：相对强度转负 / 连续 2 次周度
     检查跌出前 5 / 跌破 MA20 且量能萎缩 / 灾难止损 -12%；PE ≥90% 锁仓、
     市值拥挤度 ≥95% 分位熔断

与个股趋势子策略（trend_strategy）的关系见 strategy/overview.md：
同一动量策略族的两个粒度，门控走降杠杆版——
禁新开仓、不强制清仓（截面轮动的立身之本是弱市里也有相对强行业）。

口径说明：
- 量能/拥挤度均为行业口径真数据（AmazingData 行业 AMOUNT/CLOSE/PE）：
  资金迁移 = 行业成交额占全市场比重 5 日变化；交易热度 = 占比自身 250 日分位；
  复合拥挤度 = PE 分位 / 交易热度 / 位置热度 三分量均值（星耀因子框架的行业级
  适配），阈值 60 警戒 / 80 高度拥挤减半 / 90 熔断。
  不用长历史绝对均值（会被规模扩张污染，515010 教训）。
- 无状态设计：持仓成本用模拟仓 cost_price；仅"连续 2 次跌出前 5"需要跨周
  记忆，落盘 data/momentum_rank_history.json。
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.mx.position_utils import filter_held_positions

logger = logging.getLogger(__name__)


@dataclass
class RotationOrder:
    """卫星仓调仓订单。"""
    code: str
    name: str
    action: str      # buy / sell
    shares: int
    price: float
    amount: float
    reason: str

SATELLITE_BUDGET_RATIO = 0.10   # 卫星仓总仓位上限（总资产占比）
MAX_HOLDINGS = 2                # 最多持仓数
POOL_SIZE = 3                   # 关注池：动量排名前 N
RANK_EXIT_WINDOW = 5            # 动量退出判定窗口：跌出前 N
RANK_EXIT_STRIKES = 2           # 连续 N 次周度检查跌出前 5 → 退出
BREAKOUT_CHANGE_MIN = 3.0       # 突破日涨幅下限（%）
BREAKOUT_CHANGE_MAX = 9.5       # 突破日涨幅上限（%，排除一字板/近涨停）
VOL_MULT_5D = 2.0               # 突破量能：≥ 5 日均量倍数
VOL_MULT_20D = 1.5              # 突破量能：≥ 20 日均量倍数
PE_LOCK = 90.0                  # 行业 PE 分位锁仓线（逐只判定）
CROWD_WARN = 60.0               # 复合拥挤度警戒线（≥60%，报告提示）
CROWD_HALF = 80.0               # 复合拥挤度高度拥挤线（≥80%，新开仓减半）
CROWD_EXIT = 90.0               # 复合拥挤度熔断线（≥90%，清仓）
CATASTROPHIC_STOP = 0.88        # 灾难止损（买入价 × 0.88，动量证伪前的最后防线）
HS300_INDEX = "000300"          # 相对强度的市场基准
PLATFORM_AMPLITUDE = 0.25       # 箱体定义：近 20 日振幅上限
RANK_HISTORY_MAX = 60           # 排名历史最多保留条数（周度记录）
CROWD_LOOKBACK = 250            # 拥挤度分量指标的历史分位窗口（日）
FLOW_WINDOW = 5                 # 成交额占比变化窗口（日，资金迁移方向）

RANK_HISTORY_PATH = Path(__file__).parent.parent.parent / "data" / "momentum_rank_history.json"


def _sell_order(code: str, name: str, count: int, price: float, reason: str) -> RotationOrder:
    """构造卫星仓卖出订单（返回 RotationOrder，调用方 append 到 orders）。"""
    return RotationOrder(code=code, name=name, action="sell", shares=count,
                         price=price, amount=count * price, reason=reason)


# ── 排名历史（仅"连续 2 次跌出前 5"需要跨周记忆）──

def _load_rank_history() -> List[dict]:
    try:
        if RANK_HISTORY_PATH.exists():
            return json.loads(RANK_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("动量排名历史读取失败，按无历史处理", exc_info=True)
    return []


def _latest_data_date(results: List[dict]) -> str:
    """本次截面对应的行情日期：取任一结果日线索引的最后一值，失败兜底当天。"""
    for r in results:
        df = r.get("df")
        if df is not None and len(df) > 0:
            try:
                dt = pd.to_datetime(df.index[-1])
                # 注入式 df 可能是 RangeIndex，解析出的 1970 等非法日期跳过
                if pd.Timestamp("1990-01-01") <= dt <= pd.Timestamp.now() + pd.Timedelta(days=7):
                    return dt.strftime("%Y-%m-%d")
            except Exception:
                continue
    from datetime import date
    return date.today().strftime("%Y-%m-%d")


def _append_rank_history(top5: List[str], data_date: str):
    """追加本周 top5 记录（etf_observe 周度批次调用，诊断工具不写）。"""
    try:
        hist = _load_rank_history()
        hist.append({"date": data_date, "top5": top5})
        RANK_HISTORY_PATH.write_text(
            json.dumps(hist[-RANK_HISTORY_MAX:], ensure_ascii=False, indent=1),
            encoding="utf-8")
    except Exception:
        logger.warning("动量排名历史写入失败", exc_info=True)


def _rank_exit_triggered(industry: str, current_top5: List[str]) -> bool:
    """连续 2 次周度检查跌出前 5 → 动量退出。

    上次还在前 5、本次跌出 → 第一次，暂不退；上次已跌出、本次仍在前 5 之外 → 退。
    """
    hist = _load_rank_history()
    if industry in current_top5:
        return False
    if not hist or not hist[-1].get("top5"):
        return False
    return industry not in hist[-1]["top5"]


# ── 单 ETF 技术面（入场触发层）──

def analyze_etf(code: str, name: str = "", df: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """拉取单只 ETF 日线并计算突破触发所需技术面。失败返回 None。

    df 可注入外部日线（date/open/high/low/close/volume，日期升序），
    供非 ETF 标的复用同一套规则（momentum_check.py 用）。
    """
    if df is None:
        from data_provider import get_etf_daily
        df = get_etf_daily(code)
    if df is None or len(df) < 65:
        return None

    close = df["close"]
    vol = df["volume"]
    df["ma5"] = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["vol_ma5"] = vol.rolling(5).mean()
    df["vol_ma20"] = vol.rolling(20).mean()
    df["vol_ma60"] = vol.rolling(60).mean()

    latest = df.iloc[-1]
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else float(latest["close"])
    chg_pct = (float(latest["close"]) / prev_close - 1) * 100

    vol_ratio_5 = float(latest["volume"]) / float(latest["vol_ma5"]) if float(latest["vol_ma5"]) > 0 else 0.0
    vol_ratio_20 = float(latest["volume"]) / float(latest["vol_ma20"]) if float(latest["vol_ma20"]) > 0 else 0.0

    high_60 = float(close.iloc[-60:].max())
    high_20, low_20 = float(close.iloc[-20:].max()), float(close.iloc[-20:].min())
    new_high_60d = float(latest["close"]) >= high_60 * 0.995
    amplitude_20 = (high_20 - low_20) / low_20 if low_20 > 0 else 1.0
    platform_break = amplitude_20 < PLATFORM_AMPLITUDE and float(latest["close"]) > high_20 * 0.99

    ret_20d = (float(latest["close"]) / float(close.iloc[-21]) - 1) * 100
    ret_5d = (float(latest["close"]) / float(close.iloc[-6]) - 1) * 100

    # 技术破位（持仓退出判定）：收盘 < MA20 且 5 日均量 < 20 日均量（缩量阴跌）
    weak_break = (float(latest["close"]) < float(latest["ma20"])
                  and pd.notna(latest["vol_ma5"]) and pd.notna(latest["vol_ma20"])
                  and float(latest["vol_ma5"]) < float(latest["vol_ma20"]))

    breakout = (
        (new_high_60d or platform_break)
        and BREAKOUT_CHANGE_MIN <= chg_pct <= BREAKOUT_CHANGE_MAX
        and vol_ratio_5 >= VOL_MULT_5D
        and vol_ratio_20 >= VOL_MULT_20D
    )

    return {
        "code": code, "name": name, "df": df,
        "close": float(latest["close"]), "chg_pct": round(chg_pct, 1),
        "vol_ratio_5": round(vol_ratio_5, 2), "vol_ratio_20": round(vol_ratio_20, 2),
        "new_high_60d": new_high_60d, "platform_break": platform_break,
        "ret_20d": round(ret_20d, 1), "ret_5d": round(ret_5d, 1),
        "weak_break": weak_break,
        "breakout": breakout,
    }


# ── 行业截面排名（选行业层）──

_amt_share_cache: Optional[pd.DataFrame] = None


def _amount_share_frame() -> Optional[pd.DataFrame]:
    """全部一级行业的成交额占全市场比重序列（列=行业代码，每轮计算一次）。

    share = 行业 AMOUNT / Σ一级行业 AMOUNT，资金迁移方向的分母口径。
    """
    global _amt_share_cache
    if _amt_share_cache is not None:
        return _amt_share_cache
    from src.etf.amazing_factors import get_industry_daily, get_level1_industries
    amounts: Dict[str, pd.Series] = {}
    for ind in get_level1_industries():
        code = ind.get("index_code") or ind.get("INDEX_CODE") or ind.get("code")
        if not code:
            continue
        df = get_industry_daily(str(code))
        if df is not None and "AMOUNT" in df.columns:
            s = df["AMOUNT"].copy()
            s.index = pd.to_datetime(s.index).strftime("%Y-%m-%d")
            s = s[~s.index.duplicated(keep="last")]
            amounts[str(code)] = s
    if len(amounts) < 5:
        logger.warning("一级行业成交额数据不足（%d 个），资金迁移因子退化", len(amounts))
        return None
    total = pd.concat(amounts, axis=1).sum(axis=1)
    share = pd.concat(amounts, axis=1).div(total, axis=0)
    _amt_share_cache = share
    return share


def _pct_in_window(s: pd.Series, lookback: int) -> Optional[float]:
    """当前值在自身近 N 日窗口内的分位（0-100）。"""
    w = s.tail(lookback).dropna()
    if len(w) < 60:
        return None
    return round(float((w < w.iloc[-1]).mean() * 100), 1)


def _normalize_index(s: pd.Series) -> pd.Series:
    """索引归一化为 YYYY-MM-DD 字符串并按最后出现去重（就地修改传入 Series 的索引）。"""
    s.index = pd.to_datetime(s.index).strftime("%Y-%m-%d")
    return s[~s.index.duplicated(keep="last")]


def _industry_flow_factors(ind_code: str) -> Optional[dict]:
    """行业量能/拥挤度真口径因子（AmazingData 行业 AMOUNT/CLOSE + PE 分位）。

    Returns:
        {amt_share_5d_chg, amt_share_pct, close_pct, crowding_pct} 或 None
    - amt_share_5d_chg：成交额占全市场比重的 5 日变化（资金迁移方向）
    - amt_share_pct：成交额占比自身 250 日分位（交易热度）
    - close_pct：收盘价自身 250 日分位（位置热度）
    - crowding_pct：复合拥挤度 = PE分位/交易热度/位置热度 三分量的均值
      （星耀因子框架 5 指标拥挤度的行业级适配，阈值 60 警戒/80 减半/90 熔断）
    """
    from src.etf.amazing_factors import get_industry_daily, get_industry_pe_percentile
    df = get_industry_daily(ind_code)
    if df is None or "CLOSE" not in df.columns:
        return None
    close = _normalize_index(df["CLOSE"])

    close_pct = _pct_in_window(close, CROWD_LOOKBACK)

    amt_share_pct, amt_share_5d_chg = None, None
    share = _amount_share_frame()
    if share is not None and ind_code in share.columns:
        s = share[ind_code].dropna()
        if len(s) >= FLOW_WINDOW + 1:
            amt_share_5d_chg = round(float(s.iloc[-1] - s.iloc[-1 - FLOW_WINDOW]) * 10000, 2)  # bp
            amt_share_pct = _pct_in_window(s, CROWD_LOOKBACK)

    pe_pct = None
    pe_info = get_industry_pe_percentile(ind_code)
    if pe_info:
        pe_pct = pe_info.get("pe_pct")

    comps = [c for c in (pe_pct, amt_share_pct, close_pct) if c is not None]
    crowding_pct = round(sum(comps) / len(comps), 1) if len(comps) >= 2 else None

    return {"amt_share_5d_chg": amt_share_5d_chg, "amt_share_pct": amt_share_pct,
            "close_pct": close_pct, "pe_pct": pe_pct, "crowding_pct": crowding_pct}


def _vol_score(res: dict) -> int:
    """量能分（40 满分，行业口径）：
    - 资金迁移 20：成交额占全市场比重的 5 日变化，截面排名归一化；
    - 交易热度 20：成交额占比自身 250 日分位。
    量能不用长历史绝对均值（会被规模扩张污染，515010 教训）。
    """
    s_flow = res.get("_flow_rank_score", 0.0)
    share_pct = res.get("amt_share_pct")
    s_heat = share_pct / 5 if share_pct is not None else 0  # 0-100 → 0-20
    return round(s_flow + s_heat)


def _attach_momentum_scores(results: List[dict]) -> List[dict]:
    """给每只 ETF 附加动量分（截面排名归一化）与复合拥挤度，标注关注池。

    2026-09-16 口径切换 v2.1：价能改为 **ETF 自身收盘价** 相对沪深300 的 20/60 日
    超额（截面排名）——与交易对象一致、消除行业指数代理误差、统一风格品种口径。
    量能与拥挤度保留行业口径：资金迁移（行业成交额占全市场比重的 5 日变化）是
    真实行业资金信号，ETF 成交额会被份额扩容污染（515010 教训）。
    无行业映射的叙事组标的（风格:小市值等自造类）：量能热度/拥挤度回退为自身口径
    （成交额代理分位 / 价格位置分位），资金迁移子项记缺失，动量分上限相应降低。
    """
    from src.etf.amazing_factors import (
        get_etf_industry, get_industry_code_by_name, get_etf_group,
    )
    from data_provider import get_etf_daily, get_index_daily

    hs300 = get_index_daily(HS300_INDEX)
    hs300_close = hs300.set_index("date")["close"] if hs300 is not None else None
    if hs300_close is None:
        logger.warning("沪深300 日线获取失败，本轮动量分不可用")
    else:
        hs300_close = _normalize_index(hs300_close.copy())

    for r in results:
        r["industry"] = None
        r["group"] = get_etf_group(r["code"])
        r["excess_20d"] = None
        r["excess_60d"] = None
        r["amt_share_5d_chg"] = None
        r["amt_share_pct"] = None
        r["crowding_pct"] = None
        r["_pe_pct_ind"] = None
        r["_momentum_source"] = None
        r["_flow_rank_score"] = 0.0
        r["momentum_score"] = 0
        r["momentum_rank"] = None
        r["pool"] = False

    # 1) 价能：全部标的统一用自身收盘价（买的是 ETF，量的是 ETF）
    if hs300_close is not None:
        for r in results:
            try:
                df = get_etf_daily(r["code"])
                if df is None or len(df) < 65:
                    continue
                close = pd.Series(df["close"].values, index=df["date"].values)
                close = close[~close.index.duplicated(keep="last")]
                aligned = pd.concat([close.rename("etf"), hs300_close.rename("mkt")],
                                    axis=1).dropna()
                if len(aligned) < 65:
                    continue
                r["excess_20d"] = round((aligned["etf"].iloc[-1] / aligned["etf"].iloc[-21]
                                         - aligned["mkt"].iloc[-1] / aligned["mkt"].iloc[-21]) * 100, 4)
                r["excess_60d"] = round((aligned["etf"].iloc[-1] / aligned["etf"].iloc[-61]
                                         - aligned["mkt"].iloc[-1] / aligned["mkt"].iloc[-61]) * 100, 4)
            except Exception:
                logger.warning(f"ETF 价能计算失败 {r['code']}", exc_info=True)

    # 2) 量能/拥挤度：行业口径（有行业映射的标的）
    flow_cache: Dict[str, Optional[dict]] = {}
    for r in results:
        try:
            industry = get_etf_industry(r["code"])
            r["industry"] = industry
            ind_code = get_industry_code_by_name(industry) if industry else None
            if not ind_code:
                continue
            if ind_code not in flow_cache:
                flow_cache[ind_code] = _industry_flow_factors(ind_code)
            ff = flow_cache[ind_code]
            if ff:
                r["amt_share_5d_chg"] = ff["amt_share_5d_chg"]
                r["amt_share_pct"] = ff["amt_share_pct"]
                r["crowding_pct"] = ff["crowding_pct"]
                r["_pe_pct_ind"] = ff["pe_pct"]
        except Exception:
            logger.warning(f"行业量能因子获取失败 {r['code']}", exc_info=True)

    # 3) 回退：无行业映射的叙事组标的，量能热度/拥挤度用自身口径降级
    for r in results:
        if r.get("group") and r["amt_share_pct"] is None and r["crowding_pct"] is None:
            try:
                df = get_etf_daily(r["code"])
                if df is None or len(df) < 70:
                    continue
                amts = (df["volume"] * df["close"]).tail(250)
                r["amt_share_pct"] = _pct_in_window(amts, 250)
                closes = pd.Series(df["close"].values, index=df["date"].values).tail(250)
                r["crowding_pct"] = _pct_in_window(closes, 250)
                r["_momentum_source"] = "etf_self"
            except Exception:
                logger.warning(f"自身量能回退失败 {r['code']}", exc_info=True)

    # 4) 截面排名与动量分（n<=1 时该项记满分中位，防止除零）
    scored = [r for r in results if r["excess_20d"] is not None]
    n = len(scored)
    for key in ("excess_20d", "excess_60d"):
        ordered = sorted(scored, key=lambda x: x[key])
        for i, r in enumerate(ordered):
            r[f"_pct_{key}"] = i / (n - 1) * 30 if n > 1 else 15.0
    flowed = [r for r in scored if r["amt_share_5d_chg"] is not None]
    if len(flowed) >= 2:
        ordered = sorted(flowed, key=lambda x: x["amt_share_5d_chg"])
        for i, r in enumerate(ordered):
            r["_flow_rank_score"] = i / (len(flowed) - 1) * 20
    for r in scored:
        r["momentum_score"] = round(
            r.get("_pct_excess_20d", 0) + r.get("_pct_excess_60d", 0) + _vol_score(r))

    ranked = sorted([r for r in results if r["momentum_score"] > 0],
                    key=lambda r: r["momentum_score"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["momentum_rank"] = i
        r["pool"] = i <= POOL_SIZE
    return results


def analyze_universe(update_rank_history: bool = False) -> List[dict]:
    """分析全行业 ETF 清单：技术面 + 行业动量排名。

    update_rank_history 仅周度批次（etf_observe）开启——排名历史落盘后，
    "连续 2 次跌出前 5"的退出判定才有跨周记忆；单标的诊断不写。
    """
    from src.etf.amazing_factors import get_industry_etf_universe

    results = []
    for entry in get_industry_etf_universe():
        res = analyze_etf(entry["code"], entry["name"])
        if res is not None:
            results.append(res)
    results = _attach_momentum_scores(results)

    if update_rank_history:
        top5 = [r["industry"] or r["name"] for r in results
                if r["momentum_rank"] and r["momentum_rank"] <= RANK_EXIT_WINDOW]
        _append_rank_history(top5, _latest_data_date(results))
    return results


# ── 决策 ──

def check_pe_lock(pe_pct: Optional[float]) -> bool:
    """行业 PE 分位 ≥ 90% → 该 ETF 锁仓清仓。数据缺失时视为未锁。"""
    return pe_pct is not None and pe_pct >= PE_LOCK


def _pe_pct_for(res: dict) -> Optional[float]:
    """行业 PE 分位（量能因子阶段已随行业因子预取，缺失时兜底现取）。"""
    if res.get("_pe_pct_ind") is not None:
        return res["_pe_pct_ind"]
    try:
        from src.etf.amazing_factors import (
            get_industry_code_by_name, get_industry_pe_percentile,
        )
        industry = res.get("industry")
        if industry:
            ind_code = get_industry_code_by_name(industry)
            if ind_code:
                pe_info = get_industry_pe_percentile(ind_code)
                if pe_info:
                    return pe_info["pe_pct"]
    except Exception:
        pass
    return None


def build_sell_orders(results: List[dict], positions: List[dict], entry_map: Dict[str, str],
                      force_flat: bool = False) -> Tuple[List[dict], List[str]]:
    """生成卫星卖出订单（动量退出规则，持仓起点全部来自模拟仓数据）。

    降杠杆门控：市场状态恶化只禁新开仓，不强制清仓（截面轮动弱市里也有相对强行业，
    与个股趋势子策略的全量门控差异，见 overview.md）。
    风格状态门控：force_flat（真空/退潮期）为最高优先，直接清仓全部卫星持仓，
    覆盖所有常规退出规则——归因显示 off 期亏损来自存量持仓流血，等常规规则
    退出太慢（见 strategy/style_state.md 卫星仓门控节）。
    退出优先级：风格状态门控 > PE 锁仓 > 拥挤度熔断 > 灾难止损 > 相对强度转负
    > 技术破位 > 连续 2 次跌出前 5。

    entry_map 保留参数位（历史签名兼容，当前退出规则不再使用买入日期）。
    """
    result_map = {r["code"]: r for r in results}
    universe_codes = set(result_map)
    current_top5 = [r["industry"] or r["name"] for r in results
                    if r["momentum_rank"] and r["momentum_rank"] <= RANK_EXIT_WINDOW]

    orders: List[RotationOrder] = []
    reasons: List[str] = []

    if force_flat:
        for p in filter_held_positions(positions):
            code = p.get("code", "")
            if code not in universe_codes:
                continue
            name = p.get("name", "") or code
            count = int(p.get("count", 0) or 0)
            price = float(p.get("current_price", 0) or 0)
            reason = "风格状态门控（真空/退潮期）清仓"
            orders.append(_sell_order(code, name, count, price, reason))
            reasons.append(f"{name}({code}) {reason}")
        return orders, reasons

    for p in filter_held_positions(positions):
        code = p.get("code", "")
        if code not in universe_codes:
            continue
        entry_price = float(p.get("cost_price", 0) or 0)
        price = float(p.get("current_price", 0) or 0)
        count = int(p.get("count", 0) or 0)
        name = p.get("name", "") or code
        res = result_map[code]

        # 0. 估值/拥挤度熔断（逐只判定，全市场 PE 不穿透卫星仓）
        if check_pe_lock(res.get("pe_pct")):
            reason = f"行业 PE 分位 ≥{PE_LOCK:.0f}% 锁仓"
            orders.append(_sell_order(code, name, count, price, reason))
            reasons.append(f"{name}({code}) {reason} 清仓")
            continue
        if res.get("crowding_pct") is not None and res["crowding_pct"] >= CROWD_EXIT:
            reason = f"行业复合拥挤度 {res['crowding_pct']:.0f}% 分位熔断"
            orders.append(_sell_order(code, name, count, price, reason))
            reasons.append(f"{name}({code}) {reason} 清仓")
            continue

        # 1. 灾难止损：现价 < 成本 × 0.88（动量证伪前的最后防线）
        if entry_price > 0 and price < entry_price * CATASTROPHIC_STOP:
            orders.append(_sell_order(code, name, count, price,
                                      f"灾难止损（低于成本×{CATASTROPHIC_STOP}）"))
            reasons.append(f"{name}({code}) 灾难止损清仓")
            continue

        # 2. 相对强度转负：跑输沪深300 且 20 日绝对收益为负 → 动量证伪
        ex20 = res.get("excess_20d")
        if ex20 is not None and ex20 < 0 and res["ret_20d"] < 0:
            orders.append(_sell_order(code, name, count, price,
                                      f"相对强度转负（超额{ex20:+.1f}%，20日{res['ret_20d']:+.1f}%）"))
            reasons.append(f"{name}({code}) 相对强度转负 清仓")
            continue

        # 3. 技术破位：收盘 < MA20 且 5 日均量 < 20 日均量（缩量阴跌）
        if res.get("weak_break"):
            orders.append(_sell_order(code, name, count, price, "跌破 MA20 且缩量"))
            reasons.append(f"{name}({code}) 技术破位清仓")
            continue

        # 4. 连续 2 次周度检查跌出前 5（跨周记忆，排名历史独立于门控）
        industry = res.get("industry") or name
        if _rank_exit_triggered(industry, current_top5):
            orders.append(_sell_order(code, name, count, price,
                                      f"动量排名连续{RANK_EXIT_STRIKES}次跌出前{RANK_EXIT_WINDOW}"))
            reasons.append(f"{name}({code}) 动量排名跌出 清仓")

    return orders, reasons


def build_buy_orders(results: List[dict], held_codes: set, total_assets: float,
                     satellite_mv: float,
                     regime: str, block_new: bool = False) -> Tuple[List[dict], List[str]]:
    """生成卫星买入订单（关注池 ∩ 放量突破 ∩ 环境许可，拥挤度 ≥80% 分位减半仓）。

    环境许可 = regime 为 trending_up / weak_up（其余状态禁买、不清仓）；
    风格状态门控 block_new（真空/退潮期）同样禁新开。
    估值过滤用候选 ETF 自身行业 PE 分位 < 90%，不用全市场 PE。
    """
    if block_new:
        return [], []
    if regime not in ("trending_up", "weak_up"):
        return [], []

    budget = total_assets * SATELLITE_BUDGET_RATIO - satellite_mv
    if budget <= 0:
        return [], []

    candidates = [r for r in results
                  if r["breakout"] and r.get("pool")
                  and r["code"] not in held_codes
                  and (r["pe_pct"] is None or r["pe_pct"] < PE_LOCK)
                  and (r.get("crowding_pct") is None or r["crowding_pct"] < CROWD_EXIT)]
    candidates.sort(key=lambda r: r["momentum_score"], reverse=True)

    # 同叙事槽位只保留动量分最高的一只（行业 β 或叙事 β，不重复押注）。
    # 2026-09-16：去重键优先用叙事组 group（如 AI算力组内 AI/芯片/通信互相镜像），
    # 无组回退行业——见 update_standards.md 叙事重叠纪律。
    seen_slots: set = set()
    deduped = []
    for r in candidates:
        slot = r.get("group") or r.get("industry") or r["code"]
        if slot in seen_slots:
            continue
        seen_slots.add(slot)
        deduped.append(r)
    candidates = deduped

    slots = max(0, MAX_HOLDINGS - len(held_codes))
    selected = candidates[:slots]
    if not selected:
        return [], []

    orders, notes = [], []
    for r in selected:
        per_position = budget / len(selected)
        crowded = r.get("crowding_pct") is not None and r["crowding_pct"] >= CROWD_HALF
        if crowded:
            per_position *= 0.5  # 拥挤度 ≥80% 分位：新开仓减半
        price = r["close"]
        if price <= 0:
            continue
        shares = max(100, int(per_position / price / 100) * 100)
        crowd_note = " 拥挤≥80%分位减半" if crowded else ""
        orders.append(RotationOrder(code=r["code"], name=r["name"], action="buy",
                                    shares=shares, price=price, amount=shares * price,
                                    reason=f"动量第{r['momentum_rank']}名 + 放量突破"
                                           f"（动量分{r['momentum_score']}）{crowd_note}"))
        ex_note = f" 超额20日{r['excess_20d']:+.1f}%" if r.get("excess_20d") is not None else ""
        notes.append(f"{r['name']}({r['code']}) 动量第{r['momentum_rank']}名{ex_note} "
                     f"突破日{r['chg_pct']:+.1f}%")

    return orders, notes


# ── 汇总 ──

def analyze_satellite(positions: List[dict], total_assets: float,
                      regime: str,
                      client=None, state_gate: Optional[dict] = None) -> dict:
    """卫星仓全流程：排名 → 卖出 → 买入。不执行交易，只出指令。

    持仓成本用模拟仓 cost_price（无本地状态文件）；"连续 2 次跌出前 5"
    依赖 data/momentum_rank_history.json（analyze_universe 内落盘）。

    标的口径与再平衡引擎一致：卫星标的 = ETF_INDUSTRY_MAP − 核心基准代码，
    核心仓标的（如 516560 养老ETF）不纳入卫星买卖与市值统计。

    PE 口径：逐只行业 PE（≥90% 锁仓/禁买），全市场 PE 不参与卫星决策。
    state_gate 为风格状态门控（style_state.satellite_state_gate 输出）：
    真空/退潮期 force_flat 强制清仓 + block_new 禁新开；None 时门控不启用。
    """
    results = analyze_universe(update_rank_history=True)
    for r in results:
        r["pe_pct"] = _pe_pct_for(r)

    # 卫星标的 = 行业清单 − 核心基准（核心仓由再平衡引擎管理）
    from src.etf.config import get_rotation_universe_codes
    sat_codes = get_rotation_universe_codes()
    results = [r for r in results if r["code"] in sat_codes]

    gate = state_gate or {}
    force_flat = bool(gate.get("force_flat"))
    block_new = bool(gate.get("block_new"))

    sells, sell_notes = build_sell_orders(results, positions, {},
                                          force_flat=force_flat)

    universe_codes = {r["code"] for r in results}
    satellite_mv = sum(float(p.get("market_value", 0) or 0)
                       for p in positions if p.get("code", "") in universe_codes)
    held_codes = {p.get("code") for p in positions
                  if p.get("code", "") in universe_codes and int(p.get("count", 0) or 0) > 0}

    buys, buy_notes = build_buy_orders(results, held_codes, total_assets, satellite_mv,
                                       regime, block_new=block_new)

    # 动量排名（选行业层结果，周报观察章节与调仓共用）
    ranked = sorted([r for r in results if r["momentum_score"] > 0],
                    key=lambda r: r["momentum_score"], reverse=True)

    return {
        "results": results,
        "sells": sells,
        "buy_orders": buys,
        "sell_notes": sell_notes,
        "buy_notes": buy_notes,
        "ranked": ranked,
        "satellite_mv": satellite_mv,
        "held_codes": held_codes,
        "state_gate": state_gate,
    }
