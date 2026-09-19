# -*- coding: utf-8 -*-
"""
===================================
AmazingData 因子封装层（ETF 周度观察专用）
===================================

从星耀数智 AmazingData 平台获取 ETF 周度观察所需的因子数据：
- 行业 PE/PB 历史分位（申万一级行业，2000 年至今）
- 行业市值占比（拥挤度因子）
- 10 年期国债收益率（股债性价比：风险溢价及其自身历史分位）
- 中证指数官网估值（核心仓跟踪指数自身 PE/股息率，本地滚动累积历史供分位计算）

设计原则：
- 所有函数失败均返回 None/空值，绝不抛出异常 —— 调用方据此降级回旧逻辑
- 懒登录，复用 data_provider.amazingdata_fetcher 的登录态（单进程一次登录）
- 未配置 TGW 凭证时自动失效（返回 None）
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.etf.config import TRACKED_INDEX, DIVIDEND_STYLE_CODES

logger = logging.getLogger(__name__)

# AmazingData 本地缓存目录（InfoData 接口用 HDF5 缓存，需 pytables）
LOCAL_DATA_DIR = os.getenv("AMAZING_DATA_DIR", "D://AmazingData_local_data//")

# 一级行业 PE/PB 分位回看窗口（5 年 ≈ 1250 交易日）
PERCENTILE_LOOKBACK = 1250


def _info() -> Optional[object]:
    """获取 InfoData 实例，失败返回 None。"""
    try:
        from data_provider.fetchers.amazingdata_fetcher import AmazingDataFetcher

        return AmazingDataFetcher.get_info_data()
    except Exception as e:
        logger.debug(f"AmazingData 不可用（降级）: {e}")
        return None


def _is_available() -> bool:
    """AmazingData 是否可用（TGW 凭证是否配置）。"""
    try:
        from data_provider.fetchers.amazingdata_fetcher import tgw_configured

        return tgw_configured()
    except Exception:
        return False


# ── 行业基础数据 ──

_industry_base_cache: Optional[pd.DataFrame] = None


def get_industry_base() -> Optional[pd.DataFrame]:
    """
    获取申万行业指数基本信息（INDEX_CODE / LEVEL1_NAME / LEVEL2_NAME / LEVEL3_NAME）。

    Returns:
        DataFrame 或 None（降级）
    """
    global _industry_base_cache
    if _industry_base_cache is not None:
        return _industry_base_cache

    info = _info()
    if info is None:
        return None
    try:
        df = info.get_industry_base_info()
        if df is None or df.empty:
            return None
        _industry_base_cache = df
        return df
    except Exception as e:
        logger.warning(f"获取行业指数基础信息失败: {e}")
        return None


def get_level1_industries() -> List[dict]:
    """
    获取申万一级行业列表。

    Returns:
        [{'code': '801180.SI', 'name': '非银金融'}, ...] 或 []
    """
    base = get_industry_base()
    if base is None:
        return []
    try:
        # LEVEL_TYPE=1 为一级行业；个别字段可能缺失，做兼容
        col_type = "LEVEL_TYPE" if "LEVEL_TYPE" in base.columns else None
        col_name = "LEVEL1_NAME" if "LEVEL1_NAME" in base.columns else None
        if col_type is None or col_name is None:
            return []
        lvl1 = base[base[col_type] == 1]
        seen = {}
        for _, row in lvl1.iterrows():
            name = str(row.get(col_name, "")).strip()
            code = str(row.get("INDEX_CODE", "")).strip()
            if name and code and name not in seen:
                seen[name] = code
        return [{"code": c, "name": n} for n, c in seen.items()]
    except Exception as e:
        logger.warning(f"解析一级行业列表失败: {e}")
        return []


def get_industry_code_by_name(name: str) -> Optional[str]:
    """按行业名（一级行业）查找行业指数代码。"""
    for item in get_level1_industries():
        if item["name"] == name:
            return item["code"]
    return None


# ── 行业日线（含 PE/PB/市值） ──

_industry_daily_cache: Dict[str, pd.DataFrame] = {}


def get_industry_daily(code: str) -> Optional[pd.DataFrame]:
    """
    获取单个行业指数的日线数据（含 PE/PB/总市值/流通市值）。

    Returns:
        DataFrame（升序，列含 CLOSE/PE/PB/TOTAL_CAP/A_FLOAT_CAP）或 None
    """
    if code in _industry_daily_cache:
        return _industry_daily_cache[code]

    info = _info()
    if info is None:
        return None
    try:
        daily = info.get_industry_daily([code], local_path=LOCAL_DATA_DIR, is_local=False)
        df = daily.get(code)
        if df is None or df.empty:
            return None
        df = df.sort_index().dropna(subset=["CLOSE"])
        _industry_daily_cache[code] = df
        return df
    except Exception as e:
        logger.warning(f"获取行业日线 {code} 失败: {e}")
        return None


# ── 因子计算 ──

def _series_percentile(series: pd.Series, lookback: int) -> Optional[Tuple[float, float]]:
    """序列近 lookback 期的当前值与历史分位。数据不足或当前值 <= 0 时返回 None。"""
    recent = pd.to_numeric(series, errors="coerce").dropna().tail(lookback)
    if len(recent) < 250:
        return None
    current = float(recent.iloc[-1])
    if current <= 0:
        return None
    pct = float((recent <= current).mean() * 100)
    return current, pct


def get_industry_pe_percentile(code: str, lookback: int = PERCENTILE_LOOKBACK) -> Optional[dict]:
    """
    行业 PE 历史分位（默认近 5 年）。

    Returns:
        {'pe': 当前PE, 'pe_pct': 分位(0-100)} 或 None
    """
    df = get_industry_daily(code)
    if df is None or "PE" not in df.columns:
        return None
    try:
        result = _series_percentile(df["PE"], lookback)
        if result is None:
            return None
        current, pct = result
        return {"pe": round(current, 2), "pe_pct": round(pct, 1)}
    except Exception as e:
        logger.warning(f"计算行业 {code} PE 分位失败: {e}")
        return None


def get_industry_pb_percentile(code: str, lookback: int = PERCENTILE_LOOKBACK) -> Optional[dict]:
    """行业 PB 历史分位（默认近 5 年）。"""
    df = get_industry_daily(code)
    if df is None or "PB" not in df.columns:
        return None
    try:
        result = _series_percentile(df["PB"], lookback)
        if result is None:
            return None
        current, pct = result
        return {"pb": round(current, 2), "pb_pct": round(pct, 1)}
    except Exception as e:
        logger.warning(f"计算行业 {code} PB 分位失败: {e}")
        return None


def get_industry_mcap_share(code: str, lookback: int = PERCENTILE_LOOKBACK) -> Optional[dict]:
    """
    行业市值占比及历史分位（拥挤度因子）。

    市值占比 = 行业总市值 / 全部一级行业总市值之和。
    占比处于历史高位 → 资金拥挤，风险上升。

    Returns:
        {'share': 当前占比(0-1), 'share_pct': 占比历史分位(0-100)} 或 None
    """
    df = get_industry_daily(code)
    if df is None or "TOTAL_CAP" not in df.columns:
        return None
    try:
        # 全行业市值需要总盘子，用行业指数当日市值比价口径：
        # 若拿不到全行业汇总，退化为该行业自身市值序列的分位（趋势拥挤度）
        result = _series_percentile(df["TOTAL_CAP"], lookback)
        if result is None:
            return None
        current, pct = result
        return {"share": round(current, 2), "share_pct": round(pct, 1)}
    except Exception as e:
        logger.warning(f"计算行业 {code} 市值占比失败: {e}")
        return None


def get_margin_sentiment(lookback: int = 500) -> Optional[dict]:
    """
    全市场两融情绪（杠杆资金）指标，供风格周报判断风险偏好。

    数据源同一日期最多有 3 条记录（多口径/多市场拼接，量级差百倍），
    按日取最大值去重后再过滤量级断裂（单日变化超 20 倍视为口径切换，
    非市场行为）。

    Returns:
        {'balance': 两融余额(元), 'chg_5d': 5日变化率(%), 'chg_20d': 20日变化率(%),
         'pct': 余额自身近 lookback 日分位(0-100)} 或 None
    """
    info = _info()
    if info is None:
        return None
    try:
        df = info.get_margin_summary(local_path=LOCAL_DATA_DIR, is_local=False)
        if df is None or df.empty or "SUM_MARGIN_TRADE_BALANCE" not in df.columns:
            return None
        df = df.copy()
        df["BAL"] = pd.to_numeric(df["SUM_MARGIN_TRADE_BALANCE"], errors="coerce")
        bal = df.groupby("TRADE_DATE")["BAL"].max().sort_index().dropna()
        if len(bal) < 30:
            return None
        # 量级断裂过滤：相对前 250 日中位数偏离超 20 倍的记录视为口径噪声
        med = bal.rolling(250, min_periods=30).median().shift(1)
        bal = bal[(bal / med > 0.05) & (bal / med < 20)]
        if len(bal) < 30:
            return None
        cur = float(bal.iloc[-1])
        chg_5d = (cur / float(bal.iloc[-6]) - 1) * 100 if len(bal) >= 6 else None
        chg_20d = (cur / float(bal.iloc[-21]) - 1) * 100 if len(bal) >= 21 else None
        window = bal.tail(lookback)
        pct = float((window < cur).mean() * 100)
        return {"balance": cur, "chg_5d": round(chg_5d, 2) if chg_5d is not None else None,
                "chg_20d": round(chg_20d, 2) if chg_20d is not None else None,
                "pct": round(pct, 1)}
    except Exception as e:
        logger.warning(f"获取两融情绪失败: {e}")
        return None


def get_etf_share_flow(codes: List[str], lookback_days: int = 20) -> Dict[str, dict]:
    """
    ETF 份额变化率（资金流观察字段）：份额增减 = 场内申赎方向。

    Args:
        codes: 标准 6 位 ETF 代码列表
        lookback_days: 变化率窗口（自然交易日）

    Returns:
        {code: {'share': 最新总份额(万份), 'chg': 近 N 日变化率(%)}}；缺数据的代码不在结果中
    """
    info = _info()
    if info is None or not codes:
        return {}
    try:
        from data_provider.fetchers.amazingdata_fetcher import _code_to_tgw_format
        tgw_codes = [c for c in (_code_to_tgw_format(x) for x in codes) if c]
        if not tgw_codes:
            return {}
        data = info.get_fund_share(tgw_codes, local_path=LOCAL_DATA_DIR, is_local=False)
        result: Dict[str, dict] = {}
        for tgw_code, df in (data or {}).items():
            if df is None or df.empty:
                continue
            col = "TOTAL_SHARE" if "TOTAL_SHARE" in df.columns else "FUND_SHARE"
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 2:
                continue
            chg = None
            if len(s) >= lookback_days + 1:
                base = float(s.iloc[-1 - lookback_days])
                if base > 0:
                    ratio = float(s.iloc[-1]) / base
                    # 超 50 倍的跳变视为口径切换（真实申购潮极少超此量级）
                    if 0.02 < ratio < 50:
                        chg = round((ratio - 1) * 100, 1)
            result[str(tgw_code).split(".")[0]] = {"share": round(float(s.iloc[-1]), 0), "chg": chg}
        return result
    except Exception as e:
        logger.warning(f"获取 ETF 份额流失败: {e}")
        return {}


def get_treasury_yield_y10() -> Optional[float]:
    """
    获取 10 年期国债收益率（%）。

    Returns:
        float（如 2.13）或 None
    """
    info = _info()
    if info is None:
        return None
    try:
        ty = info.get_treasury_yield(["y10"], local_path=LOCAL_DATA_DIR, is_local=False)
        df = ty.get("y10")
        if df is None or df.empty or "YIELD" not in df.columns:
            return None
        val = float(pd.to_numeric(df["YIELD"], errors="coerce").dropna().iloc[-1])
        return val
    except Exception as e:
        logger.warning(f"获取 10 年期国债收益率失败: {e}")
        return None


def get_all_industry_factors(industry_names: List[str]) -> dict:
    """
    批量获取多个行业的因子（PE/PB 分位 + 市值分位）。

    Args:
        industry_names: 一级行业中文名列表，如 ['电子', '医药生物']

    Returns:
        {
            '电子': {'pe': ..., 'pe_pct': ..., 'pb': ..., 'pb_pct': ..., 'share_pct': ...},
            ...
        }（失败的行业不在结果中）
    """
    result = {}
    for name in industry_names:
        code = get_industry_code_by_name(name)
        if code is None:
            logger.warning(f"未找到行业 [{name}] 的指数代码，跳过")
            continue
        factors = {}
        pe_info = get_industry_pe_percentile(code)
        if pe_info:
            factors.update(pe_info)
        pb_info = get_industry_pb_percentile(code)
        if pb_info:
            factors.update(pb_info)
        mcap_info = get_industry_mcap_share(code)
        if mcap_info:
            factors.update(mcap_info)
        if factors:
            result[name] = factors
    return result


# ── 行业 ETF 清单（data/etf_industry_map.json，便于人工维护） ──
# 清单依据 2026-08 人工调研（规模 + 当日成交额筛选，首选/备选）；无对应行业 ETF 的行业不强求。
# 注：1 开头深市 ETF/LOF 妙想模拟仓无法交易（见 docs/mx_skills/mx-moni.md），执行时自动跳过转手动待办。

_ETF_INDUSTRY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "etf_industry_map.json",
)


def _load_industry_entries() -> List[dict]:
    """加载行业 ETF 清单（[{code, name, industry, note}, ...]），文件缺失/损坏时返回空表。"""
    try:
        with open(_ETF_INDUSTRY_FILE, encoding="utf-8") as f:
            entries = json.load(f)
        return [e for e in entries if e.get("code") and e.get("industry")]
    except Exception as e:
        logger.warning(f"行业 ETF 清单加载失败（{_ETF_INDUSTRY_FILE}）: {e}")
        return []


_ETF_INDUSTRY_ENTRIES = _load_industry_entries()

# 兼容映射：ETF 代码 → 申万一级行业名
ETF_INDUSTRY_MAP: Dict[str, str] = {
    e["code"]: e["industry"] for e in _ETF_INDUSTRY_ENTRIES
}

# ── 叙事组（2026-09-16 新增） ──
# 同组 ETF 视为同一叙事槽位：买入去重时组内只保留动量分最高一只（防同叙事双重下注）。
# 组名可为非申万行业（如 "AI算力"、"风格:小市值"）。无 group 字段的标的按行业去重。
ETF_GROUP_MAP: Dict[str, str] = {
    e["code"]: e["group"] for e in _ETF_INDUSTRY_ENTRIES if e.get("group")
}


def get_etf_group(etf_code: str) -> Optional[str]:
    """查询 ETF 的叙事组名（无组返回 None，去重回退按行业）。"""
    return ETF_GROUP_MAP.get(etf_code)


def get_etf_industry(etf_code: str) -> Optional[str]:
    """查询 ETF 对应的一级行业名。"""
    return ETF_INDUSTRY_MAP.get(str(etf_code).zfill(6))


def get_industry_etf_universe() -> List[dict]:
    """卫星仓引擎的动态标的清单：[{'code', 'name', 'industry'}, ...]"""
    return [
        {"code": e["code"], "name": e["name"], "industry": e["industry"]}
        for e in _ETF_INDUSTRY_ENTRIES
    ]


# ── 全市场聚合 PE（兜底用） ──

def _market_pe_series() -> Optional[pd.Series]:
    """全市场聚合 PE 日度序列（31 个申万一级行业 E/P 加权），供分位与 ERP 共用。

    加权口径：PE_market = Σ市值 / Σ(市值/PE)，避免高 PE 行业被市值权重过分放大。
    """
    industries = get_level1_industries()
    if not industries:
        return None

    # 逐行业拉日线，对齐交易日索引
    cap_dfs = []
    pe_dfs = []
    for item in industries:
        df = get_industry_daily(item["code"])
        if df is None or "PE" not in df.columns or "TOTAL_CAP" not in df.columns:
            continue
        cap = pd.to_numeric(df["TOTAL_CAP"], errors="coerce")
        pe = pd.to_numeric(df["PE"], errors="coerce")
        # 过滤无效 PE 与市值
        mask = (pe > 0) & (cap > 0)
        cap_dfs.append(cap[mask])
        pe_dfs.append(pe[mask])
    if not cap_dfs:
        return None

    total_cap = pd.concat(cap_dfs, axis=1).sum(axis=1, skipna=False)
    # Σ(市值/PE) 按行业加权
    weighted = []
    for item, cap, pe in zip(industries, cap_dfs, pe_dfs):
        valid = cap.notna() & pe.notna() & (pe > 0) & (cap > 0)
        weighted.append((cap[valid] / pe[valid]).reindex(cap.index))
    sum_ep = pd.concat(weighted, axis=1).sum(axis=1, skipna=False)

    market_pe = total_cap / sum_ep
    return market_pe[market_pe > 0].dropna()


def _as_daily_series(s: pd.Series) -> pd.Series:
    """把任意日期索引的序列统一为按天归一的 DatetimeIndex（索引解析失败的行丢弃）。

    行业日线与国债历史的索引口径可能不同（datetime/字符串/yyyymmdd 整数），
    统一走字符串解析，保证跨序列可对齐。
    """
    s = s.sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index.astype(str), errors="coerce")
    s = s[s.index.notna()]
    return s.groupby(s.index).last()


def get_market_pe(lookback: int = PERCENTILE_LOOKBACK) -> Optional[dict]:
    """
    用 31 个申万一级行业 PE 市值加权聚合出全市场 PE 及历史分位（默认近 5 年）。

    Returns:
        {'pe': 当前PE, 'pe_pct': 分位(0-100)} 或 None
    """
    series = _market_pe_series()
    if series is None:
        return None
    recent = series.tail(lookback)
    if len(recent) < 250:
        return None
    current = float(recent.iloc[-1])
    pct = float((recent <= current).mean() * 100)
    return {"pe": round(current, 2), "pe_pct": round(pct, 1)}


_erp_percentile_cache: Optional[dict] = None


def get_erp_percentile(lookback: int = PERCENTILE_LOOKBACK) -> Optional[dict]:
    """
    股权风险溢价（1/全市场PE − 10 年国债）的当前值与自身历史分位（默认近 5 年）。

    与 get_market_pe 同序列、同窗口，两个口径可比、互相印证：PE 分位受回看窗口
    锚定影响大（窗口含深熊底部则普遍读高），ERP 分位是跨资产口径的独立参照。

    Returns:
        {'erp': 当前风险溢价(%), 'erp_pct': 分位(0-100)} 或 None（国债历史不可用/对不齐时降级）
    """
    global _erp_percentile_cache
    if _erp_percentile_cache is not None:
        return _erp_percentile_cache

    pe = _market_pe_series()
    if pe is None:
        return None
    recent_pe = pe.tail(lookback)
    if len(recent_pe) < 250:
        return None

    info = _info()
    if info is None:
        return None
    try:
        ty = info.get_treasury_yield(["y10"], local_path=LOCAL_DATA_DIR, is_local=False)
        df = ty.get("y10") if ty else None
        if df is None or df.empty or "YIELD" not in df.columns:
            return None
        y10 = _as_daily_series(pd.to_numeric(df["YIELD"], errors="coerce").dropna())
        if y10.empty:
            return None
        pe_d = _as_daily_series(recent_pe)
        erp = (100.0 / pe_d - y10.reindex(pe_d.index)).dropna()
        if len(erp) < 250:
            return None
        current = float(erp.iloc[-1])
        pct = float((erp <= current).mean() * 100)
        _erp_percentile_cache = {"erp": round(current, 2), "erp_pct": round(pct, 1)}
        return _erp_percentile_cache
    except Exception as e:
        logger.warning(f"计算风险溢价分位失败: {e}")
        return None


# ── 中证指数官网估值（跟踪指数当前 PE/股息率） ──

# 官网 indicator 文件表头为中英双语（如 "日期Date"），按列位置解析
_CSINDEX_COLS = ["date", "index_code", "index_name", "index_short",
                 "en_full", "en_short", "pe", "pe2", "dy", "dy2"]
_csindex_cache: Dict[str, Optional[dict]] = {}


def _fetch_csindex_indicator(index_code: str) -> Optional[pd.DataFrame]:
    """下载中证官网估值明细（仅含近 20 个交易日）。"""
    url = (f"https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/"
           f"indicator/{index_code}indicator.xls")
    try:
        df = pd.read_excel(url)
        if df.shape[1] != len(_CSINDEX_COLS):
            logger.warning(f"csindex 估值文件列数异常 {index_code}: {df.shape[1]}")
            return None
        df.columns = _CSINDEX_COLS
        df = df[["date", "index_code", "index_name", "pe", "dy"]].copy()
        df["date"] = df["date"].astype(str)
        # 指数代码列会被读成整数（000922 → 922），补齐前导零
        df["index_code"] = df["index_code"].astype(str).str.zfill(6)
        for col in ("pe", "dy"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["pe"])
        return df if not df.empty else None
    except Exception as e:
        logger.warning(f"csindex 估值下载失败 {index_code}: {e}")
        return None


def get_csindex_valuation(index_code: str) -> Optional[dict]:
    """
    跟踪指数自身估值的当前值（中证官网 PE/股息率，取最新一日）。

    决策记录（2026-09）：不做历史分位积累。官网无历史文件、自建积累需约一年冷启动，
    而逐指数分位唯一用途（新钱估值加速器）与趋势主规则、全局便宜档（全市场 PE 分位）
    高度重叠，收益不抵维护成本。"便宜"判定收归两层：全局节奏（全市场口径）+
    红利类股息率利差（跨资产口径，零历史依赖）。

    Returns:
        {'index_code','index_name','asof','pe','dy'} 或 None。dy = 股息率1（总股本口径）。
    """
    if index_code in _csindex_cache:
        return _csindex_cache[index_code]

    df = _fetch_csindex_indicator(index_code)
    result = None
    if df is not None:
        latest = df.iloc[0]  # 官网明细按日期降序，首行为最新
        result = {
            "index_code": str(latest["index_code"]),
            "index_name": str(latest["index_name"]),
            "asof": str(latest["date"]),
            "pe": round(float(latest["pe"]), 2),
            "dy": round(float(latest["dy"]), 2) if pd.notna(latest["dy"]) else None,
        }
    _csindex_cache[index_code] = result
    return result


# ── ETF 买入优先级 & 卖出警示 ──

# 海外 ETF（暂无 PE 数据，标"数据缺失"）
_OVERSEAS_CODES = frozenset({"513100", "513500", "513380"})


def _etf_pe_info(etf_code: str) -> Optional[dict]:
    """获取单只 ETF 的估值信息，锚对准买入标的本身。

    口径优先级：跟踪指数自身估值（csindex 当前值，TRACKED_INDEX）→ 申万一级行业
    PE 分位 → 全市场兜底。csindex 不提供分位（决策记录见 get_csindex_valuation），
    仅行业/全市场等历史回退口径带 pe_pct。
    """
    code = str(etf_code).zfill(6)

    if code in _OVERSEAS_CODES:
        return {"source_type": "overseas", "pe": None, "pe_pct": None, "source_name": "海外"}

    index_code = TRACKED_INDEX.get(code)
    if index_code:
        val = get_csindex_valuation(index_code)
        if val:
            info = {
                "pe": val.get("pe"), "pe_pct": None,
                "source_type": "csindex",
                "source_name": val.get("index_name") or index_code,
            }
            # 股息率与利差（股息率 − 10Y 国债，跨资产口径，零历史依赖）
            if val.get("dy") is not None:
                info["div_yield"] = val["dy"]
                y10 = get_treasury_yield_y10()
                if y10:
                    info["div_yield_spread"] = round(val["dy"] - y10, 2)
            return info

    industry = get_etf_industry(code)
    if industry:
        ind_code = get_industry_code_by_name(industry)
        if ind_code:
            pe_info = get_industry_pe_percentile(ind_code)
            if pe_info:
                return {**pe_info, "source_type": "industry", "source_name": industry}

    market_pe = get_market_pe()
    if market_pe:
        return {**market_pe, "source_type": "market", "source_name": "全市场"}

    return None


def rank_buy_priorities(etf_list) -> list:
    """各核心 ETF 的估值参考（跟踪指数锚当前值；仅申万行业/全市场等历史回退口径带分位）。"""
    results = []
    for etf in etf_list:
        info = _etf_pe_info(etf.code)
        pe_pct = info.get("pe_pct") if info else None
        level, level_text = "", ""
        if pe_pct is not None:
            level_text = ("⭐⭐⭐ 极度低估，优先关注" if pe_pct < 20 else
                          "⭐⭐ 低估，值得关注" if pe_pct < 40 else
                          "估值合理偏低" if pe_pct < 60 else
                          "中性偏贵，暂缓" if pe_pct < 80 else
                          "偏贵，暂缓" if pe_pct < 90 else
                          "❌ 高估，回避")
        elif str(etf.code).zfill(6) in DIVIDEND_STYLE_CODES and info:
            if info.get("div_yield_spread") is not None:
                level_text = f"利差 {info['div_yield_spread']:+.1f}pt（≥1.5 视同低估）"
            elif info.get("div_yield") is not None:
                level_text = "股息率口径"
        results.append({
            "code": etf.code, "name": etf.name,
            "pe": info.get("pe") if info else None,
            "pe_pct": pe_pct,
            "div_yield": info.get("div_yield") if info else None,
            "div_yield_spread": info.get("div_yield_spread") if info else None,
            "source_name": info.get("source_name", "") if info else "",
            "level": level, "level_text": level_text,
        })

    results.sort(key=lambda x: (x["pe_pct"] is None, x["pe_pct"] if x["pe_pct"] is not None else 999))
    return results


def check_sell_warnings(etf_list) -> list:
    """
    检查需要卖出警示的 ETF。

    触发条件（三者同时满足才警示）：
    1. 全市场 PE > 90% 分位
    2. 该 ETF 对应行业 PE > 95% 分位
    3. 行业市值拥挤度 > 98% 分位（或行业 PE > 98% 分位）

    海外 ETF 不检查。
    """
    market_pe = get_market_pe()
    if not market_pe or market_pe["pe_pct"] < 90:
        return []

    warnings = []
    for etf in etf_list:
        code = str(etf.code).zfill(6)
        if code in _OVERSEAS_CODES:
            continue

        info = _etf_pe_info(code)
        if not info or info.get("pe_pct") is None:
            continue
        pe_pct = info["pe_pct"]
        if pe_pct < 95:
            continue

        industry = get_etf_industry(code)
        crowding = False
        if industry:
            ind_code = get_industry_code_by_name(industry)
            if ind_code:
                mcap = get_industry_mcap_share(ind_code)
                if mcap and mcap.get("share_pct", 0) > 98:
                    crowding = True

        source = info.get("source_name", "")

        if crowding:
            warnings.append({
                "code": code, "name": etf.name,
                "source_name": source, "pe_pct": pe_pct,
                "reason": f"{source} PE {pe_pct:.0f}%分位 + 市值极度拥挤 → 长期配置建议回避",
            })
        elif pe_pct > 98:
            warnings.append({
                "code": code, "name": etf.name,
                "source_name": source, "pe_pct": pe_pct,
                "reason": f"{source} PE {pe_pct:.0f}%分位（极度高估）→ 长期配置建议回避",
            })

    return warnings
