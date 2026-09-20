# -*- coding: utf-8 -*-
"""
===================================
妙想模拟仓 — 持仓公共工具
===================================

无状态设计：持仓起点（成本/买入日/股数）一律以模拟仓为事实来源，
趋势回踩卖出信号（src/pullback_trend/sell_rules.py）与 ETF 火箭引擎
（src/etf/industry_momentum.py）共用以下推导逻辑。
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# A 股股票代码前缀白名单（基金/ETF/债券等非股票前缀不含在内，无重叠）
_A_STOCK_PREFIXES = (
    "600", "601", "603", "605", "688", "689",  # 沪主板 + 科创板
    "000", "001", "002", "003",                # 深主板
    "300", "301",                              # 创业板
    "43", "83", "87", "88", "920",             # 北交所/新三板
)


def is_a_stock_code(code: str) -> bool:
    """判断 6 位代码是否为 A 股股票（按前缀白名单，排除 ETF/基金/债券）。"""
    c = str(code or "").strip().split(".")[0]
    return len(c) == 6 and c.startswith(_A_STOCK_PREFIXES)


def get_last_buy_dates_safe(client) -> Dict[str, str]:
    """安全推导持仓买入日期 {code: YYYY-MM-DD}。

    历史委托接口不可用时降级返回空字典，调用方跳过日期类规则
    （与 industry_momentum「无状态设计」口径一致）。
    """
    try:
        return client.get_last_buy_dates()
    except Exception:
        logger.warning("历史委托查询失败，买入日期类规则降级跳过", exc_info=True)
        return {}


def filter_held_positions(positions: List[dict], min_count: int = 0) -> List[dict]:
    """过滤有效持仓：代码非空且股数 > min_count。"""
    held: List[dict] = []
    for p in positions or []:
        code = str(p.get("code", "") or "").strip()
        if not code:
            continue
        if int(p.get("count", 0) or 0) <= min_count:
            continue
        held.append(p)
    return held


def filter_stock_positions(positions: List[dict], min_count: int = 0) -> List[dict]:
    """过滤出当前持仓的 A 股股票（排除 ETF/基金/债券等非股票持仓）。

    趋势策略只对股票持仓输出卖出信号；ETF 持仓由 ETF 系统独立管理。
    """
    return [
        p for p in filter_held_positions(positions, min_count)
        if is_a_stock_code(p.get("code", ""))
    ]


def position_profit_pct(p: dict) -> float:
    """持仓盈亏百分比；接口缺失时用成本价/现价兜底计算。"""
    pct = p.get("profit_pct")
    if pct is not None:
        try:
            return float(pct)
        except (TypeError, ValueError):
            pass
    cost = float(p.get("cost_price", 0) or 0)
    price = float(p.get("current_price", 0) or 0)
    if cost > 0 and price > 0:
        return (price / cost - 1) * 100
    return 0.0
