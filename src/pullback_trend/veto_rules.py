# -*- coding: utf-8 -*-
"""
趋势波段策略 — 负面清单（硬否决层）

定位：在"买点信号检测（signal_detector，买点 + 质量门）"之前的一道准入闸门。
任一规则触发即否决：不进信号池、不看评分。

与相邻层的职责分工：
- market_gate + cycle_stage：市场环境层，关注"今天能不能开、开多大"。
- veto_rules（本模块）：极端风险标的的"准入否决"，关注"再便宜也不能买"。
- signal_detector：买点形态与信号质量门（Cycle C1/D1 过滤 / ⑥b 中期过热等）。

规则清单（2 条硬否决，任一触发即跳过当日信号）：
    V1 [外部] 近 20 日发布过股票交易异常波动 / 风险提示公告        → skip
    V4 [行情] 近 20 日出现 ≥2 次单日跌幅 > 7%                      → skip
    V5 [外部] 近 5 日主力资金净流出 > 流通市值 1%                   → 观察项（不再否决）

动作语义：
- 统一为 skip（跳过当日信号）。选股名单每天由妙想选股重新生成，
  被否决的票当日不出信号，次日名单随新一轮选股重新判定。
- V5 为观察项：命中时以观察文案随信号展示，不拦截。

设计约定：
- 行情类规则（V4）纯本地计算，无额外 I/O，可对选股名单逐股执行。
- 外部数据规则（V1/V5）依赖妙想 API 且有日调用限额，只对"已产出信号"的候选
  惰性执行；数据缺失或解析失败一律 fail-open（记 warning 放行），避免外部
  数据源抖动导致整个系统静默黑屏。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# === 动作类型 ===
ACTION_SKIP = "skip"      # 跳过当日信号

ACTION_LABELS = {
    ACTION_SKIP: "跳过信号",
}

# === 规则阈值（集中配置，改这里即改策略）===
ANNOUNCEMENT_LOOKBACK_DAYS = 20   # V1 公告回溯天数
ANNOUNCEMENT_KEYWORDS = ("异常波动", "风险提示", "交易风险", "停牌核查")

# 当日换手率上限(%)：截面状态口径（妙想选股语句引用）。
# 妙想选股语句（trend_analysis.DEFAULT_SCREEN_KEYWORD）引用本常量，此处为单一阈值来源。
TURNOVER_DAY_MAX = 12.0
BIG_DROP_PCT = -7.0               # V4 单日跌幅阈值(%)
BIG_DROP_MIN_COUNT = 2            # V4 触发所需次数
FUND_FLOW_DAYS = 5                # V5 主力资金统计天数（已降级为观察项，见下）
FUND_FLOW_OUTFLOW_PCT = 1.0       # V5 净流出占流通市值比例阈值(%)
FROM_60D_LOW_MAX = 80.0           # 距60日最低收盘涨幅上限(%)：信号条件⑥b 用（原 V7 同口径）


@dataclass
class VetoResult:
    """否决结果。

    Attributes:
        vetoed: 是否被否决
        reasons: 触发的规则描述列表
        action: 统一为 ACTION_SKIP（跳过当日信号）
    """

    vetoed: bool = False
    reasons: List[str] = field(default_factory=list)
    action: str = ACTION_SKIP

    def add(self, reason: str) -> None:
        """记录一条触发的规则（统一跳过当日信号）。"""
        self.vetoed = True
        self.reasons.append(reason)


# ==================== 否决规则统计 ====================

@dataclass
class VetoRuleStat:
    """单条否决规则的聚合统计。"""

    rule_id: str
    name: str
    checked: int = 0       # 实际执行判定的股票数
    triggered: int = 0     # 触发否决的股票数
    skipped: int = 0       # 因数据缺失放行的股票数

    def record_checked(self) -> None:
        self.checked += 1

    def record_triggered(self) -> None:
        self.triggered += 1

    def record_skipped(self) -> None:
        self.skipped += 1

    def summary(self) -> str:
        """「检查 N 只 / 否决 N 只 / 放行 M 只」文案。"""
        parts = [f"检查 {self.checked} 只"]
        if self.triggered:
            parts.append(f"否决 {self.triggered} 只")
        if self.skipped:
            parts.append(f"放行 {self.skipped} 只（数据缺失）")
        return "，".join(parts)


class VetoStats:
    """按规则聚合负面清单检查统计。"""

    # 行情类规则
    MARKET_RULES = [
        ("V4", "20日单日跌幅>7%≥2次"),
    ]
    # 外部数据规则（V5 为观察项，不再否决）
    EXTERNAL_RULES = [
        ("V1", "20日风险公告"),
        ("V5", "5日主力净流出>1%(观察)"),
    ]

    def __init__(self) -> None:
        self._stats: Dict[str, VetoRuleStat] = {}
        for rid, rname in self.MARKET_RULES + self.EXTERNAL_RULES:
            self._stats[rid] = VetoRuleStat(rid, rname)

    def get(self, rule_id: str) -> VetoRuleStat:
        return self._stats[rule_id]

    def log_summary(self) -> None:
        """把逐条统计写进日志。"""
        for rid, _ in self.MARKET_RULES + self.EXTERNAL_RULES:
            stat = self._stats[rid]
            logger.info(f"  负面清单 {stat.rule_id} {stat.name}: {stat.summary()}")


# ==================== 行情类规则（V4）====================

def check_market_veto(
    code: str, name: str, df: Optional[pd.DataFrame],
    stats: Optional[VetoStats] = None,
) -> Tuple[VetoResult, List[str]]:
    """负面清单 — 行情类规则（V4），纯本地计算，无外部 I/O。

    Args:
        code: 股票代码
        name: 股票名称（仅用于日志）
        df: 日线 DataFrame，需含 close / date，按日期升序
        stats: 可选统计对象

    Returns:
        (VetoResult, skipped_rules) — skipped_rules 为因数据缺失未生效的规则描述列表
    """
    result = VetoResult()
    skipped: List[str] = []
    if df is None or len(df) < 2:
        return result, skipped

    df = df.sort_values('date').reset_index(drop=True)
    tag = f"{name}({code})"
    closes = df['close'].astype(float)
    pct = closes.pct_change() * 100

    # --- V4 近20日 ≥2 次单日跌幅 > 7% ---
    if stats:
        stats.get("V4").record_checked()
    if len(df) >= 20:
        big_drops = int((pct.iloc[-20:] <= BIG_DROP_PCT).sum())
        if big_drops >= BIG_DROP_MIN_COUNT:
            result.add(f"V4 近20日{big_drops}次单日跌幅>7%")

    if result.vetoed:
        logger.info(f"🚫 负面清单否决 {tag}: {'；'.join(result.reasons)} → {ACTION_LABELS[result.action]}")

    return result, skipped


# ==================== 外部数据规则（V1 公告 / V5 主力资金观察）====================

def _iter_dicts(obj: Any) -> Iterator[Dict[str, Any]]:
    """深度遍历嵌套结构，产出其中的所有 dict（用于防御式解析妙想响应）。"""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_dicts(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_dicts(item)


def _to_float(value: Any) -> Optional[float]:
    """宽松转 float，失败返回 None。"""
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def _parse_datetime(text: str) -> Optional[datetime]:
    """解析常见日期字符串，失败返回 None。"""
    text = (text or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[: len(fmt) + 2], fmt)
        except ValueError:
            continue
    return None


def _extract_news_items(resp: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """从妙想资讯搜索响应中提取 {title, date} 列表（防御式）。"""
    items: List[Dict[str, str]] = []
    if not resp:
        return items

    for obj in _iter_dicts(resp):
        title_key = next(
            (k for k in obj if "title" in str(k).lower() or "标题" in str(k)), None
        )
        if not title_key or not isinstance(obj[title_key], (str, int, float)):
            continue
        date_key = next(
            (
                k
                for k in obj
                if any(t in str(k).lower() for t in ("date", "time", "publish"))
                or "日期" in str(k)
                or "时间" in str(k)
            ),
            None,
        )
        items.append(
            {
                "title": str(obj[title_key]),
                "date": str(obj.get(date_key, "")) if date_key else "",
            }
        )
    return items


def _check_announcement_veto(
    code: str, name: str, mx_service: Any,
    stats: Optional[VetoStats] = None,
) -> Tuple[Optional[str], List[str]]:
    """V1：近 20 日发布过异常波动 / 风险提示公告 → (触发原因, skipped_rules)。"""
    skipped: List[str] = []
    if mx_service is None:
        return None, skipped

    try:
        resp = mx_service.search_news(
            f"{name}({code}) 最近一个月 股票交易异常波动公告 风险提示公告"
        )
        items = _extract_news_items(resp)
    except Exception as e:
        if stats:
            stats.get("V1").record_skipped()
        return None, skipped

    if not items:
        if stats:
            stats.get("V1").record_checked()
        return None, skipped

    cutoff = datetime.now() - timedelta(days=ANNOUNCEMENT_LOOKBACK_DAYS)
    for item in items:
        title = item.get("title", "")
        if not any(kw in title for kw in ANNOUNCEMENT_KEYWORDS):
            continue
        published = _parse_datetime(item.get("date", ""))
        if published is None or published >= cutoff:
            if stats:
                stats.get("V1").record_triggered()
            return f"V1 近{ANNOUNCEMENT_LOOKBACK_DAYS}日风险公告《{title[:24]}》", skipped

    if stats:
        stats.get("V1").record_checked()
    return None, skipped


def _check_fund_flow_observation(
    code: str, name: str, fetcher: Any,
    stats: Optional[VetoStats] = None,
) -> Tuple[Optional[str], List[str]]:
    """V5（观察项）：近 5 日主力资金净流出 > 流通市值 1% → (观察文案, skipped_rules)。

    不拦截：主力资金口径模糊、数据源漂移风险高，其价量特征已被 V4/缩量条件
    捕捉；命中仅返回观察文案，供信号描述展示。
    """
    skipped: List[str] = []
    if fetcher is None:
        return None, skipped

    tag = f"{name}({code})"

    try:
        df = fetcher.get_main_fund_flow(code, days=FUND_FLOW_DAYS)
        net_flow = df['main_net_inflow'].sum() if df is not None and not df.empty and 'main_net_inflow' in df.columns else None
    except Exception as e:
        if stats:
            stats.get("V5").record_skipped()
        return None, skipped

    if net_flow is None:
        skipped.append("V5 主力资金数据缺失")
        if stats:
            stats.get("V5").record_skipped()
        return None, skipped
    if net_flow >= 0:
        if stats:
            stats.get("V5").record_checked()
        return None, skipped

    try:
        quote = fetcher.get_realtime_quote(code)
    except Exception as e:
        if stats:
            stats.get("V5").record_skipped()
        return None, skipped

    circ_mv = getattr(quote, "circ_mv", None) if quote is not None else None
    if not circ_mv or circ_mv <= 0:
        skipped.append("V5 流通市值缺失")
        if stats:
            stats.get("V5").record_skipped()
        return None, skipped

    outflow = abs(net_flow)
    ratio = outflow / circ_mv * 100

    if ratio > FUND_FLOW_OUTFLOW_PCT:
        if stats:
            stats.get("V5").record_triggered()
        return (
            f"V5 近{FUND_FLOW_DAYS}日主力净流出{outflow / 1e8:.2f}亿，"
            f"占流通市值{ratio:.2f}%>{FUND_FLOW_OUTFLOW_PCT}%"
        ), skipped

    if stats:
        stats.get("V5").record_checked()
    return None, skipped


def check_external_veto(
    code: str,
    name: str,
    mx_service: Any = None,
    fetcher: Any = None,
    stats: Optional[VetoStats] = None,
) -> Tuple[VetoResult, List[str], List[str]]:
    """负面清单 — 外部数据规则（V1 公告否决 / V5 主力资金观察）。

    依赖妙想 API 且有日调用限额，只对已产出信号的候选惰性调用。
    数据缺失或解析失败一律 fail-open（放行）。

    Args:
        code: 股票代码
        name: 股票名称
        mx_service: MXService 实例
        fetcher: DataFetcherManager 实例
        stats: 可选统计对象

    Returns:
        (VetoResult, skipped_rules, observations) — observations 为 V5 等观察项
        命中文案（不拦截，由调用方随信号展示）
    """
    result = VetoResult()
    skipped: List[str] = []
    observations: List[str] = []

    reason, s1 = _check_announcement_veto(code, name, mx_service, stats)
    skipped.extend(s1)
    if reason:
        result.add(reason)

    reason, s2 = _check_fund_flow_observation(code, name, fetcher, stats)
    skipped.extend(s2)
    if reason:
        observations.append(reason)

    if result.vetoed:
        logger.info(f"🚫 负面清单否决 {name}({code}): {'；'.join(result.reasons)} → {ACTION_LABELS[result.action]}")
    for obs in observations:
        logger.info(f"👁️ V5 观察 {name}({code}): {obs}（已降级，不拦截）")

    return result, skipped, observations
