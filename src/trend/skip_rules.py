# -*- coding: utf-8 -*-
"""
股票跳过规则引擎（选股名单"趋势破坏"跳过）

规则清单（对应 strategy/trend_strategy.md「选股名单 → 当日跳过」）：
    R1 [已实现] 连续2天收盘跌破10日线
    R2 [已实现] 放量长阴破趋势（单日跌幅≥5% 且 量比≥2 且 收盘<MA10）
    R3 [已实现] 情绪过热（近5日换手均值 > 5% 且 > 近20日均值 × 2）
    R4 [已实现] 流动性枯竭（近5日平均换手 < 1% 且无单日 ≥ 3%）
    R5 [未实现] 板块明显退潮（跳过条件已列，本模块未做）

两条约定：
1. **逐条独立检查**：check_skip_rules_detail() 对全部规则各判一次，不短路。
   这样报告能给出每条规则「检查 N 只 / 触发 N 只」，让"跳过 N 只"这个数字可解释。
   对外主接口 check_skip_rules() 为短路语义（取优先级最高的触发项）。
2. **未执行 ≠ 未触发**：依赖换手率的 R3/R4 在 turnover_rate 列缺失时是"跳过"，
   未实现的 R5 是"未实现"，二者在报告里必须区分显示——
   否则用户会以为"规则跑了、股票没问题"，实际是根本没判。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── 阈值（集中配置，改这里即改策略）──
BIG_DROP_PCT = -5.0        # R2 单日跌幅阈值(%)
BIG_DROP_VOL_RATIO = 2.0   # R2 量比阈值
EUPHORIA_TR_5D = 5.0       # R3 近5日换手均值下限(%)
EUPHORIA_MULTIPLE = 2.0    # R3 相对近20日均值的倍数
ILLIQUID_TR_5D = 1.0       # R4 近5日平均换手上限(%)
ILLIQUID_ACTIVE_TR = 3.0   # R4 "有活跃换手"的单日换手门槛(%)

# R3/R4 需要的最小 K 线数
BARS_FOR_TR_20 = 20
BARS_FOR_TR_5 = 5


def _n(value: Optional[float], nd: int = 2) -> Optional[float]:
    """float 统一 2 位小数（报告口径）。"""
    return None if value is None else round(float(value), nd)


# ==================== 规则定义 ====================

@dataclass(frozen=True)
class SkipRuleDef:
    """跳过规则的静态定义。

    Attributes:
        rule_id: 规则编号（R1-R5）
        name: 规则名称
        implemented: 是否已实现（False 时在报告中标注「未实现」）
    """
    rule_id: str
    name: str
    implemented: bool = True


SKIP_RULES: Tuple[SkipRuleDef, ...] = (
    SkipRuleDef("R1", "连续2天收盘跌破10日线"),
    SkipRuleDef("R2", "放量长阴破趋势（跌幅≥5% + 量比≥2）"),
    SkipRuleDef("R3", "情绪过热（近5日换手>5% 且 >近20日均值×2）"),
    SkipRuleDef("R4", "流动性枯竭（近5日均换手<1% 且无活跃日）"),
    SkipRuleDef("R5", "板块明显退潮", implemented=False),
)


@dataclass
class RuleCheck:
    """单只股票在单条规则上的检查结果。

    Attributes:
        rule_id/name: 规则标识
        implemented: 规则是否已实现
        executed: 是否实际执行了判定（数据缺失/未实现时为 False）
        triggered: 是否触发跳过
        reason: 触发原因或跳过原因
    """
    rule_id: str
    name: str
    implemented: bool = True
    executed: bool = True
    triggered: bool = False
    reason: str = ""

    @property
    def status(self) -> str:
        """报告用的状态标签。"""
        if not self.implemented:
            return "未实现"
        if not self.executed:
            return "跳过（数据缺失）"
        return "触发" if self.triggered else "通过"


@dataclass
class RuleStat:
    """单条规则在选股名单上的聚合统计（「检查 N 只 / 触发 N 只」）。"""
    rule_id: str
    name: str
    implemented: bool = True
    checked: int = 0     # 实际执行判定的股票数
    triggered: int = 0   # 触发跳过的股票数
    skipped: int = 0     # 因数据缺失跳过的股票数
    reasons: List[str] = field(default_factory=list)  # 触发明细，最多留 5 条

    def add(self, check: RuleCheck, stock: str) -> None:
        if not check.implemented:
            return  # 未实现的规则不参与"检查/触发"计数
        if not check.executed:
            self.skipped += 1
            return
        self.checked += 1
        if check.triggered:
            self.triggered += 1
            if len(self.reasons) < 5:
                self.reasons.append(f"{stock}：{check.reason}")

    def summary(self) -> str:
        """「检查 N 只 / 触发 N 只」文案。"""
        if not self.implemented:
            return "未实现"
        text = f"检查 {self.checked} 只 / 触发 {self.triggered} 只"
        if self.skipped:
            text += f"（{self.skipped} 只数据缺失跳过）"
        return text


class SkipStats:
    """按规则聚合选股名单的跳过检查统计。"""

    def __init__(self) -> None:
        self._stats: Dict[str, RuleStat] = {
            r.rule_id: RuleStat(r.rule_id, r.name, r.implemented)
            for r in SKIP_RULES
        }

    def record(self, stock: str, checks: List[RuleCheck]) -> None:
        """记录一只股票的逐条检查结果。"""
        for c in checks:
            stat = self._stats.get(c.rule_id)
            if stat is not None:
                stat.add(c, stock)

    def rows(self) -> List[RuleStat]:
        """按规则编号顺序返回统计。"""
        return [self._stats[r.rule_id] for r in SKIP_RULES]

    def log_summary(self) -> None:
        """把逐条统计写进日志。"""
        for stat in self.rows():
            logger.info(f"  跳过规则 {stat.rule_id} {stat.name}: {stat.summary()}")


# ==================== 检查实现 ====================

def check_skip_rules_detail(code: str, df: Optional[pd.DataFrame]) -> List[RuleCheck]:
    """逐条检查全部跳过规则（不短路），返回每条规则各自的结果。

    与 check_skip_rules 的区别：
    - 本函数跑完全部 5 条规则，供报告输出「检查 N 只 / 触发 N 只」；
    - 数据缺失（换手率列缺失、K 线不足）时对应规则 executed=False，
      而不是静默当成"未触发"。

    Args:
        code: 股票代码（仅用于日志）
        df: 已排序并计算 MA5/MA10/MA20 的日线 DataFrame

    Returns:
        按 SKIP_RULES 顺序排列的 RuleCheck 列表
    """
    results = [
        RuleCheck(r.rule_id, r.name, implemented=r.implemented, executed=False)
        for r in SKIP_RULES
    ]
    by_id = {r.rule_id: r for r in results}

    if df is None or len(df) < 2:
        for r in results:
            r.reason = "K线数据不足（<2条）"
        return results

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    close = latest.get('close')
    prev_close = prev.get('close')
    ma10 = latest.get('ma10')
    prev_ma10 = prev.get('ma10')
    volume = latest.get('volume', 0)
    prev_volume = prev.get('volume', 1)
    volume_ratio = volume / prev_volume if prev_volume > 0 else 1

    # ── R1 连续2天收盘跌破10日线 ──
    r1 = by_id["R1"]
    if (close is not None and ma10 is not None and not pd.isna(ma10)
            and prev_close is not None and prev_ma10 is not None and not pd.isna(prev_ma10)):
        r1.executed = True
        if close < ma10 and prev_close < prev_ma10:
            r1.triggered = True
            r1.reason = (
                f"连续2天收盘跌破10日线（昨{_n(prev_close)}<{_n(prev_ma10)}，"
                f"今{_n(close)}<{_n(ma10)}）"
            )
    else:
        r1.reason = "MA10 或收盘价缺失"

    # ── R2 放量长阴破趋势 ──
    r2 = by_id["R2"]
    if close is not None and prev_close is not None and prev_close > 0:
        pct_change = (close - prev_close) / prev_close * 100
        if ma10 is not None and not pd.isna(ma10):
            r2.executed = True
            if pct_change <= BIG_DROP_PCT and volume_ratio >= BIG_DROP_VOL_RATIO and close < ma10:
                r2.triggered = True
                r2.reason = (
                    f"放量长阴破趋势（跌幅{_n(pct_change)}%，量比{_n(volume_ratio)}）"
                )
        else:
            r2.reason = "MA10 缺失"
    else:
        r2.reason = "收盘价缺失"

    # ── R3/R4 依赖换手率列 ──
    has_tr = 'turnover_rate' in df.columns

    # ── R3 情绪过热（双条件 AND：绝对值 + 相对倍数）──
    #    只看倍数会在低基数下虚高误杀（如 3.8%/1.5%=2.5 倍，换手本身并不高）
    r3 = by_id["R3"]
    if not has_tr:
        r3.reason = "换手率列缺失"
    elif len(df) < BARS_FOR_TR_20:
        r3.reason = f"K线不足{BARS_FOR_TR_20}条（仅{len(df)}条）"
    else:
        r3.executed = True
        tr_5 = df['turnover_rate'].iloc[-5:].mean()
        tr_20 = df['turnover_rate'].iloc[-20:].mean()
        if pd.notna(tr_5) and pd.notna(tr_20) and tr_20 > 0:
            if tr_5 > EUPHORIA_TR_5D and tr_5 >= tr_20 * EUPHORIA_MULTIPLE:
                r3.triggered = True
                r3.reason = (
                    f"情绪过热（近5日换手{_n(tr_5)}%>{EUPHORIA_TR_5D}% 且是近20日均"
                    f"{_n(tr_20)}%的{_n(tr_5 / tr_20, 1)}倍）"
                )
        else:
            r3.executed = False
            r3.reason = "换手率数据无效"

    # ── R4 流动性枯竭 ──
    r4 = by_id["R4"]
    if not has_tr:
        r4.reason = "换手率列缺失"
    elif len(df) < BARS_FOR_TR_5:
        r4.reason = f"K线不足{BARS_FOR_TR_5}条（仅{len(df)}条）"
    else:
        recent_tr = df['turnover_rate'].iloc[-5:]
        avg_tr = recent_tr.mean()
        if pd.notna(avg_tr):
            r4.executed = True
            if avg_tr < ILLIQUID_TR_5D and (recent_tr >= ILLIQUID_ACTIVE_TR).sum() == 0:
                r4.triggered = True
                r4.reason = f"5日平均换手{_n(avg_tr)}%过低且无活跃换手"
        else:
            r4.reason = "换手率数据无效"

    # ── R5 板块明显退潮：跳过条件已列出，本模块未实现 ──
    by_id["R5"].reason = "待实现：需接入板块行情并定义退潮阈值（见 strategy/trend_strategy.md）"

    if not has_tr:
        logger.warning(f"  {code}: 换手率列缺失，跳过规则 R3/R4 已跳过")
    skipped = [r.rule_id for r in results if r.implemented and not r.executed]
    if skipped:
        logger.warning(f"  {code}: 跳过规则 {', '.join(skipped)} 未执行（数据缺失）")

    return results


def check_skip_rules(code: str, df: Optional[pd.DataFrame]) -> Tuple[bool, str]:
    """
    检查股票是否应被跳过（短路语义：命中即返回，按 R1→R4 优先级）。

    Args:
        code: 股票代码（仅用于日志）
        df: 已排序并计算 MA5/MA10/MA20 的日线 DataFrame

    Returns:
        (是否跳过, 跳过原因)
    """
    for check in check_skip_rules_detail(code, df):
        if check.triggered:
            return True, check.reason
    return False, ""
