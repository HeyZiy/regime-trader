# -*- coding: utf-8 -*-
"""
趋势跟踪日报 — Markdown 报告生成

接收 TechnicalSignal 和市场环境数据，返回格式化 Markdown 字符串。

使用模型：T日盘后出信号 → T+1观察开盘/盘中走势 → T+1尾盘决定是否介入。
"""

import json
import logging
import string
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from src.market_state.market_gate import REGIME_CAN_OPEN, RegimeDiagnosis
from src.trend.skip_rules import SKIP_RULES, SkipStats
from src.trend.signal_detector import UNKNOWN_SECTOR, TechnicalSignal

logger = logging.getLogger(__name__)


def _style_state_display() -> Optional[str]:
    """读 data/style_state.json 返回风格状态展示文本（元层观察，A 阶段纯展示）。

    fail-soft：文件缺失/结构不符/异常一律返回 None，不阻断日报。
    """
    try:
        from src.market_state.style_state import STATE_FILE, STATE_GATE_MAX_AGE_DAYS, STATE_LABELS

        if not STATE_FILE.exists():
            return None
        st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state, data_date = st.get("state"), st.get("data_date")
        if state not in STATE_LABELS:
            return None
        streak = f"，持续第 {st.get('state_streak')} 周" if st.get("state_streak") else ""
        stale = ""
        try:
            age = (date.today() - date.fromisoformat(str(data_date))).days
            if age > STATE_GATE_MAX_AGE_DAYS:
                stale = f"，已 {age} 天未更新⚠️"
        except (TypeError, ValueError):
            pass
        return (f"{STATE_LABELS[state]}{streak}"
                f"（截至 {data_date}{stale}，规则见 strategy/style_state.md）")
    except Exception:
        return None

REGIME_DESC = {
    "trending_up":   "📈 趋势上行 — 均线多头排列",
    "weak_up":       "🌤️ 弱上行 — 站上 MA20 但非标准多头排列",
    "sideways":      "➡️ 震荡横盘 — 紧贴 MA20 震荡",
    "trending_down": "📉 趋势下行 — 均线空头排列",
    "chaos":         "🌪️ 混沌 — 方向不明",
}

# ==================== Markdown 表格渲染工具 ====================
# Markdown 表格不校验列数：多一列/少一列不会报错，只会静默渲染成一张错位的表格。
# 因此这里统一用「表头元组 + 行模板」渲染，并在每次渲染前双向校验字段与占位符。
_FORMATTER = string.Formatter()

# ── 行模板：占位符名即业务语义，加列时表头与模板必须同时改，否则 _assert_aligned 抛错 ──
_RANK_ROW = (
    "| {rank} | {stock} | {sector} | {price} | {pct} | {bias} | {vol} | {turnover} | "
    "{score} | {observation} | {levels} | {confirm} | {invalidate} | {sizing} |"
)
_RANK_HEADER = (
    "#", "股票", "板块", "价格", "涨跌", "乖离MA5", "量比", "换手", "评分",
    "操作要点", "关键价位", "确认条件", "失效条件", "仓位",
)

_PLAN_ROW = "| {rank} | {stock} | {sector} | {score} | {confirm} | {sizing} |"
_PLAN_HEADER = ("排名", "股票", "板块", "评分", "介入条件", "仓位")

# 各信号类型分表的区头与说明（与 KNOWN_SIGNAL_TYPES 对应；新增类型必须在此登记，否则 KeyError）
_SECTION_TITLES = {
    'pullback_ma5': (
        "## 🎯 缩量回踩MA5（策略首选买点）",
        "> 只做主升中的第一次像样回踩。缩量，不破5日线。",
    ),
    'pullback_ma10': (
        "## ⚠️ 回踩MA10（次优 — 需次日弱转强确认）",
        "> 已跌破5日线，回踩较深。策略要求不破5日线，此信号仅作参考。",
    ),
    'near_ma5': (
        "## 👀 缩量贴MA5（观察 — 非买点，等回踩）",
        "> 同一组 setup，但当日收涨未回踩：不追高，列入观察，等回踩MA5企稳再接。",
    ),
}

_PAIR_ROW = "| {stock} | {reason} |"
_PAIR_HEADER = ("股票", "原因")

# 跳过规则覆盖情况：逐条「检查 N 只 / 触发 N 只」，未实现的标注「未实现」
_RULE_STAT_ROW = "| {rule} | {name} | {status} | {count} | {detail} |"
_RULE_STAT_HEADER = ("规则", "内容", "状态", "检查/触发", "触发明细")

# 达标线：技术评分 ≥ 此值才算"可执行"。低于此值只作观察，不进 T+1 计划。
QUALIFY_SCORE = 60

# ── 统一仓位规则 ──
# 唯一的环境级风控约束是单笔亏损限额；仓位上限 = 亏损限额 ÷ 止损距离。
# 例：止损 5%，200 ÷ 5% = 4000 元。
LOSS_BUDGET = 200.0
MIN_STOP_LOSS_PCT = 3.0  # 止损距离下限：价格紧贴 MA10 时直接相除会算出天量仓位


def position_size(stop_loss_pct: float) -> float:
    """按亏损限额与止损距离算仓位上限（元）。

    止损距离不足 MIN_STOP_LOSS_PCT 时按下限计。
    """
    try:
        pct = max(float(stop_loss_pct), MIN_STOP_LOSS_PCT)
    except (TypeError, ValueError):
        pct = MIN_STOP_LOSS_PCT
    return LOSS_BUDGET / pct * 100


def _f(value: Any, nd: int = 2) -> str:
    """float 统一保留 2 位小数（报告口径：全篇数字精度一致，不出现 .1f/.0f 混用）。

    非数字（None / 空 / 无法转换）返回 "—"，不让异常或字面量 None 进报告。
    """
    try:
        if value is None or value == "":
            return "—"
        return f"{round(float(value), nd):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _f_pct(value: Any, nd: int = 2) -> str:
    """带正负号的百分比，同样统一 2 位小数。"""
    try:
        if value is None or value == "":
            return "—"
        return f"{round(float(value), nd):+.{nd}f}%"
    except (TypeError, ValueError):
        return "—"


def _cell(value: Any) -> str:
    """把任意值转成安全的表格单元格文本。

    - None → "—"：绝不让字面量 "None" 出现在报告里
    - 竖线/换行 → 替换：否则会撑破表格造成整行错位
    - 空串 → "—"：空单元格在视觉上就是"列少了一格"
    """
    if value is None:
        return "—"
    text = str(value).replace("|", "／").replace("\n", " ").strip()
    return text or "—"


def _placeholders(template: str) -> List[str]:
    """取出行模板中的命名占位符（'{rank}' → 'rank'）。"""
    return [name for _, name, _, _ in _FORMATTER.parse(template) if name]


def _assert_aligned(header: Tuple[str, ...], template: str) -> None:
    """渲染层断言：表头列数 == 行模板占位符数。

    加列只改表头不改模板（或反之）时立即抛错，避免产出静默错位的表格。
    """
    n_cols, n_ph = len(header), len(_placeholders(template))
    if n_cols != n_ph:
        raise ValueError(
            f"表格表头 {n_cols} 列与行模板 {n_ph} 个占位符不一致 | "
            f"表头:{header} | 模板:{template}"
        )


def _header(header: Tuple[str, ...], template: str) -> List[str]:
    """生成表头 + 分隔行（顺带做列数对齐断言）。"""
    _assert_aligned(header, template)
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]


def _row(template: str, **fields: Any) -> str:
    """按模板渲染一行，并双向校验：传入字段集合必须严格等于占位符集合。"""
    names = _placeholders(template)
    missing = [n for n in names if n not in fields]
    extra = [k for k in fields if k not in names]
    if missing or extra:
        raise ValueError(
            f"表格行字段不匹配 | 模板缺值:{missing} | 多余字段:{extra} | 模板:{template}"
        )
    return template.format(**{k: _cell(v) for k, v in fields.items()})


def _sector_label(sector: str, rules_skipped: bool = False) -> str:
    """板块列文本：未知时显式写「未知板块」，并在依赖板块的规则被跳过时标注出来。"""
    if sector == UNKNOWN_SECTOR:
        return f"{UNKNOWN_SECTOR}（板块类规则已跳过）" if rules_skipped else UNKNOWN_SECTOR
    return sector


# 操作指引的兜底值。新增信号类型时若忘了补分支，报告会渲染占位文案而不是 KeyError 崩溃。
_DEFAULT_GUIDE: Dict[str, str] = {
    "observation": "⚠️ 该信号类型未配置操作指引，按最低优先级处理",
    "confirmation": "—",
    "invalidation": "—",
    "sizing": "观望或放弃",
    "confidence": "低",
}


def _build_action_guide(s: TechnicalSignal, market_open: bool = True) -> dict:
    """
    为信号生成次日操作指引。

    Args:
        s: 技术信号
        market_open: 当前市场环境是否允许开仓（can_trade）。False 时所有信号
            仅作观察——能否开仓是市场层决策，不逐信号分化档位。

    Returns:
        dict 包含 observation（观察要点）、confirmation（确认条件）、
        invalidation（失效条件）、sizing（仓位建议）、confidence（置信度）
    """
    guide = dict(_DEFAULT_GUIDE)

    if s.signal_type == 'pullback_ma5':
        if s.pct_change > 5.0:
            guide['observation'] = "⚠️ 当日涨幅较大（长下影拉升）。次日高开≥3%放弃；平开/小幅低开按确认条件走标准介入流程"
            guide['confidence'] = "低"
        elif s.pct_change < -3.0:
            guide['observation'] = "当日跌幅较大，观察次日能否止跌企稳（高开≥3%的反弹不追）；若继续阴线下跌则放弃"
            guide['confidence'] = "低"
        else:
            guide['observation'] = "信号已确认（缩量回踩 + 不破5日线）。今日盘后 15:05-15:30 按收盘价固定价格申报买入，不等次日"
            guide['confidence'] = "中"

        # 介入时序：信号日盘后固定价格买入，15:05-15:30 按收盘价定价，不等次日确认
        # （T+1 确认后买经回测证伪，配对差 -0.67pp）。
        guide['confirmation'] = f"今日盘后 15:05-15:30：按收盘价 {_f(s.current_price)} 固定价格申报买入"
        guide['invalidation'] = (
            f"由卖出规则接管：现价 ≤ max(持仓期最高收盘, 入场价)×0.90 清仓"
            f"（初始触发 ≈ {_f(s.current_price * 0.9)}），或持满 16 个交易日到期"
        )
        guide['sizing'] = "正常仓位(50%)"

    elif s.signal_type == 'near_ma5':
        # 缩量贴MA5：同一 setup，但当日收涨且未触及MA5——是观察信号而非买点。
        # 剧本从"次日接回踩"改为"等它回下来"：等未来某日出现缩量回踩信号，
        # 触发日再按盘后固定价格纪律买入。
        guide['observation'] = (
            "⚠️ 非回踩形态（当日收涨且未触及MA5），不是买点。不追高："
            "等后续出现缩量回踩信号再按盘后纪律买入；若放量加速远离MA5则放弃观察"
        )
        guide['confirmation'] = f"出现缩量回踩信号（收盘 ≥ MA5({_f(s.ma5)})×0.995 且量比 < 1.2）→ 触发日盘后按收盘价申报买入"
        guide['invalidation'] = f"放量加速远离MA5（乖离 > 5%）或收盘价 < MA5({_f(s.ma5)}) * 0.99"
        guide['sizing'] = "轻仓(25%)或等回踩"

    elif s.signal_type == 'pullback_ma10':
        guide['observation'] = "次日观察弱转强：需收盘站上MA5(至少触碰)，若继续在MA5-MA10之间弱势震荡则等待"
        guide['confidence'] = "低"
        guide['confirmation'] = f"次日尾盘复核：收复MA5({_f(s.ma5)}) 且量比外推全天<1.0 → 确认日盘后 15:05-15:30 按收盘价申报买入"
        guide['invalidation'] = f"次日收盘跌破MA10({_f(s.ma10)}) 或继续缩量阴跌"
        guide['sizing'] = "半仓(25%)或观望"

    # 根据技术评分调节置信度与仓位基调
    if s.score >= 80:
        guide['confidence'] = "高"
        if "轻仓" not in guide['sizing']:
            guide['sizing'] = "正常仓位(50%)"
    elif s.score >= 60:
        guide['confidence'] = "中"
    else:
        guide['confidence'] = "低"
        guide['sizing'] = "观望或放弃"

    # near_ma5 兜底封顶：评分类已封 79，防止它进"盘后介入"档（与"等回踩"剧本矛盾）
    if s.signal_type == 'near_ma5':
        if guide['confidence'] == "高":
            guide['confidence'] = "中"
        if "正常仓位" in guide['sizing']:
            guide['sizing'] = "轻仓(25%)或等回踩"

    # 仓位上限 = 亏损限额 ÷ 止损距离（统一规则，不按环境分化档位）
    if not market_open:
        guide['confidence'] = "低"
        guide['sizing'] = "禁止开仓（当前市场状态不开新仓，信号仅作观察）"
    else:
        stop_pct = _stop_loss_pct(s)
        cap = position_size(stop_pct)
        guide['sizing'] = (
            f"{guide['sizing']}｜仓位上限{cap:.0f}元"
            f"（亏损限额{LOSS_BUDGET:.0f}元÷止损{stop_pct:.1f}%）"
        )

    return guide


def _stop_loss_pct(s: TechnicalSignal) -> float:
    """止损距离（%）：以 MA10 为止损参考（减仓后止损线移至 MA10，清仓亦以 MA10 为准）。

    价格贴近或跌破 MA10 时距离趋近 0，取 MIN_STOP_LOSS_PCT 为下限——
    否则「亏损限额 ÷ 止损距离」会算出天量仓位。
    """
    if s.current_price > 0 and s.ma10 > 0:
        return max((s.current_price - s.ma10) / s.current_price * 100, MIN_STOP_LOSS_PCT)
    return MIN_STOP_LOSS_PCT


def _format_rank_table(signals: List[TechnicalSignal], signal_type: str,
                       market_open: bool = True) -> List[str]:
    """生成带排名和操作指引的信号表格。"""
    lines = []
    if not signals:
        return lines

    # 按技术评分降序排列
    sorted_signals = sorted(signals, key=lambda s: s.score, reverse=True)

    # 根据信号类型取区头与说明（表头与行模板两张表共用）
    title, note = _SECTION_TITLES[signal_type]
    lines.extend([title, "", note, ""])
    lines.extend(_header(_RANK_HEADER, _RANK_ROW))

    for rank, s in enumerate(sorted_signals, 1):
        guide = _build_action_guide(s, market_open)
        rank_str = f"🥇{rank}" if rank == 1 else f"🥈{rank}" if rank == 2 else f"🥉{rank}" if rank == 3 else f"#{rank}"

        lines.append(_row(
            _RANK_ROW,
            rank=rank_str,
            stock=f"{s.name}({s.code})",
            sector=s.sector,
            price=_f(s.current_price),
            pct=_f_pct(s.pct_change),
            bias=_f_pct(s.bias_ma5),
            vol=_f(s.volume_ratio),
            turnover=f"{_f(s.turnover_rate)}%",
            score=s.score,
            observation=guide['observation'],
            levels=f"MA5={_f(s.ma5)} MA10={_f(s.ma10)}",
            confirm=guide['confirmation'],
            invalidate=guide['invalidation'],
            sizing=guide['sizing'],
        ))

    lines.append("")
    return lines


def _format_after_close_plan(signals: List[TechnicalSignal], market_open: bool = True) -> List[str]:
    """生成盘后操作计划板块（信号日盘后固定价格买入）。"""
    lines = [
        "## 📋 盘后操作计划（15:05-15:30 固定价格买入）",
        "",
        "> 信号日盘后 15:05-15:30 按收盘价固定价格申报买入，不等次日确认（T+1 确认买经回测证伪）。",
        "> 信号 2（回踩MA10）需次日弱转强确认，确认日盘后执行；以各行「确认条件」为准。",
        "",
    ]

    # 按技术评分分层（达标线用 QUALIFY_SCORE，与头条口径一致）
    high_priority = [s for s in signals if s.score >= 80]
    medium_priority = [s for s in signals if QUALIFY_SCORE <= s.score < 80]
    low_priority = [s for s in signals if s.score < QUALIFY_SCORE]

    if not signals:
        lines.extend(["> 今日无信号，无可执行的次日计划。", ""])
        return lines

    if market_open and high_priority:
        lines.extend([
            "### 🟢 优先关注（评分≥80）",
            "",
            "适合盘后直接申报买入（15:05-15:30 按收盘价定价）。",
            "",
        ])
        lines.extend(_header(_PLAN_HEADER, _PLAN_ROW))
        for rank, s in enumerate(sorted(high_priority, key=lambda x: x.score, reverse=True), 1):
            guide = _build_action_guide(s, market_open)
            lines.append(_row(
                _PLAN_ROW,
                rank=f"#{rank}",
                stock=f"{s.name}({s.code})",
                sector=s.sector,
                score=s.score,
                confirm=guide['confirmation'],
                sizing=guide['sizing'],
            ))
        lines.append("")

    if market_open and medium_priority:
        lines.extend([
            "### 🟡 备选关注（评分60-79）",
            "",
            "信号 2 需次日弱转强确认，确认日盘后执行。",
            "",
        ])
        lines.extend(_header(_PLAN_HEADER, _PLAN_ROW))
        for rank, s in enumerate(sorted(medium_priority, key=lambda x: x.score, reverse=True), 1):
            guide = _build_action_guide(s, market_open)
            lines.append(_row(
                _PLAN_ROW,
                rank=f"#{rank}",
                stock=f"{s.name}({s.code})",
                sector=s.sector,
                score=s.score,
                confirm=guide['confirmation'],
                sizing=guide['sizing'],
            ))
        lines.append("")

    if not market_open:
        lines.extend([
            "> 🚫 当前市场状态不开新仓：以下信号仅作观察，不执行介入。",
            "",
        ])

    if low_priority:
        lines.extend([
            f"### ⚪ 暂不关注（评分<{QUALIFY_SCORE}）",
            "",
            "条件不成熟。等待后续信号改善。",
            "",
        ])
        for s in sorted(low_priority, key=lambda x: x.score, reverse=True):
            lines.append(f"- {s.name}({s.code}): 评分{s.score}，{s.description}")
        lines.append("")

    return lines


def _format_veto_section(vetoed_stocks: List[Tuple[str, str, str, str]]) -> List[str]:
    """生成负面清单否决板块。

    vetoed_stocks: [(code, name, action, reason), ...]
    action: 统一为 'skip'（跳过当日信号）
    """
    lines = [
        "## 🚫 负面清单否决（不进信号池、不看评分）",
        "",
        "> 任一规则触发即跳过当日信号（V1 公告 / V2 涨幅>100% / V3 高换手 / V4 连续大跌 / "
        "V5 资金流出 / V6 涨跌停≥3天 / V7 距60日低点>80%）。",
        "",
    ]

    if not vetoed_stocks:
        lines.append("今日无标的触发负面清单。")
        lines.append("")
        return lines

    lines.extend([f"**跳过当日信号（{len(vetoed_stocks)} 只）**", ""])
    lines.extend(_header(_PAIR_HEADER, _PAIR_ROW))
    for code, name, _action, reason in vetoed_stocks[:20]:
        lines.append(_row(_PAIR_ROW, stock=f"{name}({code})", reason=reason))
    if len(vetoed_stocks) > 20:
        lines.append(_row(_PAIR_ROW, stock="...", reason=f"等共{len(vetoed_stocks)}只股票"))
    lines.append("")
    return lines


def _format_skip_stats_section(stats: Optional[SkipStats]) -> List[str]:
    """跳过规则覆盖情况表：逐条给出「检查 N 只 / 触发 N 只」。

    目的：让报告里的"跳过 N 只"这个数字可解释——用户能看到每条规则跑没跑、
    跑了多少只，而不是只看到一个总数。
    未实现的规则显式标注「未实现」，绝不让它显示成"检查 0 只 / 触发 0 只"
    （那会被读成"规则跑了，股票没问题"）。
    """
    lines = [
        "## 🔍 跳过规则覆盖情况",
        "",
        "> 逐条列出各规则的实际执行情况。"
        "「跳过」= 数据缺失该条没跑，不等于股票没问题。",
        "",
    ]
    lines.extend(_header(_RULE_STAT_HEADER, _RULE_STAT_ROW))

    if stats is None:
        # 没有统计数据时，用规则定义本身渲染，保证"未实现"仍然可见
        for r in SKIP_RULES:
            status = "未实现" if not r.implemented else "未统计"
            lines.append(_row(
                _RULE_STAT_ROW, rule=r.rule_id, name=r.name, status=status,
                count="—", detail="本次运行未提供统计数据",
            ))
        lines.extend(["", "---", ""])
        return lines

    for stat in stats.rows():
        detail = "；".join(stat.reasons) if stat.reasons else "—"
        if not stat.implemented:
            detail = "🚧 " + detail
        lines.append(_row(
            _RULE_STAT_ROW,
            rule=stat.rule_id,
            name=stat.name,
            status="未实现" if not stat.implemented else "已执行",
            count=stat.summary(),
            detail=detail,
        ))
    lines.extend(["", "---", ""])
    return lines


def _format_regime_diagnosis(diag: Optional[RegimeDiagnosis]) -> List[str]:
    """状态判定的诊断明细：均线排列 + 偏离 MA20 百分比 + 命中路径。

    回答"为什么判成这个状态"，而不是只丢一个状态名给用户猜。
    """
    if diag is None:
        return ["> ⚠️ 未提供状态判定明细（指数数据缺失）", ""]

    lines = [f"> **判定依据**：{diag.alignment}"]
    if diag.available:
        lines.extend([
            f"> **收盘偏离 MA20**：{_f_pct(diag.bias_ma20)}"
            f"（收盘 {_f(diag.close)} / MA20 {_f(diag.ma20)}）",
            f"> **均线值**：MA5 {_f(diag.ma5)} | MA10 {_f(diag.ma10)} | MA20 {_f(diag.ma20)}",
        ])
    else:
        lines.append(f"> **收盘偏离 MA20**：—（{diag.note}）")
    if diag.data_date:
        lines.append(f"> **数据日期**：{diag.data_date}")
    lines.extend([
        f"> **命中路径**：{diag.path}",
        "",
    ])
    return lines


def _format_headline(signals: List[TechnicalSignal], removed_stocks, vetoed_stocks,
                     failed_stocks) -> List[str]:
    """报告头条：先说结论——有多少信号、多少达标、能不能动手。"""
    qualified = sum(1 for s in signals if s.score >= QUALIFY_SCORE)
    lines = [
        f"> 发现 **{len(signals)}** 个信号，其中 **{qualified}** 个达标（评分≥{QUALIFY_SCORE}）",
        "",
    ]
    if not signals:
        lines.extend(["> 📭 **今日无信号**：选股名单中无标的触发回踩买点。", ""])
    elif qualified == 0:
        # 有信号但一个都没达标：必须说清"不是没跑，是都不够格"
        lines.extend([
            f"> 🚫 **今日无可执行信号**：{len(signals)} 个信号评分均 <{QUALIFY_SCORE}，"
            "按纪律不介入（详见下方信号明细）。",
            "",
        ])
    else:
        lines.extend([f"> ✅ 可执行盘后买入（15:05-15:30）：**{qualified}** 只。", ""])

    lines.extend([
        f"> 跳过（趋势破坏）**{len(removed_stocks)}** 只 | 负面清单 **{len(vetoed_stocks)}** 只 | "
        f"失败 **{len(failed_stocks)}** 只",
        "",
        "---",
        "",
    ])
    return lines


def generate_technical_report(
    signals: List[TechnicalSignal],
    removed_stocks = None,
    market_env: Optional[Tuple] = None,
    failed_stocks = None,
    vetoed_stocks = None,
    detail_level: str = "standard",
    skip_stats: Optional[SkipStats] = None,
    regime_diag: Optional[RegimeDiagnosis] = None,
) -> str:
    """
    生成 Markdown 格式的趋势跟踪日报。

    所有表格统一走 _row() / _header()：表头列数与行模板占位符数不一致、
    或传入字段与占位符不匹配，都会抛 ValueError 而不是产出静默错位的表格。

    报告结构（自上而下 = 决策优先级）：
        头条结论 → 市场环境（含状态判定明细）→ 跳过规则覆盖
        → 趋势破坏跳过 → 负面清单 → 分析失败 → 信号明细 → 盘后操作计划


    Args:
        signals: TechnicalSignal 列表
        removed_stocks: (code, name, reason) 元组列表
        market_env: (can_trade, conditions, summary, regime) 或 None
        failed_stocks: (code, name, reason) 元组列表
        vetoed_stocks: (code, name, action, reason) 元组列表，action 见 veto_rules
        detail_level: "compact"（通知精简）| "standard"（文件标准）| "full"（完整含操作计划）
        skip_stats: 跳过规则逐条统计（SkipStats）
        regime_diag: 市场状态判定明细（RegimeDiagnosis）

    Returns:
        格式化的 Markdown 字符串
    """
    removed_stocks = removed_stocks or []
    failed_stocks = failed_stocks or []
    vetoed_stocks = vetoed_stocks or []
    # 能否开仓是市场层决策（can_trade 已含状态与门控判定），统一作用于全部信号
    market_open = bool(market_env and market_env[0])
    today_str = datetime.now().strftime('%Y-%m-%d')

    # ── 头条：先给结论（发现 N 个 / 达标 M 个 / 能不能动手）──
    lines = [f"# 📊 趋势跟踪日报 ({today_str})", ""]
    lines.extend(_format_headline(signals, removed_stocks, vetoed_stocks, failed_stocks))

    # 大盘状态栏
    if market_env:
        can_trade = market_env[0]
        regime = market_env[2] if len(market_env) >= 3 else "chaos"
        regime_text = REGIME_DESC.get(regime, "❓ 状态不明")
        env_icon = "✅" if can_trade else "⛔"
        lines.extend([
            "## 🌤️ 市场环境",
            "",
            f"> **【大盘状态】{regime_text}**",
            "",
            f"> **{env_icon} {'允许开仓' if can_trade else '建议空仓'}**（纯结构口径：均线状态决定能否开仓）",
            "",
        ])
        # 风格状态（元层观察，与大盘门控不同层：大盘管能不能开仓，风格管什么打法占优）
        style_line = _style_state_display()
        if style_line:
            lines.extend([f"> **风格状态**：{style_line}", ""])
        # 判定明细：排列 + 偏离 MA20 + 命中路径，回答"为什么判成这个状态"
        lines.extend(_format_regime_diagnosis(regime_diag))

        # 当前环境是否允许开仓（能否开仓 = 市场层统一决策，作用于全部信号）
        if regime not in REGIME_CAN_OPEN:
            lines.extend([
                "",
                "> 📌 当前状态**不开新仓**：信号仅作观察（持仓卖出照常）。",
                "",
                "---",
                "",
            ])
        else:
            lines.extend([
                "",
                "> 📌 仓位上限 = 亏损限额 ÷ 止损距离（止损参考 MA10），不乘评分系数。",
                "",
                "---",
                "",
            ])

    # 跳过规则覆盖情况：逐条「检查 N 只 / 触发 N 只」，未实现项显式标注
    lines.extend(_format_skip_stats_section(skip_stats))

    # 趋势破坏跳过
    if removed_stocks:
        lines.extend([
            "## ⏭️ 跳过股票（趋势破坏）",
            "",
        ])
        lines.extend(_header(_PAIR_HEADER, _PAIR_ROW))
        for code, name, reason in removed_stocks[:20]:
            lines.append(_row(_PAIR_ROW, stock=f"{name}({code})", reason=reason))
        if len(removed_stocks) > 20:
            lines.append(_row(_PAIR_ROW, stock="...", reason=f"等共{len(removed_stocks)}只股票"))
        lines.extend(["", "---", ""])

    # 负面清单否决
    lines.extend(_format_veto_section(vetoed_stocks))
    lines.extend(["---", ""])

    # 分析失败
    if failed_stocks:
        lines.extend([
            "## ⚠️ 分析失败股票",
            "",
        ])
        lines.extend(_header(_PAIR_HEADER, _PAIR_ROW))
        for code, name, reason in failed_stocks[:20]:
            lines.append(_row(_PAIR_ROW, stock=f"{name}({code})", reason=reason))
        if len(failed_stocks) > 20:
            lines.append(_row(_PAIR_ROW, stock="...", reason=f"等共{len(failed_stocks)}只股票"))
        lines.extend(["", "---", ""])

    # ── 信号明细（按信号类型分表，含逐项操作要点）──
    lines.extend(["## 📋 信号明细", ""])
    pullback_ma5_signals = [s for s in signals if s.signal_type == 'pullback_ma5']
    pullback_ma10_signals = [s for s in signals if s.signal_type == 'pullback_ma10']
    near_ma5_signals = [s for s in signals if s.signal_type == 'near_ma5']
    lines.extend(_format_rank_table(pullback_ma5_signals, 'pullback_ma5', market_open))
    lines.extend(_format_rank_table(pullback_ma10_signals, 'pullback_ma10', market_open))
    lines.extend(_format_rank_table(near_ma5_signals, 'near_ma5', market_open))
    if not signals:
        lines.extend(["> 今日无信号。", ""])
    lines.extend(["---", ""])

    # ── 盘后操作计划：只把达标的分层列出（与信号明细互补，不重复罗列全部信号）──
    if detail_level in ("full", "standard"):
        lines.extend(_format_after_close_plan(signals, market_open))

    return "\n".join(lines)
