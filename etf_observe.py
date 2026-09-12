# -*- coding: utf-8 -*-
"""
==================================
ETF 周度观察报告 — 估值导向
==================================

定位：每周一次，默认纯观察；`--execute` 时执行统一调仓批次。

报告结构：
  1. 市场估值概览（纯数据：全市场 PE / 收益率 / 风险溢价及其分位，不给操作建议）
  2. 买入优先级（按 PE 分位排序，参考信息）
  3. 新钱投放参考（全局节奏熔断 + 逐标的补入节奏，仅建议，真实账户人工执行）
  4. 持仓对照与调仓建议（核心口径，旧钱只做阈值再平衡）
  5. 卫星仓 — 行业动量轮动（关注池 + 突破候选 + 调仓建议）
  6. 动量观察（卫星仓背景排名，仅观察不交易）
  7. 自动调仓执行结果（仅 --execute：卫星卖 → 核心卖 → 核心买 → 卫星买）

卖出警示（极端条件触发：全市场 PE>90% + 行业 PE>95% + 拥挤）。
"""

import argparse
import io
import logging
import os
import re
import sys
from datetime import datetime
from typing import List

from src.config import setup_env
from src.logging_config import setup_logging
from src.trading_calendar import is_trading_day

setup_env()

from src.etf.config import CORE_BASELINE, AssetType
from src.etf.amazing_factors import (
    get_market_pe, get_treasury_yield_y10, get_erp_percentile,
    rank_buy_priorities, check_sell_warnings,
)
from src.mx.client import is_mx_untradable

logger = logging.getLogger(__name__)

# ── 新钱全局节奏（极端熔断）阈值 ──
# 常态不表态：权益/现金整体配比观点归基准（只抄基准不抄择时），新钱默认按比例并入。
# 仅在极端估值档整笔暂缓——与卖出警示/卫星锁仓同属熔断家族，非连续择时旋钮。
GLOBAL_PAUSE_PE_PCT = 90.0   # 全市场 PE 分位（与 check_sell_warnings 同阈值）
GLOBAL_PAUSE_ERP_PCT = 10.0  # 风险溢价自身近 5 年分位


# ── 风格状态引用（元层观察，fail-soft，只展示不决策）──

def _style_state_line() -> str:
    """读取 data/style_state.json（style_report 周六产出），返回报告头部一行。"""
    import json as _json
    from pathlib import Path
    try:
        path = Path(__file__).parent / "data" / "style_state.json"
        d = _json.loads(path.read_text(encoding="utf-8"))
        label = {
            "strong": "🟢 主线强势期", "fading": "🔴 退潮期",
            "vacuum": "⚪ 风格真空期", "forming": "🟡 形成中",
        }.get(d.get("state"), d.get("state", "未知"))
        streak = d.get("state_streak", 1)
        streak_txt = f"（持续 {streak} 周）" if streak > 1 else ""
        stale = "" if (datetime.now() - datetime.strptime(d.get("data_date", ""), "%Y-%m-%d")).days <= 10 else " ⚠️ 数据陈旧"
        return f" | **风格状态**: {label}{streak_txt}{stale}（截至 {d.get('data_date')}，判定规则见 strategy/style_state.md）"
    except Exception:
        return ""


# ── 市场概览 ──

def _market_overview() -> str:
    """市场估值概览（纯数据展示，不给操作建议；整体配比观点归基准，新钱节奏见第三节）"""
    lines = ["## 一、市场估值概览", ""]

    market_pe = get_market_pe()
    if market_pe:
        pe, pe_pct = market_pe["pe"], market_pe["pe_pct"]
        level = ("极度低估" if pe_pct < 20 else
                 "低估" if pe_pct < 40 else
                 "合理" if pe_pct < 60 else
                 "偏贵" if pe_pct < 80 else "高估")
        lines.append(f"- **全市场 PE**：{pe:.1f} 倍 | 近 5 年 **{pe_pct:.0f}%** 分位 →  **{level}**")
        lines.append(f"- **股票内在收益率（1/PE）**：{100/pe:.2f}%")

        y10 = get_treasury_yield_y10()
        if y10:
            spread = 100/pe - y10
            # ERP 自身分位与 PE 分位同窗口可比：PE 分位受窗口锚定影响（含深熊底部则读高），
            # ERP 分位是跨资产口径的独立参照，两者并列展示供互相印证
            erp_note = ""
            try:
                erp = get_erp_percentile()
            except Exception:
                erp = None
            if erp:
                erp_note = f"，自身近 5 年 **{erp['erp_pct']:.0f}%** 分位"
            lines.append(
                f"- **风险溢价**：{100/pe:.2f}% − 国债 {y10:.2f}% = **{spread:+.2f}%**{erp_note} "
                f"（{'承担风险有额外回报' if spread > 0 else '股票还不如国债'}）"
            )
    else:
        lines.append("- **全市场 PE**：数据不可用")
        y10 = get_treasury_yield_y10()
        if y10:
            lines.append(f"- **10 年期国债收益率**：{y10:.2f}%")

    lines.append("")
    return "\n".join(lines)


# ── 买入优先级 ──

def _buy_priority() -> str:
    """买入优先级（按 PE 分位排序，越便宜越靠前）"""
    lines = ["## 二、买入优先级（按估值便宜度排序）", ""]

    # 收集所有核心仓权益 ETF（去重）
    equity_etfs = {}
    for a in CORE_BASELINE:
        if a.asset_type == AssetType.EQUITY and a.code not in equity_etfs:
            equity_etfs[a.code] = a

    ranked = rank_buy_priorities(list(equity_etfs.values()))

    lines.append("| ETF | 估值基准 | PE 分位 | 建议 |")
    lines.append("|-----|---------|--------|------|")

    for r in ranked:
        source = r.get("source_name", "") or "—"
        pe_display = f"{r['pe_pct']:.0f}%" if r.get("pe_pct") is not None else "—"
        lines.append(
            f"| {r['name']}（{r['code']}） | {source} | {pe_display} | "
            f"{r['level']} {r['level_text']} |"
        )

    lines.append("")
    return "\n".join(lines)


# ── 卖出警示 ──

def _sell_warning() -> str:
    """卖出警示（极端条件才触发）"""
    equity_etfs = {}
    for a in CORE_BASELINE:
        if a.asset_type == AssetType.EQUITY and a.code not in equity_etfs:
            equity_etfs[a.code] = a

    warnings = check_sell_warnings(list(equity_etfs.values()))
    if not warnings:
        return ""

    lines = ["## ⚠️ 卖出警示", ""]
    lines.append("> 以下 ETF 触发长期配置回避条件：")
    lines.append("")
    for w in warnings:
        lines.append(f"- **{w['name']}**（{w['code']}）：{w['reason']}")
    lines.append("")
    return "\n".join(lines)


# ── 持仓对照与调仓建议 ──

def _compute_allocation():
    """计算目标配比 + 调仓指令（供对照展示与自动执行复用）

    动态择时已删除：目标 = 中性基准；regime 仅选再平衡阈值松紧。
    PE 分位只用于新钱投放速度旋钮。

    Returns:
        dict 或 None（MX 不可用/无数据时）
    """
    if not os.getenv("MX_APIKEY"):
        return None

    from src.etf.rebalancer import ETFRebalancer
    from src.mx.client import MXMoniClient

    client = MXMoniClient()
    balance = client.get_balance()
    if not balance or (balance.get("total_assets") or 0) <= 0:
        return None

    positions = client.get_positions()

    # 全市场估值（新钱速度旋钮 + 报告参考，不进核心仓决策）
    market_pe = get_market_pe()
    pe_pct = market_pe.get("pe_pct") if market_pe else None
    current_pe = market_pe.get("pe") if market_pe else None

    total_assets = balance["total_assets"]
    # 入口统一构造数据 manager，传给需要行情数据的下游模块（避免各自重复构造）
    from data_provider import get_fetcher
    fm = get_fetcher()
    rebalancer = ETFRebalancer(client, fetcher=fm)
    target = rebalancer.calculate_target()

    # 市场门控：regime 只选再平衡阈值松紧 + 卫星仓许可（不再驱动核心仓位）
    regime, hard_intercept = "chaos", False
    try:
        from src.market_state.market_gate import check_market_gate, fetch_gate_inputs

        _can, _cond, _sum, regime, hard_intercept = check_market_gate(fetch_gate_inputs(fm))
    except Exception:
        logger.warning("市场门控获取失败，卫星仓禁买", exc_info=True)

    core_positions, rotation_mv, rotation_positions = rebalancer.split_rotation_positions(positions)
    orders, total_deviation = rebalancer.compare(
        target, positions, total_assets,
        gate_state=regime, hard_intercept=hard_intercept,
    )
    _should, reason = rebalancer.should_rebalance(orders, total_deviation, gate_state=regime)

    # 卫星仓 — 行业动量轮动（只出指令，执行统一走 _execute_batch）
    satellite = None
    state_gate = None
    try:
        from src.market_state.style_state import satellite_state_gate

        state_gate = satellite_state_gate()
    except Exception:
        logger.warning("风格状态门控获取失败，卫星门控 fail-open 放行", exc_info=True)
    try:
        from src.etf import industry_momentum as sat_mod

        satellite = sat_mod.analyze_satellite(positions, total_assets,
                                              hard_intercept, regime, client=client,
                                              state_gate=state_gate)
    except Exception:
        logger.warning("卫星仓分析失败", exc_info=True)

    return {
        "client": client,
        "balance": balance,
        "positions": positions,
        "pe_pct": pe_pct,
        "current_pe": current_pe,
        "total_assets": total_assets,
        "core_positions": core_positions,
        "rotation_mv": rotation_mv,
        "rotation_positions": rotation_positions,
        "rebalancer": rebalancer,
        "target": target,
        "orders": orders,
        "total_deviation": total_deviation,
        "reason": reason,
        "satellite": satellite,
        "regime": regime,
        "hard_intercept": hard_intercept,
    }


def _holding_overview(alloc=None) -> str:
    """持仓 vs 目标对照 + 调仓建议（只出建议，不自动执行）"""
    lines = ["## 四、持仓对照与调仓建议", ""]

    if alloc is None:
        alloc = _compute_allocation()
    if alloc is None:
        if not os.getenv("MX_APIKEY"):
            lines.append("- 未配置 MX_APIKEY，跳过持仓对照")
        else:
            lines.append("- 妙想账户暂无数据，跳过持仓对照")
        return "\n".join(lines)

    positions = alloc["positions"]
    pe_pct = alloc["pe_pct"]
    total_assets = alloc["total_assets"]
    rotation_mv = alloc["rotation_mv"]
    rotation_positions = alloc["rotation_positions"]
    rebalancer = alloc["rebalancer"]
    target = alloc["target"]
    orders = alloc["orders"]
    reason = alloc["reason"]

    if pe_pct is not None:
        level = ("极度低估" if pe_pct < 20 else
                 "低估" if pe_pct < 40 else
                 "合理" if pe_pct < 60 else
                 "偏贵" if pe_pct < 80 else "高估")
        gate_line = f"**PE 分位**: {pe_pct:.0f}%（{level}，仅作新钱投放速度与参考）"
    else:
        gate_line = "**PE 分位**: 数据不可用"

    assets_line = f"**总资产**: {total_assets:,.0f} 元"
    if rotation_mv > 0:
        assets_line += f" | **卫星/其他账户持仓**: {rotation_mv:,.0f} 元（由现金桶吸收）"
    lines.append(f"{assets_line} | {gate_line}")
    lines.append(f"**再平衡结论**: {reason}")
    lines.append("")

    # 当前持仓占比（资金占比即总占比；卫星等基准外持仓不产生偏离，由现金桶吸收）
    current_map: dict = {}
    for p in positions:
        mv = float(p.get("market_value", 0) or 0)
        if total_assets > 0:
            current_map[p.get("code", "")] = mv / total_assets * 100

    # 现金实际占比 = 剩余资金（妙想持仓不含现金项）
    held_sum = sum(v for k, v in current_map.items() if k != "CASH")
    current_map["CASH"] = max(0.0, 100.0 - held_sum)

    lines.append("| 资产 | 目标% | 实际% | 偏离 | 建议 |")
    lines.append("|------|-------|-------|------|------|")

    order_map = {o.code: o for o in orders}
    for asset in rebalancer.baseline:
        code = asset.code
        tgt = target.get(code, 0.0) * 100
        cur = current_map.get(code, 0.0)
        dev = tgt - cur
        order = order_map.get(code)
        action = ""
        if order:
            action = f"{'🟢买' if order.action == 'buy' else '🔴卖'} {order.quantity}股"
        lines.append(f"| {asset.name} | {tgt:.1f}% | {cur:.1f}% | {dev:+.1f}% | {action} |")

    # 卫星持仓与基准外持仓提示（由"现金（以及其他账户）"桶吸收，不产生核心偏离）
    if rotation_positions:
        lines.append("")
        lines.append("> 卫星持仓（现金桶吸收，不参与核心再平衡）："
                     + "、".join(f"{p.get('name', '')}({p.get('code', '')})" for p in rotation_positions))
    baseline_codes = {a.code for a in rebalancer.baseline}
    extra = [f"{p.get('name', '')}({p.get('code', '')})" for p in alloc["core_positions"]
             if p.get("code") not in baseline_codes]
    if extra:
        lines.append("")
        lines.append(f"> 基准外持仓：{'、'.join(extra)}（未纳入对照）")

    if orders:
        lines.append("")
        lines.append("**建议调仓指令（仅供参考，不自动执行）**")
        lines.append("")
        for o in orders:
            manual_mark = " ⚠️ 妙想无法交易，需手动" if is_mx_untradable(o.code) else ""
            lines.append(f"- {o.action.upper()} {o.name}({o.code}) {o.quantity}股 ≈ {o.amount:,.0f}元 — {o.reason}{manual_mark}")
    else:
        lines.append("")
        lines.append("无调仓需求，保持当前配置。")

    lines.append("")
    return "\n".join(lines)


def _global_rhythm_line() -> str:
    """新钱全局节奏行：回答"要不要整体先等等"。

    常态不表态——按比例并入即默认答案，权益/现金整体配比观点归基准（只抄基准不抄择时）。
    仅极端估值熔断：全市场 PE 分位 ≥ GLOBAL_PAUSE_PE_PCT（与卖出警示同阈值），
    或风险溢价自身近 5 年分位 ≤ GLOBAL_PAUSE_ERP_PCT（跨资产口径的独立确认）。
    """
    market_pe = get_market_pe()
    if not market_pe:
        return "> **全局节奏**：全市场估值数据不可用，默认按比例并入。"

    pe_pct = market_pe["pe_pct"]
    try:
        erp = get_erp_percentile()
    except Exception:
        erp = None
    erp_pct = erp.get("erp_pct") if erp else None

    triggers = []
    if pe_pct >= GLOBAL_PAUSE_PE_PCT:
        triggers.append(f"全市场 PE 分位 {pe_pct:.0f}% ≥ {GLOBAL_PAUSE_PE_PCT:.0f}%")
    if erp_pct is not None and erp_pct <= GLOBAL_PAUSE_ERP_PCT:
        triggers.append(f"风险溢价处于自身近 5 年 {erp_pct:.0f}% 分位（≤{GLOBAL_PAUSE_ERP_PCT:.0f}%）")

    if triggers:
        return (f"> ⚠️ **全局节奏：暂缓并入**（{'；'.join(triggers)}）。"
                f"新钱留在场外，等分位回落或分批；下表 🟢 仅为若必须并入时的相对优先级。")
    erp_txt = f"，风险溢价分位 {erp_pct:.0f}%" if erp_pct is not None else ""
    return (f"> ✅ **全局节奏**：无极端信号（PE 分位 {pe_pct:.0f}%{erp_txt}）→ **按比例并入**，"
            f"下表只是篮子内的先后快慢，不改变整体配比。")


def _deploy_cash_section(alloc: dict) -> str:
    """新钱投放参考（每周固定输出）：只回答节奏问题——什么可以加速补、什么不急着补。

    缺口补入不在本节：新钱入金后各资产自然欠配，按基准目标权重自行计算缺口买入即可
    （目标权重见第二节表）；模拟仓口径的缺口由旧钱再平衡（第四节）处理。
    新钱为真实资金、独立于妙想模拟仓，到账后人工在真实账户操作。
    """
    from data_provider import get_etf_daily

    lines = ["## 三、新钱投放参考（节奏判断，仅建议）", ""]
    lines.append("入金后按基准目标权重补入即可（缺口 = 目标权重 x 入金后总资产 - 实际市值）。")
    lines.append("本节先给全局节奏（要不要整体等等），再给逐标的节奏（哪些不急着补、哪些可以加速）。")
    lines.append("")
    lines.append(_global_rhythm_line())
    lines.append("")

    rows = []
    for a in CORE_BASELINE:
        if a.asset_type != AssetType.EQUITY:
            continue
        try:
            df = get_etf_daily(a.code)
        except Exception:
            df = None
        if df is None or len(df) < 25:
            continue
        close = df["close"]
        ma20 = close.rolling(20).mean().iloc[-1]
        cur = float(close.iloc[-1])
        ret20 = (cur / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
        above_ma20 = cur >= float(ma20)

        # 估值分位（ AmazingData，缺失不阻断）
        pe_pct = None
        try:
            info = _etf_pe_info_safe(a.code)
            pe_pct = info
        except Exception:
            pass

        # 节奏判定：趋势（右侧纪律）x 估值（便宜加速）
        if above_ma20:
            rhythm = ("🟢 可正常补" if (pe_pct is None or pe_pct >= 30)
                      else "🟢🟢 可加速（低估 + 企稳）")
            basis = f"站上MA20，20日{ret20:+.1f}%"
        else:
            rhythm = "⏳ 不急：趋势向下，分批或等站回 MA20"
            basis = f"MA20下方，20日{ret20:+.1f}%"
        if pe_pct is not None and pe_pct < 20 and not above_ma20:
            rhythm = "🟡 极度低估：可分批加速，不必等企稳"
        pe_txt = f"{pe_pct:.0f}%" if pe_pct is not None else "—"
        manual_mark = " ⚠️ 需手动" if is_mx_untradable(a.code) else ""
        rows.append((rhythm, f"| {a.name}({a.code}){manual_mark} | {pe_txt} | {basis} | {rhythm} |"))

    # 加速在前、不急在后
    rows.sort(key=lambda x: ("🟢🟢" not in x[0], "⏳" in x[0], "🟡" in x[0]))
    lines.append("| 标的 | 估值分位(近5年) | 趋势 | 补入节奏 |")
    lines.append("|---|---|---|---|")
    lines.extend(r for _, r in rows)
    lines.append("")
    lines.append("> 分位口径：行业/主题 ETF 用其申万一级行业 PE；红利等未映射行业的标的回退全市场口径；海外 ETF 无估值数据。")
    lines.append("> 缺口金额自己算：入金后缺口 = 目标权重 x 总资产 - 实际市值；模拟仓的缺口见第四节再平衡对照。")
    lines.append("> 到账后在真实账户人工操作（妙想模拟仓只跑旧钱再平衡，与新钱无关），不自动执行。")
    lines.append("")
    return chr(10).join(lines)


def _etf_pe_info_safe(code: str):
    """单只 ETF 的行业 PE 分位（缺失返回 None）。"""
    try:
        from src.etf.amazing_factors import _etf_pe_info
        info = _etf_pe_info(code)
        if info:
            return info.get("pe_pct")
    except Exception:
        pass
    return None


# ── 卫星仓与动量观察 ──

def _satellite_overview(satellite: dict) -> str:
    """卫星仓 — 行业动量轮动：关注池/突破候选 + 持仓 + 建议"""
    if not satellite:
        return ""

    lines = ["## 五、卫星仓 — 行业动量轮动", ""]
    lines.append(f"预算 10% | 最多 2 只 | 当前卫星持仓市值 {satellite['satellite_mv']:,.0f} 元")
    if satellite["locked"]:
        lines.append("🔒 市场门控硬拦截：卫星仓禁新开仓（已持仓不动，降杠杆门控）")
    # 风格状态门控（真空/退潮期清仓+禁新开；状态数据缺失/过期时 fail-open 放行）
    gate = satellite.get("state_gate")
    if gate and gate.get("state"):
        from src.market_state.style_state import STATE_LABELS

        gate_label = STATE_LABELS.get(gate["state"], gate["state"])
        if gate.get("force_flat") or gate.get("block_new"):
            lines.append(f"🚦 风格状态门控生效：{gate_label} → 卫星仓清仓 + 禁新开"
                         f"（数据截至 {gate.get('data_date')}）")
        else:
            lines.append(f"🚦 风格状态门控：{gate_label} → 放行（数据截至 {gate.get('data_date')}）")
    elif gate is not None and not gate.get("state"):
        lines.append("🚦 风格状态门控：状态数据缺失/过期，fail-open 放行"
                     "（请检查周日 style_report 是否正常产出）")
    lines.append("")

    lines.append("### 突破候选（关注池内放量突破）")
    lines.append("")
    cands = sorted([r for r in satellite["results"] if r["breakout"]],
                   key=lambda x: x["momentum_score"], reverse=True)
    if not cands:
        lines.append("无放量突破信号")
    for r in cands:
        pe = f"{r['pe_pct']:.0f}%" if r.get("pe_pct") is not None else "—"
        lines.append(
            f"- {r['name']}({r['code']}) 动量分{r['momentum_score']} 排名{r['momentum_rank']} 涨幅{r['chg_pct']:+.1f}% "
            f"量比5日{r['vol_ratio_5']:.1f}/20日{r['vol_ratio_20']:.1f} 行业PE分位{pe}"
        )
    lines.append("")

    # 资金流观察：关注池 + 持仓 ETF 的份额 20 日变化（申赎方向，仅观察不交易）
    lines.append("### 资金流观察（份额 20 日变化，仅观察）")
    lines.append("")
    watch_codes = {r["code"] for r in satellite["results"] if r.get("pool")}
    watch_codes |= {r["code"] for r in cands}
    watch_codes |= set(satellite.get("held_codes") or [])
    try:
        from src.etf.amazing_factors import get_etf_share_flow
        share_flow = get_etf_share_flow(sorted(watch_codes)) if watch_codes else {}
    except Exception:
        share_flow = {}
    name_map = {r["code"]: r["name"] for r in satellite["results"]}
    flow_lines = []
    for code in sorted(watch_codes):
        flow = share_flow.get(code)
        if not flow or flow.get("chg") is None:
            continue
        chg = flow["chg"]
        flag = "🔥 申购潮" if chg >= 30 else ("❄️ 赎回潮" if chg <= -20 else "")
        flow_lines.append((chg, f"- {name_map.get(code, code)}({code}) 份额20日 {chg:+.1f}% {flag}"))
    if flow_lines:
        for _, line in sorted(flow_lines, key=lambda x: x[0], reverse=True):
            lines.append(line)
        lines.append("> 份额暴增 = 场内资金追入（与价格动量同向时警惕拥挤）；份额骤减 = 资金离场")
    else:
        lines.append("无份额变动数据")
    lines.append("")

    lines.append("### 持仓与建议")
    lines.append("")
    manual_codes = {
        o.code for o in (list(satellite.get("sells", [])) + list(satellite.get("buy_orders", [])))
        if is_mx_untradable(o.code)
    }

    def _note_line(n: str, prefix: str) -> str:
        m = re.search(r"\((\d{6})\)", n)
        mark = " ⚠️ 需手动" if m and m.group(1) in manual_codes else ""
        return f"- {prefix}：{n}{mark}"

    if satellite["sell_notes"]:
        for n in satellite["sell_notes"]:
            lines.append(_note_line(n, "🔴 卖出"))
    if satellite["buy_notes"]:
        for n in satellite["buy_notes"]:
            lines.append(_note_line(n, "🟢 买入"))
    if not satellite["sell_notes"] and not satellite["buy_notes"]:
        lines.append("无调仓建议")
    lines.append("")
    return "\n".join(lines)


def _momentum_observe(satellite: dict) -> str:
    """动量观察：卫星仓背景排名（只观察，不交易）"""
    if not satellite:
        return ""
    ranked = satellite["ranked"][:6]
    if not ranked:
        return ""
    lines = ["## 六、动量观察（卫星仓背景，仅排名不交易）", ""]
    for i, r in enumerate(ranked, 1):
        pe = f"PE分位{r['pe_pct']:.0f}%" if r.get("pe_pct") is not None else "PE—"
        lines.append(
            f"{i}. {r['name']}({r['code']}) 动量分{r['momentum_score']} "
            f"20日{r['ret_20d']:+.1f}% 5日{r['ret_5d']:+.1f}% {pe}"
        )
    lines.append("")
    return "\n".join(lines)


# ── 自动调仓执行 ──

def _execute_batch(alloc: dict) -> str:
    """执行调仓批次（卫星卖 → 核心卖 → 核心买 → 卫星买，带全批次资金校验）"""
    from src.mx.client import MXMoniClient, is_mx_untradable, MX_UNTRADABLE_REASON
    from src.etf import industry_momentum as sat_mod

    core_orders = alloc["orders"]
    satellite = alloc.get("satellite")
    total_assets = alloc["total_assets"]
    avail_balance = alloc["balance"].get("avail_balance") or 0

    sat_sells = list(satellite["sells"]) if satellite else []
    sat_buys = list(satellite["buy_orders"]) if satellite else []
    core_sells = [o for o in core_orders if o.action == "sell"]
    # 新钱投放只出建议（模拟仓无法感知实际入金，不进自动执行批次，人工照建议操作）
    core_buys = [o for o in core_orders if o.action == "buy"]
    sells = sat_sells + core_sells
    buys = core_buys + sat_buys

    lines = ["## 七、自动调仓执行结果", ""]

    if not sells and not buys:
        lines.append("无调仓指令，本次不执行。")
        return "\n".join(lines)

    def _qty(o):
        return getattr(o, "quantity", None) or getattr(o, "shares", 0)

    # 安全校验 1：卖出数量不得超过实际持仓
    held = {p.get("code", ""): int(p.get("count", 0) or 0) for p in alloc["positions"]}
    for o in sells:
        if _qty(o) > held.get(o.code, 0):
            lines.append(f"❌ 中止执行：{o.name}({o.code}) 卖出 {_qty(o)} 股 > 持仓 {held.get(o.code, 0)} 股")
            return "\n".join(lines)

    # 安全校验 2：买入总额不得超过可用资金 + 卖出回款
    sell_amount = sum(o.amount for o in sells)
    buy_amount = sum(o.amount for o in buys)
    if buy_amount > avail_balance + sell_amount:
        lines.append(f"❌ 中止执行：买入总额 {buy_amount:,.0f} 元 > 可用资金 {avail_balance:,.0f} 元 + 卖出回款 {sell_amount:,.0f} 元")
        return "\n".join(lines)

    # 安全校验 3：卫星买入不超 10% 预算
    if sat_buys:
        budget = total_assets * sat_mod.SATELLITE_BUDGET_RATIO
        sat_buy_total = sum(o.amount for o in sat_buys)
        sat_sell_total = sum(o.amount for o in sat_sells)
        if sat_buy_total > budget - satellite["satellite_mv"] + sat_sell_total:
            lines.append(f"❌ 中止执行：卫星买入 {sat_buy_total:,.0f} 元 超预算上限 {budget:,.0f} 元")
            return "\n".join(lines)

    results: List[str] = []
    client = alloc["client"]

    # 1 开头深市 ETF/LOF 妙想无法交易：跳过 API，转为手动待办
    all_orders = sells + buys
    executable = [o for o in all_orders if not is_mx_untradable(o.code)]
    manual = [o for o in all_orders if is_mx_untradable(o.code)]

    # 先卖后买（卫星卖 → 核心卖 → 核心买 → 卫星买）
    for order in executable:
        qty = _qty(order)
        resp = client.trade(
            trade_type=order.action,
            stock_code=order.code,
            quantity=qty,
            use_market_price=True,
        )
        ok = resp is not None and resp.get("code") in ("0", "200")
        mark = "✅" if ok else "❌"
        msg = (resp or {}).get("message", "未知错误")
        results.append(f"{mark} {order.action.upper()} {order.name}({order.code}) {qty}股 ≈ {order.amount:,.0f}元 —— {'成功' if ok else f'失败: {msg}'}")

    lines.append(f"**总资产**: {total_assets:,.0f} 元 | **可用资金**: {avail_balance:,.0f} 元")
    lines.append(
        f"本次执行 {len(executable)} 笔（卖出 {sum(1 for o in executable if o.action == 'sell')}，"
        f"买入 {sum(1 for o in executable if o.action == 'buy')}）"
        + (f"，另有 {len(manual)} 笔需手动" if manual else "")
    )
    lines.append("")
    lines.extend(f"- {r}" for r in results)
    lines.append("")

    if manual:
        lines.append("### ⚠️ 需手动处理（妙想无法交易 1 开头深市 ETF/LOF）")
        lines.append("")
        lines.append(f"> {MX_UNTRADABLE_REASON}，请登录妙想 App 手动操作：")
        lines.append("")
        for o in manual:
            verb = "买入" if o.action == "buy" else "卖出"
            lines.append(f"- 手动{verb} {o.name}({o.code}) {_qty(o)}股 ≈ {o.amount:,.0f}元 — {o.reason}")
        lines.append("")

    # 执行后复核持仓
    fresh = MXMoniClient().get_positions()
    if fresh:
        lines.append("**执行后持仓**：")
        lines.append("")
        for p in fresh:
            mv = float(p.get("market_value", 0) or 0)
            pct = mv / total_assets * 100 if total_assets > 0 else 0
            lines.append(f"- {p.get('name', '')}({p.get('code', '')}) {p.get('count', 0)}股 ≈ {mv:,.0f}元（{pct:.1f}%）")
        lines.append("")

    return "\n".join(lines)


# ── 报告生成 ──

def _generate_report(execute: bool = False) -> str:
    """生成完整周报（execute=True 时自动执行调仓并附结果）"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    alloc = _compute_allocation()
    deploy_section = _deploy_cash_section(alloc)
    holding_section = _holding_overview(alloc)

    sections = [
        f"# ETF 周度观察 — {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"**生成时间**: {now}" + _style_state_line(),
        "",
        _market_overview(),
        _buy_priority(),
        deploy_section,
        holding_section,
        _satellite_overview(alloc.get("satellite")),
        _momentum_observe(alloc.get("satellite")),
    ]

    satellite = alloc.get("satellite")
    has_orders = bool(alloc and alloc["orders"])
    has_sat_orders = bool(satellite and (satellite["sells"] or satellite["buy_orders"]))
    if execute and alloc and (has_orders or has_sat_orders):
        sections.append(_execute_batch(alloc))

    warn = _sell_warning()
    if warn:
        sections.append(warn)

    sections.extend([
        "---",
        "",
        "*免责声明：本报告仅供观察参考，不构成投资建议。*",
    ])

    return "\n\n".join(sections)


def _save_report(report: str) -> str:
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    today_str = datetime.now().strftime('%Y%m%d')
    report_path = os.path.join(reports_dir, f"etf_weekly_{today_str}.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"周报已保存: {report_path}")
    return report_path


def _send_notification(report: str) -> bool:
    from src.notify.service import NotificationService
    notifier = NotificationService()
    if not notifier.is_available():
        logger.warning("通知服务未配置")
        return False
    success = notifier.send(report)
    if success:
        logger.info("通知发送成功")
    else:
        logger.warning("通知发送失败")
    return success


def main():
    parser = argparse.ArgumentParser(description='ETF 周度观察报告')
    parser.add_argument('--no-notify', action='store_true', help='不发送通知')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--execute', action='store_true', help='自动执行调仓指令（默认只出建议）')
    parser.add_argument('--force', action='store_true', help='跳过交易日检查（手动调试用）')
    args = parser.parse_args()

    setup_logging(log_prefix="etf_observe", debug=args.debug)

    # 交易日检查：非交易日且未 --force 时直接跳过（节假日周一）
    if not args.force and not is_trading_day():
        logger.info("今天不是 A 股交易日，跳过执行")
        print("今天不是 A 股交易日，跳过执行")
        return 0

    logger.info("=" * 60)
    logger.info("ETF 周度观察报告" + ("（自动调仓）" if args.execute else ""))
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    try:
        report = _generate_report(execute=args.execute)
    except Exception as e:
        logger.exception(f"生成报告失败: {e}")
        return 1

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print("\n" + report)
    _save_report(report)

    if not args.no_notify:
        _send_notification(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
