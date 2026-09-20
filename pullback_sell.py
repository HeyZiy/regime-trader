# -*- coding: utf-8 -*-
"""
===================================
趋势策略 — 尾盘卖出执行（每交易日 14:45 后）
===================================

每交易日 14:45 尾盘运行，承担「检测 → 执行 → 出报告 → 通知」闭环：

1. 读取妙想模拟仓股票持仓（持仓事实来源）
2. 复用 src/pullback_trend/sell_rules.py 的完全分类规则检测卖出信号
   （第一卖点减仓50%：放量破5日线/回撤≥5%/板块走弱；第二卖点清仓：破位+16日到期；
   Cycle 吸收 B1 延伸计数并入动作池）
3. 命中即自动下模拟仓市价单（委托数量为 100 整数倍，按可用股数收敛）
4. 自行渲染成交报告并推送

执行约定：
- reduce_half（减仓50%）/ clear（清仓）均自动下市价单，无需人工确认。
- 持仓峰值（peak）每日更新并落盘 data/position_exit_state.json（只升不降；加仓入场日
  推进时重置），仅作状态留痕，不触发卖出。
- 逐只隔离：单只取数或下单失败不影响其余持仓，最终汇总四类结果
  （success / failed / manual_skip / insufficient）。
- 不可交易标的（1 开头深市 ETF/LOF，见 src/mx/client.py）记为 manual_skip 单列提示。
- 下单前按 avail_count 收敛：可用股数可能小于总持仓（T+1 或已挂单），
  不足一手则跳过，避免废单。
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_provider import canonical_stock_code
from src.config import setup_env
from src.logging_config import setup_logging
from src.market_state.market_gate import check_market_gate, fetch_index_df
from src.mx.client import MXMoniClient, is_mx_untradable
from src.mx.position_utils import (
    filter_stock_positions, get_last_buy_dates_safe, position_profit_pct,
)
from src.notify.service import NotificationService
from src.pullback_trend.cycle_overlay import ExhaustionTracker
from src.pullback_trend.sell_rules import (
    HoldingRow, SellSignal, detect_sell_signals, fetch_sector_pct_map, match_sector_pct,
)
from src.pullback_trend.signal_detector import UNKNOWN_SECTOR
setup_env()

logger = logging.getLogger(__name__)

# 持仓退出状态（peak / 入场价 / 入场日），每日 14:45 运行时更新落盘。
# peak 只升不降：跨日峰值记忆不依赖行情窗口，加仓/换仓通过入场日判重自然重置。
EXIT_STATE_FILE = Path(__file__).parent / "data" / "position_exit_state.json"

# 板块行情连续失败 ≥ N 个运行日 → 板块类卖出规则（板块走弱/主线退潮）自动停用
# 并 ERROR 告警，直到某次成功拉取自动恢复。
SECTOR_HEALTH_FILE = Path(__file__).parent / "data" / "sector_fetch_health.json"
SECTOR_FAIL_DISABLE_STREAK = 5

# 执行结果状态
ST_SUCCESS = "success"          # 委托成功
ST_FAILED = "failed"            # 委托失败（含非交易时段）
ST_MANUAL = "manual_skip"       # 模拟仓不可交易，需用户手动
ST_INSUFFICIENT = "insufficient"  # 可用股数不足一手
ST_DRY_RUN = "dry_run"          # 试运行，未真正下单

ACTION_LABEL = {"reduce_half": "🟠 减仓50%", "clear": "🔴 清仓"}
STATUS_LABEL = {
    ST_SUCCESS: "✅ 已成交",
    ST_FAILED: "❌ 委托失败",
    ST_MANUAL: "⚠️ 需手动",
    ST_INSUFFICIENT: "➖ 不足一手",
    ST_DRY_RUN: "🔍 试运行",
}


@dataclass
class SellExecution:
    """单只持仓的卖出执行结果。

    组合而非复制：持仓/信号/板块事实复用 HoldingRow（sell_rules 定义的领域类型，
    sector_skipped 语义在那里定义），这里只追加执行侧字段。
    """
    row: HoldingRow                          # 持仓 + 卖出信号 + 板块判定事实
    shares: int = 0                          # 实际委托股数
    status: str = ""                         # 空 = 未执行（无信号）
    message: str = ""                        # 失败原因 / 备注
    data_ok: bool = True                     # 行情是否取到（False=判不了）

    @property
    def code(self) -> str:
        return self.row.code

    @property
    def name(self) -> str:
        return self.row.name

    @property
    def signal(self) -> Optional[SellSignal]:
        return self.row.signal

    @property
    def executed(self) -> bool:
        return self.status in (ST_SUCCESS, ST_FAILED, ST_DRY_RUN)


def _f(v) -> str:
    """价格格式化；不可用时返回 '-' 而不是抛异常。"""
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "-"


def _f_pct(v) -> str:
    """百分比格式化。"""
    try:
        return f"{float(v):+.2f}%"
    except (TypeError, ValueError):
        return "-"


def execute_sell(sig: SellSignal, position: dict, client: MXMoniClient,
                 dry_run: bool = False) -> Tuple[int, str, str]:
    """对单只持仓执行卖出委托。

    Args:
        sig: 卖出信号（suggest_shares 已为 100 整数倍）
        position: 妙想持仓 dict（取 avail_count 做可用量收敛）
        client: 妙想模拟仓客户端
        dry_run: True 时只计算不下单

    Returns:
        (shares, status, message)
    """
    code = sig.code

    # 妙想模拟仓无法识别 1 开头深市 ETF/LOF 的市场号，只能手动处理
    if is_mx_untradable(code):
        return 0, ST_MANUAL, "妙想模拟仓无法交易该标的，需手动卖出"

    avail = int(position.get("avail_count", 0) or 0)
    if avail <= 0:
        # 部分接口不返回 avail_count，退化为总持仓
        avail = int(position.get("count", 0) or 0)

    # 按可用量收敛并向下取整到 100 整数倍（妙想对非整手直接拒单）
    shares = min(sig.suggest_shares, avail)
    shares = (shares // 100) * 100
    if shares < 100:
        return 0, ST_INSUFFICIENT, f"可卖不足一手（可用{avail}股，建议{sig.suggest_shares}股）"

    if dry_run:
        return shares, ST_DRY_RUN, "试运行，未下单"

    resp = client.trade("sell", code, shares, use_market_price=True)
    if resp and resp.get("code") in ("0", "200"):
        return shares, ST_SUCCESS, ""
    message = (resp or {}).get("message", "未知错误")
    return shares, ST_FAILED, message


def _load_exit_state() -> Dict[str, dict]:
    """读取持仓退出状态（peak/入场价/入场日）。文件缺失或损坏按空状态处理。"""
    try:
        if EXIT_STATE_FILE.exists():
            return json.loads(EXIT_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"退出状态文件读取失败，按空状态处理: {e}")
    return {}


def _save_exit_state(state: Dict[str, dict]) -> None:
    try:
        EXIT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        EXIT_STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"退出状态文件写入失败: {e}")


def _build_exit_ctx(df: pd.DataFrame, position: dict, entry_date_str: str,
                    state_entry: dict) -> dict:
    """构建单只持仓的退出上下文（持有交易日数），并更新峰值。

    peak = max(历史峰值, 入场价, 入场日（含）以来最高收盘)，只升不降；
    入场日推进（加仓）时重置峰值、从新入场日重算——旧峰值不再适用新腿。
    df 最后一根 bar 为 14:45 近似收盘（尾盘口径），峰值随之更新。
    peak 仅作状态留痕（审计/未来实验），不生成触发价；退出主干为 16 日到期。

    Returns:
        {held_days, peak}
    """
    cost = float(position.get("cost_price", 0) or 0)
    ed = None
    if entry_date_str:
        try:
            ed = datetime.strptime(entry_date_str[:10], "%Y-%m-%d").date()
        except ValueError:
            logger.warning(f"入场日期无法解析：{entry_date_str!r}，到期天数检查跳过")

    stored_ed = str(state_entry.get("entry_date", ""))
    if ed is not None and stored_ed and entry_date_str[:10] > stored_ed:
        state_entry["peak"] = 0.0
        logger.info("入场日推进（加仓），持仓峰值重置后重算")

    if ed is not None:
        since_entry = df.loc[pd.to_datetime(df["date"]) >= pd.Timestamp(ed), "close"]
        hist_peak = float(since_entry.astype(float).max()) if len(since_entry) else 0.0
    else:
        # 入场日缺失：回退全窗口取最高（保守偏高，仅作峰值留痕）
        hist_peak = float(df["close"].astype(float).max()) if len(df) else 0.0

    prev_peak = float(state_entry.get("peak", 0) or 0)
    state_entry["peak"] = round(max(prev_peak, hist_peak, cost), 4)
    state_entry["entry_date"] = entry_date_str[:10] if ed else stored_ed
    state_entry["entry_price"] = cost
    state_entry["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    peak = state_entry["peak"]

    held_days = None
    if ed is not None:
        try:
            from src.trading_calendar import get_trading_dates

            # 持有交易日数不含入场日：T 日收盘买入，T+1 为第 1 个持有日
            held_days = len(get_trading_dates(ed + timedelta(days=1), datetime.now().date()))
        except Exception as e:
            logger.warning(f"持有天数计算失败，到期检查跳过: {e}")

    return {
        "held_days": held_days,
        "peak": peak,
    }


def _load_sector_health() -> Dict:
    """读取板块行情数据源健康状态（fail-soft：缺失/损坏按初始状态处理）。"""
    try:
        if SECTOR_HEALTH_FILE.exists():
            return json.loads(SECTOR_HEALTH_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"板块健康状态文件读取失败，按初始状态处理: {e}")
    return {}


def _save_sector_health(health: Dict) -> None:
    try:
        SECTOR_HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        SECTOR_HEALTH_FILE.write_text(
            json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"板块健康状态文件写入失败: {e}")


def _fetch_sector_pct_map_with_health() -> Tuple[Dict[str, float], bool]:
    """拉取板块涨跌幅并做数据源健康监控。

    板块行情多源回退数据源脆弱，连续失败达到阈值后自动停用板块类规则并
    ERROR 告警，避免每日重复报错刷屏；某次成功拉取即自动恢复。

    Returns:
        (sector_pct_map, disabled) — disabled=True 表示板块类规则已处于停用状态
    """
    health = _load_sector_health()
    sector_map = fetch_sector_pct_map()

    if sector_map:
        if int(health.get("fail_streak", 0) or 0) >= SECTOR_FAIL_DISABLE_STREAK:
            logger.info("✅ 板块行情恢复，板块类卖出规则重新启用")
        health["fail_streak"] = 0
        _save_sector_health(health)
        return sector_map, False

    streak = int(health.get("fail_streak", 0) or 0) + 1
    health["fail_streak"] = streak
    health["last_fail_date"] = datetime.now().strftime("%Y-%m-%d")
    _save_sector_health(health)

    disabled = streak >= SECTOR_FAIL_DISABLE_STREAK
    if disabled:
        logger.error(
            f"🔴 板块行情连续 {streak} 个运行日获取失败 → 板块类卖出规则"
            f"（板块走弱/主线退潮）自动停用，恢复后将自动重新启用"
        )
    return {}, disabled


def run_sell(analyzer, client: MXMoniClient, dry_run: bool = False
             ) -> Tuple[List[SellExecution], str]:
    """检测持仓卖出信号并执行。

    Returns:
        (执行结果列表, 市场状态)
    """
    _, market_summary, regime = check_market_gate(fetch_index_df())
    logger.info(market_summary)

    positions = filter_stock_positions(client.get_positions())
    if not positions:
        logger.info("妙想模拟仓当前无股票持仓")
        return [], regime

    entry_map = get_last_buy_dates_safe(client)
    sector_pct_map, sector_disabled = _fetch_sector_pct_map_with_health()
    if not sector_pct_map:
        logger.warning(
            "板块行情不可用，板块类卖出规则（板块走弱/主线退潮）跳过"
            + ("（连续多日失败，规则停用中，恢复后自动启用）" if sector_disabled else "")
        )

    exit_state = _load_exit_state()
    # Cycle 吸收：B1 延伸计数。状态寄生 exit_state 每仓 dict：新仓由
    # setdefault 创建（=on_entry）、循环末尾的 gone-codes 清理移除（=on_exit），
    # 与持仓峰值 peak 同生命周期。
    exhaustion = ExhaustionTracker()
    results: List[SellExecution] = []
    for p in positions:
        code = canonical_stock_code(p.get("code", ""))
        name = p.get("name", "") or code

        sector = analyzer._fetch_stock_sector(code)
        sector_pct = match_sector_pct(sector, sector_pct_map)
        sector_skipped = sector == UNKNOWN_SECTOR or sector_pct is None
        if sector_skipped:
            logger.warning(
                f"⚠️ {name}({code}): 板块信息不可用（板块={sector}，"
                f"板块行情={'有' if sector_pct is not None else '无'}）→ 板块类卖出规则已跳过"
            )

        row = HoldingRow(
            position=p, sector=sector,
            sector_pct=sector_pct, sector_skipped=sector_skipped,
        )
        res = SellExecution(row=row)

        try:
            df = analyzer.fetch_stock_data(code)
            if df is not None and len(df) >= 10:
                df = df.sort_values('date').reset_index(drop=True)
                df = analyzer.trend_analyzer._calculate_mas(df)
                state_entry = exit_state.setdefault(code, {})
                exit_ctx = _build_exit_ctx(df, p, entry_map.get(code, ""), state_entry)

                # Cycle 吸收：B1 延伸计数。bias 用尾盘
                # 近似收盘口径，与持仓峰值 peak 同源；动作经 ext_action 并入卖出判定。
                ext_action = None
                latest = df.iloc[-1]
                ma10_v, close_v = latest.get("ma10"), latest.get("close")
                if pd.notna(ma10_v) and float(ma10_v) > 0 and pd.notna(close_v):
                    bias10 = (float(close_v) - float(ma10_v)) / float(ma10_v) * 100
                    bar_date = str(pd.to_datetime(df["date"].iloc[-1]).date())
                    ext_action = exhaustion.update(state_entry, bias10, bar_date)
                    if ext_action:
                        logger.info(f"Cycle B1 延伸计数触发 {name}({code})：{ext_action[1]}")

                sig = detect_sell_signals(
                    code, name, df, p,
                    sector=sector, sector_pct=sector_pct,
                    entry_date=entry_map.get(code, ""),
                    exit_ctx=exit_ctx, ext_action=ext_action,
                )
            else:
                sig = None
                res.data_ok = False

            row.signal = sig
            if sig is not None:
                shares, status, message = execute_sell(sig, p, client, dry_run=dry_run)
                res.shares, res.status, res.message = shares, status, message
                label = ACTION_LABEL.get(sig.action, sig.action)
                tail = f" | {message}" if message else ""
                logger.info(
                    f"{label} {name}({code}) {shares}股 → {STATUS_LABEL.get(status, status)}"
                    f" | {'；'.join(sig.reasons)}{tail}"
                )
            elif res.data_ok:
                logger.info(f"    {name}({code}) 无卖出信号，继续持有")
            else:
                logger.warning(f"    {name}({code}) 行情获取失败，无法检测卖出信号")

        except Exception as e:
            # 单只失败不阻断其余持仓
            res.data_ok = False
            res.status = ST_FAILED
            res.message = str(e)
            logger.warning(f"处理 {name}({code}) 失败: {e}")

        results.append(res)

    # 已清仓的持仓移出退出状态，其余落盘（peak 只升不降跨日记忆）
    held_codes = {canonical_stock_code(p.get("code", "")) for p in positions}
    for gone in set(exit_state) - held_codes:
        exit_state.pop(gone, None)
    _save_exit_state(exit_state)

    return results, regime


def render_report(results: List[SellExecution], regime: str, dry_run: bool) -> str:
    """渲染成交报告（Markdown）。"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        "# 趋势策略 — 尾盘卖出执行报告",
        "",
        f"**执行时间**：{now}" + ("（试运行，未下单）" if dry_run else ""),
        f"**市场状态**：{regime}",
        f"**持仓检查**：{len(results)} 只股票持仓",
        "",
    ]

    # 单次遍历分类结果
    categorized = {
        'succeeded': [], 'failed': [], 'manual': [], 'insufficient': [],
        'holds': [], 'undetermined': []
    }
    for r in results:
        if r.signal is None:
            if r.data_ok:
                categorized['holds'].append(r)
            else:
                categorized['undetermined'].append(r)
        else:
            if r.status in (ST_SUCCESS, ST_DRY_RUN):
                categorized['succeeded'].append(r)
            elif r.status == ST_FAILED:
                categorized['failed'].append(r)
            elif r.status == ST_MANUAL:
                categorized['manual'].append(r)
            elif r.status == ST_INSUFFICIENT:
                categorized['insufficient'].append(r)

    triggered = [r for r in results if r.signal is not None]
    succeeded = categorized['succeeded']
    failed = categorized['failed']
    manual = categorized['manual']
    insufficient = categorized['insufficient']
    holds = categorized['holds']
    undetermined = categorized['undetermined']

    lines.extend([
        "## 执行概览",
        "",
        f"- 触发卖出信号：**{len(triggered)}** 只（已委托 {len(succeeded)}，失败 {len(failed)}）",
        f"- 需手动处理：{len(manual)} 只　可用不足一手：{len(insufficient)} 只",
        f"- 继续持有：{len(holds)} 只　行情缺失未判：{len(undetermined)} 只",
        "",
    ])

    if triggered:
        to_show = [r for r in triggered if r.status]
        lines.extend([
            "## 卖出执行明细",
            "",
            "| 股票 | 动作 | 现价 | 成本 | 盈亏 | 委托股数 | 结果 | 触发规则 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for r in to_show:
            sig = r.signal
            lines.append(
                f"| {r.name}({r.code}) | {ACTION_LABEL.get(sig.action, sig.action)} "
                f"| {_f(sig.current_price)} | {_f(sig.cost_price)} | {_f_pct(sig.profit_pct)} "
                f"| {r.shares}股 | {STATUS_LABEL.get(r.status, r.status)}"
                f"{f'（{r.message}）' if r.message else ''} "
                f"| {'；'.join(sig.reasons)} |"
            )
        lines.append("")

    # 板块不可用必须显式标注：否则「无卖出信号」会被误读成"板块没走弱"
    skipped = [r for r in results if r.row.sector_skipped]
    if skipped:
        shown = "、".join(f"{r.name}({r.code})" for r in skipped[:10])
        if len(skipped) > 10:
            shown += " 等"
        lines.extend([
            "> ⚠️ 以下标的板块信息不可用，板块类卖出规则（板块走弱 / 主线退潮）已跳过：",
            f"> {shown}。",
            "> 这些标的的「无卖出信号」不代表板块没走弱。",
            "",
        ])

    if holds:
        lines.extend([
            "## 继续持有（无卖出信号）",
            "",
            "| 股票 | 现价 | 成本 | 盈亏 | 板块 |",
            "| --- | --- | --- | --- | --- |",
        ])
        for r in holds:
            if r.row.sector == UNKNOWN_SECTOR:
                sector_label = f"未知板块{'（规则已跳过）' if r.row.sector_skipped else ''}"
            else:
                sector_label = r.row.sector
                if r.row.sector_pct is not None:
                    sector_label += f"（{_f_pct(r.row.sector_pct)}）"
            lines.append(
                f"| {r.name}({r.code}) | {_f(r.row.position.get('current_price', 0))} "
                f"| {_f(r.row.position.get('cost_price', 0))} "
                f"| {_f_pct(position_profit_pct(r.row.position))} "
                f"| {sector_label} |"
            )
        lines.append("")

    if undetermined:
        lines.extend([
            "## 行情缺失，未判定",
            "",
        ])
        for r in undetermined:
            reason = r.message or "行情获取失败"
            lines.append(f"- {r.name}({r.code})：{reason}")
        lines.append("")

    return "\n".join(lines)


def _save_report(report: str) -> str:
    """保存报告到 reports/ 并返回路径。"""
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, f"trend_sell_{datetime.now().strftime('%Y%m%d')}.md")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"报告已保存: {path}")
    return path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='趋势策略 — 尾盘卖出执行（检测 + 自动下模拟仓单 + 出报告）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no-notify', action='store_true', help='不发送推送通知')
    parser.add_argument('--dry-run', action='store_true', help='只检测不下单（调试用）')
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    setup_logging(log_prefix="trend_sell", debug=args.debug)

    logger.info("=" * 60)
    logger.info("趋势策略 — 尾盘卖出执行启动")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    if not os.getenv("MX_APIKEY"):
        logger.error("未配置 MX_APIKEY，无法读取持仓与下单")
        return 1

    try:
        # 复用盘后分析器的取数能力（日线 + 板块）
        from trend_analysis import SimpleTechnicalAnalyzer

        analyzer = SimpleTechnicalAnalyzer()
        client = MXMoniClient()

        results, regime = run_sell(analyzer, client, dry_run=args.dry_run)
        if not results:
            logger.info("无股票持仓，无需出报告")
            return 0

        report = render_report(results, regime, args.dry_run)
        _save_report(report)

        if not args.no_notify:
            notifier = NotificationService()
            if notifier.is_available():
                if notifier.send(report):
                    logger.info("通知发送成功")
                else:
                    logger.warning("通知发送失败")
            else:
                logger.warning("通知服务未配置")

        logger.info("运行完成")
        return 0

    except Exception as e:
        logger.exception(f"运行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
