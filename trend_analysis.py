# -*- coding: utf-8 -*-
"""
===================================
趋势交易策略 — 日度分析与信号检测
===================================

定位：趋势波段系统。只做主线中的强趋势股，只在缩量回踩时介入。

职责：
1. 每日执行妙想选股得到选股名单（截面状态条件，纯内存，不读不写妙想自选），直接技术分析
2. 市场环境过滤（调用 market_gate 模块）
3. 纯技术分析（缩量回踩MA5等规则）
4. 负面清单否决（V1/V4 硬否决 + V5 观察项，仅影响当日结果，次日名单随新一轮妙想选股自然更新）

拦截层（按执行顺序）：
- market_gate + cycle_stage：市场环境层——gate 五态决定能否开仓，Cycle A1/A2 决定开多大
- veto_rules：负面清单（V1/V4 硬否决；V5 主力资金已降级为观察项），触发即不进信号池、不看评分
- signal_detector：买点形态 + 信号质量门（Cycle C1/D1 过滤 + ⑥b 中期过热等）

核心策略：
- 买点：主升中的缩量回踩MA5（不破5日线 + 换手率>3%）
- 不做：加速追高、情绪高潮接力、连续大阳后追涨
- 环境过滤：见 strategy/market.md
- 选股名单每天由妙想选股重新生成：选股条件是信号触发条件的截面必要子集，
  票在"变得可买的那天"必然进入名单，无需名单记忆。
  持仓不受影响：卖出由 trend_sell.py 负责

使用方式：
    python trend_analysis.py                    # 正常运行
    python trend_analysis.py --debug            # 调试模式
    python trend_analysis.py --no-notify        # 不发送通知
    python trend_analysis.py --stocks CODE1,CODE2  # 指定股票分析
    python trend_analysis.py --max-stocks N     # 最多分析N只股票（按跌幅排序）
"""
import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from data_provider import DataFetcherManager, canonical_stock_code
from src.indicators import add_standard_indicators
from src.config import setup_env
from src.notify.service import NotificationService
from src.mx.service import MXService
from src.mx.client import MXMoniClient
from src.market_state.cycle_stage import (
    evaluate_open_gate, run_cycle_stage, save_cycle_state,
)
from src.market_state.market_gate import (
    check_market_gate, diagnose_regime, fetch_index_df,
)
from src.trend.analyzer import StockTrendAnalyzer
from src.trend.veto_rules import (
    check_external_veto, check_market_veto, VetoStats,
    FROM_60D_LOW_MAX, TURNOVER_DAY_MAX,
)
from src.trend.signal_detector import (
    UNKNOWN_SECTOR, TechnicalSignal, detect_pullback_signals, MA20_BIAS_MAX,
)
from src.trend.report import format_buy_signal_alert, generate_technical_report, QUALIFY_SCORE
setup_env()

logger = logging.getLogger(__name__)

# 妙想选股默认关键词：生成当日选股名单的"状态层"条件（截面状态，一次批量查询可得）。
# 分层原则：截面状态 → 妙想选股；事件/路径（公告、资金流、大跌统计）→ veto_rules；
# 盘中形态（下影线、量比、收盘位置）→ signal_detector。
# 阈值单一来源：换手当日上限为选股截面条件、乖离与信号1条件⑥同源、
# 60日位置与信号条件⑥b 同源，均引用各规则模块常量。
# 精筛由 signal_detector 负责；信号层保留同阈值防御性复查。
# 实测妙想可正常解析全部子句。
DEFAULT_SCREEN_KEYWORD = (
    f"总市值大于30亿小于500亿，5日均线大于10日均线大于20日均线，"
    f"当日换手率大于3%小于{TURNOVER_DAY_MAX:g}%，"
    f"收盘价距20日均线乖离率小于{MA20_BIAS_MAX:g}%，"
    f"距离最近60日最低收盘价的涨幅小于{FROM_60D_LOW_MAX:g}%，"
    "非科创板非创业板非北交所非ST"
)


class SimpleTechnicalAnalyzer:
    """
    简化版技术分析器
    
    不依赖 LLM，纯技术指标计算
    """
    
    def __init__(self):
        self.fetcher = DataFetcherManager()
        self.mx_service = MXService()
        self.trend_analyzer = StockTrendAnalyzer()  # 复用 StockTrendAnalyzer 的均线计算
    
    def get_trading_dates(self, start_date: date, end_date: date) -> List[date]:
        from src.trading_calendar import get_trading_dates as _get_trading_dates
        return _get_trading_dates(start_date, end_date)

    def get_stocks_pct_change(self, stock_list: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        获取股票列表的涨跌幅

        Args:
            stock_list: [(code, name), ...]

        Returns:
            {code: pct_change} 涨跌幅百分比
        """
        pct_changes = {}
        for code, name in stock_list:
            try:
                quote = self.fetcher.get_realtime_quote(code)
                if quote and hasattr(quote, 'pct_chg'):
                    pct_changes[code] = float(quote.pct_chg) if quote.pct_chg is not None else 0.0
                else:
                    pct_changes[code] = 0.0
            except Exception as e:
                logger.debug(f"获取 {code} 涨跌幅失败: {e}")
                pct_changes[code] = 0.0
        return pct_changes

    def fetch_stock_data(self, code: str, days: int = 95) -> Optional[pd.DataFrame]:
        """
        获取股票历史数据（直接从网络获取）

        天数口径为自然日：95 天约 65 个交易日，满足 60 日位置类判定
        （信号条件⑥b 需 61 根 K 线）与 MA60 的计算需求。

        Args:
            code: 股票代码
            days: 获取天数（自然日）

        Returns:
            DataFrame 或 None
        """
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            df = self.fetcher.get_daily_data(code, start_str, end_str)

            if df is not None and hasattr(df, 'empty') and not df.empty:
                df_latest_date = pd.to_datetime(df['date'].max()).date()
                trading_dates = self.get_trading_dates(end_date - timedelta(days=30), end_date)
                if trading_dates:
                    last_trading_day = trading_dates[-1]
                    if df_latest_date < last_trading_day:
                        logger.error(f"❌ {code} 网络获取的数据仍过期(最新:{df_latest_date}, 需要:{last_trading_day})")
                        return None
                # 数据层不再算指标，入口取数后统一追加（ma5/ma10/ma20/volume_ratio）
                return add_standard_indicators(df)
            else:
                logger.error(f"❌ {code} 从网络获取数据失败")
                return None

        except Exception as e:
            logger.warning(f"获取 {code} 数据失败: {e}")
            return None

    def _fetch_stock_sector(self, code: str) -> str:
        """获取股票所属板块名称。

        取不到时降级为「未知板块」并记 warning —— 不返回空串：
        空串在 Markdown 表格里是空单元格，看起来像列错位，且无法与"确实没板块"区分。
        """
        try:
            from data_provider.fetchers.efinance_fetcher import EfinanceFetcher
            ef = EfinanceFetcher()
            df = ef.get_belong_board(code)
            if df is not None and not df.empty and "板块名称" in df.columns:
                sector = str(df["板块名称"].iloc[0]).strip()
                if sector:
                    return sector
            logger.warning(
                f"⚠️ {code}: 未取到所属板块 → 降级为「{UNKNOWN_SECTOR}」，板块类规则已跳过"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ {code}: 获取所属板块失败（{e}）→ 降级为「{UNKNOWN_SECTOR}」，板块类规则已跳过"
            )
        return UNKNOWN_SECTOR

    def mx_screen(self, keyword: str, page_size: int = 30) -> List[Tuple[str, str]]:
        """执行妙想选股，返回 [(code, name), ...]

        仅做状态层粗筛（精筛由 signal_detector 负责），结果即当日选股名单。
        解析 MX 返回的中文列名，字段名可能带日期后缀，做防御式匹配。
        """
        try:
            rows, total = self.mx_service.screen_stocks(keyword, page_no=1, page_size=page_size)
            if not rows:
                logger.warning(f"妙想选股无结果（关键词: {keyword}）")
                return []
            candidates = []
            for row in rows:
                code = None
                name = ""
                for k, v in row.items():
                    ks = str(k)
                    # 代码列：列名可能是"代码"或"股票代码"，排除"市场代码简称"
                    if "市场" in ks:
                        continue
                    if "代码" in ks and code is None:
                        code = str(v or "").strip()
                    elif "简称" in ks or "名称" in ks:
                        name = str(v or "").strip()
                if code:
                    code = code.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
                    candidates.append((code, name or code))
            logger.info(f"妙想选股返回 {total} 条，解析到 {len(candidates)} 只")
            return candidates
        except Exception as e:
            logger.warning(f"妙想选股失败: {e}")
            return []

    def analyze_all_stocks(self, stock_list: List[Tuple[str, str]],
                          max_stocks: Optional[int] = None,
                          sort_by_pct: bool = True,
                          market_env: Optional[Tuple] = None,
                          notifier: Any = None,
                          ) -> Tuple[List[TechnicalSignal],
                                     List[Tuple[str, str, str]],
                                     List[Tuple[str, str, str, str]]]:
        """
        分析选股名单，返回技术信号列表、失败列表、否决列表

        准入顺序：负面清单行情类否决（V4） → 买点信号检测（含 Cycle C1/D1 过滤）
                  → 负面清单外部数据类否决/观察（仅信号候选）

        Args:
            stock_list: [(code, name), ...]
            max_stocks: 最大分析数量，超过则按跌幅排序取前N只
            sort_by_pct: 是否按跌幅排序（优先分析跌幅大的股票）

        Returns:
            (技术信号列表,
             [(code, name, 失败原因), ...],
             [(code, name, 否决动作, 否决原因), ...])
             否决动作统一为跳过当日信号
        """
        all_signals = []
        failed_stocks = []
        vetoed_stocks = []
        veto_stats = VetoStats()

        if max_stocks and len(stock_list) > max_stocks:
            if sort_by_pct:
                logger.info(f"获取涨跌幅数据，股票数量 {len(stock_list)} 超过限制 {max_stocks}，按跌幅排序...")
                pct_changes = self.get_stocks_pct_change(stock_list)
                sorted_stocks = sorted(stock_list, key=lambda x: pct_changes.get(x[0], 0))
                stock_list = sorted_stocks[:max_stocks]
                logger.info(f"已选取跌幅最大的 {max_stocks} 只股票进行分析")
            else:
                stock_list = stock_list[:max_stocks]

        logger.info(f"开始处理 {len(stock_list)} 只股票...")

        for i, (code, name) in enumerate(stock_list):
            try:
                df = self.fetch_stock_data(code)

                # 统一计算 MA，避免在负面清单检查和信号检测中重复计算
                if df is not None and len(df) >= 10:
                    df = df.sort_values('date').reset_index(drop=True)
                    df = self.trend_analyzer._calculate_mas(df)

                # 负面清单（行情类 V4）：任一规则触发即否决，不进信号池、不看评分
                market_veto, market_skipped = check_market_veto(code, name, df, veto_stats)
                if market_veto.vetoed:
                    vetoed_stocks.append(
                        (code, name, market_veto.action, '；'.join(market_veto.reasons))
                    )
                    continue

                signals = detect_pullback_signals(code, name, df)

                if signals:
                    # 负面清单（外部数据类）：只对已产出信号的候选惰性调用妙想 API
                    # （V1 公告否决；V5 主力资金已降级为观察项，observations 随信号展示）
                    ext_veto, ext_skipped, observations = check_external_veto(
                        code, name, self.mx_service, self.fetcher, veto_stats)
                    if ext_veto.vetoed:
                        vetoed_stocks.append(
                            (code, name, ext_veto.action, '；'.join(ext_veto.reasons))
                        )
                        continue

                    sector = self._fetch_stock_sector(code)
                    # 合并行情类 + 外部类未生效规则，挂到每个信号上供报告标注
                    veto_skipped = market_skipped + ext_skipped
                    for s in signals:
                        s.sector = sector
                        if veto_skipped:
                            s.veto_skipped = veto_skipped
                        for obs in observations:
                            s.description += f"；👁️ {obs}（观察项，不拦截）"
                    logger.info(f"✅ {name}({code}) [{sector}]: 发现 {len(signals)} 个信号")

                    # 即时推送：每只股票一旦检出达标买点立即发一条
                    if notifier and market_env:
                        for s in signals:
                            if s.signal_type in ("pullback_ma5", "pullback_ma10") and s.score >= QUALIFY_SCORE:
                                alert = format_buy_signal_alert([s], market_env)
                                if alert:
                                    _send_notification(alert, notifier)
                else:
                    logger.info(f"    {name}({code}) 无信号")
                all_signals.extend(signals)

                if (i + 1) % 10 == 0:
                    logger.info(f"进度: {i + 1}/{len(stock_list)}")

            except Exception as e:
                logger.warning(f"分析 {name}({code}) 失败: {e}")
                failed_stocks.append((code, name, str(e)))
                continue

        all_signals.sort(key=lambda x: x.score, reverse=True)

        kept_count = len(stock_list) - len(vetoed_stocks) - len(failed_stocks)
        logger.info(
            f"处理完成 | 保留:{kept_count} "
            f"负面清单否决:{len(vetoed_stocks)} "
            f"失败:{len(failed_stocks)} 信号:{len(all_signals)}"
        )
        # 逐条输出负面清单规则的检查/否决/放行统计
        veto_stats.log_summary()
        return all_signals, failed_stocks, vetoed_stocks


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='简化版趋势跟踪系统 - 无 LLM（集成模拟交易）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )

    parser.add_argument(
        '--no-notify',
        action='store_true',
        help='不发送推送通知'
    )

    parser.add_argument(
        '--stocks',
        type=str,
        help='指定要分析的股票代码，逗号分隔（覆盖当日选股名单）'
    )

    parser.add_argument(
        '--max-stocks',
        type=int,
        default=None,
        help='每天最多分析多少只股票（按跌幅排序优先分析跌幅大的）'
    )

    parser.add_argument(
        '--screen-keyword',
        type=str,
        default=None,
        help='妙想选股条件关键词（默认使用内置选股条件，可用 SMART_SCREEN_KEYWORD 覆盖）'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='仅列出当日妙想选股名单，不执行分析'
    )

    trade_group = parser.add_argument_group('交易模式（可选）')
    trade_group.add_argument(
        '--trade',
        action='store_true',
        help='盘后分析模式：分析技术信号并生成次日交易计划'
    )
    trade_group.add_argument(
        '--trade-execute',
        action='store_true',
        help='盘中执行模式：检查持仓止损止盈，执行买入'
    )
    trade_group.add_argument(
        '--trade-plan',
        action='store_true',
        help='查看当前交易计划'
    )

    return parser.parse_args()


def _screen_keyword(args) -> str:
    """解析妙想选股关键词：命令行参数 > 环境变量 > 内置默认。"""
    return args.screen_keyword or os.getenv('SMART_SCREEN_KEYWORD') or DEFAULT_SCREEN_KEYWORD


def _list_mx_screen(analyzer: 'SimpleTechnicalAnalyzer', keyword: str) -> int:
    """列出当日妙想选股名单，不执行分析。"""
    logger.info(f"执行妙想选股: {keyword}")
    candidates = analyzer.mx_screen(keyword)
    if not candidates:
        logger.info("妙想选股无结果，名单为空")
        return 0
    logger.info(f"当日选股名单共 {len(candidates)} 只:")
    for code, name in candidates:
        logger.info(f"  {code} {name}")
    return 0


def _fetch_portfolio_exposure() -> Tuple[float, float]:
    """读妙想账户敞口：(持仓市值, 总资产)（元），供 Cycle A1 组合档位截断（每日一次）。

    持仓市值 = 总资产 − 可用余额。取不到（未配 MX_APIKEY / 接口失败）时返回
    (0.0, 0.0) → evaluate_open_gate 的截断分支跳过（fail-open：数据缺失不放大拦截）。
    """
    if not os.getenv("MX_APIKEY"):
        logger.warning("未配置 MX_APIKEY，Cycle 档位跳过组合敞口检查（fail-open）")
        return 0.0, 0.0
    try:
        bal = MXMoniClient().get_balance() or {}
        equity = float(bal.get("total_assets", 0) or 0)
        avail = float(bal.get("avail_balance", 0) or 0)
        return max(equity - avail, 0.0), equity
    except Exception as e:
        logger.warning(f"读取妙想资金失败，Cycle 档位跳过组合敞口检查: {e}")
        return 0.0, 0.0


def _save_report(report: str) -> str:
    """将报告保存到文件并返回路径。"""
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    today_str = datetime.now().strftime('%Y%m%d')
    report_path = os.path.join(reports_dir, f"technical_simple_{today_str}.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"报告已保存: {report_path}")
    return report_path


def _send_notification(report: str,
                       notifier: Optional[NotificationService] = None) -> bool:
    """发送通知，如果已配置且可用。

    notifier 可复用（一次运行内先发买点提醒、再发完整日报，避免重复初始化渠道）。
    """
    notifier = notifier or NotificationService()
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
    """主入口"""
    args = parse_arguments()
    
    # 配置日志
    from src.logging_config import setup_logging
    setup_logging(log_prefix="stock_analysis_simple", debug=args.debug)
    
    logger.info("=" * 60)
    logger.info("趋势交易策略 — 日度分析启动")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    try:
        analyzer = SimpleTechnicalAnalyzer()

        # 0. 列出当日妙想选股名单模式
        if args.list:
            logger.info("模式: 列出当日妙想选股名单")
            return _list_mx_screen(analyzer, _screen_keyword(args))

        max_stocks = args.max_stocks
        if max_stocks is None:
            max_stocks = int(os.getenv('MAX_STOCKS_PER_DAY', '0')) or None
        
        # 1. 获取股票列表（命令行指定 > 当日妙想选股名单）
        if args.stocks:
            # 使用命令行指定的股票
            stock_codes = [canonical_stock_code(c) for c in args.stocks.split(',') if c.strip()]
            name_mapping = {code: code for code in stock_codes}
            logger.info(f"使用指定股票列表: {stock_codes}")
        else:
            keyword = _screen_keyword(args)
            logger.info(f"执行妙想选股: {keyword}")
            candidates = analyzer.mx_screen(keyword)
            if candidates:
                stock_codes = [c for c, _ in candidates]
                name_mapping = {c: n for c, n in candidates}
                logger.info(f"当日选股名单 {len(stock_codes)} 只")
            else:
                logger.warning("妙想选股无结果，名单为空")

        if not stock_codes:
            logger.error("没有获取到股票列表，退出")
            return 1

        # 4. 市场环境判定（指数取数 → 纯结构判定）
        index_df = fetch_index_df()
        can_trade, market_summary, market_regime = check_market_gate(index_df)
        logger.info(market_summary)
        # 诊断明细（均线排列 / 偏离 MA20 / 命中路径）供报告展示
        regime_diag = diagnose_regime(index_df)
        logger.info(f"市场状态判定明细 → {regime_diag.describe()}")

        # ── Cycle 吸收：指数循环定位（A1 仓位档位 + A2 快速通道）──
        # A2 可在 gate 禁开日按快速通道放行（cap=cap_bottom，仅当日）；A1 顶部延伸档位
        # 截断新开仓；stage/cap 快照进日报「市场环境」节并存 data/cycle_state.json。
        cycle_info = None
        cycle_snap = run_cycle_stage(index_df)
        if cycle_snap:
            save_cycle_state(cycle_snap)
            invested, equity = _fetch_portfolio_exposure()
            allow, _cap, cap_note = evaluate_open_gate(can_trade, cycle_snap, invested, equity)
            if cap_note:
                logger.info(f"Cycle 档位裁决：{cap_note}")
            can_trade = allow
            cycle_info = {**cycle_snap, "cap_note": cap_note}
        market_env = (can_trade, market_summary, market_regime)
        notifier = None if args.no_notify else NotificationService()

        # 2. 技术分析
        stock_list = list(zip(stock_codes, [name_mapping.get(c, c) for c in stock_codes]))
        logger.info(f"待分析列表: {len(stock_list)} 只股票")

        signals, failed_stocks, vetoed_stocks = analyzer.analyze_all_stocks(
            stock_list, max_stocks=max_stocks, sort_by_pct=False,
            market_env=market_env, notifier=notifier,
        )

        report = generate_technical_report(signals,
                                           market_env=market_env,
                                           failed_stocks=failed_stocks,
                                           vetoed_stocks=vetoed_stocks,
                                           regime_diag=regime_diag,
                                           cycle_info=cycle_info)

        # 5. 保存报告
        _save_report(report)

        # 6. 发送通知
        if notifier:
            _send_notification(report, notifier)
        
        logger.info("运行完成")
        return 0
        
    except Exception as e:
        logger.exception(f"运行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
