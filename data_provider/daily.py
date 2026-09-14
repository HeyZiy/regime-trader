# -*- coding: utf-8 -*-
"""
===================================
个股日线取数策略（多源 failover + 新鲜度校验 + 列回补）
===================================
给定 Need(kind=stock_daily) 与 fetcher 集合：
1. 按能力声明筛候选（美股日线因此只剩 YfinanceFetcher）
2. 按优先级依次取数，失败自动切换
3. 非美股校验数据新鲜度：最新 bar 必须覆盖到最近交易日
4. A 股主源缺换手率时，从其余 A 股源回补该列

边界：不含单源知识、不持有实例 —— fetcher 集合由调用方（manager / 研究脚本）传入。

ETF/指数日线不走这里（见 bars.py）。
"""
import logging
import time
from datetime import date, timedelta
from typing import Any, List, Optional

import pandas as pd

from .routing import supporting
from .types import DataFetchError, Need, summarize_exception

logger = logging.getLogger(__name__)

# 主源未提供、需从备用源补齐的关键列（当前仅换手率）。
# 各 fetcher 出口已归一化为 STANDARD_COLUMNS，故只按标准列名对齐补齐，
# 不覆盖主源已有有效数据；补齐后仍整列缺失则告警，交由下游降级/跳过处理。
# 回退前先按各源的 SUPPORTS_COLUMNS 能力声明过滤，跳过日线接口确定没有该列的源。
BACKFILL_COLUMNS = ['turnover_rate']


def _clamp_to_last_trading_day(d: date) -> date:
    """把日期收敛到 ≤ d 的最近交易日。

    新鲜度检查的目标日期不能直接用调用方传入的 end_date（常为 date.today()）：
    周末/节假日不是交易日，行情永远不会有当天的 K 线，直接比对会把整条数据源链
    误判为"数据过期"（如 2026-09-05 周六全池失败）。
    日历拉取失败时 trading_calendar 按约定返回空列表 → 退化为周一~周五兜底
    （周末回退到周五，节假日情形宁可放行也不误杀）。
    """
    from src.trading_calendar import get_trading_dates
    trading_days = get_trading_dates(d - timedelta(days=30), d)
    if trading_days:
        return trading_days[-1]
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _backfill_missing_columns(
    primary_df: pd.DataFrame,
    fetchers: List[Any],
    stock_code: str,
    start_date: Optional[str],
    end_date: Optional[str],
    days: int,
    primary_name: str,
) -> pd.DataFrame:
    """主源缺失关键列时从其余数据源补齐（按标准化 'date' 列对齐），并告警仍缺失的列。"""
    if primary_df is None or primary_df.empty:
        return primary_df
    for col in BACKFILL_COLUMNS:
        if col in primary_df.columns and not primary_df[col].isna().all():
            continue  # 主源已有有效数据，无需补齐
        for fb in fetchers:
            if fb.name == primary_name:
                continue  # 不从主源自身补齐
            if fb.SUPPORTS_COLUMNS is not None and col not in fb.SUPPORTS_COLUMNS:
                logger.debug(
                    f"[列回退] {stock_code} 跳过 [{fb.name}]：其日线接口不提供 '{col}'"
                )
                continue  # 该源确定没有此列，不发无效请求
            try:
                sub = fb.get_daily_data(
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    days=days,
                )
            except Exception as e:
                logger.debug(f"[列回退] {stock_code} 从 [{fb.name}] 补齐 '{col}' 失败: {e}")
                continue
            if sub is None or sub.empty or col not in sub.columns or sub[col].isna().all():
                continue
            filled = primary_df['date'].map(dict(zip(sub['date'], sub[col])))
            if col not in primary_df.columns:
                primary_df[col] = filled
            else:
                missing = primary_df[col].isna()
                primary_df.loc[missing, col] = filled[missing].values
            logger.info(
                f"[列回退] {stock_code} 从 [{fb.name}] 补齐缺失列 '{col}' "
                f"(主源 {primary_name} 未提供)"
            )
            break
        # 遍历所有备用源后仍缺失 → 告警，下游降级/跳过处理
        if col not in primary_df.columns or primary_df[col].isna().all():
            logger.warning(
                f"[数据缺失] {stock_code} 关键列 '{col}' 在所有数据源均缺失，"
                f"依赖该列的信号/剔除逻辑将跳过或降级处理"
            )
    return primary_df


def fetch_stock_daily(
    need: Need,
    fetchers: List[Any],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: int = 30,
) -> pd.DataFrame:
    """按需求取个股日线（多源 failover）。

    Args:
        need: kind=stock_daily 的需求（market 由 codes.classify_market 判定）
        fetchers: 已实例化的数据源列表（按优先级）
        start_date / end_date / days: 取数窗口

    Returns:
        标准化日线 DataFrame（命中哪个数据源见日志）

    Raises:
        DataFetchError: 所有候选数据源都失败或被判定为数据过期时抛出
    """
    stock_code = need.code
    candidates = supporting(fetchers, need)

    errors = []
    total_fetchers = len(candidates)
    request_start = time.time()

    for attempt, fetcher in enumerate(candidates, start=1):
        try:
            logger.info(f"[数据源尝试 {attempt}/{total_fetchers}] [{fetcher.name}] 获取 {stock_code}...")
            df = fetcher.get_daily_data(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                days=days
            )

            if df is not None and not df.empty:
                # 检查数据新鲜度：最新日期必须 >= 请求截止日对应的最近交易日
                # （end_date 本身可能是周末/节假日，须先收敛，见 _clamp_to_last_trading_day）
                # 美股不在 A 股交易日历口径内，跳过该校验
                if need.market != "us":
                    try:
                        df_latest = pd.to_datetime(df['date'].max()).date()
                        target_date = pd.to_datetime(end_date).date() if isinstance(end_date, str) else end_date
                        target_date = _clamp_to_last_trading_day(target_date)
                        # 简单判断：如果数据最新日期 < 目标日期，视为过期
                        if df_latest < target_date:
                            raise DataFetchError(
                                f"数据过期(最新:{df_latest}, 需要:{target_date})"
                            )
                    except DataFetchError:
                        raise
                    except Exception as e:
                        # 新鲜度检查出错，记录但继续使用数据
                        logger.debug(f"[{fetcher.name}] 数据新鲜度检查失败: {e}")

                elapsed = time.time() - request_start
                logger.info(
                    f"[数据源完成] {stock_code} 使用 [{fetcher.name}] 获取成功: "
                    f"rows={len(df)}, elapsed={elapsed:.2f}s"
                )
                # 主源缺失关键列时从备用源补齐；补齐后仍缺失则告警（交由下游降级/跳过）
                # 回退列（换手率）只被 A 股规则消费，且回退源全是 A 股源，非 A 股不补
                if need.market == "cn":
                    df = _backfill_missing_columns(
                        df, fetchers, stock_code, start_date, end_date, days,
                        primary_name=fetcher.name,
                    )
                return df

        except Exception as e:
            error_type, error_reason = summarize_exception(e)
            error_msg = f"[{fetcher.name}] ({error_type}) {error_reason}"
            logger.warning(
                f"[数据源失败 {attempt}/{total_fetchers}] [{fetcher.name}] {stock_code}: "
                f"error_type={error_type}, reason={error_reason}"
            )
            errors.append(error_msg)
            if attempt < total_fetchers:
                next_fetcher = candidates[attempt]
                logger.info(f"[数据源切换] {stock_code}: [{fetcher.name}] -> [{next_fetcher.name}]")
            # 继续尝试下一个数据源
            continue

    # 所有数据源都失败
    error_summary = f"所有数据源获取 {stock_code} 失败:\n" + "\n".join(errors)
    elapsed = time.time() - request_start
    logger.error(f"[数据源终止] {stock_code} 获取失败: elapsed={elapsed:.2f}s\n{error_summary}")
    raise DataFetchError(error_summary)
