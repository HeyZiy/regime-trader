# -*- coding: utf-8 -*-
"""
===================================
AkshareFetcher - 主数据源 (Priority 1)
===================================

数据来源：
1. 东方财富爬虫（通过 akshare 库） - 默认数据源
2. 新浪财经接口 - 备选数据源
3. 腾讯财经接口 - 备选数据源

特点：免费、无需 Token、数据全面
风险：爬虫机制易被反爬封禁

防封禁策略：
1. 每次请求前随机休眠 2-5 秒
2. 随机轮换 User-Agent
3. 使用 tenacity 实现指数退避重试
4. 熔断器机制：连续失败后自动冷却

增强数据：
- 实时行情：量比、换手率、市盈率、市净率、总市值、流通市值
- 筹码分布：获利比例、平均成本、筹码集中度
"""

import time

import logging
import os
import pandas as pd
import random
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from typing import Optional, Dict, Any, Tuple
from data_provider.fetchers.base import BaseFetcher
from data_provider.stats import calc_market_stats
from data_provider.types import (
    KIND_FUND_FLOW, KIND_REALTIME, KIND_STOCK_DAILY,
    DataFetchError, RateLimitError, STANDARD_COLUMNS,
    UnifiedRealtimeQuote, RealtimeSource,
    get_realtime_circuit_breaker, safe_float, safe_int,
)
from data_provider.codes import is_bse_code, is_st_stock, is_kc_cy_stock, normalize_stock_code, is_etf_code
from data_provider.codes import _is_hk_market
from data_provider.us_index_mapping import is_us_stock_code

# RealtimeQuote 别名，统一实时报价类型引用
RealtimeQuote = UnifiedRealtimeQuote


def is_hk_stock_code(stock_code: str) -> bool:
    """Public API: determine if a stock code is a Hong Kong stock."""
    return _is_hk_market(stock_code)


logger = logging.getLogger(__name__)

SINA_REALTIME_ENDPOINT = "hq.sinajs.cn/list"
TENCENT_REALTIME_ENDPOINT = "qt.gtimg.cn/q"


# User-Agent 池，用于随机轮换
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]


# 缓存实时行情数据（避免重复请求）
# TTL 设为 20 分钟 (1200秒)：
# - 批量分析场景：通常 30 只股票在 5 分钟内分析完，20 分钟足够覆盖
# - 实时性要求：股票分析不需要秒级实时数据，20 分钟延迟可接受
# - 防封禁：减少 API 调用频率
_realtime_cache: Dict[str, Any] = {
    'data': None,
    'timestamp': 0,
    'ttl': 1200  # 20分钟缓存有效期
}

# ETF 实时行情缓存
_etf_realtime_cache: Dict[str, Any] = {
    'data': None,
    'timestamp': 0,
    'ttl': 1200  # 20分钟缓存有效期
}


def _to_sina_tx_symbol(stock_code: str) -> str:
    """Convert 6-digit A-share code to sh/sz/bj prefixed symbol for Sina/Tencent APIs."""
    base = (stock_code.strip().split(".")[0] if "." in stock_code else stock_code).strip()
    if is_bse_code(base):
        return f"bj{base}"
    # Shanghai: 60xxxx, 5xxxx (ETF), 90xxxx (B-shares)
    if base.startswith(("6", "5", "90")):
        return f"sh{base}"
    return f"sz{base}"


def _classify_realtime_http_error(exc: Exception) -> Tuple[str, str]:
    """
    Classify Sina/Tencent realtime quote failures into stable categories.
    """
    detail = str(exc).strip() or type(exc).__name__
    lowered = detail.lower()

    remote_disconnect_keywords = (
        "remotedisconnected",
        "remote end closed connection without response",
        "connection aborted",
        "connection broken",
        "protocolerror",
        "chunkedencodingerror",
    )
    timeout_keywords = (
        "timeout",
        "timed out",
        "readtimeout",
        "connecttimeout",
    )
    rate_limit_keywords = (
        "banned",
        "blocked",
        "频率",
        "rate limit",
        "too many requests",
        "429",
        "限制",
        "forbidden",
        "403",
    )

    if any(keyword in lowered for keyword in remote_disconnect_keywords):
        return "remote_disconnect", detail
    if isinstance(exc, (TimeoutError, requests.exceptions.Timeout)) or any(
        keyword in lowered for keyword in timeout_keywords
    ):
        return "timeout", detail
    if any(keyword in lowered for keyword in rate_limit_keywords):
        return "rate_limit_or_anti_bot", detail
    if isinstance(exc, requests.exceptions.RequestException):
        return "request_error", detail
    return "unknown_request_error", detail


def _build_realtime_failure_message(
    source_name: str,
    endpoint: str,
    stock_code: str,
    symbol: str,
    category: str,
    detail: str,
    elapsed: float,
    error_type: str,
) -> str:
    return (
        f"{source_name} 实时行情接口失败: endpoint={endpoint}, stock_code={stock_code}, "
        f"symbol={symbol}, category={category}, error_type={error_type}, "
        f"elapsed={elapsed:.2f}s, detail={detail}"
    )


class AkshareFetcher(BaseFetcher):
    """
    Akshare 数据源实现
    
    优先级：1（最高）
    数据来源：东方财富网爬虫
    
    关键策略：
    - 每次请求前随机休眠 2.0-5.0 秒
    - 随机 User-Agent 轮换
    - 失败后指数退避重试（最多3次）
    """
    
    name = "AkshareFetcher"
    # 提升为最高优先级（有新浪财经作为备选，更稳定）
    priority = int(os.getenv("AKSHARE_PRIORITY", "0"))
    # 东财/新浪日线均含换手率列
    SUPPORTS_COLUMNS = {'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'turnover_rate'}

    # 日线/实时覆盖 A股+港股；主力资金流仅 A股。美股日线明确拒绝（复权问题，见 _fetch_raw_data）
    SUPPORTS = frozenset({
        (KIND_STOCK_DAILY, "cn"), (KIND_STOCK_DAILY, "hk"),
        (KIND_REALTIME, "cn"), (KIND_REALTIME, "hk"),
        (KIND_FUND_FLOW, "cn"),
    })
    def __init__(self, sleep_min: float = 2.0, sleep_max: float = 5.0):
        """
        初始化 AkshareFetcher
        
        Args:
            sleep_min: 最小休眠时间（秒）
            sleep_max: 最大休眠时间（秒）
        """
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self._last_request_time: Optional[float] = None
    
    def _set_random_user_agent(self) -> None:
        """
        设置随机 User-Agent
        
        通过修改 requests Session 的 headers 实现
        这是关键的反爬策略之一
        """
        try:
            import akshare as ak
            # akshare 内部使用 requests，我们通过环境变量或直接设置来影响
            # 实际上 akshare 可能不直接暴露 session，这里通过 fake_useragent 作为补充
            random_ua = random.choice(USER_AGENTS)
            logger.debug(f"设置 User-Agent: {random_ua[:50]}...")
        except Exception as e:
            logger.debug(f"设置 User-Agent 失败: {e}")
    
    def _enforce_rate_limit(self) -> None:
        """
        强制执行速率限制
        
        策略：
        1. 检查距离上次请求的时间间隔
        2. 如果间隔不足，补充休眠时间
        3. 然后再执行随机 jitter 休眠
        """
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            min_interval = self.sleep_min
            if elapsed < min_interval:
                additional_sleep = min_interval - elapsed
                logger.debug(f"补充休眠 {additional_sleep:.2f} 秒")
                time.sleep(additional_sleep)
        
        # 执行随机 jitter 休眠
        self.random_sleep(self.sleep_min, self.sleep_max)
        self._last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),  # 最多重试3次
        wait=wait_exponential(multiplier=1, min=2, max=30),  # 指数退避：2, 4, 8... 最大30秒
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Akshare 获取原始数据
        
        根据代码类型自动选择 API：
        - 美股：不支持，抛出异常由 YfinanceFetcher 处理（Issue #311）
        - 港股：使用 ak.stock_hk_hist()
        - ETF 基金：使用 ak.fund_etf_hist_em()
        - 普通 A 股：使用 ak.stock_zh_a_hist()
        
        流程：
        1. 判断代码类型（美股/港股/ETF/A股）
        2. 设置随机 User-Agent
        3. 执行速率限制（随机休眠）
        4. 调用对应的 akshare API
        5. 处理返回数据
        """
        # 根据代码类型选择不同的获取方法
        if is_us_stock_code(stock_code):
            # 美股：akshare 的 stock_us_daily 接口复权存在已知问题（参见 Issue #311）
            # 交由 YfinanceFetcher 处理，确保复权价格一致
            raise DataFetchError(
                f"AkshareFetcher 不支持美股 {stock_code}，请使用 YfinanceFetcher 获取正确的复权价格"
            )
        elif _is_hk_code(stock_code):
            return self._fetch_hk_data(stock_code, start_date, end_date)
        elif is_etf_code(stock_code):
            return self._fetch_etf_data(stock_code, start_date, end_date)
        else:
            return self._fetch_stock_data(stock_code, start_date, end_date)
    
    def _fetch_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取普通 A 股历史数据

        策略：
        1. 优先尝试东方财富接口 (ak.stock_zh_a_hist)
        2. 失败后尝试新浪财经接口 (ak.stock_zh_a_daily)
        3. 最后尝试腾讯财经接口 (ak.stock_zh_a_hist_tx)
        """
        # 尝试列表
        methods = [
            (self._fetch_stock_data_sina, "新浪财经"),
            (self._fetch_stock_data_tx, "腾讯财经"),
            (self._fetch_stock_data_em, "东方财富"),
        ]

        last_error = None

        for fetch_method, source_name in methods:
            try:
                logger.info(f"[数据源] 尝试使用 {source_name} 获取 {stock_code}...")
                df = fetch_method(stock_code, start_date, end_date)

                if df is not None and not df.empty:
                    logger.info(f"[数据源] {source_name} 获取成功")
                    return df
            except Exception as e:
                last_error = e
                logger.warning(f"[数据源] {source_name} 获取失败: {e}")
                # 继续尝试下一个

        # 所有都失败
        raise DataFetchError(f"Akshare 所有渠道获取失败: {last_error}")

    def _fetch_stock_data_em(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取普通 A 股历史数据 (东方财富)
        数据来源：ak.stock_zh_a_hist()
        """
        import akshare as ak

        # 防封禁策略 1: 随机 User-Agent
        self._set_random_user_agent()

        # 防封禁策略 2: 强制休眠
        self._enforce_rate_limit()

        logger.info(f"[API调用] ak.stock_zh_a_hist(symbol={stock_code}, ...)")

        try:
            import time as _time
            api_start = _time.time()

            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"
            )

            api_elapsed = _time.time() - api_start

            if df is not None and not df.empty:
                logger.info(f"[API返回] ak.stock_zh_a_hist 成功: {len(df)} 行, 耗时 {api_elapsed:.2f}s")
                return df
            else:
                logger.warning(f"[API返回] ak.stock_zh_a_hist 返回空数据")
                return pd.DataFrame()

        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['banned', 'blocked', '频率', 'rate', '限制']):
                raise RateLimitError(f"Akshare(EM) 可能被限流: {e}") from e
            raise e

    def _fetch_stock_data_sina(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取普通 A 股历史数据 (新浪财经)
        数据来源：ak.stock_zh_a_daily()
        """
        import akshare as ak

        # 转换代码格式：sh600000, sz000001, bj920748
        symbol = _to_sina_tx_symbol(stock_code)

        self._enforce_rate_limit()

        try:
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"
            )

            # 标准化新浪数据列名
            # 新浪返回：date, open, high, low, close, volume, amount, outstanding_share, turnover
            if df is not None and not df.empty:
                # 确保日期列存在
                if 'date' in df.columns:
                    df = df.rename(columns={'date': '日期'})

                # 映射其他列以匹配 _normalize_data 的期望
                # _normalize_data 期望：日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 换手率
                rename_map = {
                    'open': '开盘', 'high': '最高', 'low': '最低',
                    'close': '收盘', 'volume': '成交量', 'amount': '成交额',
                    'turnover': '换手率',
                }
                df = df.rename(columns=rename_map)

                # 新浪 turnover 列是小数比率（0.104325 = 10.43%），转为百分数
                if '换手率' in df.columns:
                    df['换手率'] = df['换手率'] * 100

                # 计算涨跌幅（新浪接口可能不返回）
                if '收盘' in df.columns:
                    df['涨跌幅'] = df['收盘'].pct_change() * 100
                    df['涨跌幅'] = df['涨跌幅'].fillna(0)

                return df
            return pd.DataFrame()

        except Exception as e:
            raise e

    def _fetch_stock_data_tx(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取普通 A 股历史数据 (腾讯财经)
        数据来源：ak.stock_zh_a_hist_tx()
        """
        import akshare as ak

        # 转换代码格式：sh600000, sz000001, bj920748
        symbol = _to_sina_tx_symbol(stock_code)

        self._enforce_rate_limit()

        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"
            )

            # 标准化腾讯数据列名
            # 腾讯返回：date, open, close, high, low, volume, amount
            if df is not None and not df.empty:
                rename_map = {
                    'date': '日期', 'open': '开盘', 'high': '最高',
                    'low': '最低', 'close': '收盘', 'volume': '成交量',
                    'amount': '成交额'
                }
                df = df.rename(columns=rename_map)

                # 腾讯数据通常包含 '涨跌幅'，如果没有则计算
                if 'pct_chg' in df.columns:
                    df = df.rename(columns={'pct_chg': '涨跌幅'})
                elif '收盘' in df.columns:
                    df['涨跌幅'] = df['收盘'].pct_change() * 100
                    df['涨跌幅'] = df['涨跌幅'].fillna(0)

                return df
            return pd.DataFrame()

        except Exception as e:
            raise e
    
    def _fetch_etf_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取 ETF 基金历史数据
        
        数据来源：ak.fund_etf_hist_em()
        
        Args:
            stock_code: ETF 代码，如 '512400', '159883'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            
        Returns:
            ETF 历史数据 DataFrame
        """
        import akshare as ak
        
        # 防封禁策略 1: 随机 User-Agent
        self._set_random_user_agent()
        
        # 防封禁策略 2: 强制休眠
        self._enforce_rate_limit()
        
        logger.info(f"[API调用] ak.fund_etf_hist_em(symbol={stock_code}, period=daily, "
                   f"start_date={start_date.replace('-', '')}, end_date={end_date.replace('-', '')}, adjust=qfq)")
        
        try:
            import time as _time
            api_start = _time.time()
            
            # 调用 akshare 获取 ETF 日线数据
            df = ak.fund_etf_hist_em(
                symbol=stock_code,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"  # 前复权
            )
            
            api_elapsed = _time.time() - api_start
            
            # 记录返回数据摘要
            if df is not None and not df.empty:
                logger.info(f"[API返回] ak.fund_etf_hist_em 成功: 返回 {len(df)} 行数据, 耗时 {api_elapsed:.2f}s")
                logger.info(f"[API返回] 列名: {list(df.columns)}")
                logger.info(f"[API返回] 日期范围: {df['日期'].iloc[0]} ~ {df['日期'].iloc[-1]}")
                logger.debug(f"[API返回] 最新3条数据:\n{df.tail(3).to_string()}")
            else:
                logger.warning(f"[API返回] ak.fund_etf_hist_em 返回空数据, 耗时 {api_elapsed:.2f}s")
            
            return df
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # 检测反爬封禁
            if any(keyword in error_msg for keyword in ['banned', 'blocked', '频率', 'rate', '限制']):
                logger.warning(f"检测到可能被封禁: {e}")
                raise RateLimitError(f"Akshare 可能被限流: {e}") from e
            
            raise DataFetchError(f"Akshare 获取 ETF 数据失败: {e}") from e
    
    def _fetch_us_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取美股历史数据
        
        数据来源：ak.stock_us_daily()（新浪财经接口）
        
        Args:
            stock_code: 美股代码，如 'AMD', 'AAPL', 'TSLA'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            
        Returns:
            美股历史数据 DataFrame
        """
        import akshare as ak
        
        # 防封禁策略 1: 随机 User-Agent
        self._set_random_user_agent()
        
        # 防封禁策略 2: 强制休眠
        self._enforce_rate_limit()
        
        # 美股代码直接使用大写
        symbol = stock_code.strip().upper()
        
        logger.info(f"[API调用] ak.stock_us_daily(symbol={symbol}, adjust=qfq)")
        
        try:
            import time as _time
            api_start = _time.time()
            
            # 调用 akshare 获取美股日线数据
            # stock_us_daily 返回全部历史数据，后续需要按日期过滤
            df = ak.stock_us_daily(
                symbol=symbol,
                adjust="qfq"  # 前复权
            )
            
            api_elapsed = _time.time() - api_start
            
            # 记录返回数据摘要
            if df is not None and not df.empty:
                logger.info(f"[API返回] ak.stock_us_daily 成功: 返回 {len(df)} 行数据, 耗时 {api_elapsed:.2f}s")
                logger.info(f"[API返回] 列名: {list(df.columns)}")
                
                # 按日期过滤
                df['date'] = pd.to_datetime(df['date'])
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                
                if not df.empty:
                    logger.info(f"[API返回] 过滤后日期范围: {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
                    logger.debug(f"[API返回] 最新3条数据:\n{df.tail(3).to_string()}")
                else:
                    logger.warning(f"[API返回] 过滤后数据为空，日期范围 {start_date} ~ {end_date} 无数据")
                
                # 转换列名为中文格式以匹配 _normalize_data
                # stock_us_daily 返回: date, open, high, low, close, volume
                rename_map = {
                    'date': '日期',
                    'open': '开盘',
                    'high': '最高',
                    'low': '最低',
                    'close': '收盘',
                    'volume': '成交量',
                }
                df = df.rename(columns=rename_map)
                
                # 计算涨跌幅（美股接口不直接返回）
                if '收盘' in df.columns:
                    df['涨跌幅'] = df['收盘'].pct_change() * 100
                    df['涨跌幅'] = df['涨跌幅'].fillna(0)
                
                # 估算成交额（美股接口不返回）
                if '成交量' in df.columns and '收盘' in df.columns:
                    df['成交额'] = df['成交量'] * df['收盘']
                else:
                    df['成交额'] = 0
                
                return df
            else:
                logger.warning(f"[API返回] ak.stock_us_daily 返回空数据, 耗时 {api_elapsed:.2f}s")
                return pd.DataFrame()
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # 检测反爬封禁
            if any(keyword in error_msg for keyword in ['banned', 'blocked', '频率', 'rate', '限制']):
                logger.warning(f"检测到可能被封禁: {e}")
                raise RateLimitError(f"Akshare 可能被限流: {e}") from e
            
            raise DataFetchError(f"Akshare 获取美股数据失败: {e}") from e

    def _fetch_hk_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取港股历史数据
        
        数据来源：ak.stock_hk_hist()
        
        Args:
            stock_code: 港股代码，如 '00700', '01810'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            
        Returns:
            港股历史数据 DataFrame
        """
        import akshare as ak
        
        # 防封禁策略 1: 随机 User-Agent
        self._set_random_user_agent()
        
        # 防封禁策略 2: 强制休眠
        self._enforce_rate_limit()
        
        # 确保代码格式正确（5位数字）
        code = stock_code.lower().replace('hk', '').zfill(5)
        
        logger.info(f"[API调用] ak.stock_hk_hist(symbol={code}, period=daily, "
                   f"start_date={start_date.replace('-', '')}, end_date={end_date.replace('-', '')}, adjust=qfq)")
        
        try:
            import time as _time
            api_start = _time.time()
            
            # 调用 akshare 获取港股日线数据
            df = ak.stock_hk_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"  # 前复权
            )
            
            api_elapsed = _time.time() - api_start
            
            # 记录返回数据摘要
            if df is not None and not df.empty:
                logger.info(f"[API返回] ak.stock_hk_hist 成功: 返回 {len(df)} 行数据, 耗时 {api_elapsed:.2f}s")
                logger.info(f"[API返回] 列名: {list(df.columns)}")
                logger.info(f"[API返回] 日期范围: {df['日期'].iloc[0]} ~ {df['日期'].iloc[-1]}")
                logger.debug(f"[API返回] 最新3条数据:\n{df.tail(3).to_string()}")
            else:
                logger.warning(f"[API返回] ak.stock_hk_hist 返回空数据, 耗时 {api_elapsed:.2f}s")
            
            return df
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # 检测反爬封禁
            if any(keyword in error_msg for keyword in ['banned', 'blocked', '频率', 'rate', '限制']):
                logger.warning(f"检测到可能被封禁: {e}")
                raise RateLimitError(f"Akshare 可能被限流: {e}") from e
            
            raise DataFetchError(f"Akshare 获取港股数据失败: {e}") from e
    
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Akshare 数据
        
        Akshare 返回的列名（中文）：
        日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
        
        需要映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()
        
        # 列名映射（Akshare 中文列名 -> 标准英文列名）
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '涨跌幅': 'pct_chg',
            '换手率': 'turnover_rate',
        }
        
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 添加股票代码列
        df['code'] = stock_code
        
        # 只保留需要的列
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]
        
        return df
    
    def get_realtime_quote(self, stock_code: str, source: str = "em") -> Optional[UnifiedRealtimeQuote]:
        """
        获取实时行情数据（支持多数据源）

        数据源优先级（可配置）：
        1. em: 东方财富（akshare ak.stock_zh_a_spot_em）- 数据最全，含量比/PE/PB/市值等
        2. sina: 新浪财经（akshare ak.stock_zh_a_spot）- 轻量级，基本行情
        3. tencent: 腾讯直连接口 - 单股票查询，负载小

        Args:
            stock_code: 股票/ETF代码
            source: 数据源类型，可选 "em", "sina", "tencent"

        Returns:
            UnifiedRealtimeQuote 对象，获取失败返回 None
        """
        circuit_breaker = get_realtime_circuit_breaker()

        # 根据代码类型选择不同的获取方法
        if is_us_stock_code(stock_code):
            # 美股不使用 Akshare，由 YfinanceFetcher 处理
            logger.debug(f"[API跳过] {stock_code} 是美股，Akshare 不支持美股实时行情")
            return None
        elif _is_hk_code(stock_code):
            return self._get_hk_realtime_quote(stock_code)
        elif is_etf_code(stock_code):
            source_key = "akshare_etf"
            if not circuit_breaker.is_available(source_key):
                logger.warning(f"[熔断] 数据源 {source_key} 处于熔断状态，跳过")
                return None
            return self._get_etf_realtime_quote(stock_code)
        else:
            source_key = f"akshare_{source}"
            if not circuit_breaker.is_available(source_key):
                logger.warning(f"[熔断] 数据源 {source_key} 处于熔断状态，跳过")
                return None
            # 普通 A 股：根据 source 选择数据源
            if source == "sina":
                return self._get_stock_realtime_quote_sina(stock_code)
            elif source == "tencent":
                return self._get_stock_realtime_quote_tencent(stock_code)
            else:
                return self._get_stock_realtime_quote_em(stock_code)
    
    def _get_stock_realtime_quote_em(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取普通 A 股实时行情数据（东方财富数据源）
        
        数据来源：ak.stock_zh_a_spot_em()
        优点：数据最全，含量比、换手率、市盈率、市净率、总市值、流通市值等
        缺点：全量拉取，数据量大，容易超时/限流
        """
        import akshare as ak
        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "akshare_em"
        
        try:
            # 检查缓存
            current_time = time.time()
            if (_realtime_cache['data'] is not None and 
                current_time - _realtime_cache['timestamp'] < _realtime_cache['ttl']):
                df = _realtime_cache['data']
                cache_age = int(current_time - _realtime_cache['timestamp'])
                logger.debug(f"[缓存命中] A股实时行情(东财) - 缓存年龄 {cache_age}s/{_realtime_cache['ttl']}s")
            else:
                # 触发全量刷新
                logger.info(f"[缓存未命中] 触发全量刷新 A股实时行情(东财)")
                last_error: Optional[Exception] = None
                df = None
                for attempt in range(1, 3):
                    try:
                        # 防封禁策略
                        self._set_random_user_agent()
                        self._enforce_rate_limit()

                        logger.info(f"[API调用] ak.stock_zh_a_spot_em() 获取A股实时行情... (attempt {attempt}/2)")
                        import time as _time
                        api_start = _time.time()

                        df = ak.stock_zh_a_spot_em()

                        api_elapsed = _time.time() - api_start
                        logger.info(f"[API返回] ak.stock_zh_a_spot_em 成功: 返回 {len(df)} 只股票, 耗时 {api_elapsed:.2f}s")
                        circuit_breaker.record_success(source_key)
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(f"[API错误] ak.stock_zh_a_spot_em 获取失败 (attempt {attempt}/2): {e}")
                        time.sleep(min(2 ** attempt, 5))

                # 更新缓存：成功缓存数据；失败也缓存空数据，避免同一轮任务对同一接口反复请求
                if df is None:
                    logger.error(f"[API错误] ak.stock_zh_a_spot_em 最终失败: {last_error}")
                    circuit_breaker.record_failure(source_key, str(last_error))
                    df = pd.DataFrame()
                _realtime_cache['data'] = df
                _realtime_cache['timestamp'] = current_time
                logger.info(f"[缓存更新] A股实时行情(东财) 缓存已刷新，TTL={_realtime_cache['ttl']}s")

            if df is None or df.empty:
                logger.warning(f"[实时行情] A股实时行情数据为空，跳过 {stock_code}")
                return None
            
            # 查找指定股票
            row = df[df['代码'] == stock_code]
            if row.empty:
                logger.warning(f"[API返回] 未找到股票 {stock_code} 的实时行情")
                return None
            
            row = row.iloc[0]
            
            # 使用 types.py 中的统一转换函数
            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=str(row.get('名称', '')),
                source=RealtimeSource.AKSHARE_EM,
                price=safe_float(row.get('最新价')),
                change_pct=safe_float(row.get('涨跌幅')),
                change_amount=safe_float(row.get('涨跌额')),
                volume=safe_int(row.get('成交量')),
                amount=safe_float(row.get('成交额')),
                volume_ratio=safe_float(row.get('量比')),
                turnover_rate=safe_float(row.get('换手率')),
                amplitude=safe_float(row.get('振幅')),
                open_price=safe_float(row.get('今开')),
                high=safe_float(row.get('最高')),
                low=safe_float(row.get('最低')),
                pe_ratio=safe_float(row.get('市盈率-动态')),
                pb_ratio=safe_float(row.get('市净率')),
                total_mv=safe_float(row.get('总市值')),
                circ_mv=safe_float(row.get('流通市值')),
                change_60d=safe_float(row.get('60日涨跌幅')),
                high_52w=safe_float(row.get('52周最高')),
                low_52w=safe_float(row.get('52周最低')),
            )
            
            logger.info(f"[实时行情-东财] {stock_code} {quote.name}: 价格={quote.price}, 涨跌={quote.change_pct}%, "
                       f"量比={quote.volume_ratio}, 换手率={quote.turnover_rate}%")
            return quote
            
        except Exception as e:
            logger.error(f"[API错误] 获取 {stock_code} 实时行情(东财)失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None
    
    def _get_stock_realtime_quote_sina(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取普通 A 股实时行情数据（新浪财经数据源）
        
        数据来源：新浪财经接口（直连，单股票查询）
        优点：单股票查询，负载小，速度快
        缺点：数据字段较少，无量比/PE/PB等
        
        接口格式：http://hq.sinajs.cn/list=sh600519,sz000001
        """
        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "akshare_sina"
        symbol = _to_sina_tx_symbol(stock_code)
        url = f"http://{SINA_REALTIME_ENDPOINT}={symbol}"
        api_start = time.time()
        
        try:
            headers = {
                'Referer': 'http://finance.sina.com.cn',
                'User-Agent': random.choice(USER_AGENTS)
            }
            
            logger.info(
                f"[API调用] 新浪财经接口获取 {stock_code} 实时行情: endpoint={SINA_REALTIME_ENDPOINT}, symbol={symbol}"
            )
            
            self._enforce_rate_limit()
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'gbk'
            api_elapsed = time.time() - api_start
            
            if response.status_code != 200:
                failure_message = _build_realtime_failure_message(
                    source_name="新浪",
                    endpoint=SINA_REALTIME_ENDPOINT,
                    stock_code=stock_code,
                    symbol=symbol,
                    category="http_status",
                    detail=f"HTTP {response.status_code}",
                    elapsed=api_elapsed,
                    error_type="HTTPStatus",
                )
                logger.warning(failure_message)
                circuit_breaker.record_failure(source_key, failure_message)
                return None
            
            # 解析数据：var hq_str_sh600519="贵州茅台,1866.000,1870.000,..."
            content = response.text.strip()
            if '=""' in content or not content:
                failure_message = _build_realtime_failure_message(
                    source_name="新浪",
                    endpoint=SINA_REALTIME_ENDPOINT,
                    stock_code=stock_code,
                    symbol=symbol,
                    category="empty_response",
                    detail="empty quote payload",
                    elapsed=api_elapsed,
                    error_type="EmptyResponse",
                )
                logger.warning(failure_message)
                circuit_breaker.record_failure(source_key, failure_message)
                return None
            
            # 提取引号内的数据
            data_start = content.find('"')
            data_end = content.rfind('"')
            if data_start == -1 or data_end == -1:
                failure_message = _build_realtime_failure_message(
                    source_name="新浪",
                    endpoint=SINA_REALTIME_ENDPOINT,
                    stock_code=stock_code,
                    symbol=symbol,
                    category="malformed_payload",
                    detail="quote payload missing quotes",
                    elapsed=api_elapsed,
                    error_type="MalformedPayload",
                )
                logger.warning(failure_message)
                circuit_breaker.record_failure(source_key, failure_message)
                return None
            
            data_str = content[data_start+1:data_end]
            fields = data_str.split(',')
            
            if len(fields) < 32:
                failure_message = _build_realtime_failure_message(
                    source_name="新浪",
                    endpoint=SINA_REALTIME_ENDPOINT,
                    stock_code=stock_code,
                    symbol=symbol,
                    category="insufficient_fields",
                    detail=f"field_count={len(fields)}",
                    elapsed=api_elapsed,
                    error_type="InsufficientFields",
                )
                logger.warning(failure_message)
                circuit_breaker.record_failure(source_key, failure_message)
                return None
            
            circuit_breaker.record_success(source_key)
            
            # 新浪数据字段顺序：
            # 0:名称 1:今开 2:昨收 3:最新价 4:最高 5:最低 6:买一价 7:卖一价
            # 8:成交量(股) 9:成交额(元) ... 30:日期 31:时间
            # 使用 types.py 中的统一转换函数
            price = safe_float(fields[3])
            pre_close = safe_float(fields[2])
            change_pct = None
            change_amount = None
            if price and pre_close and pre_close > 0:
                change_amount = price - pre_close
                change_pct = (change_amount / pre_close) * 100
            
            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=fields[0],
                source=RealtimeSource.AKSHARE_SINA,
                price=price,
                change_pct=change_pct,
                change_amount=change_amount,
                volume=safe_int(fields[8]),  # 成交量（股）
                amount=safe_float(fields[9]),  # 成交额（元）
                open_price=safe_float(fields[1]),
                high=safe_float(fields[4]),
                low=safe_float(fields[5]),
                pre_close=pre_close,
            )
            
            logger.info(
                f"[实时行情-新浪] {stock_code} {quote.name}: endpoint={SINA_REALTIME_ENDPOINT}, "
                f"价格={quote.price}, 涨跌={quote.change_pct}, 成交量={quote.volume}, elapsed={api_elapsed:.2f}s"
            )
            return quote
            
        except Exception as e:
            api_elapsed = time.time() - api_start
            category, detail = _classify_realtime_http_error(e)
            failure_message = _build_realtime_failure_message(
                source_name="新浪",
                endpoint=SINA_REALTIME_ENDPOINT,
                stock_code=stock_code,
                symbol=symbol,
                category=category,
                detail=detail,
                elapsed=api_elapsed,
                error_type=type(e).__name__,
            )
            logger.error(failure_message)
            circuit_breaker.record_failure(source_key, failure_message)
            return None
    
    def _get_stock_realtime_quote_tencent(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取普通 A 股实时行情数据（腾讯财经数据源）
        
        数据来源：腾讯财经接口（直连，单股票查询）
        优点：单股票查询，负载小，包含换手率
        缺点：无量比/PE/PB等估值数据
        
        接口格式：http://qt.gtimg.cn/q=sh600519,sz000001
        """
        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "akshare_tencent"
        symbol = _to_sina_tx_symbol(stock_code)
        url = f"http://{TENCENT_REALTIME_ENDPOINT}={symbol}"
        api_start = time.time()
        
        try:
            headers = {
                'Referer': 'http://finance.qq.com',
                'User-Agent': random.choice(USER_AGENTS)
            }
            
            logger.info(
                f"[API调用] 腾讯财经接口获取 {stock_code} 实时行情: endpoint={TENCENT_REALTIME_ENDPOINT}, symbol={symbol}"
            )
            
            self._enforce_rate_limit()
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'gbk'
            api_elapsed = time.time() - api_start
            
            if response.status_code != 200:
                failure_message = _build_realtime_failure_message(
                    source_name="腾讯",
                    endpoint=TENCENT_REALTIME_ENDPOINT,
                    stock_code=stock_code,
                    symbol=symbol,
                    category="http_status",
                    detail=f"HTTP {response.status_code}",
                    elapsed=api_elapsed,
                    error_type="HTTPStatus",
                )
                logger.warning(failure_message)
                circuit_breaker.record_failure(source_key, failure_message)
                return None
            
            content = response.text.strip()
            if '=""' in content or not content:
                failure_message = _build_realtime_failure_message(
                    source_name="腾讯",
                    endpoint=TENCENT_REALTIME_ENDPOINT,
                    stock_code=stock_code,
                    symbol=symbol,
                    category="empty_response",
                    detail="empty quote payload",
                    elapsed=api_elapsed,
                    error_type="EmptyResponse",
                )
                logger.warning(failure_message)
                circuit_breaker.record_failure(source_key, failure_message)
                return None
            
            # 提取数据
            data_start = content.find('"')
            data_end = content.rfind('"')
            if data_start == -1 or data_end == -1:
                failure_message = _build_realtime_failure_message(
                    source_name="腾讯",
                    endpoint=TENCENT_REALTIME_ENDPOINT,
                    stock_code=stock_code,
                    symbol=symbol,
                    category="malformed_payload",
                    detail="quote payload missing quotes",
                    elapsed=api_elapsed,
                    error_type="MalformedPayload",
                )
                logger.warning(failure_message)
                circuit_breaker.record_failure(source_key, failure_message)
                return None
            
            data_str = content[data_start+1:data_end]
            fields = data_str.split('~')

            if len(fields) < 45:
                failure_message = _build_realtime_failure_message(
                    source_name="腾讯",
                    endpoint=TENCENT_REALTIME_ENDPOINT,
                    stock_code=stock_code,
                    symbol=symbol,
                    category="insufficient_fields",
                    detail=f"field_count={len(fields)}",
                    elapsed=api_elapsed,
                    error_type="InsufficientFields",
                )
                logger.warning(failure_message)
                circuit_breaker.record_failure(source_key, failure_message)
                return None
            
            circuit_breaker.record_success(source_key)
            
            # 腾讯数据字段顺序（完整）：
            # 1:名称 2:代码 3:最新价 4:昨收 5:今开 6:成交量(手) 7:外盘 8:内盘
            # 9-28:买卖五档 30:时间戳 31:涨跌额 32:涨跌幅(%) 33:最高 34:最低 35:收盘/成交量/成交额
            # 36:成交量(手) 37:成交额(万) 38:换手率(%) 39:市盈率 43:振幅(%)
            # 44:流通市值(亿) 45:总市值(亿) 46:市净率 47:涨停价 48:跌停价 49:量比
            # 使用 types.py 中的统一转换函数
            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=fields[1] if len(fields) > 1 else "",
                source=RealtimeSource.TENCENT,
                price=safe_float(fields[3]),
                change_pct=safe_float(fields[32]),
                change_amount=safe_float(fields[31]) if len(fields) > 31 else None,
                volume=safe_int(fields[6]) * 100 if fields[6] else None,  # 腾讯返回的是手，转为股
                open_price=safe_float(fields[5]),
                high=safe_float(fields[33]) if len(fields) > 33 else None,  # 修正：字段 33 是最高价
                low=safe_float(fields[34]) if len(fields) > 34 else None,  # 修正：字段 34 是最低价
                pre_close=safe_float(fields[4]),
                turnover_rate=safe_float(fields[38]) if len(fields) > 38 else None,
                amplitude=safe_float(fields[43]) if len(fields) > 43 else None,
                volume_ratio=safe_float(fields[49]) if len(fields) > 49 else None,  # 量比
                pe_ratio=safe_float(fields[39]) if len(fields) > 39 else None,  # 市盈率
                pb_ratio=safe_float(fields[46]) if len(fields) > 46 else None,  # 市净率
                circ_mv=safe_float(fields[44]) * 100000000 if len(fields) > 44 and fields[44] else None,  # 流通市值(亿->元)
                total_mv=safe_float(fields[45]) * 100000000 if len(fields) > 45 and fields[45] else None,  # 总市值(亿->元)
            )
            
            logger.info(
                f"[实时行情-腾讯] {stock_code} {quote.name}: endpoint={TENCENT_REALTIME_ENDPOINT}, "
                f"价格={quote.price}, 涨跌={quote.change_pct}%, 量比={quote.volume_ratio}, "
                f"换手率={quote.turnover_rate}%, elapsed={api_elapsed:.2f}s"
            )
            return quote
            
        except Exception as e:
            api_elapsed = time.time() - api_start
            category, detail = _classify_realtime_http_error(e)
            failure_message = _build_realtime_failure_message(
                source_name="腾讯",
                endpoint=TENCENT_REALTIME_ENDPOINT,
                stock_code=stock_code,
                symbol=symbol,
                category=category,
                detail=detail,
                elapsed=api_elapsed,
                error_type=type(e).__name__,
            )
            logger.error(failure_message)
            circuit_breaker.record_failure(source_key, failure_message)
            return None
    
    def _get_etf_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取 ETF 基金实时行情数据
        
        数据来源：ak.fund_etf_spot_em()
        包含：最新价、涨跌幅、成交量、成交额、换手率等
        
        Args:
            stock_code: ETF 代码
            
        Returns:
            UnifiedRealtimeQuote 对象，获取失败返回 None
        """
        import akshare as ak
        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "akshare_etf"
        
        try:
            # 检查缓存
            current_time = time.time()
            if (_etf_realtime_cache['data'] is not None and 
                current_time - _etf_realtime_cache['timestamp'] < _etf_realtime_cache['ttl']):
                df = _etf_realtime_cache['data']
                logger.debug(f"[缓存命中] 使用缓存的ETF实时行情数据")
            else:
                last_error: Optional[Exception] = None
                df = None
                for attempt in range(1, 3):
                    try:
                        # 防封禁策略
                        self._set_random_user_agent()
                        self._enforce_rate_limit()

                        logger.info(f"[API调用] ak.fund_etf_spot_em() 获取ETF实时行情... (attempt {attempt}/2)")
                        import time as _time
                        api_start = _time.time()

                        df = ak.fund_etf_spot_em()

                        api_elapsed = _time.time() - api_start
                        logger.info(f"[API返回] ak.fund_etf_spot_em 成功: 返回 {len(df)} 只ETF, 耗时 {api_elapsed:.2f}s")
                        circuit_breaker.record_success(source_key)
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(f"[API错误] ak.fund_etf_spot_em 获取失败 (attempt {attempt}/2): {e}")
                        time.sleep(min(2 ** attempt, 5))

                if df is None:
                    logger.error(f"[API错误] ak.fund_etf_spot_em 最终失败: {last_error}")
                    circuit_breaker.record_failure(source_key, str(last_error))
                    df = pd.DataFrame()
                _etf_realtime_cache['data'] = df
                _etf_realtime_cache['timestamp'] = current_time

            if df is None or df.empty:
                logger.warning(f"[实时行情] ETF实时行情数据为空，跳过 {stock_code}")
                return None
            
            # 查找指定 ETF
            row = df[df['代码'] == stock_code]
            if row.empty:
                logger.warning(f"[API返回] 未找到 ETF {stock_code} 的实时行情")
                return None
            
            row = row.iloc[0]
            
            # 使用 types.py 中的统一转换函数
            # ETF 行情数据构建
            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=str(row.get('名称', '')),
                source=RealtimeSource.AKSHARE_EM,
                price=safe_float(row.get('最新价')),
                change_pct=safe_float(row.get('涨跌幅')),
                change_amount=safe_float(row.get('涨跌额')),
                volume=safe_int(row.get('成交量')),
                amount=safe_float(row.get('成交额')),
                volume_ratio=safe_float(row.get('量比')),
                turnover_rate=safe_float(row.get('换手率')),
                amplitude=safe_float(row.get('振幅')),
                open_price=safe_float(row.get('开盘价')),
                high=safe_float(row.get('最高价')),
                low=safe_float(row.get('最低价')),
                total_mv=safe_float(row.get('总市值')),
                circ_mv=safe_float(row.get('流通市值')),
                high_52w=safe_float(row.get('52周最高')),
                low_52w=safe_float(row.get('52周最低')),
            )
            
            logger.info(f"[ETF实时行情] {stock_code} {quote.name}: 价格={quote.price}, 涨跌={quote.change_pct}%, "
                       f"换手率={quote.turnover_rate}%")
            return quote
            
        except Exception as e:
            logger.error(f"[API错误] 获取 ETF {stock_code} 实时行情失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None
    
    def _get_hk_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取港股实时行情数据
        
        数据来源：ak.stock_hk_spot_em()
        包含：最新价、涨跌幅、成交量、成交额等
        
        Args:
            stock_code: 港股代码
            
        Returns:
            UnifiedRealtimeQuote 对象，获取失败返回 None
        """
        import akshare as ak
        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "akshare_hk"

        if not circuit_breaker.is_available(source_key):
            logger.warning(f"[熔断] 数据源 {source_key} 处于熔断状态，跳过")
            return None
        
        try:
            # 防封禁策略
            self._set_random_user_agent()
            self._enforce_rate_limit()
            
            # 确保代码格式正确（5位数字）
            raw_code = stock_code.strip().lower()
            if raw_code.endswith('.hk'):
                raw_code = raw_code[:-3]
            if raw_code.startswith('hk'):
                raw_code = raw_code[2:]
            code = raw_code.zfill(5)
            
            logger.info(f"[API调用] ak.stock_hk_spot_em() 获取港股实时行情...")
            import time as _time
            api_start = _time.time()
            
            df = ak.stock_hk_spot_em()
            
            api_elapsed = _time.time() - api_start
            logger.info(f"[API返回] ak.stock_hk_spot_em 成功: 返回 {len(df)} 只港股, 耗时 {api_elapsed:.2f}s")
            circuit_breaker.record_success(source_key)
            
            # 查找指定港股
            row = df[df['代码'] == code]
            if row.empty:
                logger.warning(f"[API返回] 未找到港股 {code} 的实时行情")
                return None
            
            row = row.iloc[0]
            
            # 使用 types.py 中的统一转换函数
            # 港股行情数据构建
            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=str(row.get('名称', '')),
                source=RealtimeSource.AKSHARE_EM,
                price=safe_float(row.get('最新价')),
                change_pct=safe_float(row.get('涨跌幅')),
                change_amount=safe_float(row.get('涨跌额')),
                volume=safe_int(row.get('成交量')),
                amount=safe_float(row.get('成交额')),
                volume_ratio=safe_float(row.get('量比')),
                turnover_rate=safe_float(row.get('换手率')),
                amplitude=safe_float(row.get('振幅')),
                pe_ratio=safe_float(row.get('市盈率')),
                pb_ratio=safe_float(row.get('市净率')),
                total_mv=safe_float(row.get('总市值')),
                circ_mv=safe_float(row.get('流通市值')),
                high_52w=safe_float(row.get('52周最高')),
                low_52w=safe_float(row.get('52周最低')),
            )
            
            logger.info(f"[港股实时行情] {stock_code} {quote.name}: 价格={quote.price}, 涨跌={quote.change_pct}%, "
                       f"换手率={quote.turnover_rate}%")
            return quote
            
        except Exception as e:
            logger.error(f"[API错误] 获取港股 {stock_code} 实时行情失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None

    def get_market_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取市场涨跌统计

        数据源优先级：
        1. 东财接口 (ak.stock_zh_a_spot_em)
        2. 新浪接口 (ak.stock_zh_a_spot)
        """
        import akshare as ak

        # 优先东财接口
        try:
            self._set_random_user_agent()
            self._enforce_rate_limit()

            logger.info("[API调用] ak.stock_zh_a_spot_em() 获取市场统计...")
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                return calc_market_stats(df)
        except Exception as e:
            logger.warning(f"[Akshare] 东财接口获取市场统计失败: {e}，尝试新浪接口")

        # 东财失败后，尝试新浪接口
        try:
            self._set_random_user_agent()
            self._enforce_rate_limit()

            logger.info("[API调用] ak.stock_zh_a_spot() 获取市场统计(新浪)...")
            df = ak.stock_zh_a_spot()
            if df is not None and not df.empty:
                return calc_market_stats(df)
        except Exception as e:
            logger.error(f"[Akshare] 新浪接口获取市场统计也失败: {e}")

        return None


    def get_sector_pct_map(self) -> Dict[str, float]:
        """获取全量行业板块当日涨跌幅 {板块名: 涨跌幅%}。

        数据源优先级：
        1. 新浪接口 (ak.stock_sector_spot) —— 主机稳定，默认优先
        2. 东财接口 (ak.stock_board_industry_name_em) —— 板块名与个股所属板块同口径
        全部失败返回空字典，板块类规则由调用方跳过。
        """
        import akshare as ak

        def _to_map(df: pd.DataFrame, name_col: str, pct_col: str) -> Dict[str, float]:
            if df is None or df.empty or name_col not in df.columns or pct_col not in df.columns:
                return {}
            result: Dict[str, float] = {}
            for _, row in df.iterrows():
                pct = pd.to_numeric(row[pct_col], errors="coerce")
                if pd.notna(pct):
                    result[str(row[name_col])] = float(pct)
            return result

        try:
            self._set_random_user_agent()
            self._enforce_rate_limit()

            logger.info("[API调用] ak.stock_sector_spot() 获取全量板块行情(新浪)...")
            df = ak.stock_sector_spot(indicator='行业')
            result = _to_map(df, '板块', '涨跌幅')
            if result:
                return result

        except Exception as e:
            logger.warning(f"[Akshare] 新浪接口获取板块行情失败: {e}，尝试东财接口")

        try:
            self._set_random_user_agent()
            self._enforce_rate_limit()

            logger.info("[API调用] ak.stock_board_industry_name_em() 获取全量板块行情...")
            df = ak.stock_board_industry_name_em()
            return _to_map(df, '板块名称', '涨跌幅')

        except Exception as e:
            logger.error(f"[Akshare] 东财接口获取板块行情也失败: {e}")
            return {}
    def get_main_fund_flow(self, stock_code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """
        获取主力资金流向数据

        Args:
            stock_code: 股票代码（如 '002357'）
            days: 获取天数（默认 5 个交易日）

        Returns:
            DataFrame 包含以下列：
            - date: 日期
            - main_net_inflow: 主力净流入（元，正数=净流入，负数=净流出）
            - 或 None（数据源不支持或获取失败）
        """
        import akshare as ak

        # 解析市场标识：akshare 内部 market_map = {sh:1, sz:0, bj:0}，
        # 深市与北交所最终是同一个 secid（0.xxxxxx），故 6 开头的走 sh、其余走 sz 即覆盖全市场。
        market = "sh" if stock_code.startswith("6") else "sz"

        try:
            self._set_random_user_agent()
            self._enforce_rate_limit()

            logger.info(f"[API调用] ak.stock_individual_fund_flow() 获取 {stock_code} 主力资金流...")
            df = ak.stock_individual_fund_flow(stock=stock_code, market=market)

            if df is None or df.empty:
                logger.warning(f"[Akshare] {stock_code} 主力资金流数据为空")
                return None

            # 标准化列名
            # 注意：东财接口同时返回「主力净流入-净额」与「主力净流入-净占比」，
            # 两者都含"主力/净流入"关键词，必须排除占比列，否则会重名成两列
            # main_net_inflow，调用方取列得到 DataFrame 而非 Series。
            column_mapping = {}
            for col in df.columns:
                col_str = str(col).strip()
                if '日期' in col_str or 'date' in col_str.lower():
                    column_mapping[col] = 'date'
                elif '主力' in col_str and '净流入' in col_str and '占比' not in col_str:
                    column_mapping[col] = 'main_net_inflow'

            df = df.rename(columns=column_mapping)

            # 确保必需的列存在
            if 'date' not in df.columns or 'main_net_inflow' not in df.columns:
                logger.warning(f"[Akshare] {stock_code} 主力资金流数据格式异常: {df.columns.tolist()}")
                return None

            # 转换数据类型
            df['date'] = pd.to_datetime(df['date'])
            df['main_net_inflow'] = pd.to_numeric(df['main_net_inflow'], errors='coerce')

            # 单位：东财 fflow/daykline 的 f52 原始即为「元」，akshare 原样透传，不做换算

            # 取最近 N 天，按日期升序返回
            df = df.sort_values('date', ascending=False).head(days).sort_values('date').reset_index(drop=True)

            logger.info(f"[Akshare] {stock_code} 获取主力资金流成功: {len(df)} 天")
            return df[['date', 'main_net_inflow']]

        except Exception as e:
            logger.error(f"[Akshare] {stock_code} 获取主力资金流失败: {e}")
            return None
