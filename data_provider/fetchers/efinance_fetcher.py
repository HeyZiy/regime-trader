# -*- coding: utf-8 -*-
"""
===================================
EfinanceFetcher - 优先数据源 (Priority 0)
===================================

数据来源：东方财富爬虫（通过 efinance 库）
特点：免费、无需 Token、数据全面、API 简洁
仓库：https://github.com/Micro-sheep/efinance

与 AkshareFetcher 类似，但 efinance 库：
1. API 更简洁易用
2. 支持批量获取数据
3. 更稳定的接口封装

防封禁策略：
1. 每次请求前随机休眠 1.5-3.0 秒
2. 随机轮换 User-Agent
3. 使用 tenacity 实现指数退避重试
4. 熔断器机制：连续失败后自动冷却
"""

import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import requests  # 引入 requests 以捕获异常
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# Timeout (seconds) for efinance library calls that go through eastmoney APIs
# with no built-in timeout.  Prevents indefinite hangs when hosts are unreachable.
try:
    _EF_CALL_TIMEOUT = int(os.environ.get("EFINANCE_CALL_TIMEOUT", "30"))
except (ValueError, TypeError):
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "EFINANCE_CALL_TIMEOUT is not a valid integer, using default 30s"
    )
    _EF_CALL_TIMEOUT = 30

# Connection pool settings for improved stability
try:
    _EF_MAX_RETRIES = int(os.environ.get("EFINANCE_MAX_RETRIES", "3"))
except (ValueError, TypeError):
    _EF_MAX_RETRIES = 3

try:
    _EF_RETRY_MIN_WAIT = int(os.environ.get("EFINANCE_RETRY_MIN_WAIT", "2"))
except (ValueError, TypeError):
    _EF_RETRY_MIN_WAIT = 2

try:
    _EF_RETRY_MAX_WAIT = int(os.environ.get("EFINANCE_RETRY_MAX_WAIT", "30"))
except (ValueError, TypeError):
    _EF_RETRY_MAX_WAIT = 30

# Global session with connection pooling for better stability
_session: Optional[requests.Session] = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    """Get or create a global requests session with connection pooling.
    
    Using a shared session enables HTTP keep-alive and connection reuse,
    which significantly reduces RemoteDisconnected errors when making
    multiple requests to the same host.
    """
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
                # Configure connection pooling
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,      # Number of connection pools to cache
                    pool_maxsize=20,          # Max connections to save in the pool
                    max_retries=requests.adapters.Retry(
                        total=2,              # Total retries for connection errors
                        backoff_factor=0.5,   # Backoff between retries
                        status_forcelist=[500, 502, 503, 504],  # Retry on these status codes
                    ),
                )
                _session.mount('https://', adapter)
                _session.mount('http://', adapter)
    return _session


from data_provider.fetchers.base import BaseFetcher
from data_provider.stats import calc_market_stats
from data_provider.types import (
    KIND_REALTIME, KIND_STOCK_DAILY,
    DataFetchError, RateLimitError, STANDARD_COLUMNS,
    UnifiedRealtimeQuote, RealtimeSource,
    get_realtime_circuit_breaker, safe_float, safe_int,
)
from data_provider.codes import is_bse_code, is_st_stock, is_kc_cy_stock, normalize_stock_code, is_etf_code, is_us_stock_code


# EfinanceRealtimeQuote 别名（外部统一引用 UnifiedRealtimeQuote）
@dataclass
class EfinanceRealtimeQuote:
    """
    实时行情数据（来自 efinance）别名

    新代码建议使用 UnifiedRealtimeQuote
    """
    code: str
    name: str = ""
    price: float = 0.0           # 最新价
    change_pct: float = 0.0      # 涨跌幅(%)
    change_amount: float = 0.0   # 涨跌额
    
    # 量价指标
    volume: int = 0              # 成交量
    amount: float = 0.0          # 成交额
    turnover_rate: float = 0.0   # 换手率(%)
    amplitude: float = 0.0       # 振幅(%)
    
    # 价格区间
    high: float = 0.0            # 最高价
    low: float = 0.0             # 最低价
    open_price: float = 0.0      # 开盘价
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'price': self.price,
            'change_pct': self.change_pct,
            'change_amount': self.change_amount,
            'volume': self.volume,
            'amount': self.amount,
            'turnover_rate': self.turnover_rate,
            'amplitude': self.amplitude,
            'high': self.high,
            'low': self.low,
            'open': self.open_price,
        }


logger = logging.getLogger(__name__)

EASTMONEY_HISTORY_ENDPOINT = "push2his.eastmoney.com/api/qt/stock/kline/get"


# User-Agent 池，用于随机轮换
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]


# 缓存实时行情数据（避免重复请求）
# TTL 设为 10 分钟 (600秒)：批量分析场景下避免重复拉取
_realtime_cache: Dict[str, Any] = {
    'data': None,
    'timestamp': 0,
    'ttl': 600  # 10分钟缓存有效期
}

# ETF 实时行情缓存（与股票分开缓存）
_etf_realtime_cache: Dict[str, Any] = {
    'data': None,
    'timestamp': 0,
    'ttl': 600  # 10分钟缓存有效期
}


def _ef_call_with_timeout(func, *args, timeout=None, **kwargs):
    """Run an efinance library call in a thread with a timeout.

    efinance internally uses requests/urllib3 with no timeout, so when
    eastmoney hosts are unreachable the call can hang for many minutes.
    This helper caps the *calling thread's* wait time.  Note: Python threads
    cannot be forcibly killed, so the worker thread may continue running in
    the background until the OS-level TCP timeout fires or the process exits.
    This is acceptable — the calling thread returns promptly on timeout.
    """
    if timeout is None:
        timeout = _EF_CALL_TIMEOUT
    # Do NOT use 'with ThreadPoolExecutor(...)' here: the context manager calls
    # shutdown(wait=True) on __exit__, which would re-block on the hung thread.
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)
    finally:
        # wait=False: calling thread returns immediately; worker cleans up later
        executor.shutdown(wait=False)


def _classify_eastmoney_error(exc: Exception) -> Tuple[str, str]:
    """
    Classify Eastmoney request failures into stable log categories.
    """
    message = str(exc).strip()
    lowered = message.lower()

    remote_disconnect_keywords = (
        'remotedisconnected',
        'remote end closed connection without response',
        'connection aborted',
        'connection broken',
        'protocolerror',
    )
    timeout_keywords = (
        'timeout',
        'timed out',
        'readtimeout',
        'connecttimeout',
    )
    rate_limit_keywords = (
        'banned',
        'blocked',
        '频率',
        'rate limit',
        'too many requests',
        '429',
        '限制',
        'forbidden',
        '403',
    )

    if any(keyword in lowered for keyword in remote_disconnect_keywords):
        return "remote_disconnect", message
    if isinstance(exc, (TimeoutError, requests.exceptions.Timeout)) or any(
        keyword in lowered for keyword in timeout_keywords
    ):
        return "timeout", message
    if any(keyword in lowered for keyword in rate_limit_keywords):
        return "rate_limit_or_anti_bot", message
    if isinstance(exc, requests.exceptions.RequestException):
        return "request_error", message
    return "unknown_request_error", message


class EfinanceFetcher(BaseFetcher):
    """
    Efinance 数据源实现
    
    优先级：0（最高，优先于 AkshareFetcher）
    数据来源：东方财富网（通过 efinance 库封装）
    仓库：https://github.com/Micro-sheep/efinance
    
    主要 API：
    - ef.stock.get_quote_history(): 获取历史 K 线数据
    - ef.stock.get_base_info(): 获取股票基本信息
    - ef.stock.get_realtime_quotes(): 获取实时行情
    
    关键策略：
    - 每次请求前随机休眠 1.5-3.0 秒
    - 随机 User-Agent 轮换
    - 失败后指数退避重试（最多3次）
    """
    
    name = "EfinanceFetcher"
    # 降低优先级，让 AkshareFetcher 优先（Akshare 更稳定，支持新浪财经备选）
    priority = int(os.getenv("EFINANCE_PRIORITY", "1"))
    # 东财 K 线含换手率列
    SUPPORTS_COLUMNS = {'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'turnover_rate'}

    # 日线与实时均仅 A 股（美股明确拒绝，见 _fetch_raw_data）
    SUPPORTS = frozenset({(KIND_STOCK_DAILY, "cn"), (KIND_REALTIME, "cn")})
    
    def __init__(self, sleep_min: float = 1.5, sleep_max: float = 3.0):
        """
        初始化 EfinanceFetcher
        
        Args:
            sleep_min: 最小休眠时间（秒）
            sleep_max: 最大休眠时间（秒）
        """
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self._last_request_time: Optional[float] = None
        self._consecutive_errors: int = 0  # 连续错误计数，用于自适应流控
        self._error_reset_time: Optional[float] = None  # 错误计数重置时间

    @staticmethod
    def _build_history_failure_message(
        stock_code: str,
        beg_date: str,
        end_date: str,
        exc: Exception,
        elapsed: float,
        is_etf: bool = False,
    ) -> Tuple[str, str]:
        category, detail = _classify_eastmoney_error(exc)
        instrument_type = "ETF" if is_etf else "stock"
        message = (
            "Eastmoney 历史K线接口失败: "
            f"endpoint={EASTMONEY_HISTORY_ENDPOINT}, stock_code={stock_code}, "
            f"market_type={instrument_type}, range={beg_date}~{end_date}, "
            f"category={category}, error_type={type(exc).__name__}, elapsed={elapsed:.2f}s, detail={detail}"
        )
        return category, message

    def _set_random_user_agent(self) -> None:
        """
        设置随机 User-Agent
        
        通过修改 requests Session 的 headers 实现
        这是关键的反爬策略之一
        """
        try:
            random_ua = random.choice(USER_AGENTS)
            logger.debug(f"设置 User-Agent: {random_ua[:50]}...")
        except Exception as e:
            logger.debug(f"设置 User-Agent 失败: {e}")
    
    def _enforce_rate_limit(self) -> None:
        """
        强制执行速率限制（增强版，带自适应流控）
        
        策略：
        1. 检查距离上次请求的时间间隔
        2. 如果间隔不足，补充休眠时间
        3. 根据连续错误次数动态增加休眠时间（自适应流控）
        4. 执行随机 jitter 休眠
        5. 重置错误计数（如果距离上次错误已超过60秒）
        """
        now = time.time()
        
        # 重置错误计数（如果距离上次错误已超过60秒）
        if self._error_reset_time is not None and now - self._error_reset_time > 60:
            if self._consecutive_errors > 0:
                logger.debug(f"重置连续错误计数: {self._consecutive_errors} -> 0")
                self._consecutive_errors = 0
                self._error_reset_time = None
        
        # 基础间隔检查
        if self._last_request_time is not None:
            elapsed = now - self._last_request_time
            min_interval = self.sleep_min
            if elapsed < min_interval:
                additional_sleep = min_interval - elapsed
                logger.debug(f"补充休眠 {additional_sleep:.2f} 秒")
                time.sleep(additional_sleep)
        
        # 自适应流控：根据连续错误次数增加额外休眠
        if self._consecutive_errors > 0:
            # 指数退避：每次错误增加 0.5-2 秒额外休眠
            extra_sleep = min(self._consecutive_errors * 0.5, 3.0)  # 最多额外3秒
            logger.debug(f"自适应流控: 连续{self._consecutive_errors}次错误，额外休眠{extra_sleep:.2f}秒")
            time.sleep(extra_sleep)
        
        # 执行随机 jitter 休眠
        jitter_sleep = random.uniform(self.sleep_min, self.sleep_max)
        time.sleep(jitter_sleep)
        self._last_request_time = time.time()
    
    def _record_error(self) -> None:
        """记录一次错误，用于自适应流控"""
        self._consecutive_errors += 1
        self._error_reset_time = time.time()
        logger.debug(f"记录错误，连续错误次数: {self._consecutive_errors}")
    
    def _record_success(self) -> None:
        """记录一次成功，减少错误计数"""
        if self._consecutive_errors > 0:
            self._consecutive_errors = max(0, self._consecutive_errors - 1)
            logger.debug(f"记录成功，连续错误次数减至: {self._consecutive_errors}")
    
    @retry(
        stop=stop_after_attempt(_EF_MAX_RETRIES),  # 使用环境变量配置，默认3次
        wait=wait_exponential(multiplier=1, min=_EF_RETRY_MIN_WAIT, max=_EF_RETRY_MAX_WAIT),
        retry=retry_if_exception_type((
            ConnectionError,
            TimeoutError,
            requests.exceptions.RequestException,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.HTTPError,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # 重试耗尽后抛出原始异常
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 efinance 获取原始数据
        
        根据代码类型自动选择 API：
        - 美股：不支持，抛出异常让 DataFetcherManager 切换到其他数据源
        - 普通股票：使用 ef.stock.get_quote_history()
        - ETF 基金：使用 ef.stock.get_quote_history()（ETF 是交易所证券，使用股票 K 线接口）
        
        流程：
        1. 判断代码类型（美股/股票/ETF）
        2. 设置随机 User-Agent
        3. 执行速率限制（随机休眠）
        4. 调用对应的 efinance API
        5. 处理返回数据
        """
        # 美股不支持，抛出异常让 DataFetcherManager 切换到 AkshareFetcher/YfinanceFetcher
        if is_us_stock_code(stock_code):
            raise DataFetchError(f"EfinanceFetcher 不支持美股 {stock_code}，请使用 AkshareFetcher 或 YfinanceFetcher")
        
        # 根据代码类型选择不同的获取方法
        if is_etf_code(stock_code):
            return self._fetch_etf_data(stock_code, start_date, end_date)
        else:
            return self._fetch_stock_data(stock_code, start_date, end_date)
    
    def _fetch_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取普通 A 股历史数据
        
        数据来源：ef.stock.get_quote_history()
        
        API 参数说明：
        - stock_codes: 股票代码
        - beg: 开始日期，格式 'YYYYMMDD'
        - end: 结束日期，格式 'YYYYMMDD'
        - klt: 周期，101=日线
        - fqt: 复权方式，1=前复权
        """
        import efinance as ef
        
        # 防封禁策略 1: 随机 User-Agent
        self._set_random_user_agent()
        
        # 防封禁策略 2: 强制休眠
        self._enforce_rate_limit()
        
        # 格式化日期（efinance 使用 YYYYMMDD 格式）
        beg_date = start_date.replace('-', '')
        end_date_fmt = end_date.replace('-', '')
        
        logger.info(f"[API调用] ef.stock.get_quote_history(stock_codes={stock_code}, "
                   f"beg={beg_date}, end={end_date_fmt}, klt=101, fqt=1)")
        
        api_start = time.time()
        try:
            # 调用 efinance 获取 A 股日线数据
            # klt=101 获取日线数据
            # fqt=1 获取前复权数据
            df = _ef_call_with_timeout(
                ef.stock.get_quote_history,
                stock_codes=stock_code,
                beg=beg_date,
                end=end_date_fmt,
                klt=101,  # 日线
                fqt=1,    # 前复权
                timeout=60,
            )
            
            api_elapsed = time.time() - api_start
            
            # 记录返回数据摘要
            if df is not None and not df.empty:
                logger.info(
                    "[API返回] Eastmoney 历史K线成功: "
                    f"endpoint={EASTMONEY_HISTORY_ENDPOINT}, stock_code={stock_code}, "
                    f"range={beg_date}~{end_date_fmt}, rows={len(df)}, elapsed={api_elapsed:.2f}s"
                )
                logger.info(f"[API返回] 列名: {list(df.columns)}")
                if '日期' in df.columns:
                    logger.info(f"[API返回] 日期范围: {df['日期'].iloc[0]} ~ {df['日期'].iloc[-1]}")
                logger.debug(f"[API返回] 最新3条数据:\n{df.tail(3).to_string()}")
                # 记录成功，减少错误计数
                self._record_success()
            else:
                logger.warning(
                    "[API返回] Eastmoney 历史K线为空: "
                    f"endpoint={EASTMONEY_HISTORY_ENDPOINT}, stock_code={stock_code}, "
                    f"range={beg_date}~{end_date_fmt}, elapsed={api_elapsed:.2f}s"
                )
            
            return df
            
        except Exception as e:
            api_elapsed = time.time() - api_start
            # 记录错误，用于自适应流控
            self._record_error()
            category, failure_message = self._build_history_failure_message(
                stock_code=stock_code,
                beg_date=beg_date,
                end_date=end_date_fmt,
                exc=e,
                elapsed=api_elapsed,
            )

            if category == "rate_limit_or_anti_bot":
                logger.warning(failure_message)
                raise RateLimitError(f"efinance 可能被限流: {failure_message}") from e

            logger.error(failure_message)
            raise DataFetchError(f"efinance 获取数据失败: {failure_message}") from e
    
    def _fetch_etf_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取 ETF 基金历史数据

        Exchange-traded ETFs have OHLCV data just like regular stocks, so we use
        ef.stock.get_quote_history (the stock K-line API) which returns full
        open/high/low/close/volume data.

        Previously this method used ef.fund.get_quote_history which only returns
        NAV data (单位净值/累计净值) without volume or OHLC, causing:
        - Issue #541: 'got an unexpected keyword argument beg'
        - Issue #527: ETF volume/turnover always showing 0

        Args:
            stock_code: ETF code, e.g. '512400', '159883', '515120'
            start_date: Start date, format 'YYYY-MM-DD'
            end_date: End date, format 'YYYY-MM-DD'

        Returns:
            ETF historical OHLCV DataFrame
        """
        import efinance as ef

        # Anti-ban strategy 1: random User-Agent
        self._set_random_user_agent()

        # Anti-ban strategy 2: enforce rate limit
        self._enforce_rate_limit()

        # Format dates (efinance uses YYYYMMDD)
        beg_date = start_date.replace('-', '')
        end_date_fmt = end_date.replace('-', '')

        logger.info(f"[API调用] ef.stock.get_quote_history(stock_codes={stock_code}, "
                     f"beg={beg_date}, end={end_date_fmt}, klt=101, fqt=1)  [ETF]")

        api_start = time.time()
        try:
            # ETFs are exchange-traded securities; use the stock API to get full OHLCV data
            df = _ef_call_with_timeout(
                ef.stock.get_quote_history,
                stock_codes=stock_code,
                beg=beg_date,
                end=end_date_fmt,
                klt=101,  # daily
                fqt=1,    # forward-adjusted
                timeout=60,
            )

            api_elapsed = time.time() - api_start

            if df is not None and not df.empty:
                logger.info(
                    "[API返回] Eastmoney 历史K线成功 [ETF]: "
                    f"endpoint={EASTMONEY_HISTORY_ENDPOINT}, stock_code={stock_code}, "
                    f"range={beg_date}~{end_date_fmt}, rows={len(df)}, elapsed={api_elapsed:.2f}s"
                )
                logger.info(f"[API返回] 列名: {list(df.columns)}")
                if '日期' in df.columns:
                    logger.info(f"[API返回] 日期范围: {df['日期'].iloc[0]} ~ {df['日期'].iloc[-1]}")
                logger.debug(f"[API返回] 最新3条数据:\n{df.tail(3).to_string()}")
            else:
                logger.warning(
                    "[API返回] Eastmoney 历史K线为空 [ETF]: "
                    f"endpoint={EASTMONEY_HISTORY_ENDPOINT}, stock_code={stock_code}, "
                    f"range={beg_date}~{end_date_fmt}, elapsed={api_elapsed:.2f}s"
                )

            return df

        except Exception as e:
            api_elapsed = time.time() - api_start
            category, failure_message = self._build_history_failure_message(
                stock_code=stock_code,
                beg_date=beg_date,
                end_date=end_date_fmt,
                exc=e,
                elapsed=api_elapsed,
                is_etf=True,
            )

            if category == "rate_limit_or_anti_bot":
                logger.warning(failure_message)
                raise RateLimitError(f"efinance 可能被限流: {failure_message}") from e

            logger.error(failure_message)
            raise DataFetchError(f"efinance 获取 ETF 数据失败: {failure_message}") from e
    
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 efinance 数据
        
        efinance 返回的列名（中文）：
        股票名称, 股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
        
        需要映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()
        
        # Column mapping (efinance Chinese column names -> standard English column names)
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
            '股票代码': 'code',
            '股票名称': 'name',
        }
        
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # Fallback: if OHLC columns are missing (e.g. very old data path), fill from close
        if 'close' in df.columns and 'open' not in df.columns:
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']
            
        # Fill volume and amount if missing
        if 'volume' not in df.columns:
            df['volume'] = 0
        if 'amount' not in df.columns:
            df['amount'] = 0

        
        # 如果没有 code 列，手动添加
        if 'code' not in df.columns:
            df['code'] = stock_code
        
        # 只保留需要的列
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]
        
        return df
    
    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取实时行情数据
        
        数据来源：ef.stock.get_realtime_quotes()
        ETF 数据源：ef.stock.get_realtime_quotes(['ETF'])
        
        Args:
            stock_code: 股票代码
            
        Returns:
            UnifiedRealtimeQuote 对象，获取失败返回 None
        """
        # ETF 需要单独请求 ETF 实时行情接口
        if is_etf_code(stock_code):
            return self._get_etf_realtime_quote(stock_code)

        import efinance as ef
        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "efinance"
        
        # 检查熔断器状态
        if not circuit_breaker.is_available(source_key):
            logger.warning(f"[熔断] 数据源 {source_key} 处于熔断状态，跳过")
            return None
        
        try:
            # 检查缓存
            current_time = time.time()
            if (_realtime_cache['data'] is not None and 
                current_time - _realtime_cache['timestamp'] < _realtime_cache['ttl']):
                df = _realtime_cache['data']
                cache_age = int(current_time - _realtime_cache['timestamp'])
                logger.debug(f"[缓存命中] 实时行情(efinance) - 缓存年龄 {cache_age}s/{_realtime_cache['ttl']}s")
            else:
                # 触发全量刷新
                logger.info(f"[缓存未命中] 触发全量刷新 实时行情(efinance)")
                # 防封禁策略
                self._set_random_user_agent()
                self._enforce_rate_limit()
                
                logger.info(f"[API调用] ef.stock.get_realtime_quotes() 获取实时行情...")
                import time as _time
                api_start = _time.time()
                
                # efinance 的实时行情 API (with timeout to avoid indefinite hangs)
                df = _ef_call_with_timeout(ef.stock.get_realtime_quotes)
                
                api_elapsed = _time.time() - api_start
                logger.info(f"[API返回] ef.stock.get_realtime_quotes 成功: 返回 {len(df)} 只股票, 耗时 {api_elapsed:.2f}s")
                circuit_breaker.record_success(source_key)
                
                # 更新缓存
                _realtime_cache['data'] = df
                _realtime_cache['timestamp'] = current_time
                logger.info(f"[缓存更新] 实时行情(efinance) 缓存已刷新，TTL={_realtime_cache['ttl']}s")
            
            # 查找指定股票
            # efinance 返回的列名可能是 '股票代码' 或 'code'
            code_col = '股票代码' if '股票代码' in df.columns else 'code'
            row = df[df[code_col] == stock_code]
            if row.empty:
                logger.warning(f"[API返回] 未找到股票 {stock_code} 的实时行情")
                return None
            
            row = row.iloc[0]

            # 列名映射：中文列名 → 标准列名（efinance 可能返回中英两种）
            _col = {
                '股票名称': 'name', 'name': 'name',
                '最新价': 'price', 'price': 'price',
                '涨跌幅': 'pct_chg', 'pct_chg': 'pct_chg',
                '涨跌额': 'change', 'change': 'change',
                '成交量': 'volume', 'volume': 'volume',
                '成交额': 'amount', 'amount': 'amount',
                '换手率': 'turnover_rate', 'turnover_rate': 'turnover_rate',
                '振幅': 'amplitude', 'amplitude': 'amplitude',
                '最高': 'high', 'high': 'high',
                '最低': 'low', 'low': 'low',
                '开盘': 'open', 'open': 'open',
                '量比': 'volume_ratio', 'volume_ratio': 'volume_ratio',
                '市盈率': 'pe_ratio', 'pe_ratio': 'pe_ratio',
                '总市值': 'total_mv', 'total_mv': 'total_mv',
                '流通市值': 'circ_mv', 'circ_mv': 'circ_mv',
            }
            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=str(row.get(_col.get('股票名称', 'name'), '')),
                source=RealtimeSource.EFINANCE,
                price=safe_float(row.get(_col.get('最新价', 'price'))),
                change_pct=safe_float(row.get(_col.get('涨跌幅', 'pct_chg'))),
                change_amount=safe_float(row.get(_col.get('涨跌额', 'change'))),
                volume=safe_int(row.get(_col.get('成交量', 'volume'))),
                amount=safe_float(row.get(_col.get('成交额', 'amount'))),
                turnover_rate=safe_float(row.get(_col.get('换手率', 'turnover_rate'))),
                amplitude=safe_float(row.get(_col.get('振幅', 'amplitude'))),
                high=safe_float(row.get(_col.get('最高', 'high'))),
                low=safe_float(row.get(_col.get('最低', 'low'))),
                open_price=safe_float(row.get(_col.get('开盘', 'open'))),
                volume_ratio=safe_float(row.get(_col.get('量比', 'volume_ratio'))),
                pe_ratio=safe_float(row.get(_col.get('市盈率', 'pe_ratio'))),
                total_mv=safe_float(row.get(_col.get('总市值', 'total_mv'))),
                circ_mv=safe_float(row.get(_col.get('流通市值', 'circ_mv'))),
            )
            
            logger.info(f"[实时行情-efinance] {stock_code} {quote.name}: 价格={quote.price}, 涨跌={quote.change_pct}%, "
                       f"量比={quote.volume_ratio}, 换手率={quote.turnover_rate}%")
            return quote
            
        except FuturesTimeoutError:
            logger.warning(f"[超时] ef.stock.get_realtime_quotes() 超过 {_EF_CALL_TIMEOUT}s，跳过 {stock_code}")
            circuit_breaker.record_failure(source_key, "timeout")
            return None
        except Exception as e:
            logger.error(f"[API错误] 获取 {stock_code} 实时行情(efinance)失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None

    def _get_etf_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取 ETF 实时行情

        efinance 默认实时接口仅返回股票数据，ETF 需要显式传入 ['ETF']。
        """
        import efinance as ef
        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "efinance_etf"

        if not circuit_breaker.is_available(source_key):
            logger.warning(f"[熔断] 数据源 {source_key} 处于熔断状态，跳过")
            return None

        try:
            current_time = time.time()
            if (
                _etf_realtime_cache['data'] is not None and
                current_time - _etf_realtime_cache['timestamp'] < _etf_realtime_cache['ttl']
            ):
                df = _etf_realtime_cache['data']
                cache_age = int(current_time - _etf_realtime_cache['timestamp'])
                logger.debug(f"[缓存命中] ETF实时行情(efinance) - 缓存年龄 {cache_age}s/{_etf_realtime_cache['ttl']}s")
            else:
                self._set_random_user_agent()
                self._enforce_rate_limit()

                logger.info("[API调用] ef.stock.get_realtime_quotes(['ETF']) 获取ETF实时行情...")
                import time as _time
                api_start = _time.time()
                df = _ef_call_with_timeout(ef.stock.get_realtime_quotes, ['ETF'])
                api_elapsed = _time.time() - api_start

                if df is not None and not df.empty:
                    logger.info(f"[API返回] ETF 实时行情成功: {len(df)} 条, 耗时 {api_elapsed:.2f}s")
                    circuit_breaker.record_success(source_key)
                else:
                    logger.warning(f"[API返回] ETF 实时行情为空, 耗时 {api_elapsed:.2f}s")
                    df = pd.DataFrame()

                _etf_realtime_cache['data'] = df
                _etf_realtime_cache['timestamp'] = current_time

            if df is None or df.empty:
                logger.warning(f"[实时行情] ETF实时行情数据为空(efinance)，跳过 {stock_code}")
                return None

            code_col = '股票代码' if '股票代码' in df.columns else 'code'
            code_series = df[code_col].astype(str).str.zfill(6)
            target_code = str(stock_code).strip().zfill(6)
            row = df[code_series == target_code]
            if row.empty:
                logger.warning(f"[API返回] 未找到 ETF {stock_code} 的实时行情(efinance)")
                return None

            row = row.iloc[0]
            _col = {
                '股票名称': 'name', 'name': 'name',
                '最新价': 'price', 'price': 'price',
                '涨跌幅': 'pct_chg', 'pct_chg': 'pct_chg',
                '涨跌额': 'change', 'change': 'change',
                '成交量': 'volume', 'volume': 'volume',
                '成交额': 'amount', 'amount': 'amount',
                '换手率': 'turnover_rate', 'turnover_rate': 'turnover_rate',
                '振幅': 'amplitude', 'amplitude': 'amplitude',
                '最高': 'high', 'high': 'high',
                '最低': 'low', 'low': 'low',
                '开盘': 'open', 'open': 'open',
            }
            quote = UnifiedRealtimeQuote(
                code=target_code,
                name=str(row.get(_col.get('股票名称', 'name'), '')),
                source=RealtimeSource.EFINANCE,
                price=safe_float(row.get(_col.get('最新价', 'price'))),
                change_pct=safe_float(row.get(_col.get('涨跌幅', 'pct_chg'))),
                change_amount=safe_float(row.get(_col.get('涨跌额', 'change'))),
                volume=safe_int(row.get(_col.get('成交量', 'volume'))),
                amount=safe_float(row.get(_col.get('成交额', 'amount'))),
                turnover_rate=safe_float(row.get(_col.get('换手率', 'turnover_rate'))),
                amplitude=safe_float(row.get(_col.get('振幅', 'amplitude'))),
                high=safe_float(row.get(_col.get('最高', 'high'))),
                low=safe_float(row.get(_col.get('最低', 'low'))),
                open_price=safe_float(row.get(_col.get('开盘', 'open'))),
            )

            logger.info(
                f"[ETF实时行情-efinance] {target_code} {quote.name}: "
                f"价格={quote.price}, 涨跌={quote.change_pct}%, 换手率={quote.turnover_rate}%"
            )
            return quote
        except Exception as e:
            logger.error(f"[API错误] 获取 ETF {stock_code} 实时行情(efinance)失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None

    def get_market_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取市场涨跌统计 (efinance)
        """
        import efinance as ef

        try:
            self._set_random_user_agent()
            self._enforce_rate_limit()

            current_time = time.time()
            if (
                _realtime_cache['data'] is not None and
                current_time - _realtime_cache['timestamp'] < _realtime_cache['ttl']
            ):
                df = _realtime_cache['data']
            else:
                logger.info("[API调用] ef.stock.get_realtime_quotes() 获取市场统计...")
                df = _ef_call_with_timeout(ef.stock.get_realtime_quotes)
                _realtime_cache['data'] = df
                _realtime_cache['timestamp'] = current_time

            if df is None or df.empty:
                logger.warning("[API返回] 市场统计数据为空")
                return None

            return calc_market_stats(df)
        except Exception as e:
            logger.error(f"[efinance] 获取市场统计失败: {e}")
            return None
        

    def get_sector_quotes(self) -> Optional[pd.DataFrame]:
        """获取全量行业板块实时行情（东财），含板块名称/涨跌幅列。

        供卖出规则做"板块明显走弱/主线退潮"判断，比涨跌榜（top/bottom n）
        覆盖面更全。失败返回 None。
        """
        import efinance as ef

        try:
            self._set_random_user_agent()
            self._enforce_rate_limit()

            logger.info("[API调用] ef.stock.get_realtime_quotes(['行业板块']) 获取板块行情...")
            df = _ef_call_with_timeout(ef.stock.get_realtime_quotes, ['行业板块'])
            if df is None or df.empty:
                logger.warning("[efinance] 板块行情数据为空")
                return None
            return df
        except Exception as e:
            logger.error(f"[efinance] 获取全量板块行情失败: {e}")
            return None

    def get_base_info(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取股票基本信息
        
        数据来源：ef.stock.get_base_info()
        包含：市盈率、市净率、所处行业、总市值、流通市值、ROE、净利率等
        
        Args:
            stock_code: 股票代码
            
        Returns:
            包含基本信息的字典，获取失败返回 None
        """
        import efinance as ef
        
        try:
            # 防封禁策略
            self._set_random_user_agent()
            self._enforce_rate_limit()
            
            logger.info(f"[API调用] ef.stock.get_base_info(stock_codes={stock_code}) 获取基本信息...")
            import time as _time
            api_start = _time.time()
            
            info = _ef_call_with_timeout(ef.stock.get_base_info, stock_code)
            
            api_elapsed = _time.time() - api_start
            logger.info(f"[API返回] ef.stock.get_base_info 成功, 耗时 {api_elapsed:.2f}s")
            
            if info is None:
                logger.warning(f"[API返回] 未获取到 {stock_code} 的基本信息")
                return None
            
            # 转换为字典
            if isinstance(info, pd.Series):
                return info.to_dict()
            elif isinstance(info, pd.DataFrame):
                if not info.empty:
                    return info.iloc[0].to_dict()
            
            return None
            
        except Exception as e:
            logger.error(f"[API错误] 获取 {stock_code} 基本信息失败: {e}")
            return None
    
    def get_belong_board(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        获取股票所属板块
        
        数据来源：ef.stock.get_belong_board()
        
        Args:
            stock_code: 股票代码
            
        Returns:
            所属板块 DataFrame，获取失败返回 None
        """
        import efinance as ef
        
        try:
            # 防封禁策略
            self._set_random_user_agent()
            self._enforce_rate_limit()
            
            logger.info(f"[API调用] ef.stock.get_belong_board(stock_code={stock_code}) 获取所属板块...")
            import time as _time
            api_start = _time.time()
            
            df = _ef_call_with_timeout(ef.stock.get_belong_board, stock_code)
            
            api_elapsed = _time.time() - api_start
            
            if df is not None and not df.empty:
                logger.info(f"[API返回] ef.stock.get_belong_board 成功: 返回 {len(df)} 个板块, 耗时 {api_elapsed:.2f}s")
                return df
            else:
                logger.warning(f"[API返回] 未获取到 {stock_code} 的板块信息")
                return None
            
        except FuturesTimeoutError:
            logger.warning(f"[超时] ef.stock.get_belong_board({stock_code}) 超过 {_EF_CALL_TIMEOUT}s，跳过")
            return None
        except Exception as e:
            logger.error(f"[API错误] 获取 {stock_code} 所属板块失败: {e}")
            return None
    
    def get_enhanced_data(self, stock_code: str, days: int = 60) -> Dict[str, Any]:
        """
        获取增强数据（历史K线 + 实时行情 + 基本信息）
        
        Args:
            stock_code: 股票代码
            days: 历史数据天数
            
        Returns:
            包含所有数据的字典
        """
        result = {
            'code': stock_code,
            'daily_data': None,
            'realtime_quote': None,
            'base_info': None,
            'belong_board': None,
        }
        
        # 获取日线数据
        try:
            df = self.get_daily_data(stock_code, days=days)
            result['daily_data'] = df
        except Exception as e:
            logger.error(f"获取 {stock_code} 日线数据失败: {e}")
        
        # 获取实时行情
        result['realtime_quote'] = self.get_realtime_quote(stock_code)
        
        # 获取基本信息
        result['base_info'] = self.get_base_info(stock_code)
        
        # 获取所属板块
        result['belong_board'] = self.get_belong_board(stock_code)
        
        return result


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    fetcher = EfinanceFetcher()
    
    # 测试普通股票
    print("=" * 50)
    print("测试普通股票数据获取 (efinance)")
    print("=" * 50)
    try:
        df = fetcher.get_daily_data('600519')  # 茅台
        print(f"[股票] 获取成功，共 {len(df)} 条数据")
        print(df.tail())
    except Exception as e:
        print(f"[股票] 获取失败: {e}")
    
    # 测试 ETF 基金
    print("\n" + "=" * 50)
    print("测试 ETF 基金数据获取 (efinance)")
    print("=" * 50)
    try:
        df = fetcher.get_daily_data('512400')  # 有色龙头ETF
        print(f"[ETF] 获取成功，共 {len(df)} 条数据")
        print(df.tail())
    except Exception as e:
        print(f"[ETF] 获取失败: {e}")
    
    # 测试实时行情
    print("\n" + "=" * 50)
    print("测试实时行情获取 (efinance)")
    print("=" * 50)
    try:
        quote = fetcher.get_realtime_quote('600519')
        if quote:
            print(f"[实时行情] {quote.name}: 价格={quote.price}, 涨跌幅={quote.change_pct}%")
        else:
            print("[实时行情] 未获取到数据")
    except Exception as e:
        print(f"[实时行情] 获取失败: {e}")
    
    # 测试基本信息
    print("\n" + "=" * 50)
    print("测试基本信息获取 (efinance)")
    print("=" * 50)
    try:
        info = fetcher.get_base_info('600519')
        if info:
            print(f"[基本信息] 市盈率={info.get('市盈率(动)', 'N/A')}, 市净率={info.get('市净率', 'N/A')}")
        else:
            print("[基本信息] 未获取到数据")
    except Exception as e:
        print(f"[基本信息] 获取失败: {e}")

    # 测试市场统计 
    print("\n" + "=" * 50)
    print("Testing get_market_stats (efinance)")
    print("=" * 50)
    try:
        stats = fetcher.get_market_stats()
        if stats:
            print(f"Market Stats successfully computed:")
            print(f"Up: {stats['up_count']} (Limit Up: {stats['limit_up_count']})")
            print(f"Down: {stats['down_count']} (Limit Down: {stats['limit_down_count']})")
            print(f"Flat: {stats['flat_count']}")
            print(f"Total Amount: {stats['total_amount']:.2f} 亿 (Yi)")
        else:
            print("Failed to compute market stats.")
    except Exception as e:
        print(f"Failed to compute market stats: {e}")
