# -*- coding: utf-8 -*-
"""
===================================
TushareFetcher - 备用数据源 1 (Priority 2)
===================================

数据来源：Tushare Pro API（挖地兔）
特点：需要 Token、有请求配额限制
优点：数据质量高、接口稳定

流控策略：
1. 实现"每分钟调用计数器"
2. 超过免费配额（80次/分）时，强制休眠到下一分钟
3. 使用 tenacity 实现指数退避重试
"""

import json as _json
import logging
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from data_provider.fetchers.base import BaseFetcher
from data_provider.stats import calc_market_stats
from data_provider.types import (
    KIND_REALTIME, KIND_STOCK_DAILY,
    DataFetchError, RateLimitError, STANDARD_COLUMNS, UnifiedRealtimeQuote,
)
from data_provider.codes import is_bse_code, is_st_stock, is_kc_cy_stock, normalize_stock_code, is_etf_code, is_hk_market
from data_provider.codes import is_us_stock_code
from src.config import get_config
import os
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


# ETF code prefixes by exchange
# Shanghai: 51xxxx, 52xxxx, 56xxxx, 58xxxx
# Shenzhen: 15xxxx, 16xxxx, 18xxxx
_ETF_SH_PREFIXES = ('51', '52', '56', '58')
_ETF_SZ_PREFIXES = ('15', '16', '18')


class TushareFetcher(BaseFetcher):
    """
    Tushare Pro 数据源实现
    
    优先级：2（默认；Token 配置且 API 初始化成功时提升为 -1，仅次于 AmazingData）
    数据来源：Tushare Pro API
    
    关键策略：
    - 每分钟调用计数器，防止超出配额
    - 超过 80 次/分钟时强制等待
    - 失败后指数退避重试
    
    配额说明（Tushare 免费用户）：
    - 每分钟最多 80 次请求
    - 每天最多 500 次请求
    """
    
    name = "TushareFetcher"
    priority = int(os.getenv("TUSHARE_PRIORITY", "2"))  # 默认优先级，会在 __init__ 中根据配置动态调整
    # daily/fund_daily 接口不返回换手率（换手率在 daily_basic，未接入），列回退时跳过本源
    SUPPORTS_COLUMNS = {'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg'}

    # 日线与实时均仅 A 股（美股、港股明确拒绝，见 _fetch_raw_data）
    SUPPORTS = frozenset({(KIND_STOCK_DAILY, "cn"), (KIND_REALTIME, "cn")})

    def __init__(self, rate_limit_per_minute: int = 80):
        """
        初始化 TushareFetcher

        Args:
            rate_limit_per_minute: 每分钟最大请求数（默认80，Tushare免费配额）
        """
        self.rate_limit_per_minute = rate_limit_per_minute
        self._call_count = 0  # 当前分钟内的调用次数
        self._minute_start: Optional[float] = None  # 当前计数周期开始时间
        self._api: Optional[object] = None  # Tushare API 实例
        self.date_list = None # 交易日列表缓存

        # 尝试初始化 API
        self._init_api()

        # 根据 API 初始化结果动态调整优先级
        self.priority = self._determine_priority()
    
    def _init_api(self) -> None:
        """
        初始化 Tushare API
        
        如果 Token 未配置，此数据源将不可用
        """
        config = get_config()
        
        if not config.tushare_token:
            logger.warning("Tushare Token 未配置，此数据源不可用")
            return
        
        try:
            import tushare as ts
            
            # Set Token
            ts.set_token(config.tushare_token)
            
            # Get API instance
            self._api = ts.pro_api()
            
            # Fix: tushare SDK 1.4.x hardcodes api.waditu.com/dataapi which may
            # be unavailable (503). Monkey-patch the query method to use the
            # official api.tushare.pro endpoint which posts to root URL.
            self._patch_api_endpoint(config.tushare_token)

            logger.info("Tushare API 初始化成功")
            
        except Exception as e:
            logger.error(f"Tushare API 初始化失败: {e}")
            self._api = None

    def _patch_api_endpoint(self, token: str) -> None:
        """
        Patch tushare SDK to use the official api.tushare.pro endpoint.

        The SDK (v1.4.x) hardcodes http://api.waditu.com/dataapi and appends
        /{api_name} to the URL. That endpoint may return 503, causing silent
        empty-DataFrame failures. This method replaces the query method to
        POST directly to http://api.tushare.pro (root URL, no path suffix).
        """
        import types

        TUSHARE_API_URL = "http://api.tushare.pro"
        _token = token
        _timeout = getattr(self._api, '_DataApi__timeout', 30)

        def patched_query(self_api, api_name, fields='', **kwargs):
            req_params = {
                'api_name': api_name,
                'token': _token,
                'params': kwargs,
                'fields': fields,
            }
            res = requests.post(TUSHARE_API_URL, json=req_params, timeout=_timeout)
            if res.status_code != 200:
                raise Exception(f"Tushare API HTTP {res.status_code}")
            result = _json.loads(res.text)
            if result['code'] != 0:
                raise Exception(result['msg'])
            data = result['data']
            columns = data['fields']
            items = data['items']
            return pd.DataFrame(items, columns=columns)

        self._api.query = types.MethodType(patched_query, self._api)
        logger.debug(f"Tushare API endpoint patched to {TUSHARE_API_URL}")

    def _determine_priority(self) -> int:
        """
        根据 Token 配置和 API 初始化状态确定优先级

        策略：
        - Token 配置且 API 初始化成功：优先级 -1（仅次于 AmazingData 的 -2）
        - 其他情况：优先级 2（默认）

        Returns:
            优先级数字（-2=最高，数字越大优先级越低）
        """
        config = get_config()

        if config.tushare_token and self._api is not None:
            # Token 配置且 API 初始化成功，提升为次高优先级（仅次于 AmazingData）
            logger.info("✅ 检测到 TUSHARE_TOKEN 且 API 初始化成功，Tushare 数据源优先级提升 (Priority -1，仅次于 AmazingData)")
            return -1

        # Token 未配置或 API 初始化失败，保持默认优先级
        return 2

    def is_available(self) -> bool:
        """
        检查数据源是否可用

        Returns:
            True 表示可用，False 表示不可用
        """
        return self._api is not None

    def _check_rate_limit(self) -> None:
        """
        检查并执行速率限制
        
        流控策略：
        1. 检查是否进入新的一分钟
        2. 如果是，重置计数器
        3. 如果当前分钟调用次数超过限制，强制休眠
        """
        current_time = time.time()
        
        # 检查是否需要重置计数器（新的一分钟）
        if self._minute_start is None:
            self._minute_start = current_time
            self._call_count = 0
        elif current_time - self._minute_start >= 60:
            # 已经过了一分钟，重置计数器
            self._minute_start = current_time
            self._call_count = 0
            logger.debug("速率限制计数器已重置")
        
        # 检查是否超过配额
        if self._call_count >= self.rate_limit_per_minute:
            # 计算需要等待的时间（到下一分钟）
            elapsed = current_time - self._minute_start
            sleep_time = max(0, 60 - elapsed) + 1  # +1 秒缓冲
            
            logger.warning(
                f"Tushare 达到速率限制 ({self._call_count}/{self.rate_limit_per_minute} 次/分钟)，"
                f"等待 {sleep_time:.1f} 秒..."
            )
            
            time.sleep(sleep_time)
            
            # 重置计数器
            self._minute_start = time.time()
            self._call_count = 0
        
        # 增加调用计数
        self._call_count += 1
        logger.debug(f"Tushare 当前分钟调用次数: {self._call_count}/{self.rate_limit_per_minute}")
    
    def _convert_stock_code(self, stock_code: str) -> str:
        """
        转换股票代码为 Tushare 格式

        Tushare 要求的格式：
        - 沪市股票：600519.SH
        - 深市股票：000001.SZ
        - 沪市 ETF：510050.SH, 563230.SH
        - 深市 ETF：159919.SZ
        - 北交所：920748.BJ

        Args:
            stock_code: 原始代码，如 '600519', '000001', '563230'

        Returns:
            Tushare 格式代码，如 '600519.SH', '000001.SZ', '563230.SH'
        """
        code = stock_code.strip()

        # 已有后缀：补齐 BJ（SH/SZ 已含 6 位数字可直接返回）
        if '.' in code:
            base, suffix = code.rsplit('.', 1)
            if suffix.upper() in ('SH', 'SZ') and base.isdigit() and len(base) == 6:
                return code.upper()
            if suffix.upper() == 'BJ':
                return f"{base}.BJ"

        # Tushare 不支持港股/美股，抛出异常由管理器切换数据源
        if is_hk_market(code) or is_us_stock_code(code):
            raise DataFetchError(f"TushareFetcher 不支持 {code}，请使用其他数据源")

        # ETF: 根据前缀判断交易所
        if code.startswith(_ETF_SH_PREFIXES) and len(code) == 6:
            return f"{code}.SH"
        if code.startswith(_ETF_SZ_PREFIXES) and len(code) == 6:
            return f"{code}.SZ"

        # 北交所 (BJ): 8xxxxx, 4xxxxx, 920xxx
        if is_bse_code(code):
            return f"{code}.BJ"

        # 普通 A 股
        if code.startswith(('600', '601', '603', '688')):
            return f"{code}.SH"
        elif code.startswith(('000', '001', '002', '003', '300')):
            return f"{code}.SZ"
        else:
            logger.warning(f"无法确定股票 {code} 的市场，默认使用深市")
            return f"{code}.SZ"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Tushare 获取原始数据
        
        根据代码类型选择不同接口：
        - 普通股票：daily()
        - ETF 基金：fund_daily()
        
        流程：
        1. 检查 API 是否可用
        2. 检查是否为美股（不支持）
        3. 执行速率限制检查
        4. 转换股票代码格式
        5. 根据代码类型选择接口并调用
        """
        if self._api is None:
            raise DataFetchError("Tushare API 未初始化，请检查 Token 配置")

        # Rate-limit check
        self._check_rate_limit()

        # Convert code format (also validates unsupported markets)
        ts_code = self._convert_stock_code(stock_code)
        
        # Convert date format (Tushare requires YYYYMMDD)
        ts_start = start_date.replace('-', '')
        ts_end = end_date.replace('-', '')
        
        is_etf = is_etf_code(stock_code)
        api_name = "fund_daily" if is_etf else "daily"
        logger.debug(f"调用 Tushare {api_name}({ts_code}, {ts_start}, {ts_end})")
        
        try:
            if is_etf:
                # ETF uses fund_daily interface
                df = self._api.fund_daily(
                    ts_code=ts_code,
                    start_date=ts_start,
                    end_date=ts_end,
                )
            else:
                # Regular stocks use daily interface
                df = self._api.daily(
                    ts_code=ts_code,
                    start_date=ts_start,
                    end_date=ts_end,
                )
            
            return df
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # 检测配额超限
            if any(keyword in error_msg for keyword in ['quota', '配额', 'limit', '权限']):
                logger.warning(f"Tushare 配额可能超限: {e}")
                raise RateLimitError(f"Tushare 配额超限: {e}") from e
            
            raise DataFetchError(f"Tushare 获取数据失败: {e}") from e
    
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Tushare 数据
        
        Tushare daily 返回的列名：
        ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
        
        需要映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()
        
        # 列名映射
        column_mapping = {
            'trade_date': 'date',
            'vol': 'volume',
            # open, high, low, close, amount, pct_chg 列名相同
        }
        
        df = df.rename(columns=column_mapping)
        
        # 转换日期格式（YYYYMMDD -> YYYY-MM-DD）
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # 成交量单位转换（Tushare 的 vol 单位是手，需要转换为股）
        if 'volume' in df.columns:
            df['volume'] = df['volume'] * 100
        
        # 成交额单位转换（Tushare 的 amount 单位是千元，转换为元）
        if 'amount' in df.columns:
            df['amount'] = df['amount'] * 1000
        
        # 添加股票代码列
        df['code'] = stock_code
        
        # 只保留需要的列
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]
        
        return df

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        """
        获取股票名称
        
        使用 Tushare 的 stock_basic 接口获取股票基本信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            股票名称，失败返回 None
        """
        if self._api is None:
            logger.warning("Tushare API 未初始化，无法获取股票名称")
            return None

        # HK stocks not supported by Tushare stock_basic
        if is_hk_market(stock_code):
            return None

        # 检查缓存
        cache = getattr(self, '_stock_name_cache', None)
        if cache is not None and stock_code in cache:
            return cache[stock_code]

        # 初始化缓存
        self._stock_name_cache = {}
        
        try:
            # 速率限制检查
            self._check_rate_limit()
            
            # 转换代码格式
            ts_code = self._convert_stock_code(stock_code)
            
            # ETF uses fund_basic, regular stocks use stock_basic
            if is_etf_code(stock_code):
                df = self._api.fund_basic(
                    ts_code=ts_code,
                    fields='ts_code,name'
                )
            else:
                df = self._api.stock_basic(
                    ts_code=ts_code,
                    fields='ts_code,name'
                )
            
            if df is not None and not df.empty:
                name = df.iloc[0]['name']
                self._stock_name_cache[stock_code] = name
                logger.debug(f"Tushare 获取股票名称成功: {stock_code} -> {name}")
                return name
            
        except Exception as e:
            logger.warning(f"Tushare 获取股票名称失败 {stock_code}: {e}")
        
        return None
    
    def get_stock_list(self) -> Optional[pd.DataFrame]:
        """
        获取股票列表
        
        使用 Tushare 的 stock_basic 接口获取全部股票列表
        
        Returns:
            包含 code, name 列的 DataFrame，失败返回 None
        """
        if self._api is None:
            logger.warning("Tushare API 未初始化，无法获取股票列表")
            return None
        
        try:
            # 速率限制检查
            self._check_rate_limit()
            
            # 调用 stock_basic 接口获取所有股票
            df = self._api.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,name,industry,area,market'
            )
            
            if df is not None and not df.empty:
                # 转换 ts_code 为标准代码格式
                df['code'] = df['ts_code'].apply(lambda x: x.split('.')[0])
                
                # 更新缓存
                self._stock_name_cache = getattr(self, '_stock_name_cache', {})
                for _, row in df.iterrows():
                    self._stock_name_cache[row['code']] = row['name']
                
                logger.info(f"Tushare 获取股票列表成功: {len(df)} 条")
                return df[['code', 'name', 'industry', 'area', 'market']]
            
        except Exception as e:
            logger.warning(f"Tushare 获取股票列表失败: {e}")
        
        return None
    
    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取实时行情

        策略：
        1. 优先尝试 Pro 接口（需要2000积分）：数据全，稳定性高
        2. 失败降级到旧版接口：门槛低，数据较少

        Args:
            stock_code: 股票代码

        Returns:
            UnifiedRealtimeQuote 对象，失败返回 None
        """
        if self._api is None:
            return None

        # HK stocks not supported by Tushare
        if is_hk_market(stock_code):
            logger.debug(f"TushareFetcher 跳过港股实时行情 {stock_code}")
            return None

        from data_provider.types import (
            RealtimeSource,
            safe_float, safe_int
        )

        # 速率限制检查
        self._check_rate_limit()

        # 尝试 Pro 接口
        try:
            ts_code = self._convert_stock_code(stock_code)
            # 尝试调用 Pro 实时接口 (需要积分)
            df = self._api.quotation(ts_code=ts_code)

            if df is not None and not df.empty:
                row = df.iloc[0]
                logger.debug(f"Tushare Pro 实时行情获取成功: {stock_code}")

                return UnifiedRealtimeQuote(
                    code=stock_code,
                    name=str(row.get('name', '')),
                    source=RealtimeSource.TUSHARE,
                    price=safe_float(row.get('price')),
                    change_pct=safe_float(row.get('pct_chg')),  # Pro 接口通常直接返回涨跌幅
                    change_amount=safe_float(row.get('change')),
                    volume=safe_int(row.get('vol')),
                    amount=safe_float(row.get('amount')),
                    high=safe_float(row.get('high')),
                    low=safe_float(row.get('low')),
                    open_price=safe_float(row.get('open')),
                    pre_close=safe_float(row.get('pre_close')),
                    turnover_rate=safe_float(row.get('turnover_ratio')), # Pro 接口可能有换手率
                    pe_ratio=safe_float(row.get('pe')),
                    pb_ratio=safe_float(row.get('pb')),
                    total_mv=safe_float(row.get('total_mv')),
                )
        except Exception as e:
            # 仅记录调试日志，不报错，继续尝试降级
            logger.debug(f"Tushare Pro 实时行情不可用 (可能是积分不足): {e}")

        # 降级：尝试旧版接口
        try:
            import tushare as ts

            # Tushare 旧版接口使用 6 位代码
            code_6 = stock_code.split('.')[0] if '.' in stock_code else stock_code

            # 特殊处理指数代码：旧版接口需要前缀 (sh000001, sz399001)
            # 简单的指数判断逻辑
            if code_6 == '000001':  # 上证指数
                symbol = 'sh000001'
            elif code_6 == '399001':  # 深证成指
                symbol = 'sz399001'
            elif code_6 == '399006':  # 创业板指
                symbol = 'sz399006'
            elif code_6 == '000300':  # 沪深300
                symbol = 'sh000300'
            elif is_bse_code(code_6):  # 北交所
                symbol = f"bj{code_6}"
            else:
                symbol = code_6

            # 调用旧版实时接口 (ts.get_realtime_quotes)
            df = ts.get_realtime_quotes(symbol)

            if df is None or df.empty:
                return None

            row = df.iloc[0]

            # 计算涨跌幅
            price = safe_float(row['price'])
            pre_close = safe_float(row['pre_close'])
            change_pct = 0.0
            change_amount = 0.0

            if price and pre_close and pre_close > 0:
                change_amount = price - pre_close
                change_pct = (change_amount / pre_close) * 100

            # 构建统一对象
            return UnifiedRealtimeQuote(
                code=stock_code,
                name=str(row['name']),
                source=RealtimeSource.TUSHARE,
                price=price,
                change_pct=round(change_pct, 2),
                change_amount=round(change_amount, 2),
                volume=safe_int(row['volume']) // 100,  # 转换为手
                amount=safe_float(row['amount']),
                high=safe_float(row['high']),
                low=safe_float(row['low']),
                open_price=safe_float(row['open']),
                pre_close=pre_close,
            )

        except Exception as e:
            logger.warning(f"Tushare (旧版) 获取实时行情失败 {stock_code}: {e}")
            return None

    def get_market_stats(self) -> Optional[dict]:
        """
        获取市场涨跌统计 (Tushare Pro)
        仅支持盘后 daily 接口；盘中返回 None，由管理器降级到其他数据源
        """
        if self._api is None:
            return None

        try:
            self._check_rate_limit()
            logger.info("[Tushare] ts.pro_api() 获取市场统计...")
            
            # 获取当前中国时间，判断是否在交易时间内
            china_now = datetime.now(ZoneInfo("Asia/Shanghai"))
            current_clock = china_now.strftime("%H:%M")
            current_date = china_now.strftime("%Y%m%d")

            if self.date_list is None:
                start_date = (datetime.now() - pd.Timedelta(days=20)).strftime('%Y%m%d')
                df_cal = self._api.trade_cal(exchange='SSE', start_date=start_date, end_date=current_date)

                # 过滤出 is_open == 1 (开市) 的日期，并转换为列表
                self.date_list = df_cal[df_cal['is_open'] == 1]['cal_date'].tolist()

            if current_date in self.date_list:
                if current_clock < '09:30' or current_clock > '16:30':
                    use_realtime = False
                else:
                    use_realtime = True
            else:
                use_realtime = False

            # 盘中：没有 rt_k 实时接口权限，直接返回 None，交由其他数据源（akshare/efinance）处理
            if use_realtime:
                return None

            if current_date not in self.date_list:
                last_date = self.date_list[0] # 拿最近的日期
            else:
                if current_clock < '09:30': 
                    last_date = self.date_list[1] # 拿取前一天的数据
                else:  # 即 '> 16:30'                  
                    last_date = self.date_list[0] # 拿取当天的数据

            try:
                df = self._api.daily(TS_CODE='3*.SZ,6*.SH,0*.SZ,92*.BJ',start_date=last_date, end_date=last_date)
                # 统一将列名转为小写
                df.columns = [col.lower() for col in df.columns]

                # 获取股票基础信息（包含代码和名称）
                df_basic = self._api.stock_basic(fields='ts_code,name')
                df = pd.merge(df, df_basic, on='ts_code', how='left')
                # 将 daily的 amount 列的值乘以 1000 来和其他数据源保持一致
                if 'amount' in df.columns:
                    df['amount'] = df['amount'] * 1000

                if df is not None and not df.empty:
                    return calc_market_stats(df)
            except Exception as e:
                logger.error(f"[Tushare] ts.pro_api().daily 获取数据失败: {e}")
                

            
        except Exception as e:
            logger.error(f"[Tushare] 获取市场统计失败: {e}")

        return None
    

    def get_trade_time(self,early_time='09:30',late_time='16:30') -> Optional[Tuple[str, str, list]]:
        '''
        获取当前时间可以获得数据的开始时间日期

        Args:
                early_time: 默认 '09:30'
                late_time: 默认 '16:30'
                early_time-late_time 之间为使用上一个交易日数据的时间段，其他时间为使用当天数据的时间段
        Returns:
                start_date: 可以获得数据的开始日期
        '''
        china_now = datetime.now(ZoneInfo("Asia/Shanghai"))
        china_date = china_now.strftime("%Y%m%d")
        china_clock = china_now.strftime("%H:%M")
        
        if self.date_list is None:
            start_date = (datetime.now() - pd.Timedelta(days=20)).strftime('%Y%m%d')
            df_cal = self._api.trade_cal(exchange='SSE', start_date=start_date, end_date=china_date)

            # 过滤出 is_open == 1 (开市) 的日期，并转换为列表
            self.date_list = df_cal[df_cal['is_open'] == 1]['cal_date'].tolist()

        if china_date in self.date_list:
            if  early_time < china_clock < late_time: # 使用上一个交易日数据的时间段
                use_today = False
            else:
                use_today = True
        else:
            use_today = False

        if use_today:
            start_date = self.date_list[0] # 当天
        else:
            start_date = self.date_list[1] # 前一天
            logger.info(f"[Tushare] 当前时间 {china_clock} 可能无法获取当天筹码分布，尝试获取前一个交易日的数据 {start_date}")

        return start_date
    
if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    fetcher = TushareFetcher()
    
    try:
        # 测试历史数据
        df = fetcher.get_daily_data('600519')  # 茅台
        print(f"获取成功，共 {len(df)} 条数据")
        print(df.tail())
        
        # 测试股票名称
        name = fetcher.get_stock_name('600519')
        print(f"股票名称: {name}")
        
    except Exception as e:
        print(f"获取失败: {e}")

    # 测试市场统计
    print("\n" + "=" * 50)
    print("Testing get_market_stats (tushare)")
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
