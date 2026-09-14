# -*- coding: utf-8 -*-
"""
===================================
数据源策略管理器 DataFetcherManager
===================================

持有已实例化的 fetcher（带凭证），把入口调用转成「需求 + 数据源集合」的编排调用：
- get_daily_data：个股日线 → daily.fetch_stock_daily（failover + 新鲜度校验 + 换手率回补）
- get_realtime_quote：实时报价 → 美股/港股按能力声明取源，A 股委托 realtime.merge_realtime_quotes
- get_market_stats / get_main_fund_flow：路由.routing.query_first（多源取首个非空）
- 候选筛选统一走 routing.supporting（能力声明，编排层不出现数据源类名）

分层：codes/classify_market（市场归类）→ routing（筛源 + failover）→ daily/realtime（策略）
→ 本类（构造与持有 fetcher 集合）。ETF/指数日线见 bars.py。
"""
import logging
from typing import Optional, List, Dict, Any

import pandas as pd

from data_provider.fetchers.base import BaseFetcher
from .codes import normalize_stock_code, classify_market, _is_hk_market
from .daily import fetch_stock_daily
from .routing import query_first, supporting
from .types import KIND_FUND_FLOW, KIND_REALTIME, KIND_STOCK_DAILY, Need
from .realtime import merge_realtime_quotes

logger = logging.getLogger(__name__)


def get_fetcher():
    """构造 DataFetcherManager 实例（失败返回 None）。供入口层统一调用，避免各模块重复构造。"""
    try:
        return DataFetcherManager()
    except Exception:
        return None



class DataFetcherManager:
    """
    数据源策略管理器
    
    职责：
    1. 管理多个数据源（按优先级排序）
    2. 自动故障切换（Failover）
    3. 提供统一的数据获取接口
    
    切换策略：
    - 优先使用高优先级数据源
    - 失败后自动切换到下一个
    - 所有数据源都失败时抛出异常
    """
    
    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        """
        初始化管理器
        
        Args:
            fetchers: 数据源列表（可选，默认按优先级自动创建）
        """
        self._fetchers: List[BaseFetcher] = []
        
        if fetchers:
            # 按优先级排序
            self._fetchers = sorted(fetchers, key=lambda f: f.priority)
        else:
            # 默认数据源将在首次使用时延迟加载
            self._init_default_fetchers()


    def _init_default_fetchers(self) -> None:
        """
        初始化默认数据源列表

        优先级动态调整逻辑：
        - 如果配置了 TUSHARE_TOKEN：Tushare 优先级提升为 -1（仅次于 AmazingData）
        - 否则按默认优先级：
          -2. AmazingDataFetcher (Priority -2) - 配置了 TGW 凭证时启用（最高）
          -1. TushareFetcher (Priority -1) - 配置了 Token 且初始化成功时（仅次于 AmazingData）
           0. AkshareFetcher (Priority 0)
           1. EfinanceFetcher (Priority 1)
           2. TushareFetcher (Priority 2)
           3. BaostockFetcher (Priority 3)
           4. YfinanceFetcher (Priority 4)
        """
        from data_provider.fetchers.efinance_fetcher import EfinanceFetcher
        from data_provider.fetchers.akshare_fetcher import AkshareFetcher
        from data_provider.fetchers.tushare_fetcher import TushareFetcher
        from data_provider.fetchers.baostock_fetcher import BaostockFetcher
        from data_provider.fetchers.yfinance_fetcher import YfinanceFetcher
        # 创建所有数据源实例（优先级在各 Fetcher 的 __init__ 中确定）
        efinance = EfinanceFetcher()
        akshare = AkshareFetcher()
        tushare = TushareFetcher()  # 会根据 Token 配置自动调整优先级
        baostock = BaostockFetcher()
        yfinance = YfinanceFetcher()

        # 初始化数据源列表
        self._fetchers = [
            efinance,
            akshare,
            tushare,
            baostock,
            yfinance,
        ]

        # 配置了 TGW 凭证时启用 AmazingData（优先数据源）
        try:
            from data_provider.fetchers.amazingdata_fetcher import AmazingDataFetcher, tgw_configured

            if tgw_configured():
                amazing = AmazingDataFetcher()
                self._fetchers.append(amazing)
                logger.info("已启用 AmazingDataFetcher（星耀数智，TGW 凭证已配置）")
            else:
                logger.debug("未配置 TGW 凭证，跳过 AmazingDataFetcher")
        except Exception as e:
            logger.warning(f"AmazingDataFetcher 初始化失败，已跳过: {e}")

        # 按优先级排序（Tushare 如果配置了 Token 且初始化成功，优先级为 -1，仅次于 AmazingData）
        self._fetchers.sort(key=lambda f: f.priority)

        # 构建优先级说明
        priority_info = ", ".join([f"{f.name}(P{f.priority})" for f in self._fetchers])
        logger.info(f"已初始化 {len(self._fetchers)} 个数据源（按优先级）: {priority_info}")

    
    def get_daily_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        获取日线数据（自动切换数据源）
        
        数据源筛选与切换：
        1. 按市场归类（codes.classify_market）+ 各源能力声明（BaseFetcher.SUPPORTS）筛出候选，
           不按数据源类名判断——美股日线因此天然只剩 YfinanceFetcher
        2. 候选中从最高优先级数据源开始尝试
        3. 捕获异常后自动切换到下一个
        4. 记录每个数据源的失败原因
        5. 所有数据源失败后抛出详细异常
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            days: 获取天数
            
        Returns:
            DataFrame: 标准化日线数据（命中哪个数据源见日志）
            
        Raises:
            DataFetchError: 所有数据源都失败时抛出
        """
        code = normalize_stock_code(stock_code)
        need = Need(KIND_STOCK_DAILY, code, classify_market(code))
        return fetch_stock_daily(
            need, self._fetchers, start_date=start_date, end_date=end_date, days=days
        )
    


    
    def get_realtime_quote(self, stock_code: str):
        """
        获取实时行情数据（自动故障切换）
        
        取数路径（按市场分流）：
        1. 美股/美股指数 → 能力声明 (realtime, us) 的源（当前仅 YfinanceFetcher）
        2. 港股 → 能力声明 (realtime, hk) 的源（当前仅 AkshareFetcher，走 source="hk"）
        3. A 股 → 委托 realtime.merge_realtime_quotes 按 source_priority 跨源合并
        4. 全部失败返回 None（降级兜底）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            UnifiedRealtimeQuote 对象，所有数据源都失败则返回 None
        """
        # Normalize code (strip SH/SZ prefix etc.)
        stock_code = normalize_stock_code(stock_code)

        from data_provider.codes import is_us_stock_code
        from .us_index_mapping import is_us_index_code
        from src.config import get_config

        config = get_config()

        # 如果实时行情功能被禁用，直接返回 None
        if not config.enable_realtime_quote:
            logger.debug(f"[实时行情] 功能已禁用，跳过 {stock_code}")
            return None

        # 美股指数由 YfinanceFetcher 处理（在美股股票检查之前）
        if is_us_index_code(stock_code):
            return self._quote_from_yfinance(stock_code, "美股指数")

        # 美股单独处理，使用 YfinanceFetcher
        if is_us_stock_code(stock_code):
            return self._quote_from_yfinance(stock_code, "美股")

        # 港股实时行情只走港股专用入口，避免按 A 股 source_priority
        # 反复触发同一个 ak.stock_hk_spot_em() 接口。
        # source="hk" 是 akshare 的港股入口；当前仅 AkshareFetcher 声明 (realtime, hk)
        if _is_hk_market(stock_code):
            for fetcher in supporting(self._fetchers, Need(KIND_REALTIME, stock_code, "hk")):
                try:
                    quote = fetcher.get_realtime_quote(stock_code, source="hk")
                    if quote is not None and quote.has_basic_data():
                        logger.info(f"[实时行情] 港股 {stock_code} 成功获取 (来源: {fetcher.name})")
                        return quote
                except Exception as e:
                    logger.warning(f"[实时行情] 港股 {stock_code} 获取失败: {e}")
            logger.warning(f"[实时行情] 港股 {stock_code} 无可用数据源")
            return None
        
        # 跨源合并（A 股按 source_priority）抽离为与 manager 解耦的纯函数。
        # 未来剪除日线杂活后，调用方只需持有 fetcher 集合即可直接调用，不再依赖本类。
        source_priority = config.realtime_source_priority.split(',')
        return merge_realtime_quotes(stock_code, self._fetchers, source_priority)

    def _quote_from_yfinance(self, stock_code: str, label: str) -> Optional["UnifiedRealtimeQuote"]:
        """美股/美股指数实时行情：按能力声明找候选（当前仅 YfinanceFetcher 声明 (realtime, us)）。"""
        for fetcher in supporting(self._fetchers, Need(KIND_REALTIME, stock_code, "us")):
            try:
                quote = fetcher.get_realtime_quote(stock_code)
                if quote is not None:
                    logger.info(f"[实时行情] {label} {stock_code} 成功获取 (来源: {fetcher.name})")
                    return quote
            except Exception as e:
                logger.warning(f"[实时行情] {label} {stock_code} 获取失败: {e}")
        logger.warning(f"[实时行情] {label} {stock_code} 无可用数据源")
        return None

    def get_market_stats(self) -> Dict[str, Any]:
        """获取市场涨跌统计（自动切换数据源）；全部源无数据返回 {}。"""
        stats = query_first("市场统计", self._fetchers, "get_market_stats")
        if stats is None:
            logger.warning("[市场统计] 无可用数据源")
            return {}
        return stats


    def get_main_fund_flow(self, stock_code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """
        获取个股主力资金流向（自动切换数据源）。

        当前仅 AkshareFetcher 声明 (fund_flow, cn)，其余源由能力筛选直接排除。

        Args:
            stock_code: 股票代码
            days: 统计天数（默认 5 个交易日）

        Returns:
            标准化 DataFrame（列：date / main_net_inflow，单位元）；全部无数据时返回 None
        """
        code = normalize_stock_code(stock_code)
        df = query_first(
            f"主力资金流 {code}", self._fetchers, "get_main_fund_flow", code, days=days,
            need=Need(KIND_FUND_FLOW, code, classify_market(code)),
        )
        if df is None:
            logger.warning(f"[主力资金流] {stock_code} 无可用数据源，返回 None")
        return df



