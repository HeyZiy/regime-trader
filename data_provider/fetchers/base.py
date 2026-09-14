# -*- coding: utf-8 -*-
"""
===================================
数据源抽象基类 BaseFetcher
===================================

策略模式抽象基类：定义统一的数据获取接口与标准化流程。
所有数据源实现（*_fetcher.py）继承本基类；代码判定见 codes.py、
契约/异常见 types.py、管理器见 manager.py。
"""
import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from data_provider.types import DataFetchError, Need, summarize_exception

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    """
    数据源抽象基类
    
    职责：
    1. 定义统一的数据获取接口
    2. 提供数据标准化方法
    子类实现：
    - _fetch_raw_data(): 从具体数据源获取原始数据
    - _normalize_data(): 将原始数据转换为标准格式
    """
    
    name: str = "BaseFetcher"
    priority: int = 99  # 优先级数字越小越优先

    # 能力声明：本数据源「日线接口确实提供」的标准列。
    # None = 未声明（视为可能提供所有标准列）；
    # 列回退 (_backfill_missing_columns) 会先按此过滤，跳过确定没有该列的源，避免无效请求。
    SUPPORTS_COLUMNS: Optional[set] = None

    # 能力声明：本数据源覆盖的 (kind, market) 组合，元素形如 ("stock_daily", "cn")。
    # None = 未声明（不限制）；编排层按此筛源，避免编排逻辑出现具体数据源类名。
    # 声明过宽 → 无效请求 + 误导性日志；声明过窄 → 可用源被跳过：按实测能力如实填。
    SUPPORTS: Optional[frozenset] = None

    def supports(self, need: "Need") -> bool:
        """本数据源能否满足该需求（能力声明的唯一判定入口）。"""
        if self.SUPPORTS is None:
            return True
        return (need.kind, need.market) in self.SUPPORTS
    
    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从数据源获取原始数据（子类必须实现）
        
        Args:
            stock_code: 股票代码，如 '600519', '000001'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            
        Returns:
            原始数据 DataFrame（列名因数据源而异）
        """
        pass
    
    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化数据列名（子类必须实现）

        将不同数据源的列名统一为：
        ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        """
        pass

    def get_market_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取市场涨跌统计

        Returns:
            Dict: 包含:
                - up_count: 上涨家数
                - down_count: 下跌家数
                - flat_count: 平盘家数
                - limit_up_count: 涨停家数
                - limit_down_count: 跌停家数
                - total_amount: 两市成交额
        """
        return None

    def get_realtime_quote(self, stock_code: str, **kwargs) -> Optional["UnifiedRealtimeQuote"]:
        """
        获取实时行情（基类兜底实现）。

        默认返回 None，表示「该数据源不支持实时行情」。子类若支持实时行情应覆写此方法；
        基类提供此兜底后，调用方无需 hasattr 预检，统一以 None 判断「无数据」。
        """
        return None

    def get_main_fund_flow(self, stock_code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """
        获取主力资金流向数据（默认抽象方法）

        Args:
            stock_code: 股票代码
            days: 获取天数（默认 5 个交易日）

        Returns:
            DataFrame 包含以下列：
            - date: 日期
            - main_net_inflow: 主力净流入（元，正数=净流入，负数=净流出）
            - 或 None（数据源不支持）

        基类默认返回 None（不声明支持），子类若提供此数据应覆写此方法。
        """
        return None

    def get_daily_data(
        self,
        stock_code: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        获取日线数据（统一入口）

        流程：
        1. 计算日期范围
        2. 调用子类获取原始数据
        3. 标准化列名
        4. 数据清洗

        Args:
            stock_code: 股票代码
            start_date: 开始日期（可选）
            end_date: 结束日期（可选，默认今天）
            days: 获取天数（当 start_date 未指定时使用）

        Returns:
            标准化的 DataFrame（仅行情列；技术指标由调用方按需追加）
        """
        # 计算日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # 默认获取最近 30 个交易日（按日历日估算，多取一些）
            from datetime import timedelta
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)
            start_date = start_dt.strftime('%Y-%m-%d')

        request_start = time.time()
        logger.info(f"[{self.name}] 开始获取 {stock_code} 日线数据: 范围={start_date} ~ {end_date}")
        
        try:
            # Step 1: 获取原始数据
            raw_df = self._fetch_raw_data(stock_code, start_date, end_date)
            
            if raw_df is None or raw_df.empty:
                raise DataFetchError(f"[{self.name}] 未获取到 {stock_code} 的数据")
            
            # Step 2: 标准化列名
            df = self._normalize_data(raw_df, stock_code)

            # Step 3: 数据清洗
            df = self._clean_data(df)

            elapsed = time.time() - request_start
            logger.info(
                f"[{self.name}] {stock_code} 获取成功: 范围={start_date} ~ {end_date}, "
                f"rows={len(df)}, elapsed={elapsed:.2f}s"
            )
            return df
            
        except Exception as e:
            elapsed = time.time() - request_start
            error_type, error_reason = summarize_exception(e)
            logger.error(
                f"[{self.name}] {stock_code} 获取失败: 范围={start_date} ~ {end_date}, "
                f"error_type={error_type}, elapsed={elapsed:.2f}s, reason={error_reason}"
            )
            raise DataFetchError(f"[{self.name}] {stock_code}: {error_reason}") from e
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        处理：
        1. 确保日期列格式正确
        2. 数值类型转换
        3. 去除空值行
        4. 按日期排序
        """
        df = df.copy()
        
        # 确保日期列为 datetime 类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 数值列类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'turnover_rate']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 去除关键列为空的行
        df = df.dropna(subset=['close', 'volume'])
        
        # 按日期升序排序
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    
    @staticmethod
    def random_sleep(min_seconds: float = 1.0, max_seconds: float = 3.0) -> None:
        """
        智能随机休眠（Jitter）
        
        防封禁策略：模拟人类行为的随机延迟
        在请求之间加入不规则的等待时间
        """
        sleep_time = random.uniform(min_seconds, max_seconds)
        logger.debug(f"随机休眠 {sleep_time:.2f} 秒...")
        time.sleep(sleep_time)


