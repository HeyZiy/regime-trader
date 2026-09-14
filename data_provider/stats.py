# -*- coding: utf-8 -*-
"""
===================================
市场统计工具（多 Fetcher 共享）
===================================

从行情 DataFrame 计算涨跌统计的公共实现。
EfinanceFetcher / AkshareFetcher / TushareFetcher 共用。
"""
import logging

import numpy as np
import pandas as pd

from .codes import is_bse_code, is_kc_cy_stock, is_st_stock, normalize_stock_code

logger = logging.getLogger(__name__)


# 兼容不同接口返回的列名
_COL_CANDIDATES = {
    'code': ['代码', '股票代码', 'ts_code', 'stock_code'],
    'name': ['名称', '股票名称', 'name'],
    'close': ['最新价', 'close', 'lastPrice'],
    'pre_close': ['昨收', '昨日收盘', 'pre_close', 'lastClose'],
    'amount': ['成交额', 'amount'],
}


def _resolve_column(df: pd.DataFrame, candidates: list) -> str:
    """返回 DataFrame 中首个命中的列名。"""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"未找到匹配列: {candidates}")


def calc_market_stats(df: pd.DataFrame) -> dict:
    """从行情 DataFrame 计算涨跌统计。

    Args:
        df: 含行情数据的 DataFrame（列名来自各数据源，自动兼容中英文）。

    Returns:
        dict: up_count / down_count / flat_count / limit_up_count /
              limit_down_count / total_amount(亿)
    """
    df = df.copy()

    code_col = _resolve_column(df, _COL_CANDIDATES['code'])
    name_col = _resolve_column(df, _COL_CANDIDATES['name'])
    close_col = _resolve_column(df, _COL_CANDIDATES['close'])
    pre_close_col = _resolve_column(df, _COL_CANDIDATES['pre_close'])
    amount_col = _resolve_column(df, _COL_CANDIDATES['amount'])

    limit_up_count = 0
    limit_down_count = 0
    up_count = 0
    down_count = 0
    flat_count = 0

    for code, name, current_price, pre_close, amount in zip(
        df[code_col], df[name_col], df[close_col], df[pre_close_col], df[amount_col]
    ):
        if pd.isna(current_price) or pd.isna(pre_close) or current_price in ['-'] or pre_close in ['-'] or amount == 0:
            continue

        current_price = float(current_price)
        pre_close = float(pre_close)

        pure_code = normalize_stock_code(str(code))

        if is_bse_code(pure_code):
            ratio = 0.30
        elif is_kc_cy_stock(pure_code):
            ratio = 0.20
        elif is_st_stock(name):
            ratio = 0.05
        else:
            ratio = 0.10

        limit_up_price = np.floor(pre_close * (1 + ratio) * 100 + 0.5) / 100.0
        limit_down_price = np.floor(pre_close * (1 - ratio) * 100 + 0.5) / 100.0

        tolerance_up = round(abs(pre_close * (1 + ratio) - limit_up_price), 10)
        tolerance_down = round(abs(pre_close * (1 - ratio) - limit_down_price), 10)

        if current_price > 0:
            is_limit_up = abs(current_price - limit_up_price) <= tolerance_up
            is_limit_down = abs(current_price - limit_down_price) <= tolerance_down

            if is_limit_up:
                limit_up_count += 1
            if is_limit_down:
                limit_down_count += 1

            if current_price > pre_close:
                up_count += 1
            elif current_price < pre_close:
                down_count += 1
            else:
                flat_count += 1

    stats = {
        'up_count': up_count,
        'down_count': down_count,
        'flat_count': flat_count,
        'limit_up_count': limit_up_count,
        'limit_down_count': limit_down_count,
        'total_amount': 0.0,
    }

    if amount_col in df.columns:
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
        stats['total_amount'] = df[amount_col].sum() / 1e8

    return stats
