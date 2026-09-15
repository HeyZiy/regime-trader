# -*- coding: utf-8 -*-
"""
===================================
股票代码规范化与市场/类型判定
===================================

纯函数，不碰网络。集中存放项目所有代码判定规则：
- normalize_stock_code / canonical_stock_code：代码归一化
- classify_market：市场归类 cn/hk/us（编排层按此匹配数据源能力声明）
- is_bse_code / is_st_stock / is_kc_cy_stock / is_hk_market / is_etf_code：市场/类型判定
- ETF_PREFIXES：ETF 码族常量
- _split_prefix：sh/sz/bj 前缀拆分

个股日线多源与实时报价见 manager.py / realtime.py；ETF/指数日线见 bars.py。
"""
import re
from typing import Optional, Tuple

def normalize_stock_code(stock_code: str) -> str:
    """
    Normalize stock code by stripping exchange prefixes/suffixes.

    Accepted formats and their normalized results:
    - '600519'      -> '600519'   (already clean)
    - 'SH600519'    -> '600519'   (strip SH prefix)
    - 'SZ000001'    -> '000001'   (strip SZ prefix)
    - 'BJ920748'    -> '920748'   (strip BJ prefix, BSE)
    - 'sh600519'    -> '600519'   (case-insensitive)
    - '600519.SH'   -> '600519'   (strip .SH suffix)
    - '000001.SZ'   -> '000001'   (strip .SZ suffix)
    - '920748.BJ'   -> '920748'   (strip .BJ suffix, BSE)
    - 'HK00700'     -> 'HK00700'  (keep HK prefix for HK stocks)
    - '1810.HK'     -> 'HK01810'  (normalize HK suffix to canonical prefix form)
    - 'AAPL'        -> 'AAPL'     (keep US stock ticker as-is)

    This function is applied at the DataProviderManager layer so that
    all individual fetchers receive a clean 6-digit code (for A-shares/ETFs).
    """
    code = stock_code.strip()
    upper = code.upper()

    # Normalize HK prefix to a canonical 5-digit form (e.g. hk1810 -> HK01810)
    if upper.startswith('HK') and not upper.startswith('HK.'):
        candidate = upper[2:]
        if candidate.isdigit() and 1 <= len(candidate) <= 5:
            return f"HK{candidate.zfill(5)}"

    # Strip SH/SZ/BJ prefix (e.g. SH600519 -> 600519, BJ920748 -> 920748)
    if upper.startswith(('SH', 'SZ', 'BJ')) and not upper.startswith(('SH.', 'SZ.', 'BJ.')):
        candidate = code[2:]
        if candidate.isdigit() and len(candidate) in (5, 6):
            return candidate

    # Strip .SH/.SZ/.BJ suffix (e.g. 600519.SH -> 600519, 920748.BJ -> 920748)
    if '.' in code:
        base, suffix = code.rsplit('.', 1)
        if suffix.upper() == 'HK' and base.isdigit() and 1 <= len(base) <= 5:
            return f"HK{base.zfill(5)}"
        if suffix.upper() in ('SH', 'SZ', 'BJ') and base.isdigit():
            return base

    return code


ETF_PREFIXES = ("51", "52", "56", "58", "15", "16", "18")



def is_hk_market(code: str) -> bool:
    """
    判定是否为港股代码。

    支持 `HK00700` 及纯 5 位数字形式（A 股 ETF/股票常见为 6 位）。
    """
    normalized = (code or "").strip().upper()
    if normalized.endswith(".HK"):
        base = normalized[:-3]
        return base.isdigit() and 1 <= len(base) <= 5
    if normalized.startswith("HK"):
        digits = normalized[2:]
        return digits.isdigit() and 1 <= len(digits) <= 5
    if normalized.isdigit() and len(normalized) == 5:
        return True
    return False



def classify_market(code: str) -> str:
    """把代码归到 cn / hk / us 三类市场，供编排层做数据源能力匹配。

    判定口径与 DataFetcherManager 的路由一致：港股走 is_hk_market，
    美股（含美股指数）走 us_index_mapping，其余归 A 股（含 ETF/北交所）。
    """
    if is_hk_market(code):
        return "hk"
    from .us_index_mapping import is_us_index_code, is_us_stock_code
    if is_us_index_code(code) or is_us_stock_code(code):
        return "us"
    return "cn"


def is_bse_code(code: str) -> bool:
    """
    Check if the code is a Beijing Stock Exchange (BSE) A-share code.

    BSE rules:
    - Old format (pre-2024): 8xxxxx (e.g. 838163), 4xxxxx (e.g. 430047)
    - New format (2024+, post full migration Oct 2025): 920xxx+
    Note: 900xxx are Shanghai B-shares, NOT BSE — must return False.
    """
    c = (code or "").strip().split(".")[0]
    if len(c) != 6 or not c.isdigit():
        return False
    return c.startswith(("8", "4")) or c.startswith("92")

def is_st_stock(name: str) -> bool:
    """
    Check if the stock is an ST or *ST stock based on its name.

    ST stocks have special trading rules and typically a ±5% limit.
    """
    n = (name or "").upper()
    return 'ST' in n

def is_kc_cy_stock(code: str) -> bool:
    """
    Check if the stock is a STAR Market (科创板) or ChiNext (创业板) stock based on its code.

    - STAR Market: Codes starting with 688
    - ChiNext: Codes starting with 300
    Both have a ±20% limit.
    """
    c = (code or "").strip().split(".")[0]
    return c.startswith("688") or c.startswith("30")


def canonical_stock_code(code: str) -> str:
    """
    Return the canonical (uppercase) form of a stock code.

    This is a display/storage layer concern, distinct from normalize_stock_code
    which strips exchange prefixes. Apply at system input boundaries to ensure
    consistent case across BOT, WEB UI, API, and CLI paths (Issue #355).

    Examples:
        'aapl'    -> 'AAPL'
        'AAPL'    -> 'AAPL'
        '600519'  -> '600519'  (digits are unchanged)
        'hk00700' -> 'HK00700'
    """
    return (code or "").strip().upper()



def _split_prefix(code: str) -> Tuple[str, Optional[str]]:
    """返回 (numeric_code, market_prefix)。prefix 为 sh/sz/bj（无前缀时 None）。"""
    raw = (code or "").strip().lower()
    m = re.match(r"^(sh|sz|bj)([0-9]{6})$", raw)
    if m:
        return m.group(2), m.group(1)
    return raw, None


def is_etf_code(code: str) -> bool:
    """按码族判定是否为 ETF（51/52/56/58/15/16/18，码族→市场无歧义）。"""
    num, _ = _split_prefix(code)
    return num[:2] in ETF_PREFIXES


def is_us_stock_code(code: str) -> bool:
    """判定代码是否为美股股票（排除美股指数）。

    美股代码规则：
    - 1-5个大写字母，如 'AAPL', 'TSLA'
    - 可能包含 '.'，如 'BRK.B'
    """
    c = (code or "").strip().upper()
    return bool(re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', c))

