# -*- coding: utf-8 -*-
"""
===================================
多源编排（纯函数）
===================================
给定「需求 + 数据源集合」，做候选筛选与"取第一个非空"：
- supporting：按 BaseFetcher.SUPPORTS 能力声明筛候选，编排逻辑不出现数据源类名
- query_first：跨源 failover 的公共形状（市场统计 / 主力资金流 / 板块行情等直接用）

边界（与 realtime.py 一致）：
- 不持有 fetcher 实例、不碰 SDK：fetcher 集合由调用方传入
- 不含单源知识、不含日线/实时等策略语义（那些在 daily.py / realtime.py）
"""
import logging
from typing import Any, List, Optional

import pandas as pd

from .types import Need

logger = logging.getLogger(__name__)


def supporting(fetchers: List[Any], need: Need) -> List[Any]:
    """按能力声明筛出满足该需求的候选数据源（保持传入顺序，即优先级顺序）。"""
    return [f for f in fetchers if f.supports(need)]


def is_empty(result: Any) -> bool:
    """判定取数结果是否为空：None / 空 DataFrame / 空 dict。"""
    if result is None:
        return True
    if isinstance(result, pd.DataFrame):
        return result.empty
    if isinstance(result, dict):
        return not result
    return False


def query_first(label: str, fetchers: List[Any], method: str, *args,
                need: Optional[Need] = None, **kwargs) -> Optional[Any]:
    """遍历数据源，返回第一个非空结果（多源 failover 的公共形状）。

    Args:
        label: 日志标签（建议带标的，如 "主力资金流 603175"）
        fetchers: 已实例化的数据源列表（按优先级）
        method: 要调用的数据源方法名
        need: 传则先按能力声明筛候选；不传则遍历全部源（如市场统计这类不限市场的接口）

    Returns:
        首个非空结果；全部无数据返回 None（该源抛异常只记 debug 并继续，
        异常细节由数据源自己记录）
    """
    candidates = supporting(fetchers, need) if need is not None else fetchers
    for fetcher in candidates:
        try:
            result = getattr(fetcher, method)(*args, **kwargs)
        except Exception as e:
            logger.debug(f"[{label}] [{fetcher.name}] 失败: {e}")
            continue
        if not is_empty(result):
            logger.info(f"[{label}] 成功获取 (来源: {fetcher.name})")
            return result
    return None
