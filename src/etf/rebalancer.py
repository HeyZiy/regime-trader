# -*- coding: utf-8 -*-
"""
===================================
ETF 再平衡引擎
===================================

职责：
1. 中性基准 → 目标配比（无偏移，动态择时已删除，见 strategy/etf_allocation.md）
2. 比较 mx-moni 实际持仓 vs 目标 → 生成调仓指令（旧钱唯一动作）
3. 执行由 etf_observe --execute 统一批次完成（本模块只出指令）

新钱投放参考（全局节奏 + 逐标的节奏）由 etf_observe 报告生成，仅建议不自动执行。
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.etf.config import (
    AssetAllocation, AssetType, NEUTRAL_BASELINE, PROTECTED_TYPES,
    get_rebalance_threshold, get_rotation_universe_codes,
    MIN_TRADE_DEVIATION, REBALANCE_TOTAL_THRESHOLD,
)
from src.mx.client import MXMoniClient

logger = logging.getLogger(__name__)


@dataclass
class RebalanceOrder:
    code: str
    name: str
    action: str        # "buy" 或 "sell"
    amount: float      # 交易金额（元）
    quantity: int       # 交易股数
    current_pct: float  # 当前占比
    target_pct: float   # 目标占比
    reason: str         # 原因


class ETFRebalancer:
    """ETF 再平衡引擎"""

    def __init__(self, mx_client: MXMoniClient = None, fetcher=None):
        self.mx_client = mx_client or MXMoniClient()
        # 数据 manager 由入口层注入（用于 _fetch_etf_price 的实时行情回退）；
        # 为 None 时该回退跳过，不自行构造，避免分析模块各自重复初始化数据源
        self._fetcher = fetcher
        self.baseline = NEUTRAL_BASELINE

    # ── 计算目标配比 ──

    def calculate_target(self) -> Dict[str, float]:
        """目标配比 = 中性基准权重（无偏移；估值观点在基准里，趋势信息只影响阈值松紧）

        Returns:
            {code: target_weight} 目标权重（0.0 ~ 1.0）
        """
        return {a.code: a.neutral_weight for a in self.baseline}

    # ── 比较持仓 ──

    def _fetch_etf_price(self, code: str) -> float:
        """获取 ETF 日线收盘价（盘后分析用，数据稳定）"""
        try:
            from data_provider.bars import get_etf_daily
            df = get_etf_daily(code)
            if df is not None and not df.empty:
                return float(df.iloc[-1]['close'])
        except Exception as e:
            logger.debug(f"get_etf_daily 获取 {code} 价格失败: {e}")
        # 回退：入口注入的 manager 实时行情（未注入则跳过）
        if self._fetcher is not None:
            try:
                quote = self._fetcher.get_realtime_quote(code)
                if quote and hasattr(quote, 'price') and quote.price:
                    return float(quote.price)
            except Exception:
                pass
        return 0.0

    def _fill_missing_prices(self, current: Dict[str, dict]):
        """为未持仓的 ETF 补齐行情价格"""
        needed = [
            a.code for a in self.baseline
            if a.code != "CASH"
            and (a.code not in current or (current[a.code].get("current_price", 0) or 0) <= 0)
        ]
        if not needed:
            return
        logger.info(f"获取 {len(needed)} 只 ETF 的行情价格...")
        for code in needed:
            price = self._fetch_etf_price(code)
            if price > 0:
                if code not in current:
                    current[code] = {
                        "market_value": 0.0, "current_pct": 0.0,
                        "count": 0, "name": "", "current_price": price,
                    }
                else:
                    current[code]["current_price"] = price
                logger.debug(f"{code} 价格: {price:.3f}")

    def _build_current_map(self, positions: List[dict], capital: float) -> Dict[str, dict]:
        """将持仓列表转为 {code: {market_value, current_pct}}（占比按给定资金口径）"""
        result = {}
        for p in positions:
            code = p.get("code", "")
            mv = float(p.get("market_value", 0) or 0)
            pct = mv / capital if capital > 0 else 0.0
            result[code] = {
                "market_value": mv,
                "current_pct": pct,
                "count": p.get("count", 0),
                "current_price": p.get("current_price", 0),
                "name": p.get("name", ""),
            }
        return result

    def split_rotation_positions(self, positions: List[dict]) -> Tuple[List[dict], float, List[dict]]:
        """拆分核心持仓与卫星（非核心）持仓。

        卫星持仓独立预算，不参与核心仓偏离计算；
        核心资金 = 总资产 − 卫星持仓市值。

        Returns:
            (core_positions, rotation_mv, rotation_positions)
        """
        rotation_codes = get_rotation_universe_codes()
        core_positions, rotation_positions = [], []
        rotation_mv = 0.0
        for p in positions:
            if p.get("code", "") in rotation_codes:
                rotation_mv += float(p.get("market_value", 0) or 0)
                rotation_positions.append(p)
            else:
                core_positions.append(p)
        return core_positions, rotation_mv, rotation_positions

    def compare(self, target: Dict[str, float], positions: List[dict],
                total_assets: float, gate_state: str, hard_intercept: bool) -> Tuple[List[RebalanceOrder], float]:
        """比较目标 vs 实际，生成调仓指令（旧钱唯一动作：阈值再平衡）

        资金口径：核心资金 = 总资产（资金占比即总占比）。卫星仓与其他账户持仓
        不在基准代码集内，不产生偏离；其资金被"现金（以及其他账户）"桶吸收。

        Returns:
            (orders, total_deviation) 调仓指令列表 + 总偏离度
        """
        core_positions, rotation_mv, _ = self.split_rotation_positions(positions)
        current = self._build_current_map(positions, total_assets)
        # 补齐未持仓 ETF 的行情价格
        self._fill_missing_prices(current)
        threshold = get_rebalance_threshold(gate_state)
        orders: List[RebalanceOrder] = []
        total_deviation = 0.0

        for asset in self.baseline:
            code = asset.code
            if code == "CASH":
                continue
            target_pct = target.get(code, 0.0)
            cur = current.get(code, {"current_pct": 0.0, "market_value": 0.0, "count": 0, "current_price": 0.0, "name": asset.name})
            cur_pct = cur["current_pct"]
            deviation = target_pct - cur_pct
            total_deviation += abs(deviation)

            if abs(deviation) < MIN_TRADE_DEVIATION:
                continue

            amount = deviation * total_assets
            cur_price = cur.get("current_price", 0) or 0

            if deviation > 0:
                # 需要加仓
                quantity = self._round_lot(abs(amount) / cur_price, "buy") if cur_price > 0 else 0
                if quantity >= 100:
                    orders.append(RebalanceOrder(
                        code=code, name=asset.name, action="buy",
                        amount=abs(amount), quantity=quantity,
                        current_pct=cur_pct, target_pct=target_pct,
                        reason=f"低于目标{deviation*100:.1f}%"
                    ))
            else:
                # 需要减仓
                if gate_state in ("trending_down", "chaos") or hard_intercept:
                    if asset.asset_type in PROTECTED_TYPES:
                        continue  # 黄金/债券不减
                cur_count = cur.get("count", 0) or 0
                quantity = self._round_lot(min(abs(int(abs(amount) / cur_price)) if cur_price > 0 else 0, cur_count), "sell")
                if quantity >= 100:
                    orders.append(RebalanceOrder(
                        code=code, name=asset.name, action="sell",
                        amount=abs(amount), quantity=quantity,
                        current_pct=cur_pct, target_pct=target_pct,
                        reason=f"高于目标{abs(deviation)*100:.1f}%"
                    ))

        # 按 volatility_rank 排序（卖出优先高波动，买入优先低波动）
        vol_map = {a.code: a.volatility_rank for a in self.baseline}
        sells = sorted([o for o in orders if o.action == "sell"], key=lambda o: vol_map.get(o.code, 99), reverse=True)
        buys = sorted([o for o in orders if o.action == "buy"], key=lambda o: vol_map.get(o.code, 99))
        orders = sells + buys

        return orders, total_deviation

    # ── 判断是否触发再平衡 ──

    def should_rebalance(self, orders: List[RebalanceOrder], total_deviation: float,
                         gate_state: str) -> Tuple[bool, str]:
        """判断是否应该执行再平衡"""
        if not orders:
            return False, "无偏离，无需再平衡"

        threshold = get_rebalance_threshold(gate_state)
        if total_deviation > REBALANCE_TOTAL_THRESHOLD:
            return True, f"总偏离度{total_deviation*100:.1f}% > 强制阈值{REBALANCE_TOTAL_THRESHOLD*100:.0f}%"

        if any(abs(o.target_pct - o.current_pct) > threshold for o in orders):
            return True, f"存在单类偏离 > {threshold*100:.0f}%"

        return False, f"偏离在阈值{threshold*100:.0f}%以内"

    # ── 工具函数 ──

    @staticmethod
    def _round_lot(qty: float, action: str) -> int:
        lot = int(qty // 100) * 100
        return max(100, lot)
