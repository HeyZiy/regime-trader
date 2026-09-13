# -*- coding: utf-8 -*-
"""
===================================
ETF 长期配置 — 中性基准
===================================

中性基准反映长期信念，半年至一年审视一次。每日 gate 状态只在基准上做战术偏移，
不改变基准本身。

类别归属：
  equity    — 权益类（进攻弹性来源）
  bond      — 债券类（安全垫）
  gold      — 黄金（极端风险对冲，不减仓）
  cash      — 现金/货币（弹药 + 流动性缓冲）
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List

# ── 类别枚举 ──

class AssetType:
    EQUITY = "equity"
    BOND = "bond"
    GOLD = "gold"
    CASH = "cash"


@dataclass
class AssetAllocation:
    code: str           # 证券代码（6 位）
    name: str           # 名称
    asset_type: str     # AssetType
    neutral_weight: float  # 中性基准权重（0.0 ~ 1.0）
    volatility_rank: int   # 波动率排名（1=最高波动，用于减仓优先级）


# ── 中性基准配置 ──
# 核心仓位：长期持有，来源=有知有行基准（适配规则见 strategy/etf_allocation.md 第二节），
# 半年人工对齐一次。权重为总资产占比；"现金"桶吸收卫星仓与其他账户的资金。
# 卫星仓（行业动量轮动，标的集见 get_rotation_universe_codes）独立于核心基准，
# 动态扫描 ETF_INDUSTRY_MAP，其持仓被"现金（以及其他账户）"桶吸收，不产生核心偏离。

CORE_BASELINE: List[AssetAllocation] = [
    # ── A股宽基 ──
    AssetAllocation("563360", "A500ETF",                AssetType.EQUITY, 0.17, 8),
    AssetAllocation("159680", "中证1000增强ETF",         AssetType.EQUITY, 0.01, 6),
    AssetAllocation("515180", "红利ETF",                AssetType.EQUITY, 0.14, 10),
    # ── 海外 ──
    AssetAllocation("513100", "纳指ETF",                AssetType.EQUITY, 0.05, 9),
    AssetAllocation("513500", "标普500ETF",              AssetType.EQUITY, 0.05, 9),
    AssetAllocation("513380", "恒生科技ETF",            AssetType.EQUITY, 0.10, 5),
    # ── 行业/主题 ──
    AssetAllocation("159938", "医药ETF",                AssetType.EQUITY, 0.04, 4),
    AssetAllocation("516560", "养老ETF",                AssetType.EQUITY, 0.02, 7),
    AssetAllocation("159928", "消费ETF",                AssetType.EQUITY, 0.08, 7),
    # ── 黄金 ──
    AssetAllocation("159934", "黄金ETF",                AssetType.GOLD,   0.05, 11),
    # ── 现金（国债逆回购，自动理财，不买货基） ──
    AssetAllocation("CASH",   "现金/逆回购",              AssetType.CASH,   0.29, 13),
]

# 再平衡模块使用核心仓位
NEUTRAL_BASELINE = CORE_BASELINE

# ── 核心仓 ETF 跟踪指数（估值锚对准买入标的本身） ──
# 口径 = 中证指数官网估值（PE/股息率，见 amazing_factors.get_csindex_valuation），
# 指数代码已逐只经官网实测核对（2026-09）。海外 ETF（513100/513500/513380）无 csindex
# 数据；黄金为无现金流资产，不适用估值锚。红利类估值以股息率为主锚（股息是现金流本体）。
TRACKED_INDEX: Dict[str, str] = {
    "563360": "000510",  # A500ETF → 中证A500指数
    "159680": "000852",  # 中证1000增强ETF → 中证1000指数
    "515180": "000922",  # 红利ETF → 中证红利指数
    "159938": "000991",  # 医药ETF → 中证全指医药卫生指数
    "516560": "399812",  # 养老ETF → 中证养老产业指数
    "159928": "000932",  # 消费ETF → 中证主要消费指数
}

# 股息策略类 ETF：估值以股息率为主锚（股息是现金流本体，PE/全市场口径易反向），
# 新钱节奏用"股息率 − 10Y 国债利差"做加速判定（分位历史积累后可切换分位口径）。
DIVIDEND_STYLE_CODES: FrozenSet[str] = frozenset({"515180"})

# 减仓优先级：按 volatility_rank 从高到低（创业板先减，国债/现金后减）
# gold 和 bond 在 trending_down/chaos 时不减
PROTECTED_TYPES = {AssetType.GOLD, AssetType.BOND}

# 再平衡阈值——触发与执行分层的两段式设计：
# 触发层（should_rebalance）：单类偏离 > 分状态阈值（get_rebalance_threshold）
#   或总偏离 > 15%，决定整批是否执行；
# 执行层（compare）：整批一起修，单笔只按 MIN_TRADE_DEVIATION 过滤碎股。
# 分状态阈值不作用于单笔订单：上行期 3~5% 的漂移保留（让盈利奔跑），扳机扣下才随批修齐。
REBALANCE_SINGLE_THRESHOLD = 0.05     # sideways 等其余状态的触发阈值
REBALANCE_TOTAL_THRESHOLD = 0.15      # 所有偏离绝对值之和 > 15% 强制触发（无视状态）
REBALANCE_LOOSE_THRESHOLD = 0.08      # trending_up/weak_up 时的放宽阈值
REBALANCE_TIGHT_THRESHOLD = 0.03      # trending_down/chaos 时的收紧阈值
MIN_TRADE_DEVIATION = 0.02            # 碎股偏差过滤：执行批内 < 2% 的不修


def get_neutral_baseline() -> List[AssetAllocation]:
    return NEUTRAL_BASELINE


def get_rotation_universe_codes() -> set:
    """卫星仓（非核心）标的代码集（剔除核心基准代码，避免与核心仓资金口径重叠）。

    核心仓再平衡以"核心资金 = 总资产 − 卫星持仓市值"为口径，
    卫星标的独立预算、独立进出，不参与核心偏离计算。
    """
    try:
        from src.etf.amazing_factors import ETF_INDUSTRY_MAP
    except Exception:
        return set()
    baseline_codes = {a.code for a in CORE_BASELINE}
    return set(ETF_INDUSTRY_MAP) - baseline_codes



def get_rebalance_threshold(gate_state: str) -> float:
    """根据 gate 状态返回再平衡阈值（gate 的残值：只选阈值松紧，不产生仓位动作）"""
    if gate_state in ("trending_up", "weak_up"):
        return REBALANCE_LOOSE_THRESHOLD
    if gate_state in ("trending_down", "chaos"):
        return REBALANCE_TIGHT_THRESHOLD
    return REBALANCE_SINGLE_THRESHOLD

