# -*- coding: utf-8 -*-
"""
===================================
市场环境开仓门控模块
===================================

职责：
1. fetch_index_df(): 【入口层调用】取上证指数日线——AmazingData 单源（K 线 + 快照补
   当日 bar + 数据日期断言），与个股主源同源；无 akshare 回退
2. diagnose_regime(): 根据均线结构判断市场状态（5 级）+ 可解释诊断
3. check_market_gate(): 纯判定——市场状态决定能否开仓

市场状态（纯结构口径，与 research/trend_bt 回测完全一致）：
trending_down > trending_up > sideways > weak_up > chaos（判定优先级）
只有 trending_up / weak_up 允许开仓（REGIME_CAN_OPEN）。


"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

import pandas as pd

from data_provider.types import DataFetchError

logger = logging.getLogger(__name__)

# 状态判定的命中路径文案（与 diagnose_regime 的判定顺序一一对应，供报告诊断用）
REGIME_PATHS = {
    "trending_down": "① 均线空头排列(MA5<MA10<MA20) + 收盘<MA10",
    "trending_up":   "② 均线多头排列(MA5>MA10>MA20) + 收盘>MA10",
    "sideways":      "③ 收盘紧贴MA20（偏离<1.5%）",
    "weak_up":       "④ 收盘>MA20（非标准多头排列）",
    "chaos":         "⑤ 收盘<MA20 的乱序区（或数据不足）",
}

SIDEWAYS_BIAS = 0.015  # sideways 判定阈值：收盘偏离 MA20 < 1.5%

# 可开仓的市场状态（对应 strategy/trend_strategy.md「市场状态分级与响应动作」）；
# trending_down（均线空头）、chaos（收盘<MA20 乱序）与 sideways（横盘）禁止开新仓。
REGIME_CAN_OPEN = ("trending_up", "weak_up")


def _n(value: Optional[float]) -> Optional[float]:
    """float 统一 2 位小数（报告口径，避免各处精度不一）。"""
    return None if value is None else round(float(value), 2)


def _alignment_text(ma5: float, ma10: float, ma20: float) -> str:
    """均线排列描述，如 'MA5>MA10>MA20（多头排列）'。"""
    order = sorted((("MA5", ma5), ("MA10", ma10), ("MA20", ma20)),
                   key=lambda kv: kv[1], reverse=True)
    seq = ">".join(name for name, _ in order)
    if seq == "MA5>MA10>MA20":
        label = "（多头排列）"
    elif seq == "MA20>MA10>MA5":
        label = "（空头排列）"
    else:
        label = "（交织，无明确排列）"
    return f"{seq}{label}"


@dataclass
class RegimeDiagnosis:
    """市场状态判定的诊断信息（回答"为什么判成这个状态"）。

    Attributes:
        regime: 判定结果
        ma5/ma10/ma20/close: 判定所依据的均线与收盘（均为 2 位小数）
        bias_ma20: 收盘偏离 MA20 的百分比（正=在 MA20 上方）
        alignment: 均线排列描述
        path: 命中的判定路径文案
        note: 补充说明（如数据不足）
        data_date: 判定所用数据的最新交易日（P0 验收：日报强制展示，杜绝 T-1 静默流入）
    """
    regime: str
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    close: Optional[float] = None
    bias_ma20: Optional[float] = None
    alignment: str = "数据不足，无法判定"
    path: str = ""
    note: str = ""
    data_date: str = ""

    @property
    def available(self) -> bool:
        return self.ma20 is not None and self.close is not None

    def describe(self) -> str:
        """人类可读的诊断行（供日志与报告共用）。"""
        date_part = f"｜数据日期 {self.data_date}" if self.data_date else ""
        if not self.available:
            return f"状态={self.regime}｜均线数据不足，无法给出排列与偏离（{self.note}）{date_part}"
        return (
            f"状态={self.regime}｜{self.alignment}｜"
            f"收盘{self.close} 偏离MA20 {self.bias_ma20:+.2f}%｜命中：{self.path}{date_part}"
        )


def diagnose_regime(index_df) -> RegimeDiagnosis:
    """根据指数均线结构判断市场状态，并给出可解释的诊断信息。

    判定优先级：trending_down > trending_up > sideways > weak_up > chaos

    Returns:
        RegimeDiagnosis：含 regime、MA5/MA10/MA20 排列、偏离 MA20 百分比、命中路径
        trending_up   — 均线多头排列 + 收盘在 MA10 上方
        trending_down — 均线空头排列 + 收盘在 MA10 下方
        sideways      — 收盘紧贴 MA20（偏离 < 1.5%）
        weak_up       — 收盘在 MA20 上方，但非明确多头
        chaos         — 收盘 < MA20 的乱序区，或数据不足
    """
    try:
        if index_df is None or len(index_df) < 20:
            return RegimeDiagnosis("chaos", path=REGIME_PATHS["chaos"],
                                   note="指数日线缺失或不足20条")
        ma5 = index_df['close'].rolling(5).mean().iloc[-1]
        ma10 = index_df['close'].rolling(10).mean().iloc[-1]
        ma20 = index_df['close'].rolling(20).mean().iloc[-1]
        close = index_df['close'].iloc[-1]
        if any(pd.isna(x) for x in [ma5, ma10, ma20]):
            return RegimeDiagnosis("chaos", path=REGIME_PATHS["chaos"],
                                   note="MA5/MA10/MA20 存在 NaN")

        ma5, ma10, ma20, close = float(ma5), float(ma10), float(ma20), float(close)
        bias_ma20 = (close - ma20) / ma20 * 100 if ma20 > 0 else 0.0
        data_date = ""
        try:
            data_date = str(pd.to_datetime(index_df['date'].iloc[-1]).date())
        except Exception:
            pass
        base = dict(
            ma5=_n(ma5), ma10=_n(ma10), ma20=_n(ma20), close=_n(close),
            bias_ma20=_n(bias_ma20), alignment=_alignment_text(ma5, ma10, ma20),
            data_date=data_date,
        )

        # ① trending_down — 均线空头，最高优先级
        if ma5 < ma10 < ma20 and close < ma10:
            return RegimeDiagnosis("trending_down", path=REGIME_PATHS["trending_down"],
                                   note=f"收盘{_n(close)} < MA10 {_n(ma10)}", **base)

        # ② trending_up — 均线多头
        if ma5 > ma10 > ma20 and close > ma10:
            return RegimeDiagnosis("trending_up", path=REGIME_PATHS["trending_up"],
                                   note=f"收盘{_n(close)} > MA10 {_n(ma10)}", **base)

        # ③ sideways — 紧贴 MA20 震荡
        if abs(close - ma20) / ma20 < SIDEWAYS_BIAS:
            return RegimeDiagnosis("sideways", path=REGIME_PATHS["sideways"],
                                   note=f"偏离MA20 {_n(bias_ma20):+.2f}%", **base)

        # ④ weak_up — 在 MA20 上方，但不是标准多头排列
        if close > ma20:
            return RegimeDiagnosis("weak_up", path=REGIME_PATHS["weak_up"],
                                   note="收盘在MA20上方，均线非标准多头", **base)

        # ⑤ chaos — 收盘 < MA20 的乱序区
        return RegimeDiagnosis("chaos", path=REGIME_PATHS["chaos"],
                               note="收盘<MA20 且未满足以上任一结构", **base)
    except Exception as e:
        logger.warning(f"市场状态判定失败，降级为 chaos：{e}")
    return RegimeDiagnosis("chaos", path=REGIME_PATHS["chaos"], note="判定异常，降级处理")


def fetch_index_df() -> Optional[pd.DataFrame]:
    """取上证指数日线（市场状态判定的唯一数据源 = AmazingData，无回退源）。

    指数与个股主源统一走 AmazingData（服务器 cron 已配 TGW 凭证）——
    公开数据源（csindex/新浪日线盘后晚间更新、东财限流）无法保证收盘即含当日 bar。
    TGW 未配置（本地调试）或取数失败时返回 None → regime=chaos →
    当日不开仓：宁可不出信号，不拿过期数据做开仓判断。
    """
    try:
        from data_provider.fetchers.amazingdata_fetcher import AmazingDataFetcher

        fetcher = AmazingDataFetcher()
    except Exception as e:
        logger.warning(f"指数数据不可用（TGW 未配置或初始化失败）: {e}")
        return None

    try:
        return _build_index_df(fetcher)
    except DataFetchError as e:
        logger.error(f"指数日线获取失败: {e}")
        return None
    except Exception as e:
        logger.error(f"指数日线组装异常: {e}")
        return None


def _expected_latest_trade_date(now=None) -> Optional[date]:
    """预期指数最新交易日：交易日取当天，节假日取此前最近一个交易日。

    15:10 买入分析 / 14:45 卖出任务都在交易日盘中后段运行，
    因此交易日的"预期最新交易日"就是当天。
    """
    d = (now or datetime.now()).date()
    try:
        from src.trading_calendar import get_trading_dates

        dates = get_trading_dates(d - timedelta(days=15), d)
        return dates[-1] if dates else None
    except Exception as e:
        logger.warning(f"交易日历获取失败，跳过数据日期断言: {e}")
        return None


def _build_index_df(fetcher) -> Optional[pd.DataFrame]:
    """指数日线组装：K 线 + 快照补当日 bar + 数据日期断言。

    1. query_kline 拉指数日线；
    2. 若最后一根 bar 落后于预期交易日（当日 bar 未入库的已知缺口），
       用 query_snapshot 当日最后一笔快照（close/volume）补一根——
       盘中为最新近似（与尾盘 14:45 近似收盘口径一致），收盘后为官方值；
    3. 断言：补齐后仍落后 → 返回 None 并显式报错，杜绝 T-1 数据流入判定。
    """
    df = fetcher.get_index_daily("sh000001")
    if df is None or df.empty:
        return None

    expected = _expected_latest_trade_date()
    if expected is None:
        return df  # 日历不可用时跳过断言（fail-open），日期已在日志可见

    last_date = df["date"].iloc[-1].date()
    if last_date >= expected:
        return df

    # 当日 bar 缺失 → 用当日最后一笔快照补
    snap = fetcher.get_index_snapshot("sh000001", expected)
    if not snap or not snap.get("close"):
        logger.error(
            f"🔴 指数数据过期且无法补齐：K 线最新 {last_date} < 预期交易日 {expected}，"
            f"快照补齐失败 → 拒绝提供（regime 将降级 chaos，当日不开仓）"
        )
        return None

    close = float(snap["close"])
    bar = {
        "date": pd.Timestamp(expected),
        "open": float(snap.get("open") or close),
        "high": float(snap.get("high") or close),
        "low": float(snap.get("low") or close),
        "close": close,
        "volume": float(snap.get("volume") or 0),
        "amount": float(snap.get("amount") or 0),
    }
    df = pd.concat([df, pd.DataFrame([bar])], ignore_index=True)
    logger.info(
        f"当日 bar 缺失，已用 {snap.get('trade_time', expected)} 快照补齐 "
        f"{expected}（close={close}）"
    )

    if df["date"].iloc[-1].date() < expected:
        logger.error(
            f"🔴 指数数据过期：补齐后最新 {df['date'].iloc[-1].date()} < 预期 {expected} → 拒绝提供"
        )
        return None
    return df


def check_market_gate(index_df: Optional[pd.DataFrame]) -> Tuple[bool, str, str]:
    """市场状态判定 → 能否开仓（纯结构口径，与 research/trend_bt 回测完全一致）。

    Args:
        index_df: 上证指数日线（fetch_index_df() 的返回）

    Returns:
        can_trade — trending_up/weak_up 允许开仓，其余状态禁止
        summary   — 人类可读的判定摘要（日志用）
        regime    — 市场状态：trending_up | trending_down | sideways | weak_up | chaos
    """
    diag = diagnose_regime(index_df)
    regime = diag.regime
    can_trade = regime in REGIME_CAN_OPEN

    data_date = ""
    if index_df is not None and "date" in index_df.columns and len(index_df):
        data_date = f"｜数据日期 {index_df['date'].iloc[-1].date()}"

    if regime == "trending_down":
        # 均线空头时禁止开仓：空头结构下"高成交+高情绪"是下跌中继/放量出货的典型
        # 特征，不是反转信号。趋势策略坚持"底部偏右进场"，等收盘重回 MA20 再参与。
        action = "📉 均线空头排列，禁止开仓（等收盘重回MA20）"
    elif not can_trade:
        # sideways（横盘）/ chaos（收盘<MA20 乱序）：禁止开新仓；持仓卖出照常按正常版输出。
        action = f"🌪️ 状态{regime}：不开新仓（等方向明确）"
    else:
        action = "✅ 结构确认，允许开仓"

    summary = (
        f"市场状态判定：{regime} → {'✅ 允许开仓' if can_trade else '❌ 禁止开仓'}{data_date}\n"
        f"{action}\n{diag.describe()}"
    )

    if can_trade:
        logger.info(f"✅ 市场状态 {regime}，允许开仓")
    else:
        logger.warning(f"⛔ 市场状态 {regime}，不开新仓")

    return can_trade, summary, regime
