"""网格接网试点回测骨架（Grid Pilot）。

规格文档: strategy/grid_pilot.md（2026-09-22 规格冻结）
状态: 骨架，未验证；估值分位数据源 TODO；禁止回测后改参数，改参数 = 规格修订。

四组对照: always-on 网格 / 带开关网格 / 买入持有 / 空仓
用法:
    python bt_grid_switch.py --code 510300 --start 2013-01-01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 冻结参数（来自规格第二节/第三节；改动须走规格修订）
# ---------------------------------------------------------------------------
PARAMS = {
    "net_step": 0.05,            # 网距 5%（扫描 {0.03, 0.05, 0.08}）
    "cap_nets": 2,               # 最大超买网数（扫描 {1, 2, 3}）
    "sleeve_pct": 0.20,          # 现金预算 = 持仓市值 × 20%（每网 = 持仓 10%）
    "adx_on": 25.0,              # 闸2: ADX < 25 允许
    "adx_off": 30.0,             # 闸2: ADX > 30 关闭
    "adx_period": 14,
    "bbw_period": 20,            # 闸3: 布林带宽窗口
    "bbw_pctile_win": 500,       # 闸3: 分位窗口（约两年）
    "bbw_pctile_th": 0.30,       # 闸3: 分位阈值
    "confirm_days": 3,           # 确认期（扫描 {3, 5}）
    "ma_hard": 60,               # 硬风控: MA60
    "break_buf": 0.03,           # 硬风控: 近 60 日最低收盘 × (1-3%)
    "fee_rate": 0.0001,          # 免5 佣金万 1（已确认免 5）
    "slippage_bp": 2,            # 滑点 2bp
}


# ---------------------------------------------------------------------------
# 数据加载（TODO: 估值分位数据源待定，见规格第八节）
# ---------------------------------------------------------------------------
def load_etf_daily(code: str, start: str) -> pd.DataFrame:
    """加载 ETF 日线（前复权）。列: open/high/low/close/volume，index=DatetimeIndex。

    TODO: 接入 data_provider（akshare fund_etf_hist_em / baostock），
    与 data_provider/daily.py 的复权口径保持一致并在台账记录。
    """
    raise NotImplementedError("接入 data_provider 后替换此桩")


def load_pe_percentile(index_code: str, window_years: int = 10) -> pd.Series:
    """标的指数 PE 十年滚动分位（闸1）。返回 0~1 的日度分位序列。

    TODO: 数据源待定（akshare 指数估值 / 中证官网 / 理杏仁）。
    确定口径后在此记录: 指数代码、PE 口径（静态/TTM）、分位窗口。
    """
    raise NotImplementedError("估值分位数据源待定")


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ADX。只测趋势强度，不测方向。"""
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def bbw_percentile(df: pd.DataFrame, n: int, win: int) -> pd.Series:
    """布林带宽 (上轨-下轨)/中轨 的滚动分位（0~1）。"""
    mid = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std()
    width = (4 * std) / mid
    return width.rolling(win).apply(lambda x: (x[-1] <= x).mean(), raw=True)


# ---------------------------------------------------------------------------
# 开关状态机（三道闸 + 确认期 + 硬风控）
# ---------------------------------------------------------------------------
@dataclass
class Switch:
    """输出 state ∈ {ON, OFF} 与 transition 事件。允许=可接新网；关闭=只还网不接网。"""

    params: dict
    state: str = "OFF"
    _cand: str = field(default="OFF")
    _cand_days: int = 0

    def update(self, row: pd.Series, adx_v: float, bbw_p: float, pe_p: float | None) -> str:
        """每根日线收盘调用一次；返回当日状态转换（无转换为 ''）。"""
        p = self.params
        # 硬风控（最高优先级，立即生效，无确认期）
        hard = row["close"] < row["ma60"] or row["close"] < row["low60"] * (1 - p["break_buf"])
        if hard:
            if self.state != "OFF":
                self.state, self._cand, self._cand_days = "OFF", "OFF", 0
                return "HARD_OFF"
            return ""

        # 闸1 估值否决: 禁止新开（不强制关已开，交由闸2/3管理）
        gate1_ok = pe_p is None or pe_p <= 0.80
        # 闸2 趋势强度（滞回）
        if adx_v < p["adx_on"]:
            gate2 = True
        elif adx_v > p["adx_off"]:
            gate2 = False
        else:
            gate2 = self.state == "ON" or self._cand == "ON"
        # 闸3 波动收敛
        gate3 = bbw_p is not None and bbw_p < p["bbw_pctile_th"]

        want = "ON" if (gate1_ok and gate2 and gate3) else "OFF"
        if want == self.state:
            self._cand, self._cand_days = self.state, 0
            return ""
        if want == self._cand:
            self._cand_days += 1
        else:
            self._cand, self._cand_days = want, 1
        if self._cand_days >= p["confirm_days"]:
            self.state = self._cand
            self._cand_days = 0
            return f"TURN_{self.state}"
        return ""


# ---------------------------------------------------------------------------
# 网格阶梯（模式一·下阶梯变体）
# ---------------------------------------------------------------------------
@dataclass
class GridLadder:
    """核心仓不动；本层仅管理超额定份额: 跌接网、涨还网，cap 封顶。

    reference: 开关转 ON 当日收盘价（重启重置）。
    买入: reference × (1 - step×k)，k=1..cap；卖出: 该网买入价 × (1 + step)。
    """

    params: dict
    cash: float = 0.0
    net_value: float = 0.0     # 每网金额（按持仓市值×10% 在初始化时设定）
    reference: float = 0.0
    lots: list = field(default_factory=list)  # 每手买入价

    def activate(self, ref_close: float, net_value: float, cash: float) -> None:
        self.reference, self.net_value, self.cash = ref_close, net_value, cash

    def on_bar(self, open_px: float) -> tuple[list, list]:
        """T+1: 信号次日开盘价成交。返回 (buys, sells)，元素为 (价格, 金额)。"""
        p, buys, sells = self.params, [], []
        step = p["net_step"]
        # 卖出: 任意持仓手满足 现价 >= 买入价×(1+step)
        while self.lots and open_px >= min(self.lots) * (1 + step):
            px = min(self.lots) * (1 + step)
            self.lots.remove(min(self.lots))
            amt = self.net_value * (1 - p["fee_rate"] - p["slippage_bp"] / 1e4)
            self.cash += amt
            sells.append((px, amt))
        # 买入: 阶梯 + cap
        if self.reference > 0:
            k = len(self.lots) + 1
            if k <= p["cap_nets"]:
                buy_px = self.reference * (1 - step * k)
                if open_px <= buy_px:
                    amt = self.net_value * (1 + p["fee_rate"] + p["slippage_bp"] / 1e4)
                    if self.cash >= amt:
                        self.cash -= amt
                        self.lots.append(open_px)
                        buys.append((open_px, amt))
        return buys, sells

    def force_clear(self, open_px: float) -> float:
        """硬风控: 全部清仓（可能亏损），现金回池。"""
        for _ in self.lots:
            self.cash += self.net_value * (1 - p_fee(self.params) )
        self.lots.clear()
        return self.cash


def p_fee(params: dict) -> float:
    return params["fee_rate"] + params["slippage_bp"] / 1e4


# ---------------------------------------------------------------------------
# 四组对照
# ---------------------------------------------------------------------------
def run_switch_grid(df: pd.DataFrame, params: dict) -> pd.Series:
    """带开关网格: 状态机 + 阶梯；核心仓价值用买入持有曲线表示，网格层损益叠加。"""
    raise NotImplementedError("骨架: 组合 Switch + GridLadder 逐日推进，输出网格层净值序列")


def run_always_on_grid(df: pd.DataFrame, params: dict) -> pd.Series:
    """always-on 网格: 同一阶梯，无开关，全程运行。"""
    raise NotImplementedError("骨架")


def run_buy_hold(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].iloc[0]


# ---------------------------------------------------------------------------
# 指标与硬门槛检查
# ---------------------------------------------------------------------------
def max_drawdown(nav: pd.Series) -> float:
    return (nav / nav.cummax() - 1).min()


def worst_month(nav: pd.Series) -> float:
    m = nav.resample("M").last().pct_change()
    return m.min()


def annual_return(nav: pd.Series) -> float:
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return (nav.iloc[-1]) ** (1 / years) - 1


def safety_gates(sw_nav: pd.Series, bh_nav: pd.Series, ao_nav: pd.Series,
                 switches_per_year: float, params: dict) -> dict:
    """规格第五节安全硬门槛。返回 {gate: bool}，全 True 才通过。"""
    seg = lambda nav, a, b: nav.loc[a:b]  # 2018-01-01~2018-12-31, 2022-01-01~2022-12-31
    return {
        "bear2018_dd_le_bh": True,   # TODO: max_drawdown(seg(sw,'2018','2018')) >= max_drawdown(seg(bh,...))（回撤为负值，比较符号注意）
        "bear2022_dd_le_bh": True,
        "worst_month_le_alwayson": worst_month(sw_nav) >= worst_month(ao_nav),
        "no_lookahead": True,       # 代码审查项，人工确认
        "switches_per_year_le_8": switches_per_year <= 8,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="510300")
    ap.add_argument("--start", default="2013-01-01")
    args = ap.parse_args()
    print(f"[grid-pilot] code={args.code} start={args.start}")
    print("[grid-pilot] TODO: load_etf_daily / load_pe_percentile 数据源未接入，回测未可运行")
    print("[grid-pilot] 规格: strategy/grid_pilot.md（冻结）；改动参数 = 规格修订")


if __name__ == "__main__":
    main()
