# -*- coding: utf-8 -*-
"""
===================================
AmazingDataFetcher - 星耀数智数据源
===================================

数据来源：中国银河证券 星耀数智 AmazingData 行情平台（基于 tgw 行情网关）
特点：
- 交易所直连数据，稳定性高，无爬虫封禁风险
- 支持 A股/ETF/可转债/期货/期权/港股通
- 依赖 TGW 账号（.env 配置 TGW_USERNAME/TGW_PASSWORD/TGW_HOST/TGW_PORT）

设计：
- 懒登录：首次使用时登录，登录失败抛出异常让 DataFetcherManager 切换
- 未配置 TGW 凭证时不注册（_init_default_fetchers 检查）
- 换手率由 _normalize_data 自算（volume ÷ 流通A股），指标计算已移出数据层
"""

import contextlib
import io
import logging
import os
import threading
from datetime import date as date_cls, timedelta
from pathlib import Path
from typing import ClassVar, Optional, Dict, Any

import numpy as np
import pandas as pd

from data_provider.fetchers.base import BaseFetcher
from data_provider.types import KIND_STOCK_DAILY, DataFetchError, STANDARD_COLUMNS
from data_provider.codes import normalize_stock_code

logger = logging.getLogger(__name__)

# 幂等加载 .env（项目入口已调用 setup_env 时无副作用；独立使用本模块也能读到凭证）
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# === 配置 ===
_TGW_HOST = os.getenv("TGW_HOST", "").strip()
_TGW_PORT = os.getenv("TGW_PORT", "").strip()
_TGW_USERNAME = os.getenv("TGW_USERNAME", "").strip()
_TGW_PASSWORD = os.getenv("TGW_PASSWORD", "").strip()


def tgw_configured() -> bool:
    """是否已配置 TGW 登录凭证。"""
    return bool(_TGW_HOST and _TGW_PORT and _TGW_USERNAME and _TGW_PASSWORD)


def _index_code_to_tgw_format(code: str) -> Optional[str]:
    """
    指数代码转换为 tgw 格式（sh000001 / 000001 -> 000001.SH）。

    指数与个股走独立通道：000001 在个股通道是平安银行(000001.SZ)，
    在指数通道是上证指数(000001.SH)——两者 tgw 代码本就不同市场后缀，
    但转换入口必须分开，避免裸码歧义导致误查个股 K 线。

    Returns:
        tgw 格式代码；不支持的代码返回 None
    """
    code = (code or "").strip().lower()
    if code.startswith(("sh", "sz", "bj")) and len(code) == 8:
        pref, num = code[:2], code[2:]
    else:
        # 裸码按确定规则推断：399 → 深证/国证指数，其余 → 上证指数
        pref, num = ("sz" if code.startswith("399") else "sh"), code
    if not (num.isdigit() and len(num) == 6):
        return None
    if pref == "bj":
        return None  # 北交所指数暂不支持
    return f"{num}.{pref.upper()}"


def _code_to_tgw_format(code: str) -> str:
    """
    将标准 6 位代码转换为 tgw 格式（600519 -> 600519.SH）。

    Returns:
        tgw 格式代码；不支持的代码返回 None
    """
    code = normalize_stock_code(code)
    if not code.isdigit() or len(code) != 6:
        return None
    if code.startswith(("6", "51", "52", "56", "58")):
        return f"{code}.SH"
    if code.startswith(("0", "3", "15", "16", "18")):
        return f"{code}.SZ"
    return None  # 北交所等其他市场暂不支持


# 项目根目录：本文件位于 <repo>/data_provider/fetchers/ 下，故 parents[2] 是仓库根
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_cache_dir() -> str:
    """InfoData（get_equity_structure 等）本地 HDF5 缓存目录。

    与 src/etf/amazing_factors.py 共用同一份缓存（同读 AMAZING_DATA_DIR），
    避免两处各自落盘、各自冷启动。相对路径按「项目根」解析 —— cron 以项目根为
    工作目录，但手动从别处运行或被 import 复用时 CWD 会变，直接用相对路径会解析到别处。

    返回值必须带尾部分隔符：SDK 把 local_path 当字符串前缀拼 "infodata/..."，
    少了分隔符会拼成 AmazingData_local_datainfodata/（文档示例亦为 'D://...//'）。
    """
    raw = (os.getenv("AMAZING_DATA_DIR") or "").strip()
    if raw:
        path = Path(raw)
        base = path if path.is_absolute() else _PROJECT_ROOT / path
    else:
        base = _PROJECT_ROOT / "data" / "AmazingData_local_data"
    return str(base) + os.sep


AMAZINGDATA_CACHE_DIR = _resolve_cache_dir()


class AmazingDataFetcher(BaseFetcher):
    """
    AmazingData 数据源实现

    优先级：-2（最高，需要 TGW 凭证，高于 Tushare 的 -1）
    数据来源：星耀数智行情平台（query_kline）
    """

    name = "AmazingDataFetcher"
    # 默认最高优先级（-2，高于 Tushare(-1)/Akshare(0)/Efinance(1)）；
    # 未配置凭证时不注册。失败时由 DataFetcherManager 自动切换到下一数据源
    priority = int(os.getenv("AMAZINGDATA_PRIORITY", "-2"))

    # query_kline 原生不返回换手率，故不声明 turnover_rate：主源成功时由 _normalize_data
    # 自算该列（不触发回退）；自算失败时回退循环本就跳过主源自身，直接走 akshare，
    # 不把本源当候选回退源可避免"整段日线重拉一遍再自算一次"的无效请求
    SUPPORTS_COLUMNS = {'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg'}

    # 仅 A 股日线（交易所直连，不覆盖港股/美股/北交所）
    SUPPORTS = frozenset({(KIND_STOCK_DAILY, "cn")})

    # 登录单例
    _login_lock = threading.Lock()
    _login_attempted = False
    _login_success = False
    _market_data = None
    _calendar = None

    def __init__(self):
        super().__init__()
        if not tgw_configured():
            raise DataFetchError(
                "AmazingDataFetcher 未配置 TGW 凭证（TGW_HOST/TGW_PORT/TGW_USERNAME/TGW_PASSWORD）"
            )

    # ---------- 登录管理 ----------

    @classmethod
    def _ensure_login(cls) -> None:
        """确保已登录 tgw 并拿到 MarketData 实例。"""
        with cls._login_lock:
            if cls._login_success and cls._market_data is not None:
                return
            if cls._login_attempted and not cls._login_success:
                raise DataFetchError("AmazingData 登录已尝试失败，跳过该数据源")

            cls._login_attempted = True
            try:
                # 登录会打印大量 logon json，临时静默
                with contextlib.redirect_stdout(io.StringIO()):
                    import AmazingData as ad

                    ad.login(
                        username=_TGW_USERNAME,
                        password=_TGW_PASSWORD,
                        host=_TGW_HOST,
                        port=int(_TGW_PORT),
                    )
                    base = ad.BaseData()
                    cal = base.get_calendar()
                    market = ad.MarketData(cal)
                cls._calendar = cal
                cls._market_data = market
                cls._login_success = True
                logger.info(f"AmazingData 登录成功，交易日历 {len(cal)} 天")
            except SystemExit:
                # ad.login 失败时会 sys.exit(0)，转为异常交给上层切换数据源
                cls._login_success = False
                raise DataFetchError("AmazingData 登录失败（账号/密码/网络），切换其他数据源")
            except Exception as e:
                cls._login_success = False
                raise DataFetchError(f"AmazingData 登录异常: {e}") from e

    @classmethod
    def ensure_login(cls) -> None:
        """公共登录入口（供其他模块复用登录态）。未配置凭证时不报错。"""
        if not tgw_configured():
            raise DataFetchError("AmazingData 未配置 TGW 凭证")
        cls._ensure_login()

    @classmethod
    def get_info_data(cls):
        """获取 InfoData 实例（复用已建立的登录态）。"""
        cls._ensure_login()
        with contextlib.redirect_stdout(io.StringIO()):
            import AmazingData as ad
            return ad.InfoData()

    # ---------- BaseFetcher 接口 ----------

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 AmazingData 获取日 K 数据。

        数据来源：MarketData.query_kline()
        返回原始列：code, kline_time, open, high, low, close, volume, amount
        """
        self._ensure_login()

        tgw_code = _code_to_tgw_format(stock_code)
        if tgw_code is None:
            raise DataFetchError(
                f"AmazingDataFetcher 不支持的代码 {stock_code}（仅支持沪深 A 股与 ETF）"
            )

        begin = int(start_date.replace("-", ""))
        end = int(end_date.replace("-", ""))

        logger.info(f"[API调用] query_kline({tgw_code}, {begin}~{end}, period=day)")
        try:
            from AmazingData.utils.constant import Period

            kline_dict = self._market_data.query_kline(
                [tgw_code],
                begin_date=begin,
                end_date=end,
                period=Period.day.value,
            )
        except SystemExit:
            raise DataFetchError("AmazingData 查询被中断")
        except Exception as e:
            raise DataFetchError(f"AmazingData query_kline 失败: {e}") from e

        df = kline_dict.get(tgw_code)
        if df is None or df.empty:
            raise DataFetchError(f"AmazingData 未返回 {tgw_code} 的数据")

        logger.info(f"[API返回] query_kline({tgw_code}) 成功: rows={len(df)}")
        return df

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 AmazingData 数据。

        AmazingData 列：code, kline_time, open, high, low, close, volume, amount
        映射到：date, open, high, low, close, volume, amount, pct_chg, turnover_rate
        """
        df = df.copy()
        df = df.rename(columns={"kline_time": "date"})

        if "date" not in df.columns:
            raise DataFetchError("AmazingData 数据缺少日期列")

        df["date"] = pd.to_datetime(df["date"])

        # 计算涨跌幅（数据源不直接提供）
        df["pct_chg"] = df["close"].pct_change() * 100

        # 自行计算换手率（volume ÷ 流通股本），避免主源缺失时回退到其他数据源二次拉取。
        # 计算失败不影响主流程，缺失列仍由 DataFetcherManager 的列回退兜底。
        try:
            tor = self._compute_turnover_rate(df, stock_code)
            if tor is not None:
                df["turnover_rate"] = tor
        except Exception as e:
            logger.warning(f"[AmazingData] {stock_code} 换手率计算失败，将依赖回退源补齐: {e}")

        # 只保留需要的列
        keep_cols = ["date"] + [c for c in STANDARD_COLUMNS if c != "date"]
        existing_cols = [c for c in keep_cols if c in df.columns]
        return df[existing_cols]

    # 流通股本（万股）进程级缓存（类属性，全实例共享）：key = tgw 代码，value = 按变动日索引的 Series
    _float_shares_cache: ClassVar[Dict[str, pd.Series]] = {}

    def _fetch_float_shares_series(self, tgw_code: str) -> Optional[pd.Series]:
        """
        获取个股流通A股（万股）的「变动日 → 万股」序列，用于按交易日 ffill。

        数据源：InfoData.get_equity_structure 的 FLOAT_A_SHARE（流通A股，单位万股）。
        股本仅在解禁/增发/送转等变动日更新，故需按变动日 ffill 到每个交易日。
        会话内按代码缓存，避免重复请求。

        取数约定（与 src/etf/amazing_factors.py 同款，该组合在本项目已验证可用）：
        只传 local_path + is_local=False，从服务端取全量并更新本地缓存。
        不传 begin_date/end_date —— 那是「变动日期」过滤（文档里与本地缓存模式
        二选一、不可混用），拿 K 线窗口去框会因股本变动稀疏而整段查空。
        """
        if tgw_code in self._float_shares_cache:
            return self._float_shares_cache[tgw_code]

        info = self.get_info_data()
        os.makedirs(AMAZINGDATA_CACHE_DIR, exist_ok=True)
        eq = info.get_equity_structure(
            [tgw_code],
            local_path=AMAZINGDATA_CACHE_DIR,
            is_local=False,
        )
        if eq is None or eq.empty or "FLOAT_A_SHARE" not in eq.columns:
            return None

        eq = eq.copy()
        eq["CHANGE_DATE"] = pd.to_datetime(eq["CHANGE_DATE"], errors="coerce")
        eq = eq.dropna(subset=["CHANGE_DATE", "FLOAT_A_SHARE"])
        if eq.empty:
            return None
        # 同一变动日可能有多行，取最后一条；按变动日排序
        eq = eq.sort_values("CHANGE_DATE").drop_duplicates("CHANGE_DATE", keep="last")
        series = eq.set_index("CHANGE_DATE")["FLOAT_A_SHARE"].sort_index()
        self._float_shares_cache[tgw_code] = series
        return series

    def _compute_turnover_rate(self, df: pd.DataFrame, stock_code: str) -> Optional[pd.Series]:
        """
        计算官方口径换手率（%）= 成交量(股) ÷ 流通股本(股) × 100。

        关键处理：
        1. 单位自校准：AmazingData K线 volume 单位（股/手）文档未明确，用「成交额 ÷ 流通市值」
           这一单位无关的中位比值做交叉校验，自动识别手→股（×100），杜绝 100× 静默错误。
        2. 流通股本按股本结构变动日 ffill 到每个交易日（解禁/增发导致阶跃）。
        3. 仅支持沪深 A 股/ETF（需能映射 tgw 代码且能取到股本结构）。
        """
        if not {"volume", "close", "amount"}.issubset(df.columns):
            return None

        tgw_code = _code_to_tgw_format(stock_code)
        if tgw_code is None:
            return None

        float_wan = self._fetch_float_shares_series(tgw_code)
        if float_wan is None:
            return None

        # 股本（万股）→ 股，按变动日 ffill 到每个交易日（取该日之前最近一次股本结构）
        idx = pd.to_datetime(df["date"])
        float_shares = float_wan.reindex(idx, method="ffill").bfill() * 1e4  # 万股 -> 股
        # 按位置对齐到 df 行索引（df 的 volume/close/amount 用默认 RangeIndex，
        # 而 reindex 后是日期值索引，直接相除会因索引不对齐全部变 NaN）
        float_shares = pd.Series(float_shares.values, index=df.index)
        float_shares = float_shares.replace(0, pd.NA)
        if float_shares.isna().all():
            return None

        volume = pd.to_numeric(df["volume"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        amount = pd.to_numeric(df["amount"], errors="coerce")

        # 单位无关参考值：换手率 ≈ 成交额 ÷ 流通市值（元/元，无量纲）
        tor_amt = (amount / (close * float_shares)).replace([pd.NA, 0], pd.NA)
        tor_vol_raw = (volume / float_shares).replace([pd.NA, 0], pd.NA)

        # 比值 median ≈ 1 → volume 单位为股；≈0.01 → 单位为手（需 ×100）
        ratio = (tor_vol_raw / tor_amt).replace([pd.NA, np.inf, -np.inf], pd.NA).dropna()
        unit_factor = 1.0
        if not ratio.empty:
            med = float(ratio.median())
            if med < 0.2:  # 约 0.01，手
                unit_factor = 100.0
                logger.info(f"[AmazingData] {stock_code} 检测到 K线 volume 单位为'手'，换手率计算已×100 转股")
            elif med > 5:  # 约 100，百股或其他异常，退回金额法
                logger.warning(
                    f"[AmazingData] {stock_code} volume/流通股本 比值异常(median={med:.3f})，"
                    f"换手率改用成交额÷流通市值估算"
                )
                tor = (tor_amt * 100).round(4)
                return tor

        tor = (volume * unit_factor / float_shares * 100).round(4)
        return tor

    def get_realtime_quote(self, stock_code: str):
        """
        AmazingData 快照为分时 5 档数据（单日单股 5000+ 行），
        不适合实时行情单点查询，返回 None 交由其他数据源处理。
        """
        return None

    def get_market_stats(self) -> Optional[Dict[str, Any]]:
        """AmazingData 未提供市场统计接口，返回 None。"""
        return None

    # ---------- 指数数据（市场状态判定专用，单源无回退） ----------

    def get_index_daily(self, code: str = "sh000001", days: int = 120) -> Optional[pd.DataFrame]:
        """
        指数日线（query_kline，市场状态判定唯一数据源，无回退源）。

        指数码族与个股独立转换（见 _index_code_to_tgw_format）。
        当日 bar 收盘后何时入库手册未写，由 market_gate 层用指数快照
        补当日 bar + 数据日期断言兜底。

        Returns:
            DataFrame(date/open/high/low/close/volume/amount，按日期升序)；取不到返回 None
        """
        tgw_code = _index_code_to_tgw_format(code)
        if tgw_code is None:
            logger.warning(f"[AmazingData] 不支持的指数代码 {code}")
            return None

        self._ensure_login()
        end = int(date_cls.today().strftime("%Y%m%d"))
        begin = int((date_cls.today() - timedelta(days=days * 2)).strftime("%Y%m%d"))

        try:
            from AmazingData.utils.constant import Period

            logger.info(f"[API调用] query_kline({tgw_code}, {begin}~{end}, period=day, 指数)")
            kline_dict = self._market_data.query_kline(
                [tgw_code],
                begin_date=begin,
                end_date=end,
                period=Period.day.value,
            )
        except SystemExit:
            raise DataFetchError("AmazingData 查询被中断")
        except Exception as e:
            raise DataFetchError(f"AmazingData 指数 query_kline 失败: {e}") from e

        df = kline_dict.get(tgw_code) if isinstance(kline_dict, dict) else None
        if df is None or df.empty:
            logger.warning(f"[AmazingData] 指数 {tgw_code} 未返回 K 线数据")
            return None

        df = df.rename(columns={"kline_time": "date"})
        df["date"] = pd.to_datetime(df["date"])
        keep = ["date", "open", "high", "low", "close", "volume", "amount"]
        df = df[[c for c in keep if c in df.columns]].sort_values("date").reset_index(drop=True)
        logger.info(f"[API返回] 指数 {tgw_code} 日线 {len(df)} 根，最新 {df['date'].iloc[-1]}")
        return df

    def get_index_snapshot(self, code: str = "sh000001",
                           day: Optional[date_cls] = None) -> Optional[Dict[str, Any]]:
        """
        指数单点快照（query_snapshot，取 day 当日最后一笔）。

        SnapshotIndex 字段（附录 4.2.4）：last=最新价、close=收盘价（仅上海有效，
        盘中即随最新价跳动）、volume=成交总量（上交所:手）、amount=成交总金额。
        用于补齐日 K 的当日 bar（收盘后 close/volume 即官方值；盘中为最新近似，
        与尾盘 14:45 近似收盘的既有口径一致）。

        Returns:
            {open, high, low, close, last, volume, amount, trade_time}；取不到返回 None
        """
        tgw_code = _index_code_to_tgw_format(code)
        if tgw_code is None:
            logger.warning(f"[AmazingData] 不支持的指数代码 {code}")
            return None

        self._ensure_login()
        day = day or date_cls.today()
        day_int = int(day.strftime("%Y%m%d"))

        try:
            snapshot_dict = self._market_data.query_snapshot(
                [tgw_code], begin_date=day_int, end_date=day_int,
            )
        except SystemExit:
            raise DataFetchError("AmazingData 查询被中断")
        except Exception as e:
            raise DataFetchError(f"AmazingData 指数 query_snapshot 失败: {e}") from e

        df = snapshot_dict.get(tgw_code) if isinstance(snapshot_dict, dict) else None
        if df is None or df.empty:
            logger.warning(f"[AmazingData] 指数 {tgw_code} {day} 无快照数据")
            return None

        row = df.iloc[-1]  # 当日最后一笔快照 = 收盘定格状态（盘中为最新状态）
        snap = {k: row[k] for k in ("open", "high", "low", "close", "last", "volume", "amount")
                if k in df.columns}
        snap["trade_time"] = row.get("trade_time", row.name)
        logger.info(f"[API返回] 指数 {tgw_code} {day} 快照: close={snap.get('close')}, "
                    f"last={snap.get('last')}, volume={snap.get('volume')}")
        return snap


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    f = AmazingDataFetcher()
    df = f.get_daily_data("600519", start_date="2026-07-01", end_date="2026-08-07")
    print(df.tail(5))
    print("\n列:", list(df.columns))
