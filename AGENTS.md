# AGENTS.md — Regime Trader

## Architecture

```
style_report.py     → 风格状态周报：周度判定 主线强势期/退潮期/真空期/形成中 + 主导风格，
                       落盘 data/style_state.json；--backtest 历史回放验证标签

src/trend/                       ← 趋势策略全链路：分析器(analyzer)、信号检测(signal_detector)、
                                   负面清单硬否决(veto_rules，V1/V4 硬否决 + V5 观察项，位于信号检测之前)、
                                   卖出规则(sell_rules，只判定不下单)、
                                   Cycle 个股组件(cycle_overlay：B1延伸计数/C1 ATR过滤/D1方向门，全开)、
                                   日报生成(report，买入侧)
trend_sell.py                    ← 尾盘卖出任务(14:45)：读妙想持仓→sell_rules 判定→
                                   自动下模拟仓市价单→自出成交报告并推送
src/market_state/                ← 市场环境判断（跨策略共享）：趋势状态判定(market_gate，
                                   指数均线纯结构 5 级；指数数据 AmazingData 单源——K线+快照补当日bar+数据日期断言，无 akshare 回退)、风格状态判定(style_state)、Cycle 指数循环定位(cycle_stage：A1仓位档位/A2快速通道，全开)，
                                   文档见 strategy/style_state.md
src/indicators.py                ← numba 指标算子封装（纯计算，无交易语义，根级别共享工具）
src/etf/                         ← ETF 配置：再平衡+新钱投放(rebalancer)、
                                   卫星仓行业动量轮动(industry_momentum)、因子封装(amazing_factors)、基准(config)
src/notify/                     ← 多渠道通知（飞书/邮件）
src/mx/                         ← 妙想模拟仓 API 客户端 + 持仓公共工具(position_utils)
data_provider/                  ← 多源行情数据（AmazingData > tushare > akshare > efinance > baostock > yfinance）

data/etf_industry_map.json      ← 行业 ETF 清单（申万行业 → 首选/备选 ETF，卫星仓标的池）
```

## Commands

```bash
# 风格状态周报（纯规则，无 LLM；cron 暂未启用）
python style_report.py                    # 周报：风格状态 + 下周怎么办 + 主线明细 + 风格指标
python style_report.py --backtest 2021-01-01  # 历史回放状态时间线（不落盘不通知）

# 趋势交易（每交易日：14:45 卖出执行 + 15:10 买入分析）
python trend_sell.py                        # 尾盘卖出：检测→自动下模拟仓市价单→出成交报告并推送
python trend_sell.py --dry-run              # 只检测不下单（调试用）
python trend_analysis.py                    # 日度分析（选股名单=当日妙想选股结果，不读妙想自选）
python trend_analysis.py --screen-keyword "..."  # 自定义妙想选股条件
python trend_analysis.py --stocks 000001,600519  # 指定股票（覆盖当日选股名单）
python trend_analysis.py --list             # 仅列出当日妙想选股名单
python trend_analysis.py --debug --no-notify

# ETF 长期配置（每周一 9:35）
python etf_observe.py                    # 周度观察报告（只出建议，不下单）
python etf_observe.py --execute          # 执行统一批次：核心再平衡 + 卫星动量调仓（妙想市价单）
python etf_observe.py --force            # 跳过交易日检查（调试用）
python etf_observe.py --no-notify --debug
```

组件单测：`python -m pytest tests/`（Cycle 吸收组件 + 开关全关等价性）；无 lint/typecheck 命令。

## 部署与定时任务

- GitHub Actions （当前disable） / 云服务器（Linux），Python 环境用项目内 `.conda/`
- 定时任务：`deploy/crontab.server`（crontab 格式，服务器时区须为 Asia/Shanghai）
- 每个任务用 `flock` 防重入；节假日由 `src/trading_calendar.py:is_trading_day()` 处理（etf 任务）或 cron 的 `1-5` 限定（trend 任务）

## Design Decisions

- **文档分工**：
  - `strategy/overview.md`：总览
  - `strategy/*.md` — 项目自身的策略设计文档（总览/趋势/ETF 配置/行业动量轮动），**必须与代码同步**。 改代码中的阈值、交易逻辑、信号优先级时，必须在同一提交里更新对应策略文档；反之改文档时也要同步代码。
- 当文档与代码出现矛盾、或需要决策阈值/逻辑时，**以投资/交易逻辑为准** 思考什么对策略合理，而不是"文档说了什么"或"代码现在怎么写的"。

## Key Conventions

- Chinese docstrings and comments throughout
- `data/` holds cached state (e.g. `style_state.json`, `position_exit_state.json`, `cycle_state.json`, `momentum_rank_history.json`)
- Logging via `src/logging_config.py:setup_logging()` — console + file + debug file handlers
- All stock codes normalized via `data_provider.base:canonical_stock_code()`
- Cycle 吸收（A1/A2/B1/C1/D1，证据与决策台账存档 `research/cycle_absorption/`）：阈值见
  `strategy/trend_strategy.md`「Cycle 吸收组件」节，改动须同步文档并重跑证据；组件作为整体
  使用，勿单独启停（消融证明部分启用有害）
- 目录归属规则：只服务一个策略 → 进该策略的包（`src/trend/`、`src/etf/`）；
  跨策略共享且有交易语义 → 共享概念包（`src/market_state/`）；
  纯计算/无交易语义 → `src/` 根级别工具（如 `indicators.py`）

## docs 外部工具说明书

- mx_skills
- 星耀数智