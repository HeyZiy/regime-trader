# 接回真实代码库 — 最小侵入补丁指南

> 本包的回测实现是按 `trend_strategy.md` 规格的 1:1 重建（用于验证），生产代码库在用户交易机上。本文档把五个吸收模块映射到真实工程的接入点：**新增 2 个文件、改动 3 个文件的少量行、不改任何既有判定语义**。
> 对应回测实现：`cycle_overlay.py`（组件）+ `engine.py`（合并点），可对照阅读。

## 0. 原则

1. gate 五态判定、信号 1/2 触发条件、既有卖出规则的语义**一字不动**。
2. 吸收组件全部可开关（`config.cycle`），默认关闭 = 现行为逐 bit 不变（回测已验证等价性，`selfcheck.py` switch_equivalence=True）。
3. 新组件只消费日线收盘后可得字段，兼容"盘后买入申报 / 14:45 尾盘卖出"时序。

## 1. 新增文件

### 1.1 `src/market_state/cycle_stage.py`（A1 + A2）

从本包 `cycle_overlay.py` 的 `CycleStage` 类整体迁入，依赖只有 numpy/pandas。它消费指数日线（gate 同源数据），每日收盘后调用一次：

```python
# 每日 15:05 日报流程内, gate 判定完成后:
from market_state.cycle_stage import CycleStage
stage = CycleStage(cfg.cycle, idx_dict)   # idx_dict 与 market_gate 同一份指数数据
stage.update(i, gate)                     # gate = market_gate 的当日判定结果
# 读取: stage.stage / stage.cap / stage.allow_override
```

落盘：`data/cycle_state.json` 存当日 `{stage, cap, allow_override, panic_pending_until, panic_low}`，供 14:45 卖出流程与次日盘前复核读取（与 `position_exit_state.json` 同风格，fail-soft）。

### 1.2 `src/trend/cycle_overlay.py`（B1 + C1 + D1）

从本包 `cycle_overlay.py` 迁入 `ExhaustionTracker` / `atr_filter` / `slope_gate`。其中 `ExhaustionTracker` 需要持仓级持久化——在 `data/position_exit_state.json` 的每仓记录里追加两个字段（见 §3.2）。

## 2. 改动点 A — `src/trend/report.py`（开仓放行 + 仓位上限）

在买入名单生成处（信号评分排序之后、输出"优先关注"档之前）插入：

```python
# --- Cycle 吸收: A1 仓位档位 + A2 快速通道 ---
cycle_state = load_cycle_state()            # data/cycle_state.json, fail-soft
allow = gate_allows_open                    # 既有 gate 放行逻辑, 不动
if not allow and cfg.cycle.a2_enabled and cycle_state.get("allow_override"):
    allow = True                            # 右侧快速通道放行(仅当日)
cap = 1.0
if cfg.cycle.a2_enabled and cycle_state.get("allow_override") and not gate_allows_open:
    cap = cfg.cycle.cap_bottom              # 40%
elif cfg.cycle.a1_enabled:
    cap = cycle_state.get("cap", 1.0)       # 顶部 30% / 主升 100%

invested = 当前持仓市值(equity 口径)
if allow and invested >= cap * equity:
    allow = False                            # 组合仓位上限约束, 档位截断
# 信号候选循环内同样检查: invested + 本笔金额 <= cap * equity
```

## 3. 改动点 B — `src/trend/sell_rules.py`（B1 延伸动作合并）

在每仓卖出判定返回动作前合并延伸计数：

```python
# --- Cycle 吸收: B1 延伸计数 ---
from trend.cycle_overlay import ExhaustionTracker
ext = ExhaustionTracker(cfg.cycle)                      # 或从缓存恢复的单例
ext.sync(position_exit_state)                           # 每仓 episodes/in_episode 状态
b1 = ext.update(code, stock_bars, today_idx)            # None | ('reduce_half'|'clear', reason)
if b1:
    acts.append(b1)                                     # 与既有 acts 同池
# 既有"取最强动作"逻辑不变: clear > reduce_half
```

持久化（`data/position_exit_state.json` 每仓追加）：

```json
{ "...既有 peak/entry 字段...",
  "ext_episodes": 2, "ext_in_episode": true }
```

买入成交确认处调用 `ext.on_entry(code)`；平仓确认处调用 `ext.on_exit(code)`。

## 4. 改动点 C — `signal_detector`（C1 + D1 后置过滤）

信号产出之后、评分与展示之前：

```python
from trend.cycle_overlay import atr_filter, slope_gate
if cfg.cycle.c1_enabled and atr_filter(bars, i, cfg.cycle):
    return None          # 剔除, 不评分
if cfg.cycle.d1_enabled and slope_gate(bars, i, cfg.cycle):
    return None
```

对信号 2 的"次日弱转强确认"分支同样应用两个过滤器（回测口径与之一致）。

## 5. 配置开关（config.py 或日报配置节）

```python
@dataclass
class CycleConfig:
    a1_enabled: bool = True;  ext_idx_thr: float = 4.5;  cap_top: float = 0.30;  cap_bottom: float = 0.40
    a2_enabled: bool = True;  panic_thr: float = -4.0;   fp_window: int = 10
    b1_enabled: bool = True;  ext_stk_thr: float = 10.0; ext_stk_reset: float = 4.0
    c1_enabled: bool = True;  atr_exp_thr: float = 1.3
    d1_enabled: bool = True
```

**必须全开或全关落地**（消融结论：A1/B1 单开有害）；影子模式下可 `enabled` 拆成"计算+展示"与"参与拦截"两级。

## 6. 回归验证清单（合入后必做）

1. **等价性**：全开关关闭，连续 3 个交易日日报输出与合入前逐字节一致（含卖出信号与买入名单）。
2. **组件单测**（用合成 bar）：
   - A2：构造 bias≤−4% + 放量反转 bar → 登记；次日收复 MA10 阳线 → allow_override=True 恰一日。
   - B1：构造 bias 序列 12%→3%→11%→3%→11% → episodes=3，第三次返回 clear。
   - C1：ATR5/ATR20=1.4 → 剔除；NaN → 放行。
   - D1：MA5 连续下行日 → 剔除。
3. **台账抽查**：启用后第一周，人工核对日报中 top 档提示、延伸减仓动作与对应 K 线。
4. **影子模式**：先只计算展示（stage/cap 写入日报"市场环境"节），观察 2 周，再切入拦截。

## 7. 上线路径

影子模式（只展示）→ 2 周无口径事故 → 开 B1+C1+D1（卖出与信号侧，影响温和）→ 再开 A1+A2（仓位侧）。任一环节回撤或行为异常，开关逐项回退即可（等价性已验证）。
