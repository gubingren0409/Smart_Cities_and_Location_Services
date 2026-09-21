# 任务三消融实验结果（修复版）

正式数字以 [修复版实验报告](experiments/llm_assisted_ecnu_20260921_v2/实验报告.md)、[summary.json](experiments/llm_assisted_ecnu_20260921_v2/summary.json) 和 [case 级 JSONL](experiments/llm_assisted_ecnu_20260921_v2/raw_results.jsonl) 为准。修复前实验完整保留在 `experiments/llm_assisted_ecnu_20260921/`，只作为 pre-fix evidence。

## 1. 实验范围

- 模型：`ecnu-plus`；固定 12 条 demo、12 条不重叠 holdout、3 次重复。
- 主实验活动参数统一为 `dp_tolerance`、`dist_threshold`、`max_speed_mps`。
- LLM、Search、Verifier 和 Memory 使用相同的三维参数空间。
- `search-verifier` 只提供内部有界搜索参考，不修改 LLM proposal。
- `det-search-output` 才把搜索参考参数作为最终输出。
- 质量分移除 runtime；runtime 单独报告。road 不可用时不占分母。
- CI 以 `vehicle_id` 为 cluster，同一车辆的三次 repetition 整组抽取。
- 静止短轨迹保留在 L1 审计账本，不进入可检索 Memory 或 L2；L2 `min_samples=3`。

## 2. 修复版主结果

每个模式有 36 条 holdout 记录，其中 8 辆独立车辆、24 条适用观测进入得分统计。

| 模式 | 平均得分变化 | vehicle-cluster 95% CI | 超过默认参数比例 | LLM calls |
|---|---:|---:|---:|---:|
| `llm-only` | -0.000702 | [-0.015137, 0.012893] | 50.0% | 40 |
| `prior-only+search-verifier` | -0.000818 | [-0.001914, -0.000046] | 0.0% | 0 |
| `det-search-output` | +0.131676 | [0.023094, 0.338599] | 100.0% | 0 |
| `llm+search-verifier` | +0.001397 | [-0.013348, 0.015262] | 58.3% | 43 |
| `llm+memory+search-verifier` | +0.001347 | [-0.013158, 0.014805] | 62.5% | 40 |

`det-search-output` 的较高内部得分不能解释为真实清洗质量提升。当前 objective 以去噪后轨迹作为 DP 参考，尚无人工漂移标签或道路真值独立判断清洗删除是否正确。因此它只是 bounded-search reference。

## 3. Memory 成本与质量

按车辆先对三轮 paired difference 求平均，Memory − no-memory 的均值为 **-0.000049**，95% CI 为 **[-0.002513, 0.002879]**；8 辆车的胜/平/负为 **1/2/5**。没有检测到稳定质量增益。

| 指标 | 无 Memory | 有 Memory |
|---|---:|---:|
| LLM calls | 43 | 40 |
| prompt tokens | 103,871 | 108,196 |
| median case latency | 6.29 s | 6.26 s |
| direction accuracy | 0.708 | 0.686 |

调用次数与延迟受独立模型请求和一次失败重试影响，不能解释为 Memory 带来稳定成本下降。Memory 模式使用更多 prompt tokens，方向准确率略低，质量区间覆盖 0。

三轮 Memory 状态：

| repetition | L1案例 | admitted | L2 regions |
|---:|---:|---:|---:|
| 1 | 12 | 3 | 0 |
| 2 | 12 | 3 | 0 |
| 3 | 12 | 4 | 0 |

修复后静止样本不再自动准入，`min_samples=3` 后没有任何分层参数区间获得足够支持。36 条 Memory holdout 仍可检索 admitted episodic cases，但没有 L2 推荐区间。

## 4. 修复前后变化

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 有 LLM 模式的约束违规率 | 两个模式 5.6% | 所有模式 0% |
| Demo admitted 总数（三轮） | 24 | 10 |
| L2 regions（三轮） | 80 | 0 |
| Memory paired mean | +0.001236 | -0.000049 |
| Memory paired CI | 行级 [-0.000851, 0.003498] | 车辆级 [-0.002513, 0.002879] |

前后 objective、参数空间、模式语义和统计单位均已变化，这张表用于追踪修复影响，不能解释为同一条件下的因果差异。原报告里的 `5.6% constraint violation` 混入了合法整数取整，旧 Memory coverage 又包含无独立验证的静止案例，因此必须用修复版重算结果替代。

## 5. 仍然有效与必须废弃的旧结果

仍然有效：固定 demo/holdout 清单、二者零重叠、真实 provider 可运行、逐 case checkpoint、holdout 只读、修复前原始调用与 token 记录。

必须废弃或重算：旧约束违规率、旧 L2 region 数、旧 Memory admitted/coverage、行级 bootstrap CI、旧 `search-only` 标签解释、被 proposal 覆盖后的 search best，以及包含 runtime 抖动和缺失 road 权重的旧 score。

## 6. 解释边界

1. LLM 模式使用独立请求，模式差异包含模型生成波动。
2. 每个分层只有 2 辆独立车辆、6 条重复观测，分层均值只作为探索性信号。
3. 有界搜索不是现实真值，也不保证得到参数空间的全局解。
4. 当前没有人工漂移标签、独立道路匹配指标或 stationary validator。
5. 本轮没有执行 24/48/100/200 demo 扩展。
