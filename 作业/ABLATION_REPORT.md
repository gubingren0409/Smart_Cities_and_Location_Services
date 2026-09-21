# 消融实验结果

本页把当前真实 ECNU 实验与历史 Mock 对照分开记录。正式数字以 [真实实验 summary.json](experiments/llm_assisted_ecnu_20260921/summary.json) 和 [case 级 raw_results.jsonl](experiments/llm_assisted_ecnu_20260921/raw_results.jsonl) 为准。

## 1. 真实 ECNU 三轮实验

运行入口：

```bash
python experiments/llm_assisted_ecnu_20260921/run_real_ablation.py --run
```

Provider 为 `OpenAICompatProvider`，模型为 `ecnu-plus`。Demo 固定为 `0 1 3 7 18 62 68 129 101 149 170 187`，holdout 固定为 `2 4 10 22 111 137 153 154 165 194 201 208`，两组无重叠。四模式重复 3 轮，每轮创建新的 Memory，holdout 期间只读。

共有 180 条 case 记录、167 次真实 LLM 请求，使用 529,980 prompt tokens 和 37,761 completion tokens。case 时间合计 1,416.83 秒；失败 0，重试 0，provider 错误 0。`search-only` 的 LLM 调用为 0。

### 四模式主结果

每模式有 36 条 holdout 记录，其中 24 条 `applicable=True` 进入得分统计。跨模式比较使用 `proposal_score - baseline_score`，不使用口径随搜索开关变化的平均 regret 排名。

| 模式 | 平均变化 | 中位数 | 95% bootstrap CI | 超过基线比例 | JSON成功率 | 约束违规率 | LLM调用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llm-only` | -0.000050 | +0.003153 | [-0.006178, 0.005441] | 58.3% | 100% | 5.6% | 36 |
| `search-only` | -0.000555 | -0.000036 | [-0.001081, -0.000142] | 37.5% | N/A | 0% | 0 |
| `llm+search` | +0.000429 | +0.000179 | [-0.005874, 0.006810] | 54.2% | 100% | 0% | 39 |
| `llm+memory+search` | +0.001645 | +0.002193 | [-0.004523, 0.007489] | 58.3% | 100% | 5.6% | 48 |

三种 LLM 模式相对基线变化的 95% CI 都覆盖 0，当前实验没有支持“真实 LLM 稳定优于默认参数”的结论。`search-only` 的区间略低于 0，但该行衡量的是数据驱动 proposal 相对固定默认值；有界确定性搜索结果仍作为 verifier 内部参考，不能把二者混成一个概念。

### 三轮波动

| repetition | llm-only | search-only | llm+search | llm+memory+search |
|---:|---:|---:|---:|---:|
| 1 | +0.000201 | -0.000597 | +0.001166 | +0.001184 |
| 2 | -0.000095 | -0.000501 | +0.000600 | +0.002022 |
| 3 | -0.000257 | -0.000566 | -0.000481 | +0.001728 |

### Memory 配对比较

相同 holdout case 上，`llm+memory+search - llm+search` 的平均差值为 +0.001236，中位数 +0.000021，95% bootstrap CI 为 [-0.000851, 0.003498]，胜/平/负为 16/0/8。区间覆盖 0，所以当前没有测出稳定 Memory 增益。

| repetition | L1案例 | admitted | L2 region | stationary/ok | stationary/degraded | mixed/ok | mixed/degraded | moving/ok | moving/degraded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 7 | 35 | 2 | 2 | 1 | 1 | 1 | 0 |
| 2 | 12 | 9 | 21 | 2 | 2 | 1 | 1 | 2 | 1 |
| 3 | 12 | 8 | 24 | 2 | 2 | 1 | 1 | 2 | 0 |

36 条 Memory holdout 全部取得先验，邻居数分布为 1 个 4 条、2 个 16 条、3 个 4 条、4 个 12 条，邻居相似度均值 0.602。三轮 80 个参数 region 的 `n_samples` 全部只有 1 或 2，`moving/degraded` 合计只准入 1 条。证据覆盖仍然薄，因此应表述为“当前证据不足以建立稳定 Memory 增益”，不能表述为“Memory 机制无效”。

### Demo 扩展成本

24/48/100/200 条分层候选已按固定种子生成并排除 holdout。本轮只估算，不自动执行高成本扩展：

| 每轮 demo 数 | 预计 LLM calls | prompt tokens | completion tokens | 预计时间（s） |
|---:|---:|---:|---:|---:|
| 24 | 88 | 281,366 | 17,026 | 618.8 |
| 48 | 176 | 562,732 | 34,052 | 1,237.7 |
| 100 | 367 | 1,172,358 | 70,942 | 2,578.5 |
| 200 | 733 | 2,344,717 | 141,883 | 5,157.1 |

估计只覆盖三轮 demo，不含 holdout。下一轮可先比较 24 与 48 条的 coverage 变化，再决定是否继续扩大。

## 2. 历史 Mock 对照

历史 `MockProvider` 使用相同 12 条 demo 与 12 条 holdout，三种含 Mock 模式的平均提议分都约为 0.3075，相对基线约 +0.0025；Memory 与无 Memory 的提议逐位相同。该结果用于验证流程、泄漏防护和模式开关，不能替代真实模型实验。

当时 moving 类没有已验证 Memory 区域，这解释了 Mock 运行中检索先验为空。真实 ECNU 运行已取得少量 moving 记忆，但每个 region 仍只有 1–2 个样本，配对区间依然覆盖 0。

## 3. 共同解释边界

1. 确定性搜索从默认参数独立运行，建立同一内部 objective 下的参考。它不是现实道路真值。
2. 静止短轨迹的普通压缩目标不适用，标记为 `applicable=False`，不强行进入均值。
3. 当前没有接入独立 OSM 中心线或人工漂移标签，不能把内部得分直接解释成道路清洗质量。
4. L1 可以留档失败案例；只有 `admitted=True` 案例会被默认检索并进入 L2。
5. Demo 与 holdout 零重叠；holdout 只读；`search-only` 确实跳过 LLM。这些约束均有回归测试。
