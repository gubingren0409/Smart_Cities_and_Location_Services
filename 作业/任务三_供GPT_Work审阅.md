# 任务三：供 GPT Work 审阅的修复版材料说明

更新日期：2026-09-21。当前结论以 `experiments/llm_assisted_ecnu_20260921_v2/` 为准；`llm_assisted_ecnu_20260921/` 完整保留为修复前证据。

## 1. 课程范围与本轮边界

任务三让 LLM 根据不含坐标的诊断卡提出参数和预期效果，再由确定性代码执行、搜索和核验。本轮没有重构 Agent、没有修改数据集，也没有运行 24/48/100/200 demo 扩展，只修复实验语义并重跑固定的 `12 demo + 12 holdout × 3 repetitions`。

原始数据 `traj_dict.json` 含 11,386 辆车；真实模型验证规模仍只有固定的 24 辆不同用途样本，不能把数据集规模写成 Agent 验证规模。

## 2. 关键修复

1. 参数记录拆成 `raw_proposal_params`、`normalized_proposal_params` 和 `execution_params`；合法整数取整只进入 `rounded_params`，真越界进入 `out_of_bounds_params`。
2. 主实验参数空间统一为 `dp_tolerance`、`dist_threshold`、`max_speed_mps`，LLM、Search、Verifier、Memory 共用同一三维空间。
3. 纯确定性搜索参考不再被 LLM proposal 覆盖；另存 signed gap 与 best-observed 来源。
4. 模式名称改为真实语义；新增 `det-search-output`。带 `search-verifier` 的 LLM 模式不会用 Search 修改 proposal。
5. 主质量 score 移除 runtime；runtime 单独报告。road 不可用时从分子和分母同时移除。
6. `applicable=False` 的静止短轨迹只进入 L1 审计账本，不进入可检索 Memory/L2。
7. L2 `min_samples=3`；Bootstrap 以车辆为 cluster，明确区分 8 辆独立车辆、24 条观测和3次重复。
8. retry-failed 追加的失败证据保留在 JSONL；正式统计按 case key 取最后结果，成本仍累计所有尝试。

修复代码提交为 `13f86cd3572464131fab8fd548f82a2fb10dd907`，分析代码记录在 `config.json` 的 `analysis_code_commit_sha`。

## 3. 修复版实验设计

- Provider：`OpenAICompatProvider`；模型：`ecnu-plus`。
- Demo：`0 1 3 7 18 62 68 129 101 149 170 187`。
- Holdout：`2 4 10 22 111 137 153 154 165 194 201 208`；与 demo 无重叠。
- 每轮使用新 Memory；holdout 只读。
- 模式：`llm-only`、`prior-only+search-verifier`、`det-search-output`、`llm+search-verifier`、`llm+memory+search-verifier`。
- JSONL 共217行：216个唯一 case 最终成功，另有1条失败尝试记录；断点重跑后有效失败数为0。
- 全部实际尝试共159次模型请求、406,605 prompt tokens、31,659 completion tokens；1条失败尝试中记录1次内部重试和2次 provider error。

## 4. 主结果

每个模式有24条适用观测，来自8辆独立车辆的3次重复。95% CI 使用 vehicle-cluster bootstrap。

| 模式 | mean score delta | 95% CI | baseline beaten | LLM calls |
|---|---:|---:|---:|---:|
| `llm-only` | -0.000702 | [-0.015137, 0.012893] | 50.0% | 40 |
| `prior-only+search-verifier` | -0.000818 | [-0.001914, -0.000046] | 0.0% | 0 |
| `det-search-output` | +0.131676 | [0.023094, 0.338599] | 100.0% | 0 |
| `llm+search-verifier` | +0.001397 | [-0.013348, 0.015262] | 58.3% | 43 |
| `llm+memory+search-verifier` | +0.001347 | [-0.013158, 0.014805] | 62.5% | 40 |

三个 LLM 模式的区间均覆盖 0。当前结果没有支持“LLM 稳定优于默认参数”。`det-search-output` 的高内部得分只说明它优化了当前 bounded objective；objective 以去噪后轨迹为 DP 参考，不能独立判断清洗删除是否正确，因此不能把该结果当作真实质量提升。

## 5. Memory 成本—收益

Memory − no-memory 的车辆级平均 paired difference 为 **-0.000049**，95% CI 为 **[-0.002513, 0.002879]**；8辆车的胜/平/负为 **1/2/5**。

| 指标 | 无 Memory | 有 Memory |
|---|---:|---:|
| LLM calls | 43 | 40 |
| prompt tokens | 103,871 | 108,196 |
| median case latency | 6.29 s | 6.26 s |
| direction accuracy | 0.708 | 0.686 |

没有检测到稳定质量增益。Memory 使用更多 prompt tokens；调用次数与延迟受独立请求和失败重试影响，不能解释为稳定成本下降。

修复前后三轮 admitted 总数从24降到10，L2 regions 从80降到0。修复版三轮分别 admitted 3/3/4；由于每个 `(regime, timeline_quality)` 分层支持不足3条，没有形成可推荐的 L2 region。36条 Memory holdout 仍能检索 admitted episodic cases。

## 6. 修复前后结论变化

- 旧 `5.6% constraint violation` 混入了合法整数取整，修复后所有模式为0%。
- 旧 Search best 可能被 proposal 覆盖，旧的 proposal-vs-search 结论必须废弃；修复版中适用样本上的 LLM proposal 均未超过纯有界搜索参考。
- 旧 Memory paired mean `+0.001236` 使用24行伪独立观测；修复版车辆级 mean 为 `-0.000049`，区间仍覆盖0。
- 旧80个 L2 region 只有1–2条支持且含静止自动准入；修复版 `min_samples=3` 后为0。
- 旧 score 含 runtime抖动且缺失road仍占权重；所有 score、delta、regret 均应以修复版重算结果为准。

仍然有效的是：固定样本清单、demo/holdout零重叠、真实 provider 工程链路、逐case checkpoint、holdout只读，以及修复前真实调用和token的历史记录。

## 7. 建议 GPT Work 审阅的问题

1. 如何解释 `det-search-output` 明显提高内部 score，但清洗删除正确性尚无独立验证这一限制。
2. 在 L2 region 为0的情况下，是否先设计独立 stationary validator 或人工标注的 moving 子集，再讨论扩大 demo。
3. 是否需要把 episodic Memory 与 procedural Memory 分开做新的消融；当前 Memory 模式只有前者实际可用。
4. 分层每类只有2辆独立车辆、6条观测，`mixed/ok`、`moving/ok` 等差异只能称为 exploratory signal。

建议阅读顺序：本文件 → [修复版实验报告](experiments/llm_assisted_ecnu_20260921_v2/实验报告.md) → `summary.json` → `raw_results.jsonl` → runner/analyzer → verifier 与 memory 源码 → 修复前目录。
