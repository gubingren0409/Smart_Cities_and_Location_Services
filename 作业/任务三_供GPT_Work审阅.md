# 任务三：供 GPT Work 审阅的本地材料说明

更新日期：2026-09-21。本文件用于帮助审阅者定位事实来源，不能替代源码、case 级 JSONL 或课程材料。

## 1. 课程要求与系统目标

根目录的 `实验课1.pptx` 是课程要求的第一手材料。第 17–22 页介绍 GPS 中断分段、漂移去噪和 Douglas–Peucker 简化；第 24–25 页的 95 m 距离阈值是骑行案例，不能直接当成所有车辆的统一阈值。第 26 页把三步预处理、评价指标和实验报告列为必做，把基于助教系统的 LLM 辅助质量提升与思考讨论列为选做；本项目把选做内容也作为交付范围。

任务三系统让 LLM 根据不含坐标的诊断卡提出参数和预期效果。确定性代码负责执行参数、从默认参数独立搜索内部参考、计算得分与 regret，并决定是否允许案例进入后续推荐。LLM 候选与确定性搜索是两条独立路径，搜索没有以 LLM 参数为起点。

原始数据 `traj_dict.json` 约 44.8 MB，含 11,386 辆车。本次真实模型实验仅使用固定的 12 条 demo 和 12 条 holdout，不能把原始数据规模写成智能体的验证规模。

## 2. 当前材料及属性

| 材料 | 属性 | 使用边界 |
|---|---|---|
| `实验课1.pptx`、`作业.zip` | 课程要求和原始文件快照 | 用于确定任务范围与助教框架 |
| [任务3_LLM辅助评估清洗.ipynb](任务3_LLM辅助评估清洗.ipynb) | 不产生 API 费用的课程演示 Notebook | 明确使用 Mock；读取独立目录中的 ECNU 汇总 |
| [traj_agent](traj_agent) | 确定性算法、工具、agent、verifier、memory 和实验模块 | 机制与边界以代码和测试为准 |
| [experiments/llm_assisted_ecnu_20260921](experiments/llm_assisted_ecnu_20260921) | 真实 `ecnu-plus` 三轮四模式实验 | 当前真实模型结论的主要证据 |
| [experiments/llm_assisted_20260920](experiments/llm_assisted_20260920) | 固定样本 Mock 对照 | 只能证明离线流程和对照口径，不代表真实模型 |
| [ABLATION_REPORT.md](ABLATION_REPORT.md) | 汇总真实实验并保留 Mock 历史对照 | 跨模式不直接比较平均 regret |
| [DECISIONS.md](DECISIONS.md) | 投影、异常规则、时间轴和指标设计记录 | 样本统计必须看清版本与抽样方式 |
| [tests](tests) | 298 项单元、回归与实验工程测试 | 测试通过不等于内部目标代表道路真值 |
| [基于LLM的轨迹数据清洗评估.pptx](基于LLM的轨迹数据清洗评估.pptx) | 课堂汇报材料 | 数值应回到 `summary.json` 核对 |

真实 API key 不在仓库中。`config.json` 只记录 provider、model、base URL 和运行配置等可公开信息。

## 3. 真实 ECNU 实验设计

- Provider：`OpenAICompatProvider`；模型：`ecnu-plus`；Base URL：`https://chat.ecnu.edu.cn/open/api/v1`。
- Demo：`0 1 3 7 18 62 68 129 101 149 170 187`。
- Holdout：`2 4 10 22 111 137 153 154 165 194 201 208`；与 demo 无重叠。
- 模式：`llm-only`、`search-only`、`llm+search`、`llm+memory+search`。
- 三轮使用相同样本与搜索预算，每轮建立新的 Memory；holdout 只读。
- `search-only` 没有调用 LLM。所有模式继续沿用原 objective、参数边界、默认值、搜索预算、regret 定义、工具子集与 `direct_first=True`。
- runner 每个 case 后追加并同步 `raw_results.jsonl`，支持断点恢复、有限重试与渐进退避；运行保持串行。

本次共有 180 条 case 记录，其中 demo 36 条、holdout 144 条。实际调用 LLM 167 次，使用 529,980 prompt tokens 和 37,761 completion tokens，case 时间合计 1,416.83 秒；0 个失败、0 次重试、0 个 provider 错误。

## 4. 四模式结果

每个模式有 36 条 holdout 记录。每轮 4 条静止短轨迹标记为 `applicable=False`，所以普通得分统计每模式使用 24 条。

| 模式 | 相对固定基线平均变化 | 中位数 | 95% bootstrap CI | 超过基线比例 | 约束/截断率 | LLM 调用 |
|---|---:|---:|---:|---:|---:|---:|
| `llm-only` | -0.000050 | +0.003153 | [-0.006178, 0.005441] | 58.3% | 5.6% | 36 |
| `search-only` | -0.000555 | -0.000036 | [-0.001081, -0.000142] | 37.5% | 0% | 0 |
| `llm+search` | +0.000429 | +0.000179 | [-0.005874, 0.006810] | 54.2% | 0% | 39 |
| `llm+memory+search` | +0.001645 | +0.002193 | [-0.004523, 0.007489] | 58.3% | 5.6% | 48 |

三种 LLM 模式的 JSON 成功率均为 100%。`llm-only` 与 Memory 模式各有 2 条记录的 `dt_threshold` 超界并在执行前截断，原始超界事实仍保存在核验记录中。

以上表格的主比较量是 `proposal_score - baseline_score`。搜索开关会改变 regret 的参考口径，所以不能用四种模式的平均 regret 排名。`search-only` 行描述其数据驱动 proposal 相对固定默认值的表现；有界确定性搜索结果另由 verifier 用作内部参考。

## 5. Memory 结果与证据覆盖

相同 holdout 上，`llm+memory+search - llm+search` 的平均配对差值为 +0.001236，中位数 +0.000021，95% bootstrap CI 为 [-0.000851, 0.003498]，胜/平/负为 16/0/8。区间覆盖 0，因此没有检出稳定的 Memory 增益。

Demo 准入结构如下：

| 轮次 | L1案例 | admitted | L2 region | stationary/ok | stationary/degraded | mixed/ok | mixed/degraded | moving/ok | moving/degraded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 7 | 35 | 2 | 2 | 1 | 1 | 1 | 0 |
| 2 | 12 | 9 | 21 | 2 | 2 | 1 | 1 | 2 | 1 |
| 3 | 12 | 8 | 24 | 2 | 2 | 1 | 1 | 2 | 0 |

36 条 Memory holdout 记录都取得记忆先验。检索邻居数分布为 1 个邻居 4 条、2 个 16 条、3 个 4 条、4 个 12 条；96 个邻居相似度的均值为 0.602。三轮共形成 80 个参数 region，每个 region 的 `n_samples` 仅为 1 或 2；`moving/degraded` 三轮合计只准入 1 条。当前 Memory evidence 较薄，实验只支持“尚未检出稳定增益”，不支持“Memory 机制无效”。每条邻居的 similarity、suggested regions 以及每个 region 的 `n_samples` 都在 `summary.json` 和 `run_state.json` 中。

## 6. 已修正的旧口径

- 固定随机种子抽取的 200 条样本中，stationary/mixed/moving 为 94/48/58，即 47%/24%/29%。这只是该样本分布，不是全量比例，也不能与历史排序样本的统计混用。
- 当前数据中车辆 352 有 170 点、54 个不同时间戳、543 秒跨度，正时间间隔中位数 10 秒、P95 20 秒，零间隔比例 68.64%，轨迹长 7,632.11 m，最大表观速度 242.04 m/s。旧版“全部位于 3 秒、仅 4 个时间戳”已撤销。
- L1 可保存 `admitted=False` 案例用于审计；默认检索和 L2 蒸馏只使用 `admitted=True` 案例。

## 7. 结论边界与下一步讨论点

当前证据支持：真实 provider 工程链路能稳定运行；四种模式结构确实不同；`search-only` 为 0 次 LLM 调用；三轮 JSON 解析与运行没有失败；Memory 检索真实发生；候选参数偶尔会越界，核验器能保留证据并在执行前截断。

当前证据不支持：真实 LLM 稳定优于固定默认参数或确定性搜索；Memory 已带来稳定收益；内部目标函数等价于现实道路清洗质量；11,386 辆轨迹已经完成真实模型验证。

已按固定种子生成 24/48/100/200 条分层 demo 候选，优先覆盖 moving，且全部排除 holdout。按现有 demo 实测估计，三轮 demo 阶段的调用数约为 88/176/367/733，prompt tokens 约为 0.28M/0.56M/1.17M/2.34M。本轮没有执行 100/200 条高成本实验。

请 GPT Work 重点讨论两项决定：

1. 是否先运行 24 或 48 条分层 demo，以判断增加 moving evidence 后配对区间是否收窄，再决定是否扩大到 100/200。
2. 是否设计一个小规模、独立于当前 objective 的质量核验，例如人工标注 moving 子集或真实 OSM 中心线距离。该指标应作为 secondary evaluation，暂不写入 primary objective。

建议审阅顺序：`实验课1.pptx` 相关页 → 本文件 → 真实实验 `实验报告.md` 与 `summary.json` → runner/analyzer 源码 → verifier、memory 源码 → Notebook 与 PPT。
