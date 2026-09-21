# traj_agent — LLM 辅助的轨迹清洗评估工作流

对应课程作业的**任务③**：让 LLM 选择评估工具并提出参数建议，再用定量指标核验建议。

完整文件导航见 [任务三_代码与目录清单.md](任务三_代码与目录清单.md)。当前真实模型实验结论见 [experiments/llm_assisted_ecnu_20260921/实验报告.md](experiments/llm_assisted_ecnu_20260921/实验报告.md)；`llm_assisted_20260920` 保留为 Mock 对照。

---

## 1. 它解决的三个问题

| 问题 | 做法 |
|---|---|
| **LLM 猜参数不准** | 物理先验约束空间 → LLM 独立提候选 → 确定性程序执行 → 独立搜索建立内部参考 |
| **无法判断 LLM 的建议好不好** | `regret` = 相对有界确定性搜索参考的差距，**可自动判分** |
| **11386 条轨迹全跑 LLM 太贵** | 记忆分层：跑少数、复用多数 |

## 2. 闭环流程

```
[1] 载入轨迹      → 诊断卡（只含标量统计，不含坐标）
[2] 检索记忆      → L2 参数区间 + L3 人类知识（Obsidian）
[3] LLM 提议      → {params, expected_effect, rationale}
[4] 核验约束      → 纯代码，无 LLM
[5] 执行提议      → 实测指标
[6] 确定性搜索    → 从默认参数开始，建立内部参考
[7] regret        → LLM 候选 vs 内部参考
[8] 记忆留档      → 全部进 L1；仅达标案例可检索并进入 L2
```

**第 3 步与第 7–8 步之间是全部价值所在**：它把「LLM 说得好不好」
变成一个可自动判分的数字，而不是让人去读一段解释。

## 3. 快速开始

```bash
# 离线跑通（无需 API key）
python -m pip install -r requirements.txt
python -m pytest tests -q

# 接 ECNU 真实模型
export DSH_TRAJ_LLM_API_KEY=sk-xxx
export DSH_TRAJ_LLM_BASE_URL=https://chat.ecnu.edu.cn/open/api/v1
export DSH_TRAJ_LLM_MODEL=ecnu-plus
```

```python
from traj_agent.core import traj as tm
from traj_agent.agent.loop import TrajCleaningAgent
from traj_agent.agent import provider as prov
from traj_agent.memory.store import MemoryStore

raw = tm.load_raw("traj_dict.json")
mem = MemoryStore("memory.sqlite")
agent = TrajCleaningAgent(llm=prov.build_provider(), memory=mem)
agent.attach_dataset(raw)

result = agent.run("246")
print(result.proposal_params)          # LLM 提的参数
print(result.verification["regret"])   # 相对最优的差距
print(result.verification["admitted"]) # 是否够格进记忆
```

演示 notebook：`任务3_LLM辅助评估清洗.ipynb`

## 4. 目录结构

```
traj_agent/
├── core/       纯确定性算法（零 LLM 依赖，可单测）
│   ├── geo.py         坐标与距离（局部等距投影）
│   ├── traj.py        轨迹数据模型
│   ├── segment.py     按时间/空间阈值切分        ← 任务①
│   ├── anomalies.py   异常点规则 + 原因字段      ← 任务①
│   ├── clean.py       去噪（保留原因账本）        ← 任务①
│   ├── simplify.py    Douglas-Peucker            ← 任务①
│   ├── metrics.py     点数/长度/Hausdorff/Fréchet/耗时 ← 任务①
│   ├── diagnosis.py   诊断卡（喂给 LLM 的唯一视图）
│   └── params.py      参数物理先验
├── road/       路网匹配协议 + 离线实现           ← 任务①新需求（接口就位）
├── tools/      JSON-Schema 工具层（LLM 唯一能碰的层）
├── verifier/   目标函数、knee point、搜索、核验
├── memory/     SQLite 四层记忆 + 特征 kNN
├── agent/      ReAct 主循环 + provider
└── report/     六类图
```

## 5. 四层记忆

| 层 | 存储 | 谁写 | 检索方式 |
|---|---|---|---|
| L0 工作记忆 | 上下文 | agent | — |
| L1 情景记忆 | SQLite 全量 `(诊断→参数→指标)` | **核验器** | — |
| L2 程序记忆 | SQLite 蒸馏 `诊断签名→参数区间` | **核验器** | 12 维特征 kNN |
| L3 人类知识 | Obsidian `.md` | **人** | 只读遍历 |

L1 是完整实验账本，admitted=False 的失败案例也会保存；默认检索只读取
admitted=1，L2 也只从这些已准入案例蒸馏。这样可以分析失败原因，同时
避免未通过核验的参数影响后续推荐。

**agent 对 Obsidian vault 只有读权限，永远不能写。**
自动导出（`memory/export.py`）落到 `00-Inbox/`，人工 review 后才升格。
这样 agent 的任何 bug 都不可能损坏你的第二大脑。

设置 vault 路径：

```bash
export DSH_TRAJ_VAULT_DIR=/path/to/YourVault
```

未设置时回退到工作目录下的 `vault_stub/`（含 3 篇示例笔记）。
Obsidian vault 就是一堆 `.md`，Python 用标准库直接读写，
不需要 Obsidian 在运行，也不需要任何插件。

## 6. 关键设计约束

1. **LLM 不碰数值，也不当优化器。** 所有几何/统计计算在 `core/` 里。
2. **LLM 不能写记忆。** 没有 `write_memory` 工具——这是设计，不是遗漏。
3. **坐标永不进上下文。** 只传句柄 `246#0@v3` 与诊断卡。
4. **提议与核验分离。** 两个独立步骤，专门制造 generator–verifier gap。
5. **`core/` 不 import 任何上层模块。** 由 `tests/test_architecture.py` 强制。

## 7. 抽样数据中观察到的两个现象

这两点来自不同阶段的抽样分析，详见 `DECISIONS.md` 与正式实验报告：

- 静止、混合和行驶轨迹的统计会随抽样方式明显变化。早期排序样本、早期分层样本和固定随机 200 条样本不能混为全量比例。正式实验的固定随机样本中，静止、混合和行驶分别为 47.0%、24.0% 和 29.0%。
  三类轨迹的参数需求不同，诊断卡需要先识别轨迹状态。
- **连续重复点占比中位数 52.8%**，静止轨迹可达 90%。
  折叠重复点是收益最高的一步，应先做它再压缩。

另有若干实现陷阱（墨卡托把距离放大 17%、DP 偏差事后估计错 3 个量级、
离散顶点集 Hausdorff 在简化场景下给出 548m 而真实是 4.77m）
全部记录在 `DECISIONS.md` 并配了回归测试。

## 8. 消融实验

| 模式 | LLM | 记忆 | 搜索 | 说明 |
|---|---|---|---|---|
| `llm-only` | 有 | 无 | 无 | 单靠 LLM，无实测依据 |
| `search-only` | **无** | 无 | 有 | 纯确定性基线，`use_llm=False` |
| `llm+search` | 有 | 无 | 有 | 加实测依据 |
| `llm+memory+search` | 有 | 有 | 有 | 加历史经验 |

`search-only` 必须真的不调用 LLM，否则与含 LLM 的模式不可比。

**注意**：关闭搜索时 regret 是绝对口径（最优=基线），
**不能**与含搜索模式的归一化 regret 直接比较。代码会在
`result.search["caveat"]` 里标注这一点。

### 真实 ECNU 三轮结果

固定使用 12 条 demo、12 条不重叠 holdout，四种模式各重复三轮。目标函数适用的样本每模式为 24 个；12 个静止短轨迹标为 `applicable=False`，不进入普通得分均值。

| 模式 | 相对固定基线平均变化 | 95% bootstrap CI | 超过基线比例 | LLM 调用 |
|---|---:|---:|---:|---:|
| `llm-only` | -0.000050 | [-0.006178, 0.005441] | 58.3% | 36 |
| `search-only` | -0.000555 | [-0.001081, -0.000142] | 37.5% | **0** |
| `llm+search` | +0.000429 | [-0.005874, 0.006810] | 54.2% | 39 |
| `llm+memory+search` | +0.001645 | [-0.004523, 0.007489] | 58.3% | 48 |

Memory 与无 Memory 的配对差值均值为 +0.001236，95% CI 为 [-0.000851, 0.003498]，区间覆盖 0。三轮 L2 参数区域各自只有 1–2 个准入样本，其中 `moving/degraded` 三轮合计仅准入 1 条，因此当前实验尚未检出稳定的 Memory 增益，也不能据此判断 Memory 机制无效。

运行入口：

```bash
python experiments/llm_assisted_ecnu_20260921/run_real_ablation.py --run
```

脚本逐 case 写入 JSONL，可从中断处恢复；若 provider 回退为 Mock 会立即终止。完整调用、token、延迟、分层与置信区间均保存在该实验目录。
