# traj_agent — LLM 辅助的轨迹清洗评估工作流

对应课程作业的**任务③**：让 LLM 选择评估工具并提出参数建议，再用定量指标核验建议。

---

## 1. 它解决的三个问题

| 问题 | 做法 |
|---|---|
| **LLM 猜参数不准** | 物理先验定区间 → LLM 提起点与方向 → 确定性搜索精调 |
| **无法判断 LLM 的建议好不好** | `regret` = 相对确定性搜索最优的差距，**可自动判分** |
| **11386 条轨迹全跑 LLM 太贵** | 记忆分层：跑少数、复用多数 |

## 2. 闭环流程

```
[1] 载入轨迹      → 诊断卡（只含标量统计，不含坐标）
[2] 检索记忆      → L2 参数区间 + L3 人类知识（Obsidian）
[3] LLM 提议      → {params, expected_effect, rationale}
[4] 核验约束      → 纯代码，无 LLM
[5] 执行提议      → 实测指标
[6] 确定性搜索    → ground truth 最优
[7] regret        → 提议 vs 最优（这是判据）
[8] 记忆准入      → 仅当 regret 达标才写入 SQLite
```

**第 3 步与第 7–8 步之间是全部价值所在**：它把「LLM 说得好不好」
变成一个可自动判分的数字，而不是让人去读一段解释。

## 3. 快速开始

```bash
# 离线跑通（无需 API key）
python3 -m pytest tests/ -q          # 148+ 项测试

# 接真实模型（任何 OpenAI 兼容端点）
export DSH_TRAJ_LLM_API_KEY=sk-xxx
export DSH_TRAJ_LLM_BASE_URL=https://api.deepseek.com/v1
export DSH_TRAJ_LLM_MODEL=deepseek-chat
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

## 7. 本数据集实测的两个意外事实

这两点是实现过程中被数据推翻的假设，详见 `DECISIONS.md`：

- **约 63% 的车辆全程静止**（位移中位 42 m），37% 在真实行驶。
  对这两类用同一套阈值不可能合理——这是智能体应当自动识别的第一件事。
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
