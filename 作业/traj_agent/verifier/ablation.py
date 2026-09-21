"""消融实验：先在示范集上积累记忆，再在**留出集**上评测。

为什么必须分两组
---------------
若在评测轨迹上边跑边攒记忆，agent 会把「这条轨迹自己的上一次结果」
检索回来当作先验——这是信息泄漏，会让消融表得出完全错误的结论。
本模块强制两阶段：

    phase 1（demo 集）：跑 agent 并写入记忆，产出 L2 参数区间
    phase 2（holdout 集）：只读记忆，**不再写入**，评测各模式

另外，regret 的口径必须按模式区分并显式标注：
  - 含搜索：最优来自确定性搜索 -> 归一化 regret 可跨轨迹比较
  - 关搜索：本模式可达上限就是基线 -> 绝对口径，**不可**与前者直接比较
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..agent import provider as provider_mod
from ..agent.loop import TrajCleaningAgent
from ..memory.store import MemoryStore
from ..tools.registry import ToolRegistry, attach_dataset
from . import verify as verify_mod

# 四种模式的配置。regret_threshold 用于 phase 1 的准入，
# 设得合理（而非 1.0）才能让 access 门控真正起作用。
MODE_SPECS: Dict[str, Dict[str, Any]] = {
    "llm-only": dict(use_llm=True, use_memory=False, use_search=False,
                     regret_threshold=0.05),
    "search-only": dict(use_llm=False, use_memory=False, use_search=True,
                        regret_threshold=0.05),
    "llm+search": dict(use_llm=True, use_memory=False, use_search=True,
                       regret_threshold=0.05),
    "llm+memory+search": dict(use_llm=True, use_memory=True, use_search=True,
                              regret_threshold=0.05),
}


@dataclass
class AblationRun:
    """一次完整消融实验的结果。"""

    rows: List[Dict[str, Any]] = field(default_factory=list)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    demo_vehicles: List[str] = field(default_factory=list)
    holdout_vehicles: List[str] = field(default_factory=list)
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    baseline_accuracy: Optional[float] = None
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "demo_vehicles": self.demo_vehicles,
            "holdout_vehicles": self.holdout_vehicles,
            "memory_stats": self.memory_stats,
            "baseline_accuracy": self.baseline_accuracy,
            "caveats": self.caveats,
            "n_cases": len(self.cases),
        }


def run_ablation(raw: Dict[str, Any],
                 demo_vehicles: Sequence[str],
                 holdout_vehicles: Sequence[str],
                 memory_path: str = ":memory:",
                 modes: Optional[Sequence[str]] = None,
                 llm: Optional[Any] = None,
                 vault_dir: Optional[str] = None) -> AblationRun:
    """执行两阶段消融实验。"""
    mode_names = list(modes) if modes else list(MODE_SPECS)
    llm = llm or provider_mod.MockProvider()
    overlap = set(demo_vehicles) & set(holdout_vehicles)
    if overlap:
        raise ValueError(
            f"demo 集与 holdout 集不能重叠（发现 {sorted(overlap)}）；"
            "重叠会让记忆直接看到评测答案，消融结果无效")
    out = AblationRun(demo_vehicles=list(demo_vehicles),
                      holdout_vehicles=list(holdout_vehicles))

    # ---------- phase 1：在 demo 集上积累记忆 ----------
    # 只有 llm+memory+search 模式写入记忆；其余模式不写，
    # 否则「无记忆」模式会因为别人写的记忆而受益，混淆归因。
    mem = MemoryStore(memory_path)
    writer = TrajCleaningAgent(llm=llm, memory=mem,
                               vault_dir=vault_dir,
                               mode="llm+memory+search",
                               **MODE_SPECS["llm+memory+search"])
    writer.attach_dataset(raw)
    for vid in demo_vehicles:
        writer.run(vid)
    mem.rebuild_procedural(min_samples=1)
    out.memory_stats = mem.stats()

    # ---------- phase 2：在 holdout 集上评测（不再写入） ----------
    for mode in mode_names:
        spec = dict(MODE_SPECS[mode])
        if not spec.get("use_memory"):
            # 无记忆模式共享同一个只读库不会污染（因为不检索），
            # 但保持 None 更干净，也避免误用
            pass
        for vid in holdout_vehicles:
            reg = ToolRegistry()
            attach_dataset(reg.ctx, raw)
            # use_memory=True 的模式读到 phase 1 攒下的记忆；
            # read_only=True 保证评测阶段**不写回**，
            # 否则第二条 holdout 轨迹就能检索到第一条的结果（跨样本泄漏）。
            agent_mem = mem if spec.get("use_memory") else None
            ag = TrajCleaningAgent(registry=reg, llm=llm, memory=agent_mem,
                                   vault_dir=vault_dir, mode=mode,
                                   read_only=True, **spec)
            r = ag.run(vid)
            if not r.ok:
                continue
            d = r.to_dict(with_trace=False)
            v = d["verification"]
            out.cases.append({
                "mode": mode, "vehicle_id": vid,
                "regret": v["regret"], "regret_basis": v["regret_basis"],
                "proposal_score": v["proposal_score"],
                # baseline_score 提供跨模式通用的比较基线
                "baseline_score": v["baseline_score"],
                "n_evaluations": d["llm_tool_calls"],
                "direction_accuracy": v["direction_accuracy"],
                "admitted": v["admitted"],
                "applicable": v["applicable"],
                "n_neighbors_available": bool(d["memory_prior_available"]),
            })

    out.rows = verify_mod.ablation_summary(out.cases)
    # 标注方向准确率的可用性
    accs = [c["direction_accuracy"] for c in out.cases
            if c["direction_accuracy"] is not None]
    out.baseline_accuracy = sum(accs) / len(accs) if accs else None

    out.caveats = [
        "评测在**留出集**上进行：phase 1 在 demo 集积累记忆，phase 2 只读不写（read_only=True）。",
        "检索排除自身（exclude_self=True），demo 与 holdout 重叠会直接报错。",
        "**regret 口径随模式变化**：关闭搜索时为绝对口径（该模式可达上限即基线），",
        "不可与含搜索模式的归一化 regret 直接比较。跨模式结论请用",
        "mean_gap_to_baseline 与 baseline_beaten_rate —— 这两列是绝对口径，恒可比。",
        "search-only 不含任何 LLM 调用（use_llm=False），是纯确定性基线。",
    ]
    if mem.path != ":memory:":
        mem.close()
    else:
        mem.close()
    return out
