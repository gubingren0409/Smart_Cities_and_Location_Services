"""核验器：把 LLM 的建议变成一个可判分的数字。

这是任务③「再用定量指标核验建议」的核心，也是整个工作流的准入闸门：
**只有通过核验的建议才允许写入记忆。**

核验维度
-------
1. 约束满足  —— 提议是否落在物理先验区间内（越界即夹紧并记录）
2. 可行性    —— 是否违反长度/偏差硬约束
3. 方向一致  —— LLM 预测的升降方向与实测符号是否一致
4. regret    —— 相对确定性搜索最优解的差距（关键判据）
5. 样本效率  —— 达到最优 95% 所需的评估次数
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core import params as params_mod
from . import objective as obj_mod
from . import search as search_mod


@dataclass
class DirectionCheck:
    """方向一致性检查结果。"""

    param: str
    predicted: str                  # "up" | "down" | "same" | "unknown"
    observed: str
    agree: Optional[bool] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "param": self.param,
            "predicted": self.predicted,
            "observed": self.observed,
            "agree": self.agree,
        }


def _sign_word(before: float, after: float, tol: float = 1e-9) -> str:
    d = float(after) - float(before)
    if d > tol:
        return "up"
    if d < -tol:
        return "down"
    return "same"


def check_direction(predicted: Dict[str, str],
                    observed: Dict[str, float],
                    baseline: Dict[str, float]) -> List[DirectionCheck]:
    """对比 LLM 预测的方向与实测方向。

    predicted 形如 {"dp_tolerance": "up", "quality": "down"}；
    observed/baseline 给出各量的实际取值。
    只有同时出现在三个字典里的键才参与判定。
    """
    out: List[DirectionCheck] = []
    for key, pred in predicted.items():
        if key not in observed or key not in baseline:
            continue
        pred_norm = str(pred).strip().lower()
        if pred_norm not in ("up", "down", "same"):
            out.append(DirectionCheck(key, pred_norm, "unknown", None))
            continue
        obs = _sign_word(baseline[key], observed[key])
        out.append(DirectionCheck(key, pred_norm, obs, pred_norm == obs))
    return out


def direction_accuracy(checks: Sequence[DirectionCheck]) -> Optional[float]:
    """方向准确率。无有效检查项时返回 None（不要用 0 冒充）。"""
    judged = [c for c in checks if c.agree is not None]
    if not judged:
        return None
    return sum(1 for c in judged if c.agree) / len(judged)


@dataclass
class VerificationResult:
    """一次建议的完整核验结果。"""

    # 约束
    clamped_params: List[str] = field(default_factory=list)
    constraint_ok: bool = True

    # 可行性
    feasible: bool = True
    violations: List[str] = field(default_factory=list)

    # 分数
    proposal_score: float = 0.0
    baseline_score: float = 0.0
    best_score: float = 0.0
    regret: float = 0.0                 # best - proposal（>=0 表示不如最优）

    # 方向
    direction_checks: List[DirectionCheck] = field(default_factory=list)
    direction_accuracy: Optional[float] = None

    # 样本效率
    n_evaluations: int = 0
    evals_to_95pct: Optional[int] = None

    # 目标函数是否适用于本条轨迹
    applicable: bool = True
    inapplicable_reason: str = ""

    # regret 口径（normalized / absolute(small_headroom) / not_applicable）
    regret_basis: str = ""

    # 记忆准入
    admitted: bool = False
    admit_reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "constraint_ok": self.constraint_ok,
            "clamped_params": list(self.clamped_params),
            "feasible": self.feasible,
            "violations": list(self.violations),
            "proposal_score": round(self.proposal_score, 6),
            "baseline_score": round(self.baseline_score, 6),
            "best_score": round(self.best_score, 6),
            "regret": round(self.regret, 6),
            "regret_basis": self.regret_basis,
            "applicable": self.applicable,
            "inapplicable_reason": self.inapplicable_reason,
            "direction_checks": [c.to_dict() for c in self.direction_checks],
            "direction_accuracy": (None if self.direction_accuracy is None
                                   else round(self.direction_accuracy, 4)),
            "n_evaluations": self.n_evaluations,
            "evals_to_95pct": self.evals_to_95pct,
            "admitted": self.admitted,
            "admit_reason": self.admit_reason,
        }


# 当归一化分母（headroom）过小时，归一化 regret 会失去意义，
# 此时退回绝对 regret 并据此判定。阈值取目标函数分数的一个小量。
MIN_HEADROOM = 0.02
ABSOLUTE_REGRET_TOL = 0.01


def compute_regret(proposal_score: float, best_score: float,
                   baseline_score: float = 0.0,
                   normalize: bool = True,
                   applicable: bool = True) -> Tuple[float, str]:
    """返回 (regret, 口径说明)。

    regret = 最优分 - 提议分。normalize=True 时按可达增益归一化，
    使 regret 成为「丢掉了多少比例的可得收益」，可跨轨迹比较。

    **headroom 退化**：当 best 与 baseline 几乎相等（差 < MIN_HEADROOM）时，
    分母趋零会把任何微小差距放大成 regret=1.0。
    实测案例：车辆 246 上 best=0.6366、baseline=0.6159、提议=0.6159，
    headroom 仅 0.021，提议与基线完全持平却被判为 regret=1.0 —— 明显不合理。
    此时改用绝对 regret（阈值 ABSOLUTE_REGRET_TOL），
    含义回到「提议是否与最优相差无几」。
    """
    gap = max(0.0, float(best_score) - float(proposal_score))
    if not normalize:
        return gap, "absolute(disabled)"
    if not applicable:
        # 目标函数不适用时 regret 无意义，一律返回 0，
        # 由调用方按 regime 正确性与约束合规性判定。
        return 0.0, "not_applicable"
    headroom = float(best_score) - float(baseline_score)
    if headroom < MIN_HEADROOM:
        # 分母过小会把微小差距放大成 regret=1.0
        return gap, "absolute(small_headroom)"
    return max(0.0, min(1.0, gap / headroom)), "normalized"


def evals_to_reach(results: Sequence[Tuple[Dict[str, float], float]],
                   best_score: float, ratio: float = 0.95) -> Optional[int]:
    """按评估顺序，首次达到 `ratio × 最优分` 时的累计评估次数。"""
    if not results:
        return None
    if best_score <= 0:
        target = best_score * ratio
    else:
        target = best_score * ratio
    for i, (_, s) in enumerate(results, start=1):
        if s >= target:
            return i
    return None


def verify_proposal(proposal_params: Dict[str, float],
                    proposal_objective: obj_mod.ObjectiveResult,
                    baseline_objective: obj_mod.ObjectiveResult,
                    search_trace: search_mod.SearchTrace,
                    predicted_effects: Optional[Dict[str, str]] = None,
                    baseline_metrics: Optional[Dict[str, float]] = None,
                    observed_metrics: Optional[Dict[str, float]] = None,
                    regret_threshold: float = 0.05) -> VerificationResult:
    """核验一条建议，并判定是否准予写入记忆。

    参数
    ----
    proposal_params     : LLM 提议的参数（已经过夹紧）
    proposal_objective  : 执行该提议得到的目标函数结果
    baseline_objective  : 默认/基线参数的结果（regret 的归一化基准）
    search_trace        : 确定性搜索轨迹（提供最优分与评估序列）
    predicted_effects   : LLM 预测的各量方向
    baseline_metrics / observed_metrics : 方向判定所需的实测值
    regret_threshold    : 归一化 regret 上限，超过则不准入记忆
    """
    res = VerificationResult()
    res.proposal_score = float(proposal_objective.score)
    res.baseline_score = float(baseline_objective.score)
    res.best_score = float(search_trace.best_score)

    # 1) 约束满足：提议参数是否越界
    _, clamped = params_mod.clamp_params(proposal_params)
    res.clamped_params = list(clamped)
    res.constraint_ok = len(clamped) == 0

    # 2) 可行性
    res.feasible = bool(proposal_objective.feasible)
    res.violations = list(proposal_objective.violations)
    res.applicable = bool(getattr(proposal_objective, "applicable", True))
    res.inapplicable_reason = str(
        getattr(proposal_objective, "inapplicable_reason", "") or "")

    # 3) regret
    # 分母优先用搜索记录的固定参照系分数；退回参数里的 baseline_score。
    # 不这样做的话，若调用方把 baseline 传成了「搜索起点」（恰好等于提议），
    # headroom 会被压到接近 0，归一化 regret 虚高成 1.0。
    ref_baseline = float(getattr(search_trace, "baseline_score", 0.0) or 0.0)
    if ref_baseline <= 0.0:
        ref_baseline = res.baseline_score
    res.regret, res.regret_basis = compute_regret(
        res.proposal_score, res.best_score, ref_baseline,
        applicable=res.applicable)

    # 4) 方向一致性
    if predicted_effects:
        res.direction_checks = check_direction(
            predicted_effects,
            observed_metrics or {},
            baseline_metrics or {},
        )
        res.direction_accuracy = direction_accuracy(res.direction_checks)

    # 5) 样本效率
    res.n_evaluations = search_trace.n_evaluations
    res.evals_to_95pct = evals_to_reach(search_trace.evaluations, res.best_score, 0.95)

    # 6) 记忆准入闸门：三条同时满足才准入
    reasons: List[str] = []
    if not res.applicable:
        # 目标函数不适用：改按「参数是否落在物理先验内 + 未违规」判定，
        # 不再要求 regret —— 对静止轨迹要求 regret 是没有意义的。
        if res.constraint_ok:
            res.admitted = True
            res.admit_reason = (
                f"目标函数不适用（{res.inapplicable_reason}）；"
                "参数合规且无硬约束违规，按 regime 正确性准入")
        else:
            res.admitted = False
            res.admit_reason = (
                f"目标函数不适用且参数越界: {', '.join(res.clamped_params)}")
        return res
    if not res.feasible:
        reasons.append("提议不可行（违反长度或偏差约束）")
    if not res.constraint_ok:
        reasons.append(f"参数越界被夹紧: {', '.join(res.clamped_params)}")
    # 口径不同则阈值不同：绝对口径下要求提议与最优的分数差小于绝对容差
    effective_threshold = (regret_threshold if res.regret_basis == "normalized"
                           else ABSOLUTE_REGRET_TOL)
    if res.regret > effective_threshold:
        reasons.append(
            f"regret {res.regret:.4f}（{res.regret_basis}）超过阈值 {effective_threshold}")
    if reasons:
        res.admitted = False
        res.admit_reason = "；".join(reasons)
    else:
        res.admitted = True
        res.admit_reason = (
            f"可行、参数未越界、regret {res.regret:.4f}（{res.regret_basis}）"
            f"<= {effective_threshold}")
    return res


def ablation_summary(cases: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """消融表：对比 LLM-only / search-only / LLM+search / LLM+memory+search。

    **关于跨模式可比性（重要）**

    `regret` 的口径随模式变化：
      - 含搜索：最优来自确定性搜索 -> 归一化 regret
      - 关搜索：本模式可达上限就是基线 -> 绝对口径
    两者不能直接比较，表中已按口径分列。

    为提供一个**全模式通用**的判据，本表同时给出：
      mean_score              提议参数的实测目标函数得分（原始口径，恒可比）
      mean_gap_to_baseline    提议得分 - 物理先验默认参数的得分（绝对，恒可比）
      baseline_beaten_rate    提议显著优于基线的比例（绝对，恒可比）

    结论页应当以这三列为准；regret 只在同口径的模式之间比较。
    """
    groups: Dict[str, List[Dict[str, object]]] = {}
    for c in cases:
        groups.setdefault(str(c.get("mode", "unknown")), []).append(c)

    def mean(vals: List[float]) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    rows: List[Dict[str, object]] = []
    for mode, items in sorted(groups.items()):
        # 跨模式通用判据：提议得分相对基线的绝对提升
        gaps = []
        for i in items:
            ps = _num(i.get("proposal_score"))
            bs = _num(i.get("baseline_score"))
            if ps is not None and bs is not None:
                gaps.append(ps - bs)
        beat = [1.0 if g > 1e-9 else 0.0 for g in gaps]
        rows.append({
            "mode": mode,
            "n": len(items),
            "mean_regret": mean([_num(i.get("regret")) for i in items]),
            "regret_basis_mixed": len({str(i.get("regret_basis")) for i in items}) > 1,
            "mean_score": mean([_num(i.get("proposal_score")) for i in items]),
            "mean_baseline_score": mean([_num(i.get("baseline_score")) for i in items]),
            "mean_gap_to_baseline": mean(gaps),
            "baseline_beaten_rate": mean(beat),
            "mean_evals": mean([_num(i.get("n_evaluations")) for i in items]),
            "mean_direction_accuracy": mean([_num(i.get("direction_accuracy")) for i in items]),
            "admit_rate": mean([1.0 if i.get("admitted") else 0.0 for i in items]),
            "n_with_memory": sum(1 for i in items if i.get("n_neighbors_available")),
        })
    for r in rows:
        for k in ("mean_regret", "mean_score", "mean_baseline_score",
                  "mean_gap_to_baseline", "baseline_beaten_rate", "mean_evals",
                  "mean_direction_accuracy", "admit_rate"):
            if r[k] is not None:
                r[k] = round(r[k], 4)
    return rows


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v
