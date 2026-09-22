"""固定 Objective 下的 teacher data、区域提议与 warm-start 搜索。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
import random
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..agent import provider as provider_mod
from ..agent.loop import TrajCleaningAgent
from ..core import diagnosis, params as params_mod, traj as traj_mod
from ..memory import features as feature_mod, retrieve as retrieve_mod
from ..memory.store import MemoryStore
from ..tools.registry import ToolRegistry, attach_dataset
from ..verifier import search as search_mod
from ..verifier.verify import MIN_HEADROOM


TEACHER_SOURCE = "deterministic-search-reference"
LLM_SOURCE = "llm-verified"
DEFAULT_TEACHER_SEED = 20260922
DEFAULT_NEAR_EPSILON = 0.01
DEFAULT_BUDGETS: Tuple[int, ...] = (3, 5, 10, 20)
DEFAULT_TEACHER_SIZES: Tuple[int, ...] = (12, 24, 48, 100, 200)
MEMORY_MODES: Tuple[str, ...] = (
    "none",
    "episodic-only",
    "procedural-only",
    "episodic+procedural",
)


def active_bounds() -> search_mod.SearchBounds:
    return search_mod.full_parameter_bounds(params_mod.ACTIVE_EXECUTION_PARAMS)


def teacher_manifest(
    raw: Mapping[str, Any],
    *,
    holdout: Sequence[str],
    excluded_demo: Sequence[str] = (),
    size: int = 200,
    seed: int = DEFAULT_TEACHER_SEED,
) -> Dict[str, Any]:
    """固定种子按六个诊断层轮转抽样，且排除 v2 demo/holdout。"""
    excluded = {str(x) for x in holdout} | {str(x) for x in excluded_demo}
    buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for vehicle_id in sorted(raw, key=lambda x: (len(str(x)), str(x))):
        vehicle_id = str(vehicle_id)
        if vehicle_id in excluded:
            continue
        card = diagnosis.diagnose(
            traj_mod.traj_from_raw(vehicle_id, *raw[vehicle_id]))
        buckets[(card.regime, card.timeline_quality)].append(vehicle_id)

    rng = random.Random(int(seed))
    strata = [
        (regime, timeline)
        for regime in ("stationary", "mixed", "moving")
        for timeline in ("ok", "degraded")
    ]
    for key in strata:
        rng.shuffle(buckets[key])

    selected: List[str] = []
    offsets: Dict[Tuple[str, str], int] = defaultdict(int)
    while len(selected) < int(size):
        progressed = False
        for key in strata:
            position = offsets[key]
            if position < len(buckets[key]):
                selected.append(buckets[key][position])
                offsets[key] += 1
                progressed = True
                if len(selected) == int(size):
                    break
        if not progressed:
            raise ValueError("可用轨迹不足以建立 teacher set")

    counts: Dict[str, int] = defaultdict(int)
    for vehicle_id in selected:
        card = diagnosis.diagnose(
            traj_mod.traj_from_raw(vehicle_id, *raw[vehicle_id]))
        counts[f"{card.regime}/{card.timeline_quality}"] += 1
    return {
        "seed": int(seed),
        "strategy": "六个(regime,timeline_quality)层固定种子打乱后等权轮转",
        "vehicle_ids": selected,
        "n_unique_vehicles": len(set(selected)),
        "strata": dict(sorted(counts.items())),
        "excluded_v2_demo": [str(x) for x in excluded_demo],
        "holdout": [str(x) for x in holdout],
        "teacher_holdout_overlap": sorted(set(selected) & {str(x) for x in holdout}),
        "teacher_v2_demo_overlap": sorted(
            set(selected) & {str(x) for x in excluded_demo}),
        "nested_sizes": {
            str(n): selected[:n]
            for n in DEFAULT_TEACHER_SIZES if n <= len(selected)
        },
    }


@dataclass
class TrajectoryEvaluator:
    """复用现有 Agent 执行器和 Objective，避免复制评分语义。"""

    raw: Mapping[str, Any]
    vehicle_id: str
    registry: ToolRegistry = field(init=False)
    agent: TrajCleaningAgent = field(init=False)
    handle: str = field(init=False)
    card: diagnosis.DiagnosisCard = field(init=False)
    cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vehicle_id = str(self.vehicle_id)
        self.registry = ToolRegistry()
        attach_dataset(self.registry.ctx, dict(self.raw))
        self.agent = TrajCleaningAgent(
            registry=self.registry,
            llm=provider_mod.MockProvider(),
            memory=None,
            use_llm=False,
            use_memory=False,
            use_search=False,
            read_only=True,
            mode="objective-evaluator",
        )
        loaded = self.registry.call(
            "load_trajectory", vehicle_id=self.vehicle_id, segment_index=0)
        if not loaded.get("ok"):
            raise ValueError(loaded.get("error") or "轨迹载入失败")
        self.handle = str(loaded["handle"])
        self.card = diagnosis.diagnose(self.registry.ctx.store.get(self.handle))

    @staticmethod
    def _key(params: Mapping[str, float]) -> str:
        return json.dumps(
            {k: round(float(v), 8) for k, v in sorted(params.items())},
            sort_keys=True,
        )

    def evaluate(self, params: Dict[str, float], *, cache: bool = True) -> float:
        key = self._key(params)
        if cache and key in self.cache:
            return float(self.cache[key]["objective"]["score"])
        measured, _ = self.agent._execute(self.handle, params)
        objective = self.agent._objective_of(measured, params).to_dict()
        if cache:
            self.cache[key] = {
                "params": dict(params),
                "measured": measured,
                "objective": objective,
            }
        return float(objective["score"])

    def details(self, params: Dict[str, float]) -> Dict[str, Any]:
        self.evaluate(params)
        return dict(self.cache[self._key(params)])


def run_teacher_case(
    raw: Mapping[str, Any],
    vehicle_id: str,
    *,
    max_evals: int = 60,
) -> Dict[str, Any]:
    evaluator = TrajectoryEvaluator(raw, str(vehicle_id))
    defaults = params_mod.active_default_params()
    started = time.perf_counter()
    trace = search_mod.coordinate_descent(
        base_params=defaults,
        search_params=params_mod.ACTIVE_EXECUTION_PARAMS,
        evaluator=evaluator.evaluate,
        n_steps=5,
        rounds=3,
        max_evals=int(max_evals),
    )
    default_details = evaluator.details(defaults)
    best_details = evaluator.details(trace.best_params)
    trace.baseline_params = dict(defaults)
    trace.baseline_score = float(default_details["objective"]["score"])
    return {
        "record_type": "teacher",
        "vehicle_id": str(vehicle_id),
        "seg_id": evaluator.card.seg_id,
        "regime": evaluator.card.regime,
        "timeline_quality": evaluator.card.timeline_quality,
        "diagnosis": evaluator.card.to_dict(),
        "diagnosis_features": feature_mod.vector_to_dict(
            feature_mod.featurize(evaluator.card)),
        "default_params": defaults,
        "default_score": trace.baseline_score,
        "default_objective": default_details["objective"],
        "search_best_params": dict(trace.best_params),
        "search_best_score": float(trace.best_score),
        "search_best_objective": best_details["objective"],
        "search_best_measured": best_details["measured"],
        "search_trace": [
            {"params": dict(p), "score": float(score)}
            for p, score in trace.evaluations
        ],
        "n_evaluations": int(trace.n_evaluations),
        "search_budget": int(max_evals),
        "source": TEACHER_SOURCE,
        "applicable": bool(best_details["objective"].get("applicable")),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "llm_calls": 0,
    }


def populate_search_memory(
    store: MemoryStore,
    raw: Mapping[str, Any],
    teacher_rows: Sequence[Mapping[str, Any]],
    *,
    min_samples: int = 3,
) -> Dict[str, Any]:
    """把 bounded reference 明确写为独立来源，不伪装成 LLM 经验。"""
    for row in teacher_rows:
        vehicle_id = str(row["vehicle_id"])
        card = diagnosis.diagnose(
            traj_mod.traj_from_raw(vehicle_id, *raw[vehicle_id]))
        applicable = bool(row.get("applicable"))
        store.write_case(
            card=card,
            params=dict(row["search_best_params"]),
            metrics=dict(row.get("search_best_measured") or {}),
            objective=dict(row.get("search_best_objective") or {}),
            verification={
                "source": TEACHER_SOURCE,
                "search_budget": int(row.get("search_budget") or 0),
                "applicable": applicable,
            },
            admitted=applicable,
            admit_reason=(
                "固定 Objective 下的 bounded internal reference"
                if applicable else "Objective 对该轨迹不适用，仅保留记录"
            ),
            mode="search-teacher",
            run_id="objective-optimization-teacher",
            score=float(row["search_best_score"]),
            regret=0.0 if applicable else None,
            source=TEACHER_SOURCE,
        )
    region_count = store.rebuild_procedural(
        param_names=params_mod.ACTIVE_EXECUTION_PARAMS,
        min_samples=int(min_samples),
        source=TEACHER_SOURCE,
    )
    memory_stats = store.stats()
    memory_stats.pop("path", None)
    return {
        "min_samples": int(min_samples),
        "region_count": int(region_count),
        "stats": memory_stats,
        "regions": [
            region.to_dict()
            for region in store.query_regions(source=TEACHER_SOURCE)
        ],
    }


def memory_flags(mode: str) -> Tuple[bool, bool]:
    mapping = {
        "none": (False, False),
        "episodic-only": (True, False),
        "procedural-only": (False, True),
        "episodic+procedural": (True, True),
    }
    if mode not in mapping:
        raise ValueError(f"未知 Memory 模式: {mode}")
    return mapping[mode]


def memory_prior(
    card: diagnosis.DiagnosisCard,
    store: Optional[MemoryStore],
    mode: str,
) -> Dict[str, Any]:
    use_episodic, use_procedural = memory_flags(mode)
    if mode == "none":
        return retrieve_mod.MemoryPrior(
            regime=card.regime,
            timeline_quality=card.timeline_quality,
            caution="No Memory 模式",
        ).to_dict()
    return retrieve_mod.prior_for(
        card,
        store,
        k=5,
        max_params=len(params_mod.ACTIVE_EXECUTION_PARAMS),
        use_episodic=use_episodic,
        use_procedural=use_procedural,
        sources=(TEACHER_SOURCE,),
    ).to_dict()


def build_region_messages(
    card: diagnosis.DiagnosisCard,
    *,
    default_score: float,
    prior: Mapping[str, Any],
) -> List[Dict[str, str]]:
    bounds = {
        name: list(active_bounds()[name])
        for name in params_mod.ACTIVE_EXECUTION_PARAMS
    }
    compact_prior = {
        "neighbors": [
            {
                "similarity": row.get("similarity"),
                "params": row.get("params"),
                "score": row.get("score"),
                "source": row.get("source"),
            }
            for row in list(prior.get("neighbors") or [])[:5]
        ],
        "high_score_regions": list(prior.get("suggested_regions") or []),
    }
    system = (
        "你在固定课程 Objective 下提出参数搜索区域。目标是在合法参数空间内提高"
        "给定 Objective，并满足现有 feasible constraints。你只负责缩小区域，"
        "确定性搜索负责最终数值优化。不要输出精确最优点、解释或 Markdown。"
        "只返回 JSON：{\"regions\":{\"dp_tolerance\":[low,high],"
        "\"dist_threshold\":[low,high],\"max_speed_mps\":[low,high]}}。"
    )
    payload = {
        "diagnosis": card.to_dict(),
        "active_parameter_bounds": bounds,
        "default_params": params_mod.active_default_params(),
        "default_score": round(float(default_score), 6),
        "memory": compact_prior,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def _json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = re.sub(r"```(?:json)?|```", "", str(text or ""),
                      flags=re.IGNORECASE).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_region_proposal(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """把模型区域转换为合法搜索边界，并保留原始/夹紧证据。"""
    raw_regions = payload.get("regions") if isinstance(payload, Mapping) else None
    raw_regions = raw_regions if isinstance(raw_regions, Mapping) else {}
    global_bounds = active_bounds()
    proposed: search_mod.SearchBounds = {}
    invalid: List[str] = []
    for name in params_mod.ACTIVE_EXECUTION_PARAMS:
        value = raw_regions.get(name)
        try:
            if isinstance(value, Mapping):
                low, high = float(value["low"]), float(value["high"])
            else:
                low, high = float(value[0]), float(value[1])
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError("non-finite")
            proposed[name] = (low, high)
        except (TypeError, ValueError, KeyError, IndexError):
            proposed[name] = global_bounds[name]
            invalid.append(name)
    normalized = search_mod.normalize_search_bounds(proposed)
    clamped = [
        name for name in params_mod.ACTIVE_EXECUTION_PARAMS
        if tuple(proposed[name]) != tuple(normalized[name])
    ]
    start = {
        name: (bounds[0] + bounds[1]) / 2.0
        for name, bounds in normalized.items()
    }
    return {
        "raw_regions": dict(raw_regions),
        "normalized_regions": {
            name: [float(bounds[0]), float(bounds[1])]
            for name, bounds in normalized.items()
        },
        "start_params": start,
        "invalid_regions": invalid,
        "clamped_regions": clamped,
        "valid": not invalid,
    }


def propose_regions(
    provider: Any,
    card: diagnosis.DiagnosisCard,
    *,
    default_score: float,
    prior: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Any]:
    reply = provider.chat(build_region_messages(
        card, default_score=default_score, prior=prior))
    payload = _json_object(getattr(reply, "content", "")) or {}
    return normalize_region_proposal(payload), reply


def optimization_metrics(
    *,
    method_score: float,
    default_score: float,
    search_score: float,
    epsilon: float = DEFAULT_NEAR_EPSILON,
    headroom_threshold: float = MIN_HEADROOM,
) -> Dict[str, Any]:
    headroom = float(search_score) - float(default_score)
    gap = float(search_score) - float(method_score)
    recovery: Optional[float] = None
    if headroom >= float(headroom_threshold):
        recovery = (float(method_score) - float(default_score)) / headroom
    return {
        "gap_to_search": gap,
        "search_headroom": headroom,
        "gain_recovery": recovery,
        "gain_recovery_applicable": recovery is not None,
        "near_search": float(method_score) >= float(search_score) - float(epsilon),
        "near_search_epsilon": float(epsilon),
    }


def evaluations_to_gain(
    trace: search_mod.SearchTrace,
    *,
    default_score: float,
    search_score: float,
    fraction: float,
    headroom_threshold: float = MIN_HEADROOM,
) -> Optional[int]:
    headroom = float(search_score) - float(default_score)
    if headroom < float(headroom_threshold):
        return None
    target = float(default_score) + float(fraction) * headroom
    running = -float("inf")
    for index, (_, score) in enumerate(trace.evaluations, start=1):
        running = max(running, float(score))
        if running >= target:
            return index
    return None


def run_budgeted_method(
    evaluator: TrajectoryEvaluator,
    *,
    bounds: search_mod.SearchBounds,
    start_params: Mapping[str, float],
    budget: int,
    reference_score: float,
    reference_params: Mapping[str, float],
    epsilon: float = DEFAULT_NEAR_EPSILON,
) -> Dict[str, Any]:
    defaults = params_mod.active_default_params()
    default_score = evaluator.evaluate(defaults)
    started = time.perf_counter()
    trace = search_mod.budgeted_region_search(
        dict(start_params),
        evaluator.evaluate,
        int(budget),
        bounds=bounds,
        search_params=params_mod.ACTIVE_EXECUTION_PARAMS,
    )
    metrics = optimization_metrics(
        method_score=trace.best_score,
        default_score=default_score,
        search_score=float(reference_score),
        epsilon=epsilon,
    )
    return {
        "budget": int(budget),
        "n_evaluations": int(trace.n_evaluations),
        "default_score": float(default_score),
        "reference_score": float(reference_score),
        "reference_params": dict(reference_params),
        "start_score": float(trace.evaluations[0][1]),
        "method_score": float(trace.best_score),
        "method_params": dict(trace.best_params),
        "trace": [
            {"params": dict(params), "score": float(score)}
            for params, score in trace.evaluations
        ],
        **metrics,
        "evaluations_to_90_gain": evaluations_to_gain(
            trace, default_score=default_score,
            search_score=float(reference_score), fraction=0.90),
        "evaluations_to_95_gain": evaluations_to_gain(
            trace, default_score=default_score,
            search_score=float(reference_score), fraction=0.95),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def sensitivity_from_teacher(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """从 trace 中的一维相邻变化估计各活动参数的 Objective 敏感性。"""
    deltas: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        trace = list(row.get("search_trace") or [])
        for left, right in zip(trace, trace[1:]):
            p0 = left.get("params") or {}
            p1 = right.get("params") or {}
            changed = [
                name for name in params_mod.ACTIVE_EXECUTION_PARAMS
                if abs(float(p0.get(name, 0.0)) - float(p1.get(name, 0.0))) > 1e-12
            ]
            if len(changed) == 1:
                deltas[changed[0]].append(
                    abs(float(right["score"]) - float(left["score"])))
    out: Dict[str, Any] = {}
    for name in params_mod.ACTIVE_EXECUTION_PARAMS:
        values = sorted(deltas.get(name, []))
        out[name] = {
            "n_transitions": len(values),
            "mean_abs_score_delta": (
                sum(values) / len(values) if values else None),
            "median_abs_score_delta": _quantile(values, 0.5),
            "p95_abs_score_delta": _quantile(values, 0.95),
        }
    return out


def parameter_distributions(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = {"all": list(rows)}
    for row in rows:
        groups.setdefault(f"regime:{row.get('regime')}", []).append(row)
        groups.setdefault(
            f"timeline:{row.get('timeline_quality')}", []).append(row)
        groups.setdefault(
            f"stratum:{row.get('regime')}/{row.get('timeline_quality')}",
            []).append(row)
    out: Dict[str, Any] = {}
    for group, members in sorted(groups.items()):
        out[group] = {"n": len(members), "parameters": {}}
        for name in params_mod.ACTIVE_EXECUTION_PARAMS:
            values = sorted(
                float(row["search_best_params"][name]) for row in members)
            out[group]["parameters"][name] = {
                "min": values[0] if values else None,
                "p25": _quantile(values, 0.25),
                "median": _quantile(values, 0.5),
                "p75": _quantile(values, 0.75),
                "max": values[-1] if values else None,
            }
    return out


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
