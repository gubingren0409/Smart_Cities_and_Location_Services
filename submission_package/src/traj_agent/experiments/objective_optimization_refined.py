"""Objective warm-start 收尾实验的确定性计算组件。

本模块只复用既有 Search Teacher、LLM 区域提案和固定预算结果；不会调用
provider，也不会改变 Objective 或活动参数空间。新增搜索只包括更强的确定性
reference 与两种直接由 Memory 构造区域的消融。
"""
from __future__ import annotations

from collections import defaultdict
import math
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core import params as params_mod
from ..verifier import search as search_mod
from . import objective_optimization as opt


STRONG_GLOBAL_BUDGET = 512
STRONG_MULTI_STARTS = 4
STRONG_CD_MAX_EVALS = 60
EPISODIC_MIN_WIDTH_FRACTION = 0.10


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


def midpoint(bounds: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    normalized = search_mod.normalize_search_bounds(dict(bounds))
    return {
        name: (float(low) + float(high)) / 2.0
        for name, (low, high) in normalized.items()
    }


def region_contains(
    params: Mapping[str, float],
    bounds: Mapping[str, Sequence[float]],
    *,
    atol: float = 1e-9,
) -> bool:
    """判断参数点是否落在所有活动维度的闭区间内。"""
    normalized = search_mod.normalize_search_bounds(dict(bounds))
    for name in params_mod.ACTIVE_EXECUTION_PARAMS:
        if name not in params:
            return False
        low, high = normalized[name]
        value = float(params[name])
        if value < low - atol or value > high + atol:
            return False
    return True


def normalized_region_volume(bounds: Mapping[str, Sequence[float]]) -> float:
    """区域体积除以完整活动参数空间体积。"""
    normalized = search_mod.normalize_search_bounds(dict(bounds))
    global_bounds = opt.active_bounds()
    volume = 1.0
    for name in params_mod.ACTIVE_EXECUTION_PARAMS:
        low, high = normalized[name]
        global_low, global_high = global_bounds[name]
        width = max(0.0, float(high) - float(low))
        global_width = float(global_high) - float(global_low)
        volume *= width / global_width
    return float(volume)


def episodic_region_bounds(
    prior: Mapping[str, Any],
    *,
    min_width_fraction: float = EPISODIC_MIN_WIDTH_FRACTION,
) -> search_mod.SearchBounds:
    """按 top-k Episodic 参数的 P25--P75 构造固定规则局部区域。

    某维无历史值时退回完整区间；IQR 退化时，以该维完整范围的 10% 为最小
    宽度并围绕退化点展开。规则与常量在读取 holdout 结果前固定。
    """
    full = opt.active_bounds()
    neighbors = list(prior.get("neighbors") or [])
    proposed: search_mod.SearchBounds = {}
    min_fraction = max(0.0, min(1.0, float(min_width_fraction)))
    for name in params_mod.ACTIVE_EXECUTION_PARAMS:
        values: List[float] = []
        for row in neighbors:
            value = (row.get("params") or {}).get(name)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values.append(number)
        if not values:
            proposed[name] = full[name]
            continue
        low = float(_quantile(values, 0.25))
        high = float(_quantile(values, 0.75))
        global_low, global_high = full[name]
        min_width = (global_high - global_low) * min_fraction
        if high - low < min_width:
            center = (low + high) / 2.0
            low = center - min_width / 2.0
            high = center + min_width / 2.0
        proposed[name] = (low, high)
    return search_mod.normalize_search_bounds(proposed)


def procedural_region_bounds(
    prior: Mapping[str, Any],
) -> search_mod.SearchBounds:
    """直接使用 Procedural L2 的 IQR；缺失维度退回完整参数范围。"""
    proposed = dict(opt.active_bounds())
    for row in list(prior.get("suggested_regions") or []):
        name = str(row.get("param") or "")
        if name not in proposed:
            continue
        try:
            proposed[name] = (float(row["low"]), float(row["high"]))
        except (TypeError, ValueError, KeyError):
            continue
    return search_mod.normalize_search_bounds(proposed)


def select_strong_reference(
    old_reference: Mapping[str, Any],
    global_trace: search_mod.SearchTrace,
    multistart_traces: Sequence[search_mod.SearchTrace],
) -> Dict[str, Any]:
    """按定义取 CD60、global 与 multi-start 三者最高分。"""
    candidates: List[Tuple[str, float, Dict[str, float]]] = [
        (
            "old_cd60",
            float(old_reference["search_best_score"]),
            dict(old_reference["search_best_params"]),
        ),
        ("global_halton512", float(global_trace.best_score), dict(global_trace.best_params)),
    ]
    for index, trace in enumerate(multistart_traces, start=1):
        candidates.append((
            f"multistart_cd:{index}",
            float(trace.best_score),
            dict(trace.best_params),
        ))
    # 稳定 tie-break：相同分数时保留列表中更早、语义更简单的来源。
    source, score, params = max(
        enumerate(candidates), key=lambda item: (item[1][1], -item[0]))[1]
    return {
        "source_method": source,
        "strong_reference_score": score,
        "strong_reference_params": params,
    }


def run_strong_reference_case(
    raw: Mapping[str, Any],
    vehicle_id: str,
    old_reference: Mapping[str, Any],
    *,
    global_budget: int = STRONG_GLOBAL_BUDGET,
    multi_starts: int = STRONG_MULTI_STARTS,
    cd_max_evals: int = STRONG_CD_MAX_EVALS,
) -> Dict[str, Any]:
    """运行约 500--1000 次/车的纯确定性 stronger bounded reference。"""
    evaluator = opt.TrajectoryEvaluator(raw, str(vehicle_id))
    if not bool(old_reference.get("applicable")):
        raise ValueError("strong reference 只对 Objective applicable 车辆运行")
    started = time.perf_counter()
    defaults = params_mod.active_default_params()
    full_bounds = opt.active_bounds()
    global_trace = search_mod.budgeted_region_search(
        defaults,
        evaluator.evaluate,
        int(global_budget),
        bounds=full_bounds,
        search_params=params_mod.ACTIVE_EXECUTION_PARAMS,
    )

    ranked: List[Tuple[float, Dict[str, float]]] = []
    seen = set()
    for params, score in sorted(
        global_trace.evaluations, key=lambda item: item[1], reverse=True):
        key = opt.TrajectoryEvaluator._key(params)
        if key in seen:
            continue
        seen.add(key)
        ranked.append((float(score), dict(params)))
        if len(ranked) >= int(multi_starts):
            break

    cd_traces: List[search_mod.SearchTrace] = []
    for _, start_params in ranked:
        cd_traces.append(search_mod.coordinate_descent(
            base_params=start_params,
            search_params=params_mod.ACTIVE_EXECUTION_PARAMS,
            evaluator=evaluator.evaluate,
            n_steps=5,
            rounds=3,
            max_evals=int(cd_max_evals),
        ))

    selected = select_strong_reference(old_reference, global_trace, cd_traces)
    new_logical = int(global_trace.n_evaluations) + sum(
        int(trace.n_evaluations) for trace in cd_traces)
    old_evals = int(old_reference.get("n_evaluations") or 0)
    return {
        "record_type": "strong_reference",
        "vehicle_id": str(vehicle_id),
        "seg_id": evaluator.card.seg_id,
        "regime": evaluator.card.regime,
        "timeline_quality": evaluator.card.timeline_quality,
        "applicable": True,
        "old_reference_score": float(old_reference["search_best_score"]),
        "old_reference_params": dict(old_reference["search_best_params"]),
        **selected,
        "score_improvement": (
            float(selected["strong_reference_score"])
            - float(old_reference["search_best_score"])),
        "component_scores": {
            "old_cd60": float(old_reference["search_best_score"]),
            "global_halton512": float(global_trace.best_score),
            "multistart_cd": max(
                (float(trace.best_score) for trace in cd_traces),
                default=-float("inf"),
            ),
        },
        "component_best_params": {
            "global_halton512": dict(global_trace.best_params),
            "multistart_cd": [dict(trace.best_params) for trace in cd_traces],
        },
        "old_evaluations_reused": old_evals,
        "new_logical_evaluations": new_logical,
        "new_unique_objective_evaluations": len(evaluator.cache),
        "n_evaluations": old_evals + new_logical,
        "global_budget": int(global_budget),
        "multi_starts": len(cd_traces),
        "multistart_evaluations": [int(trace.n_evaluations) for trace in cd_traces],
        "llm_calls": 0,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def recompute_reference_metrics(
    row: Mapping[str, Any],
    strong_reference: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = opt.optimization_metrics(
        method_score=float(row["method_score"]),
        default_score=float(row["default_score"]),
        search_score=float(strong_reference["strong_reference_score"]),
    )
    out = dict(row)
    out.update(metrics)
    out["reference_score"] = float(strong_reference["strong_reference_score"])
    out["reference_params"] = dict(strong_reference["strong_reference_params"])
    out["reference_kind"] = "strong_bounded_reference"
    for fraction, field in ((0.90, "evaluations_to_90_gain"), (0.95, "evaluations_to_95_gain")):
        headroom = float(strong_reference["strong_reference_score"]) - float(row["default_score"])
        reached: Optional[int] = None
        if headroom >= opt.MIN_HEADROOM:
            target = float(row["default_score"]) + fraction * headroom
            running = -float("inf")
            for index, step in enumerate(list(row.get("trace") or []), start=1):
                running = max(running, float(step["score"]))
                if running >= target:
                    reached = index
                    break
        out[field] = reached
    return out


def normalized_sensitivity_from_teacher(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """在参数完整范围尺度上归一化 Teacher trace 的局部敏感性。"""
    old_values: Dict[str, List[float]] = defaultdict(list)
    normalized_values: Dict[str, List[float]] = defaultdict(list)
    full = opt.active_bounds()
    for row in rows:
        trace = list(row.get("search_trace") or [])
        for left, right in zip(trace, trace[1:]):
            p0 = left.get("params") or {}
            p1 = right.get("params") or {}
            changed = [
                name for name in params_mod.ACTIVE_EXECUTION_PARAMS
                if abs(float(p0.get(name, 0.0)) - float(p1.get(name, 0.0))) > 1e-12
            ]
            if len(changed) != 1:
                continue
            name = changed[0]
            delta_score = abs(float(right["score"]) - float(left["score"]))
            delta_x = abs(float(p1[name]) - float(p0[name]))
            global_width = float(full[name][1]) - float(full[name][0])
            normalized_step = delta_x / global_width
            if normalized_step <= 1e-12:
                continue
            old_values[name].append(delta_score)
            normalized_values[name].append(delta_score / normalized_step)

    out: Dict[str, Any] = {}
    for name in params_mod.ACTIVE_EXECUTION_PARAMS:
        old = sorted(old_values.get(name, []))
        normalized = sorted(normalized_values.get(name, []))
        out[name] = {
            "n_transitions": len(normalized),
            "mean_abs_score_delta": sum(old) / len(old) if old else None,
            "median_abs_score_delta": _quantile(old, 0.5),
            "p95_abs_score_delta": _quantile(old, 0.95),
            "mean_normalized_sensitivity": (
                sum(normalized) / len(normalized) if normalized else None),
            "median_normalized_sensitivity": _quantile(normalized, 0.5),
            "p95_normalized_sensitivity": _quantile(normalized, 0.95),
        }
    return out


def online_cost(
    *,
    search_elapsed_ms: float,
    proposal: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """统一在线成本口径；Teacher 离线成本不在此函数中出现。"""
    proposal = proposal or {}
    llm_elapsed = float(proposal.get("llm_elapsed_ms") or 0.0)
    proposal_elapsed = float(proposal.get("elapsed_ms") or llm_elapsed)
    search_elapsed = float(search_elapsed_ms)
    return {
        "llm_latency_ms": llm_elapsed,
        "llm_calls": int(proposal.get("llm_calls") or 0),
        "prompt_tokens": int(proposal.get("prompt_tokens") or 0),
        "completion_tokens": int(proposal.get("completion_tokens") or 0),
        "objective_search_ms": search_elapsed,
        "total_online_ms": llm_elapsed + search_elapsed,
        "end_to_end_online_ms": proposal_elapsed + search_elapsed,
        "retries": int(proposal.get("retry_count") or 0),
        "provider_errors": len(proposal.get("provider_errors") or []),
    }


def validate_paired_alignment(
    left_rows: Iterable[Mapping[str, Any]],
    right_rows: Iterable[Mapping[str, Any]],
) -> bool:
    fields = ("vehicle_id", "repetition", "budget")
    left = {tuple(row.get(field) for field in fields) for row in left_rows}
    right = {tuple(row.get(field) for field in fields) for row in right_rows}
    return bool(left) and left == right
