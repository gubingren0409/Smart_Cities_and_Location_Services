"""Objective warm-start 精炼实验的可恢复 runner。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ..core import params as params_mod, traj as traj_mod
from ..memory.store import MemoryStore
from . import objective_optimization as opt
from . import objective_optimization_refined as refined


BASELINE_COMMIT = "6204b5bae33adf6debbc1e66339c42a0bd18ef07"
OLD_EXPERIMENT = "objective_optimization_20260922"
V2_EXPERIMENT = "llm_assisted_ecnu_20260921_v2"
HOLDOUT = ("2", "4", "10", "22", "111", "137", "153", "154", "165", "194", "201", "208")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest_by(
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    out: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        out[tuple(row.get(field) for field in fields)] = dict(row)
    return out


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def tree_manifest(path: Path) -> Dict[str, str]:
    return {
        item.relative_to(path).as_posix(): file_sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and not any(part.startswith(".runtime") for part in item.parts)
    }


def manifest_digest(manifest: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(sorted(manifest.items())), ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def git_sha(repo_dir: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir,
        text=True, encoding="utf-8").strip()


def build_config(experiment_dir: Path) -> Dict[str, Any]:
    repo_dir = experiment_dir.parents[2]
    experiments_dir = experiment_dir.parent
    old_dir = experiments_dir / OLD_EXPERIMENT
    v2_dir = experiments_dir / V2_EXPERIMENT
    objective_path = repo_dir / "作业" / "traj_agent" / "verifier" / "objective.py"
    old_config = json.loads((old_dir / "config.json").read_text(encoding="utf-8"))
    old_manifest = tree_manifest(old_dir)
    v2_manifest = tree_manifest(v2_dir)
    return {
        "experiment": "objective-optimization-refined",
        "date": "2026-09-22",
        "execution_base_commit": git_sha(repo_dir),
        "frozen_baseline_commit": BASELINE_COMMIT,
        "objective_version": file_sha256(objective_path),
        "expected_objective_version": old_config["objective_version"],
        "objective_status": "unchanged",
        "active_parameter_space": old_config["active_parameter_space"],
        "holdout_manifest": list(HOLDOUT),
        "teacher_sizes": list(opt.DEFAULT_TEACHER_SIZES),
        "fixed_budgets": list(opt.DEFAULT_BUDGETS),
        "repetitions": 3,
        "strong_reference": {
            "definition": "max(old_cd60, global_halton512, four_multistart_cd60)",
            "global_budget": refined.STRONG_GLOBAL_BUDGET,
            "multi_starts": refined.STRONG_MULTI_STARTS,
            "coordinate_descent_max_evals": refined.STRONG_CD_MAX_EVALS,
            "llm_calls": 0,
            "scope": "fixed holdout Objective-applicable vehicles only",
        },
        "deterministic_memory_rules": {
            "episodic": (
                "top-5 retrieved teacher params per dimension P25-P75; missing dimension uses "
                "full range; width below 10% global range expands around interval midpoint"
            ),
            "procedural": "direct L2 IQR; missing dimension uses full range",
            "fixed_before_holdout_execution": True,
            "llm_calls": 0,
        },
        "cost_definition": {
            "total_online_ms": "llm_elapsed_ms + objective_search_elapsed_ms",
            "end_to_end_online_ms": "proposal outer elapsed_ms + objective_search_elapsed_ms",
            "teacher_offline_excluded_from_query_cost": True,
        },
        "frozen_artifacts": {
            OLD_EXPERIMENT: {
                "tree_digest": manifest_digest(old_manifest),
                "files": old_manifest,
            },
            V2_EXPERIMENT: {
                "tree_digest": manifest_digest(v2_manifest),
                "files": v2_manifest,
            },
        },
        "no_new_llm_calls": True,
    }


def prepare_config(experiment_dir: Path) -> Dict[str, Any]:
    path = experiment_dir / "config.json"
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        config = build_config(experiment_dir)
        atomic_json(path, config)
    if config["objective_version"] != config["expected_objective_version"]:
        raise RuntimeError("Objective hash 与冻结基线不一致")
    return config


def assert_frozen_artifacts(experiment_dir: Path) -> None:
    config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    for name in (OLD_EXPERIMENT, V2_EXPERIMENT):
        expected = config["frozen_artifacts"][name]["tree_digest"]
        actual = manifest_digest(tree_manifest(experiment_dir.parent / name))
        if actual != expected:
            raise RuntimeError(f"冻结目录发生变化：{name}")


def run_strong_reference_stage(experiment_dir: Path, data_path: Path) -> None:
    prepare_config(experiment_dir)
    assert_frozen_artifacts(experiment_dir)
    raw = traj_mod.load_raw(str(data_path))
    old_dir = experiment_dir.parent / OLD_EXPERIMENT
    old_refs = latest_by(read_jsonl(old_dir / "holdout_reference.jsonl"), ("vehicle_id",))
    output_path = experiment_dir / "strong_reference.jsonl"
    completed = latest_by(read_jsonl(output_path), ("vehicle_id",))
    applicable = [
        vehicle_id for vehicle_id in HOLDOUT
        if bool(old_refs[(vehicle_id,)].get("applicable"))
    ]
    for index, vehicle_id in enumerate(applicable, start=1):
        existing = completed.get((vehicle_id,))
        if existing and not existing.get("error"):
            continue
        try:
            row = refined.run_strong_reference_case(
                raw, vehicle_id, old_refs[(vehicle_id,)])
            row["error"] = ""
        except Exception as exc:
            row = {
                "record_type": "strong_reference",
                "vehicle_id": vehicle_id,
                "llm_calls": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        append_jsonl(output_path, row)
        completed[(vehicle_id,)] = row
        print(
            f"strong-reference {index}/{len(applicable)} vehicle={vehicle_id} "
            f"evals={row.get('n_evaluations', 0)} "
            f"improvement={row.get('score_improvement')} error={bool(row.get('error'))}",
            flush=True,
        )
    assert_frozen_artifacts(experiment_dir)


def write_region_quality(experiment_dir: Path) -> None:
    old_dir = experiment_dir.parent / OLD_EXPERIMENT
    strong = latest_by(
        read_jsonl(experiment_dir / "strong_reference.jsonl"), ("vehicle_id",))
    proposals = latest_by(
        read_jsonl(old_dir / "region_proposals.jsonl"),
        ("repetition", "teacher_size", "memory_mode", "vehicle_id"),
    )
    rows: List[Dict[str, Any]] = []
    for key in sorted(proposals, key=lambda value: tuple(str(x) for x in value)):
        proposal = proposals[key]
        bounds = proposal["proposal"]["normalized_regions"]
        reference = strong.get((str(proposal["vehicle_id"]),))
        volume = refined.normalized_region_volume(bounds)
        containment = (
            refined.region_contains(reference["strong_reference_params"], bounds)
            if reference and not reference.get("error") else None
        )
        rows.append({
            "record_type": "region_quality",
            "repetition": int(proposal["repetition"]),
            "teacher_size": int(proposal["teacher_size"]),
            "memory_mode": str(proposal["memory_mode"]),
            "vehicle_id": str(proposal["vehicle_id"]),
            "regime": proposal.get("regime"),
            "timeline_quality": proposal.get("timeline_quality"),
            "applicable": reference is not None,
            "normalized_regions": bounds,
            "strong_reference_params": (
                reference.get("strong_reference_params") if reference else None),
            "reference_containment": containment,
            "normalized_region_volume": volume,
            "region_efficiency": (
                float(containment) / volume
                if containment is not None and volume > 1e-12 else None),
            "source_proposal_key": list(key),
            "llm_calls_reused": int(proposal.get("llm_calls") or 0),
            "new_llm_calls": 0,
        })
    output = experiment_dir / "region_quality.jsonl"
    temp = output.with_suffix(".jsonl.tmp")
    temp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temp.replace(output)


def _reference_for_vehicle(
    vehicle_id: str,
    strong: Mapping[Tuple[Any, ...], Mapping[str, Any]],
    old_refs: Mapping[Tuple[Any, ...], Mapping[str, Any]],
) -> Dict[str, Any]:
    value = strong.get((vehicle_id,))
    if value:
        return dict(value)
    old = old_refs[(vehicle_id,)]
    return {
        "strong_reference_score": float(old["search_best_score"]),
        "strong_reference_params": dict(old["search_best_params"]),
    }


def run_deterministic_memory_stage(experiment_dir: Path, data_path: Path) -> None:
    prepare_config(experiment_dir)
    assert_frozen_artifacts(experiment_dir)
    raw = traj_mod.load_raw(str(data_path))
    old_dir = experiment_dir.parent / OLD_EXPERIMENT
    teacher_rows = read_jsonl(old_dir / "search_teacher.jsonl")
    teacher_rows = [row for row in teacher_rows if not row.get("error")]
    if len(teacher_rows) != 200:
        raise RuntimeError(f"Search Teacher 应为200条，实际{len(teacher_rows)}条")
    old_refs = latest_by(read_jsonl(old_dir / "holdout_reference.jsonl"), ("vehicle_id",))
    strong = latest_by(read_jsonl(experiment_dir / "strong_reference.jsonl"), ("vehicle_id",))
    expected_applicable = sum(bool(old_refs[(vehicle_id,)].get("applicable")) for vehicle_id in HOLDOUT)
    if len([row for row in strong.values() if not row.get("error")]) != expected_applicable:
        raise RuntimeError("strong reference 尚未完成")

    output_path = experiment_dir / "deterministic_memory_ablation.jsonl"
    completed = latest_by(
        read_jsonl(output_path),
        ("repetition", "teacher_size", "method", "budget", "vehicle_id"),
    )
    total_cases = len(opt.DEFAULT_TEACHER_SIZES) * 2 * len(HOLDOUT)
    case_index = 0
    for teacher_size in opt.DEFAULT_TEACHER_SIZES:
        store = MemoryStore(":memory:")
        opt.populate_search_memory(store, raw, teacher_rows[:teacher_size], min_samples=3)
        for vehicle_id in HOLDOUT:
            reference = _reference_for_vehicle(vehicle_id, strong, old_refs)
            applicable = bool(old_refs[(vehicle_id,)].get("applicable"))
            for method, mode in (
                ("episodic-region-search", "episodic-only"),
                ("procedural-region-search", "procedural-only"),
            ):
                case_index += 1
                needed = [
                    (repetition, teacher_size, method, budget, vehicle_id)
                    for repetition in range(1, 4)
                    for budget in opt.DEFAULT_BUDGETS
                    if (repetition, teacher_size, method, budget, vehicle_id) not in completed
                ]
                if not needed:
                    continue
                evaluator = opt.TrajectoryEvaluator(raw, vehicle_id)
                prior = opt.memory_prior(evaluator.card, store, mode)
                bounds = (
                    refined.episodic_region_bounds(prior)
                    if mode == "episodic-only"
                    else refined.procedural_region_bounds(prior)
                )
                start = refined.midpoint(bounds)
                evaluator.cache.clear()
                cumulative_elapsed_ms = 0.0
                for budget in opt.DEFAULT_BUDGETS:
                    outcome = opt.run_budgeted_method(
                        evaluator,
                        bounds=bounds,
                        start_params=start,
                        budget=budget,
                        reference_score=float(reference["strong_reference_score"]),
                        reference_params=reference["strong_reference_params"],
                    )
                    incremental = float(outcome["elapsed_ms"])
                    cumulative_elapsed_ms += incremental
                    outcome["elapsed_ms"] = round(cumulative_elapsed_ms, 2)
                    outcome["incremental_elapsed_ms"] = round(incremental, 2)
                    outcome["execution_reuse"] = "cumulative_budget_prefix"
                    for repetition in range(1, 4):
                        key = (repetition, teacher_size, method, budget, vehicle_id)
                        if key in completed:
                            continue
                        row = {
                            "record_type": "deterministic_memory_ablation",
                            "repetition": repetition,
                            "teacher_size": teacher_size,
                            "method": method,
                            "memory_mode": mode,
                            "budget": budget,
                            "vehicle_id": vehicle_id,
                            "regime": evaluator.card.regime,
                            "timeline_quality": evaluator.card.timeline_quality,
                            "applicable": applicable,
                            "prior_available": bool(prior.get("available")),
                            "episodic_neighbor_count": len(prior.get("neighbors") or []),
                            "procedural_region_count": len(prior.get("suggested_regions") or []),
                            "bounds": {name: list(value) for name, value in bounds.items()},
                            "normalized_region_volume": refined.normalized_region_volume(bounds),
                            "llm_calls": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "llm_elapsed_ms": 0.0,
                            "retry_count": 0,
                            "provider_errors": [],
                            "deterministic_reuse_across_repetitions": True,
                            **outcome,
                            "error": "",
                        }
                        append_jsonl(output_path, row)
                        completed[key] = row
                print(
                    f"deterministic {case_index}/{total_cases} size={teacher_size} "
                    f"method={method} vehicle={vehicle_id} volume="
                    f"{refined.normalized_region_volume(bounds):.6f}",
                    flush=True,
                )
        store.close()
    assert_frozen_artifacts(experiment_dir)


def run_all_generation(experiment_dir: Path, data_path: Path) -> None:
    run_strong_reference_stage(experiment_dir, data_path)
    write_region_quality(experiment_dir)
    run_deterministic_memory_stage(experiment_dir, data_path)
