"""Objective-Oriented Optimization 的可恢复真实实验 runner。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..agent import provider as provider_mod
from ..core import diagnosis, params as params_mod, traj as traj_mod
from ..memory.store import MemoryStore
from ..verifier import objective as objective_mod
from ..verifier import search as search_mod
from . import objective_optimization as opt
from .real_ablation import transient_error


BASELINE_COMMIT = "cc9b3f0668046bf190ce4113f3b132c958c34bb7"
V2_DEMO = ("0", "1", "3", "7", "18", "62", "68", "129", "101", "149", "170", "187")
V2_HOLDOUT = ("2", "4", "10", "22", "111", "137", "153", "154", "165", "194", "201", "208")


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
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def latest_by(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    out: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        out[tuple(row.get(name) for name in fields)] = dict(row)
    return out


def git_sha(repo_dir: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir,
        text=True, encoding="utf-8").strip()


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_config(repo_dir: Path, provider: Optional[Any] = None) -> Dict[str, Any]:
    objective_path = repo_dir / "作业" / "traj_agent" / "verifier" / "objective.py"
    v2_summary_path = (
        repo_dir / "作业" / "experiments" /
        "llm_assisted_ecnu_20260921_v2" / "summary.json")
    v2_summary = json.loads(v2_summary_path.read_text(encoding="utf-8"))
    return {
        "experiment": "objective-oriented-optimization",
        "date": "2026-09-22",
        "baseline_commit": BASELINE_COMMIT,
        "implementation_commit": git_sha(repo_dir),
        "objective_version": file_sha256(objective_path),
        "objective": {
            "weights": objective_mod.ObjectiveWeights().__dict__,
            "min_applicable_length_m": objective_mod.MIN_APPLICABLE_LENGTH_M,
            "status": "frozen; formula, weights and feasibility unchanged",
        },
        "search_budget": {
            "bounded_reference": {
                "method": "coordinate_descent",
                "start": "active defaults",
                "n_steps": 5,
                "rounds": 3,
                "max_evals": 60,
            },
            "fixed_budget_comparison": {
                "method": "budgeted_region_halton",
                "budgets": list(opt.DEFAULT_BUDGETS),
            },
        },
        "active_parameter_space": {
            "names": list(params_mod.ACTIVE_EXECUTION_PARAMS),
            "defaults": params_mod.active_default_params(),
            "bounds": params_mod.param_bounds_table(
                params_mod.ACTIVE_EXECUTION_PARAMS),
        },
        "demo_manifest": {
            "v2_demo": list(V2_DEMO),
            "teacher_sizes": list(opt.DEFAULT_TEACHER_SIZES),
            "teacher_policy": "new samples; v2 demo and holdout excluded",
        },
        "holdout_manifest": list(V2_HOLDOUT),
        "v2_primary_results": {
            mode: {
                "mean_score_delta": row["score_delta"]["mean"],
                "unique_vehicle_n": row["unique_vehicle_n"],
                "observation_n": row["observation_n"],
            }
            for mode, row in v2_summary["mode_summary"].items()
        },
        "teacher_seed": opt.DEFAULT_TEACHER_SEED,
        "teacher_size": 200,
        "teacher_source": opt.TEACHER_SOURCE,
        "procedural_min_samples_primary": 3,
        "procedural_min_samples_sensitivity": [3, 5],
        "near_search_epsilon": opt.DEFAULT_NEAR_EPSILON,
        "gain_recovery_headroom": 0.02,
        "repetitions": 3,
        "memory_modes": list(opt.MEMORY_MODES),
        "provider": None if provider is None else {
            "type": "OpenAICompatProvider",
            "model": str(getattr(provider, "model", "")),
            "base_url": str(getattr(provider, "base_url", "") or "(default)"),
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "prohibitions_observed": [
            "objective.py unchanged",
            "v2 experiment unchanged",
            "holdout excluded from teacher and memory construction",
            "bounded reference not described as reality truth",
        ],
    }


def run_teacher_stage(
    experiment_dir: Path,
    data_path: Path,
    *,
    size: int = 200,
    seed: int = opt.DEFAULT_TEACHER_SEED,
) -> None:
    experiment_dir = experiment_dir.resolve()
    repo_dir = experiment_dir.parents[2]
    raw = traj_mod.load_raw(str(data_path))
    config = build_config(repo_dir)
    atomic_json(experiment_dir / "config.json", config)
    manifest = opt.teacher_manifest(
        raw,
        holdout=V2_HOLDOUT,
        excluded_demo=V2_DEMO,
        size=int(size),
        seed=int(seed),
    )
    if manifest["teacher_holdout_overlap"]:
        raise RuntimeError("teacher set 与 holdout 重叠")
    atomic_json(experiment_dir / "teacher_manifest.json", manifest)

    path = experiment_dir / "search_teacher.jsonl"
    completed = {
        str(row["vehicle_id"])
        for row in read_jsonl(path)
        if not row.get("error")
    }
    for index, vehicle_id in enumerate(manifest["vehicle_ids"], start=1):
        if vehicle_id in completed:
            continue
        try:
            row = opt.run_teacher_case(raw, vehicle_id, max_evals=60)
            row["error"] = ""
        except Exception as exc:
            row = {
                "record_type": "teacher",
                "vehicle_id": vehicle_id,
                "source": opt.TEACHER_SOURCE,
                "llm_calls": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        append_jsonl(path, row)
        print(
            f"teacher {index}/{len(manifest['vehicle_ids'])} vehicle={vehicle_id} "
            f"evals={row.get('n_evaluations', 0)} error={bool(row.get('error'))}",
            flush=True,
        )
    analyze_teacher(experiment_dir, raw)


def analyze_teacher(experiment_dir: Path, raw: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = json.loads(
        (experiment_dir / "teacher_manifest.json").read_text(encoding="utf-8"))
    latest = latest_by(
        read_jsonl(experiment_dir / "search_teacher.jsonl"), ("vehicle_id",))
    rows = [
        latest[(vehicle_id,)] for vehicle_id in manifest["vehicle_ids"]
        if (vehicle_id,) in latest and not latest[(vehicle_id,)].get("error")
    ]
    memory_scaling: Dict[str, Any] = {}
    for size in opt.DEFAULT_TEACHER_SIZES:
        subset = rows[:size]
        memory_scaling[str(size)] = {}
        for min_samples in (3, 5):
            store = MemoryStore(":memory:")
            diagnostics = opt.populate_search_memory(
                store, raw, subset, min_samples=min_samples)
            coverage = {
                "episodic": 0,
                "procedural": 0,
                "episodic+procedural": 0,
            }
            for vehicle_id in V2_HOLDOUT:
                card = diagnosis.diagnose(
                    traj_mod.traj_from_raw(vehicle_id, *raw[vehicle_id]))
                for mode in coverage:
                    if opt.memory_prior(card, store, mode)["available"]:
                        coverage[mode] += 1
            region_widths: Dict[str, List[float]] = {}
            for region in diagnostics["regions"]:
                region_widths.setdefault(region["param"], []).append(
                    float(region["high"]) - float(region["low"]))
            diagnostics["holdout_coverage"] = {
                key: {
                    "n": value,
                    "rate": value / len(V2_HOLDOUT),
                }
                for key, value in coverage.items()
            }
            diagnostics["parameter_interval_width"] = {
                name: {
                    "n_regions": len(values),
                    "mean": sum(values) / len(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
                for name, values in region_widths.items()
            }
            memory_scaling[str(size)][str(min_samples)] = diagnostics
            store.close()
    analysis = {
        "n_unique_vehicles": len({str(row["vehicle_id"]) for row in rows}),
        "n_records": len(rows),
        "errors": len(manifest["vehicle_ids"]) - len(rows),
        "teacher_holdout_overlap": manifest["teacher_holdout_overlap"],
        "total_llm_calls": sum(int(row.get("llm_calls") or 0) for row in rows),
        "sensitivity": opt.sensitivity_from_teacher(rows),
        "parameter_distributions": opt.parameter_distributions(rows),
        "memory_scaling": memory_scaling,
    }
    atomic_json(experiment_dir / "teacher_analysis.json", analysis)
    return analysis


def _redact_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    secret = os.environ.get(provider_mod.DEFAULT_API_KEY_ENV, "")
    if secret:
        text = text.replace(secret, "<redacted>")
    return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "<redacted>", text)


def call_region_proposal(
    provider: Any,
    evaluator: opt.TrajectoryEvaluator,
    prior: Mapping[str, Any],
    *,
    repetition: int,
    teacher_size: int,
    memory_mode: str,
    max_retries: int = 2,
    backoff_s: float = 2.0,
) -> Dict[str, Any]:
    totals = {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_elapsed_ms": 0.0,
        "provider_errors": [],
        "retry_count": 0,
    }
    default_score = evaluator.evaluate(params_mod.active_default_params())
    started = time.perf_counter()
    proposal: Optional[Dict[str, Any]] = None
    raw_content = ""
    for attempt in range(max_retries + 1):
        totals["llm_calls"] += 1
        try:
            proposal, reply = opt.propose_regions(
                provider,
                evaluator.card,
                default_score=default_score,
                prior=prior,
            )
            raw_content = str(getattr(reply, "content", ""))
            totals["prompt_tokens"] += int(
                getattr(reply, "prompt_tokens", 0) or 0)
            totals["completion_tokens"] += int(
                getattr(reply, "completion_tokens", 0) or 0)
            totals["llm_elapsed_ms"] += float(
                getattr(reply, "elapsed_ms", 0.0) or 0.0)
            break
        except Exception as exc:
            message = _redact_error(exc)
            totals["provider_errors"].append(message)
            if attempt >= max_retries or not transient_error(message):
                break
            totals["retry_count"] += 1
            time.sleep(backoff_s * (2 ** attempt))
    error = ""
    if proposal is None:
        proposal = opt.normalize_region_proposal({})
        error = (
            totals["provider_errors"][-1]
            if totals["provider_errors"] else "LLM未返回区域")
    return {
        "record_type": "region_proposal",
        "repetition": int(repetition),
        "teacher_size": int(teacher_size),
        "memory_mode": memory_mode,
        "vehicle_id": evaluator.vehicle_id,
        "regime": evaluator.card.regime,
        "timeline_quality": evaluator.card.timeline_quality,
        "default_score": float(default_score),
        "prior_available": bool(prior.get("available")),
        "episodic_neighbor_count": len(prior.get("neighbors") or []),
        "procedural_region_count": len(prior.get("suggested_regions") or []),
        "prior": dict(prior),
        "proposal": proposal,
        "json_success": bool(proposal.get("valid")),
        "raw_response": raw_content,
        **totals,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "error": error,
    }


def _proposal_key(row: Mapping[str, Any]) -> Tuple[int, int, str, str]:
    return (
        int(row.get("repetition") or 0),
        int(row.get("teacher_size") or 0),
        str(row.get("memory_mode") or ""),
        str(row.get("vehicle_id") or ""),
    )


def _result_key(row: Mapping[str, Any]) -> Tuple[int, int, str, int, str]:
    return (
        int(row.get("repetition") or 0),
        int(row.get("teacher_size") or 0),
        str(row.get("method") or ""),
        int(row.get("budget") or 0),
        str(row.get("vehicle_id") or ""),
    )


def run_formal_stage(
    experiment_dir: Path,
    data_path: Path,
    *,
    repetitions: int = 3,
    retry_failed: bool = False,
) -> None:
    from dotenv import load_dotenv

    experiment_dir = experiment_dir.resolve()
    work_dir = experiment_dir.parents[1]
    repo_dir = work_dir.parent
    load_dotenv(work_dir / ".env", override=False)
    status = provider_mod.provider_status()
    if status.get("would_use") != "OpenAICompatProvider":
        raise RuntimeError("未检测到真实 ECNU provider")
    provider = provider_mod.build_provider()
    if not isinstance(provider, provider_mod.OpenAICompatProvider):
        raise RuntimeError("正式实验必须使用 OpenAICompatProvider")

    raw = traj_mod.load_raw(str(data_path))
    config = build_config(repo_dir, provider)
    atomic_json(experiment_dir / "config.json", config)
    manifest = json.loads(
        (experiment_dir / "teacher_manifest.json").read_text(encoding="utf-8"))
    if manifest["teacher_holdout_overlap"]:
        raise RuntimeError("teacher set 与 holdout 重叠")
    teacher_latest = latest_by(
        read_jsonl(experiment_dir / "search_teacher.jsonl"), ("vehicle_id",))
    teacher_rows = [
        teacher_latest[(vehicle_id,)] for vehicle_id in manifest["vehicle_ids"]
        if (vehicle_id,) in teacher_latest
        and not teacher_latest[(vehicle_id,)].get("error")
    ]
    if len(teacher_rows) < max(opt.DEFAULT_TEACHER_SIZES):
        raise RuntimeError("teacher data 尚未完成200条")

    runtime_dir = experiment_dir / ".runtime"
    runtime_dir.mkdir(exist_ok=True)
    stores: Dict[int, MemoryStore] = {}
    memory_diagnostics: Dict[str, Any] = {}
    for size in opt.DEFAULT_TEACHER_SIZES:
        path = runtime_dir / f"teacher_{size}.sqlite"
        if path.exists():
            path.unlink()
        store = MemoryStore(str(path))
        diagnostics = opt.populate_search_memory(
            store, raw, teacher_rows[:size], min_samples=3)
        stores[size] = store
        memory_diagnostics[str(size)] = diagnostics
    atomic_json(experiment_dir / "memory_diagnostics.json", memory_diagnostics)

    reference_path = experiment_dir / "holdout_reference.jsonl"
    references = latest_by(read_jsonl(reference_path), ("vehicle_id",))
    for vehicle_id in V2_HOLDOUT:
        if (vehicle_id,) in references and not references[(vehicle_id,)].get("error"):
            continue
        try:
            row = opt.run_teacher_case(raw, vehicle_id, max_evals=60)
            row["record_type"] = "holdout_reference"
            row["error"] = ""
        except Exception as exc:
            row = {
                "record_type": "holdout_reference",
                "vehicle_id": vehicle_id,
                "llm_calls": 0,
                "error": _redact_error(exc),
            }
        append_jsonl(reference_path, row)
        references[(vehicle_id,)] = row
        print(f"reference vehicle={vehicle_id} error={bool(row['error'])}", flush=True)

    proposal_path = experiment_dir / "region_proposals.jsonl"
    proposal_rows = read_jsonl(proposal_path)
    proposals = latest_by(proposal_rows, (
        "repetition", "teacher_size", "memory_mode", "vehicle_id"))
    for repetition in range(1, int(repetitions) + 1):
        for vehicle_id in V2_HOLDOUT:
            evaluator = opt.TrajectoryEvaluator(raw, vehicle_id)
            combinations = [(0, "none", None)]
            for size in opt.DEFAULT_TEACHER_SIZES:
                combinations.extend([
                    (size, "episodic-only", stores[size]),
                    (size, "procedural-only", stores[size]),
                    (size, "episodic+procedural", stores[size]),
                ])
            for size, memory_mode, store in combinations:
                key = (repetition, size, memory_mode, vehicle_id)
                existing = proposals.get(key)
                if existing and (not retry_failed or not existing.get("error")):
                    continue
                prior = opt.memory_prior(evaluator.card, store, memory_mode)
                row = call_region_proposal(
                    provider,
                    evaluator,
                    prior,
                    repetition=repetition,
                    teacher_size=size,
                    memory_mode=memory_mode,
                )
                append_jsonl(proposal_path, row)
                proposals[key] = row
                print(
                    f"proposal r{repetition} size={size} mode={memory_mode} "
                    f"vehicle={vehicle_id} calls={row['llm_calls']} "
                    f"error={bool(row['error'])}",
                    flush=True,
                )

    results_path = experiment_dir / "raw_results.jsonl"
    results = latest_by(
        read_jsonl(results_path),
        ("repetition", "teacher_size", "method", "budget", "vehicle_id"),
    )
    global_bounds = opt.active_bounds()
    defaults = params_mod.active_default_params()
    for repetition in range(1, int(repetitions) + 1):
        for teacher_size in opt.DEFAULT_TEACHER_SIZES:
            for vehicle_id in V2_HOLDOUT:
                reference = references[(vehicle_id,)]
                if reference.get("error"):
                    continue
                evaluator = opt.TrajectoryEvaluator(raw, vehicle_id)
                method_specs: List[Tuple[str, search_mod.SearchBounds, Mapping[str, float], Optional[Mapping[str, Any]]]] = [
                    ("pure-search", global_bounds, defaults, None),
                ]
                for memory_mode in opt.MEMORY_MODES:
                    proposal_size = 0 if memory_mode == "none" else teacher_size
                    proposal = proposals.get((
                        repetition, proposal_size, memory_mode, vehicle_id))
                    if not proposal or proposal.get("error"):
                        continue
                    normalized = proposal["proposal"]
                    bounds = {
                        name: tuple(value)
                        for name, value in normalized["normalized_regions"].items()
                    }
                    method_specs.append((
                        f"llm-warmstart:{memory_mode}",
                        bounds,
                        normalized["start_params"],
                        proposal,
                    ))
                for method, bounds, start, proposal in method_specs:
                    for budget in opt.DEFAULT_BUDGETS:
                        key = (repetition, teacher_size, method, budget, vehicle_id)
                        existing = results.get(key)
                        if existing and (not retry_failed or not existing.get("error")):
                            continue
                        try:
                            evaluator.cache.clear()
                            outcome = opt.run_budgeted_method(
                                evaluator,
                                bounds=bounds,
                                start_params=start,
                                budget=budget,
                                reference_score=float(reference["search_best_score"]),
                                reference_params=reference["search_best_params"],
                            )
                            row = {
                                "record_type": "method_result",
                                "repetition": repetition,
                                "teacher_size": teacher_size,
                                "method": method,
                                "memory_mode": (
                                    "none" if method == "pure-search"
                                    else method.split(":", 1)[1]),
                                "budget": budget,
                                "vehicle_id": vehicle_id,
                                "regime": evaluator.card.regime,
                                "timeline_quality": evaluator.card.timeline_quality,
                                "applicable": bool(reference.get("applicable")),
                                "proposal_key": None if proposal is None else list(
                                    _proposal_key(proposal)),
                                **outcome,
                                "error": "",
                            }
                        except Exception as exc:
                            row = {
                                "record_type": "method_result",
                                "repetition": repetition,
                                "teacher_size": teacher_size,
                                "method": method,
                                "budget": budget,
                                "vehicle_id": vehicle_id,
                                "error": _redact_error(exc),
                            }
                        append_jsonl(results_path, row)
                        results[key] = row
                        print(
                            f"result r{repetition} size={teacher_size} {method} "
                            f"b={budget} vehicle={vehicle_id} error={bool(row['error'])}",
                            flush=True,
                        )

    for store in stores.values():
        store.close()
    for path in runtime_dir.glob("*.sqlite*"):
        path.unlink(missing_ok=True)
    runtime_dir.rmdir()

