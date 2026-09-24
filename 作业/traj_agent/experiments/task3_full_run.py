"""任务三全量 11,386 辆车可恢复部署运行器。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import psutil
from dotenv import load_dotenv

from ..agent import provider as provider_mod
from ..core import diagnosis, params as params_mod, traj as traj_mod
from ..memory.store import MemoryStore
from ..road.osm import OSMRoadMatcher
from ..verifier import search as search_mod
from . import objective_optimization as opt
from . import objective_optimization_refined as refined
from . import objective_optimization_runner as opt_runner


TERMINAL_STATUSES = {"success", "not_applicable", "fallback_success", "permanent_error"}
DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_LLM_CASE_CAP = 8
DEFAULT_SEARCH_BUDGET = 3
POLICY_VERSION = "task3-full-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    # Windows 上杀毒/索引器可能短暂持有目标文件，os.replace 会报 WinError 5。
    # terminal JSONL 已先 fsync，因此这里只需有限重试状态快照，不应中断批处理。
    for attempt in range(10):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1 * (attempt + 1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return "unknown"


def redact_secret(value: Any, secrets: Sequence[str] = ()) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "<redacted>")
    return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "<redacted>", text)


def load_terminal_rows(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows, by_id
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            vehicle_id = str(row.get("vehicle_id"))
            if row.get("terminal_status") not in TERMINAL_STATUSES:
                raise ValueError(f"第 {line_number} 行不是终态记录")
            if vehicle_id in by_id:
                raise ValueError(f"重复 terminal vehicle_id: {vehicle_id}")
            rows.append(row)
            by_id[vehicle_id] = row
    return rows, by_id


def append_terminal(path: Path, row: Mapping[str, Any]) -> None:
    if row.get("terminal_status") not in TERMINAL_STATUSES:
        raise ValueError("只能追加 terminal record")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def coverage_report(source_ids: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    source = [str(value) for value in source_ids]
    completed = [str(row.get("vehicle_id")) for row in rows]
    source_counts = Counter(source)
    completed_counts = Counter(completed)
    source_set = set(source)
    completed_set = set(completed)
    return {
        "n_source_vehicle_ids": len(source),
        "n_unique_source_vehicle_ids": len(source_set),
        "n_terminal_records": len(rows),
        "n_unique_terminal_vehicle_ids": len(completed_set),
        "coverage_rate": len(source_set & completed_set) / len(source_set) if source_set else 0.0,
        "missing_vehicle_ids": sorted(source_set - completed_set, key=lambda x: (len(x), x)),
        "unexpected_vehicle_ids": sorted(completed_set - source_set, key=lambda x: (len(x), x)),
        "duplicate_source_ids": sorted(key for key, count in source_counts.items() if count > 1),
        "duplicate_terminal_ids": sorted(key for key, count in completed_counts.items() if count > 1),
        "pending": len(source_set - completed_set),
    }


def _mean(values: Iterable[Any]) -> Optional[float]:
    numbers: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return statistics.fmean(numbers) if numbers else None


def _provider_config(provider: Any) -> Dict[str, Any]:
    description = dict(provider.describe()) if hasattr(provider, "describe") else {}
    description.pop("api_key", None)
    description["api_key_set"] = bool(getattr(provider, "api_key", None))
    return description


def _is_auth_error(errors: Sequence[Any]) -> bool:
    text = " ".join(str(value) for value in errors).lower()
    return any(token in text for token in (
        "authentication", "invalid api key", "incorrect api key", "401", "403"))


def _attach_cached_road_metrics(
    row: Dict[str, Any], evaluator: opt.TrajectoryEvaluator,
    params: Mapping[str, float], road_cache_dir: Optional[Path],
) -> None:
    if road_cache_dir is None:
        return
    cache_path = Path(road_cache_dir) / f"vehicle_{evaluator.vehicle_id}.geojson.gz"
    if not cache_path.exists():
        return
    matcher = OSMRoadMatcher.from_geojson(cache_path, match_tolerance_m=30.0)
    _, simplified = evaluator.agent._execute(evaluator.handle, dict(params))
    match = matcher.match(simplified.traj)
    row["road_backend"] = "OSMRoadMatcher/EPSG:32651/STRtree"
    row["road_metrics"] = match.stats()


def _terminal_case(
    raw: Mapping[str, Any],
    vehicle_id: str,
    store: MemoryStore,
    provider: Any,
    *,
    similarity_threshold: float,
    llm_allowed: bool,
    search_budget: int,
    road_cache_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], bool]:
    """运行一个 case。返回终态记录和是否消耗了一个 LLM route slot。"""
    started = time.perf_counter()
    evaluator = opt.TrajectoryEvaluator(raw, vehicle_id)
    defaults = params_mod.active_default_params()
    default_details = evaluator.details(defaults)
    default_objective = default_details["objective"]
    prior = opt.memory_prior(evaluator.card, store, "episodic+procedural")
    similarities = [float(row.get("similarity") or 0.0) for row in prior.get("neighbors") or []]
    top_similarity = max(similarities, default=0.0)
    base = {
        "record_type": "full_run_terminal",
        "policy_version": POLICY_VERSION,
        "vehicle_id": str(vehicle_id),
        "seg_id": evaluator.card.seg_id,
        "regime": evaluator.card.regime,
        "timeline_quality": evaluator.card.timeline_quality,
        "diagnosis": evaluator.card.to_dict(),
        "applicable": bool(default_objective.get("applicable")),
        "memory_available": bool(prior.get("available")),
        "episodic_neighbor_count": len(prior.get("neighbors") or []),
        "top_neighbor_similarity": top_similarity,
        "similarity_threshold": float(similarity_threshold),
        "default_params": defaults,
        "default_objective": default_objective,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_latency_ms": 0.0,
        "provider_errors": [],
        "retry_count": 0,
        "objective_search_evaluations": 0,
        "objective_unique_evaluations": len(evaluator.cache),
        "road_backend": "none",
        "road_metrics": None,
        "created_at_utc": _now(),
    }
    if not base["applicable"]:
        base.update({
            "terminal_status": "not_applicable",
            "route": "not_applicable",
            "route_reason": str(default_objective.get("inapplicable_reason") or "Objective不适用"),
            "selected_params": defaults,
            "selected_objective": default_objective,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        })
        _attach_cached_road_metrics(base, evaluator, defaults, road_cache_dir)
        return base, False

    llm_slot_consumed = False
    terminal_status = "success"
    proposal: Optional[Dict[str, Any]] = None
    if prior.get("available") and top_similarity >= float(similarity_threshold):
        route = "episodic-region-search"
        route_reason = "同 regime 的 Search Teacher 邻居达到预设相似度阈值"
        bounds = refined.episodic_region_bounds(prior)
        start_params = refined.midpoint(bounds)
    elif llm_allowed:
        route = "llm-region-search"
        route_reason = "Episodic evidence 低于预设阈值，进入真实 LLM 区域提议"
        llm_slot_consumed = True
        proposal = opt_runner.call_region_proposal(
            provider, evaluator, prior,
            repetition=1, teacher_size=200, memory_mode="episodic+procedural",
            max_retries=2, backoff_s=1.0,
        )
        for key in ("llm_calls", "prompt_tokens", "completion_tokens",
                    "llm_elapsed_ms", "retry_count"):
            base[key] = proposal.get(key, 0)
        base["llm_latency_ms"] = float(proposal.get("llm_elapsed_ms") or 0.0)
        base["provider_errors"] = list(proposal.get("provider_errors") or [])
        base["llm_json_success"] = bool(proposal.get("json_success"))
        base["llm_proposal"] = proposal.get("proposal")
        if proposal.get("json_success"):
            normalized = proposal["proposal"]
            bounds = search_mod.normalize_search_bounds(normalized["normalized_regions"])
            start_params = dict(normalized["start_params"])
        else:
            route = "llm-failed-deterministic-fallback"
            route_reason = "真实 LLM 未返回合法区域，使用完整参数域确定性 fallback"
            terminal_status = "fallback_success"
            bounds = opt.active_bounds()
            start_params = defaults
    else:
        route = "deterministic-capacity-fallback"
        route_reason = "Episodic evidence 不足且真实 LLM 成本上限已用尽"
        terminal_status = "fallback_success"
        bounds = opt.active_bounds()
        start_params = defaults

    search_started = time.perf_counter()
    trace = search_mod.budgeted_region_search(
        dict(start_params), evaluator.evaluate, int(search_budget),
        bounds=bounds, search_params=params_mod.ACTIVE_EXECUTION_PARAMS)
    search_ms = (time.perf_counter() - search_started) * 1000.0
    selected_details = evaluator.details(trace.best_params)
    base.update({
        "terminal_status": terminal_status,
        "route": route,
        "route_reason": route_reason,
        "selected_params": dict(trace.best_params),
        "selected_objective": selected_details["objective"],
        "selected_measured": selected_details["measured"],
        "objective_search_evaluations": int(trace.n_evaluations),
        "objective_unique_evaluations": len(evaluator.cache),
        "objective_search_ms": search_ms,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    })

    _attach_cached_road_metrics(base, evaluator, trace.best_params, road_cache_dir)
    return base, llm_slot_consumed


def _state(
    total: int, rows: Sequence[Mapping[str, Any]], started_epoch: float,
    last_vehicle_id: Optional[str], llm_disabled_reason: str = "",
) -> Dict[str, Any]:
    counts = Counter(str(row.get("terminal_status")) for row in rows)
    elapsed = max(0.0, time.time() - started_epoch)
    completed = len(rows)
    rate = completed / elapsed if elapsed > 0 else 0.0
    return {
        "total": int(total),
        "completed": completed,
        "pending": max(0, int(total) - completed),
        "success": counts["success"],
        "not_applicable": counts["not_applicable"],
        "fallback": counts["fallback_success"],
        "errors": counts["permanent_error"],
        "last_vehicle_id": last_vehicle_id,
        "elapsed_s": elapsed,
        "estimated_remaining_s": ((total - completed) / rate) if rate > 0 else None,
        "llm_disabled_reason": llm_disabled_reason,
        "rss_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "updated_at_utc": _now(),
    }


def _aggregate_summary(
    source_ids: Sequence[str], rows: Sequence[Mapping[str, Any]], wall_s: float,
    resumed_records: int,
) -> Dict[str, Any]:
    coverage = coverage_report(source_ids, rows)
    statuses = Counter(str(row.get("terminal_status")) for row in rows)
    routes = Counter(str(row.get("route")) for row in rows)
    quality: Dict[str, Any] = {}
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("route"))].append(row)
    for route, group in sorted(grouped.items()):
        objectives = [row.get("selected_objective") or row.get("default_objective") or {} for row in group]
        quality[route] = {
            "n": len(group),
            "objective_mean": _mean(item.get("score") for item in objectives if item.get("applicable")),
            "feasible_rate": (
                sum(bool(item.get("feasible")) for item in objectives if item.get("applicable"))
                / max(1, sum(bool(item.get("applicable")) for item in objectives))
            ),
            "reference_gap": None,
            "reference_gap_note": "全量部署未运行每车 strong reference，故不伪造 gap",
        }
    llm_rows = [row for row in rows if int(row.get("llm_calls") or 0) > 0]
    road_rows = [row for row in rows if row.get("road_metrics")]
    expected = len(source_ids)
    pass_state = (
        coverage["n_unique_source_vehicle_ids"] == expected
        and coverage["n_terminal_records"] == expected
        and coverage["n_unique_terminal_vehicle_ids"] == expected
        and not coverage["missing_vehicle_ids"]
        and not coverage["duplicate_terminal_ids"]
        and statuses["permanent_error"] == 0
    )
    return {
        "status": "PASS" if pass_state else "FAIL",
        "coverage": coverage,
        "terminal_status": dict(sorted(statuses.items())),
        "routing": dict(sorted(routes.items())),
        "n_total": len(source_ids),
        "n_not_applicable": statuses["not_applicable"],
        "n_memory_only": routes["episodic-region-search"],
        "n_procedural": routes["procedural-region-search"],
        "n_llm_routed": len(llm_rows),
        "n_deterministic_fallback": (
            routes["deterministic-capacity-fallback"]
            + routes["llm-failed-deterministic-fallback"]),
        "n_llm_success": sum(row.get("route") == "llm-region-search" for row in rows),
        "n_llm_failed_then_fallback": routes["llm-failed-deterministic-fallback"],
        "n_fallback_success": statuses["fallback_success"],
        "n_permanent_error": statuses["permanent_error"],
        "actual_api_calls": sum(int(row.get("llm_calls") or 0) for row in rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "llm_latency_ms": sum(float(row.get("llm_latency_ms") or 0) for row in rows),
        "provider_errors": sum(len(row.get("provider_errors") or []) for row in rows),
        "retries": sum(int(row.get("retry_count") or 0) for row in rows),
        "deterministic_search_evaluations": sum(
            int(row.get("objective_search_evaluations") or 0) for row in rows),
        "total_wall_clock_s": wall_s,
        "resumed_terminal_records": int(resumed_records),
        "checkpoint_resume_verified": resumed_records > 0,
        "quality_by_route": quality,
        "road_cached_case_count": len(road_rows),
        "road_match_rate_mean": _mean(
            row["road_metrics"].get("match_rate") for row in road_rows),
        "objective_definition_modified": False,
        "historical_experiments_overwritten": False,
    }


def _plots(rows: Sequence[Mapping[str, Any]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    route_counts = Counter(str(row.get("route")) for row in rows)
    labels = list(route_counts)
    values = [route_counts[key] for key in labels]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.barh(labels, values, color="#0072B2")
    for index, value in enumerate(values):
        ax.text(value, index, f" {value:,}", va="center")
    ax.set_xlabel("Vehicles")
    ax.set_title("Full-run routing (11,386 vehicles)")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(out / "routing_counts.png", dpi=180)
    fig.savefig(out / "routing_counts.svg")
    plt.close(fig)

    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        objective = row.get("selected_objective") or row.get("default_objective") or {}
        if objective.get("applicable") and objective.get("score") is not None:
            grouped[str(row.get("route"))].append(float(objective["score"]))
    if grouped:
        fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        names = list(grouped)
        ax.boxplot([grouped[name] for name in names], tick_labels=names, showfliers=False)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("Frozen Objective score")
        ax.set_title("Objective distribution by deployment route")
        ax.grid(axis="y", alpha=.2)
        fig.savefig(out / "objective_by_route.png", dpi=180)
        fig.savefig(out / "objective_by_route.svg")
        plt.close(fig)


def _reports(out: Path, summary: Mapping[str, Any]) -> None:
    coverage = summary["coverage"]
    report = f"""# 任务三全量 11,386 辆车运行报告

## 部署策略

每辆车先生成 Diagnosis Card 并用冻结 Objective 评估默认参数。Objective 不适用时记为 `not_applicable`；适用且 top-1 Search Teacher 相似度不低于 0.75 时运行预算 3 的 `episodic-region-search`；低相似样本前 8 条进入真实 LLM 区域提议，随后仍由确定性搜索选点；其余低相似样本进入显式 deterministic capacity fallback。真实 LLM 失败时有限重试并转 fallback。

该策略在全量运行前写入 `config.json`。运行期间没有按结果修改阈值、预算或 Objective。

## 覆盖与可靠性

- 原始唯一 vehicle_id：{coverage['n_unique_source_vehicle_ids']:,}
- terminal records：{coverage['n_terminal_records']:,}
- 覆盖率：{coverage['coverage_rate']:.2%}
- missing：{len(coverage['missing_vehicle_ids'])}
- duplicate：{len(coverage['duplicate_terminal_ids'])}
- permanent error：{summary['n_permanent_error']}
- not applicable：{summary['n_not_applicable']:,}
- Memory 主路径：{summary['n_memory_only']:,}
- real LLM 路径：{summary['n_llm_routed']:,}
- fallback success：{summary['n_fallback_success']:,}

## 成本

- 实际 API attempts：{summary['actual_api_calls']}
- prompt tokens：{summary['prompt_tokens']:,}
- completion tokens：{summary['completion_tokens']:,}
- LLM latency：{summary['llm_latency_ms'] / 1000:.2f} s
- 确定性搜索 evaluations：{summary['deterministic_search_evaluations']:,}
- 全量 wall-clock：{summary['total_wall_clock_s']:.2f} s

## 结果边界

全量实验用于验证部署覆盖、路由、恢复和成本，不为每辆车额外运行 strong reference，因此不报告伪造的 per-case gap。路网只对已有真实 OSM 缓存的 holdout 记录指标；不在线下载全部上海路网。历史 Objective、200 Search Teacher、真实 LLM 消融和 strong bounded reference 均未修改。
"""
    (out / "全量运行报告.md").write_text(report, encoding="utf-8")
    feedback = f"""# 全量运行完成反馈

- 状态：**{summary['status']}**
- 覆盖：{coverage['n_terminal_records']:,} / {coverage['n_unique_source_vehicle_ids']:,}（{coverage['coverage_rate']:.2%}）
- missing / duplicate / pending：{len(coverage['missing_vehicle_ids'])} / {len(coverage['duplicate_terminal_ids'])} / {coverage['pending']}
- permanent error：{summary['n_permanent_error']}
- checkpoint/resume 实际验证：{summary['checkpoint_resume_verified']}
- 真实 LLM API attempts：{summary['actual_api_calls']}
- Objective 修改：否
- 历史实验覆盖：否

证据：`sample_manifest.json`、`run_state.json`、`raw_results.jsonl`、`failures.jsonl`、`summary.json`、`figures/`。
"""
    (out / "完成反馈.md").write_text(feedback, encoding="utf-8")


def run(
    data_path: Path,
    teacher_path: Path,
    out: Path,
    repo: Path,
    *,
    limit: Optional[int] = None,
    stop_after: Optional[int] = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    llm_case_cap: int = DEFAULT_LLM_CASE_CAP,
    search_budget: int = DEFAULT_SEARCH_BUDGET,
    road_cache_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    raw = traj_mod.load_raw(str(data_path))
    all_ids = sorted(raw, key=lambda value: (len(str(value)), str(value)))
    selected_ids = all_ids[:int(limit)] if limit is not None else all_ids
    if limit is None and len(selected_ids) != 11386:
        raise RuntimeError(f"正式全量数据不是 11,386 辆: {len(selected_ids)}")
    teacher_rows = [
        json.loads(line) for line in teacher_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({str(row["vehicle_id"]) for row in teacher_rows}) != 200:
        raise RuntimeError("Search Teacher 必须恰好为既有 200 辆")

    load_dotenv(data_path.parent / ".env", override=False)
    provider = provider_mod.build_provider()
    if not isinstance(provider, provider_mod.OpenAICompatProvider):
        raise RuntimeError("全量正式策略中的 LLM 路径必须使用真实 OpenAICompatProvider")
    config = {
        "experiment": "task3_full_11386",
        "policy_version": POLICY_VERSION,
        "created_at_utc": _now(),
        "implementation_commit": _git_head(repo),
        "data_file": data_path.name,
        "data_sha256": _sha256(data_path),
        "teacher_file": str(teacher_path),
        "teacher_count": 200,
        "similarity_threshold": float(similarity_threshold),
        "similarity_threshold_basis": (
            "固定 holdout 中 applicable 样本 top-1 similarity 为 0.7389--0.9173；"
            "0.75 在运行全量前冻结，并保留至少一个真实 LLM 低相似路径"),
        "llm_case_cap": int(llm_case_cap),
        "llm_cap_reason": "完成度优先的有界真实调用；超额低相似样本显式 deterministic fallback",
        "search_budget": int(search_budget),
        "memory_mode": "episodic+procedural",
        "memory_source": "existing 200 Search Teacher",
        "provider": _provider_config(provider),
        "terminal_statuses": sorted(TERMINAL_STATUSES),
        "road_backend": "OSMRoadMatcher for cases with existing task3_real_road cache",
        "objective_definition_modified": False,
        "limit": limit,
    }
    config_path = out / "config.json"
    if config_path.exists():
        frozen = json.loads(config_path.read_text(encoding="utf-8"))
        immutable = ("policy_version", "data_sha256", "similarity_threshold",
                     "llm_case_cap", "search_budget", "teacher_count", "limit")
        mismatched = [key for key in immutable if frozen.get(key) != config.get(key)]
        if mismatched:
            raise RuntimeError(f"resume 配置与冻结 config 不一致: {mismatched}")
        config = frozen
        current_commit = _git_head(repo)
        if current_commit != config.get("implementation_commit"):
            recovery_commits = list(config.get("resume_implementation_commits") or [])
            if current_commit not in recovery_commits:
                recovery_commits.append(current_commit)
                config["resume_implementation_commits"] = recovery_commits
                _json(config_path, config)
    else:
        _json(config_path, config)

    sample_manifest = {
        "source_raw_file": data_path.name,
        "source_raw_sha256": config["data_sha256"],
        "source_unique_vehicle_ids": len(all_ids),
        "manifest_vehicle_ids": len(selected_ids),
        "manifest_unique_vehicle_ids": len(set(selected_ids)),
        "duplicate_vehicle_ids": [],
        "missing_from_source": [],
        "selection": "all source vehicle ids" if limit is None else f"smoke first {limit} stable IDs",
        "vehicle_ids": selected_ids,
    }
    _json(out / "sample_manifest.json", sample_manifest)
    raw_results_path = out / "raw_results.jsonl"
    existing_rows, existing_by_id = load_terminal_rows(raw_results_path)
    resumed_records = len(existing_rows)

    memory_path = out / "search_memory.sqlite"
    store = MemoryStore(str(memory_path))
    if store.count() == 0:
        opt.populate_search_memory(store, raw, teacher_rows, min_samples=3)

    start_epoch = time.time()
    llm_used = sum(int(row.get("llm_calls") or 0) > 0 for row in existing_rows)
    state_path = out / "run_state.json"
    old_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    llm_disabled_reason = str(old_state.get("llm_disabled_reason") or "")
    processed_this_invocation = 0
    rows = list(existing_rows)
    secrets = [os.environ.get(provider_mod.DEFAULT_API_KEY_ENV, "")]
    failures_path = out / "failures.jsonl"

    try:
        for vehicle_id in selected_ids:
            if vehicle_id in existing_by_id:
                continue
            try:
                row, llm_slot = _terminal_case(
                    raw, vehicle_id, store, provider,
                    similarity_threshold=float(similarity_threshold),
                    llm_allowed=(llm_used < int(llm_case_cap) and not llm_disabled_reason),
                    search_budget=int(search_budget),
                    road_cache_dir=road_cache_dir,
                )
                if llm_slot:
                    llm_used += 1
                    if _is_auth_error(row.get("provider_errors") or []):
                        llm_disabled_reason = "authentication/config error; later LLM routes use deterministic fallback"
                row["provider_errors"] = [
                    redact_secret(value, secrets) for value in row.get("provider_errors") or []]
            except Exception as exc:
                error = redact_secret(f"{type(exc).__name__}: {exc}", secrets)
                row = {
                    "record_type": "full_run_terminal",
                    "policy_version": POLICY_VERSION,
                    "vehicle_id": vehicle_id,
                    "terminal_status": "permanent_error",
                    "route": "permanent_error",
                    "error": error,
                    "created_at_utc": _now(),
                }
                with failures_path.open("a", encoding="utf-8") as failures:
                    failures.write(json.dumps(row, ensure_ascii=False) + "\n")
                    failures.flush()
                    os.fsync(failures.fileno())
            append_terminal(raw_results_path, row)
            rows.append(row)
            existing_by_id[vehicle_id] = row
            processed_this_invocation += 1
            current = _state(
                len(selected_ids), rows, start_epoch, vehicle_id, llm_disabled_reason)
            _json(state_path, current)
            if processed_this_invocation % 100 == 0 or processed_this_invocation == 1:
                print(
                    f"full {len(rows)}/{len(selected_ids)} id={vehicle_id} "
                    f"status={row['terminal_status']} route={row.get('route')} "
                    f"rss={current['rss_mb']:.1f}MB", flush=True)
            if stop_after is not None and processed_this_invocation >= int(stop_after):
                print(f"intentional checkpoint stop after {processed_this_invocation}", flush=True)
                return None
    finally:
        store.close()

    wall_s = float(old_state.get("elapsed_s") or 0.0) + (time.time() - start_epoch)
    final_rows, _ = load_terminal_rows(raw_results_path)
    summary = _aggregate_summary(selected_ids, final_rows, wall_s, resumed_records)
    _json(out / "summary.json", summary)
    _json(state_path, {
        **_state(len(selected_ids), final_rows, time.time() - wall_s,
                 selected_ids[-1] if selected_ids else None, llm_disabled_reason),
        "estimated_remaining_s": 0.0,
    })
    failures_path.touch(exist_ok=True)
    _plots(final_rows, out / "figures")
    _reports(out, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("作业/traj_dict.json"))
    parser.add_argument("--teacher", type=Path, default=Path(
        "作业/experiments/objective_optimization_20260922/search_teacher.jsonl"))
    parser.add_argument("--out", type=Path, default=Path(
        "作业/experiments/task3_full_11386_20260924"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--llm-case-cap", type=int, default=DEFAULT_LLM_CASE_CAP)
    parser.add_argument("--search-budget", type=int, default=DEFAULT_SEARCH_BUDGET)
    parser.add_argument("--road-cache-dir", type=Path, default=Path(
        "作业/experiments/task3_real_road_20260924/road_cache"))
    args = parser.parse_args()
    run(
        args.data.resolve(), args.teacher.resolve(), args.out.resolve(), args.repo.resolve(),
        limit=args.limit, stop_after=args.stop_after,
        similarity_threshold=args.similarity_threshold,
        llm_case_cap=args.llm_case_cap, search_budget=args.search_budget,
        road_cache_dir=args.road_cache_dir.resolve(),
    )


if __name__ == "__main__":
    main()
