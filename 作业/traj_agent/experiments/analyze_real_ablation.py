"""真实消融结果统计、图表和报告。"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .real_ablation import (
    DEFAULT_SEED, MODES, STAGE1_HOLDOUT, STRATA, atomic_json, case_key,
    read_jsonl,
)
from ..verifier.ablation import MODE_SPECS


def bootstrap_ci(values: Sequence[float], seed: int = DEFAULT_SEED,
                 n_boot: int = 4000) -> Optional[list[float]]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not xs:
        return None
    if len(xs) == 1:
        return [xs[0], xs[0]]
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(xs) for _ in xs) / len(xs))
    means.sort()
    return [
        means[int(0.025 * (len(means) - 1))],
        means[int(0.975 * (len(means) - 1))],
    ]


def stats(values: Sequence[float], seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {
        "n": len(xs),
        "mean": (sum(xs) / len(xs)) if xs else None,
        "median": statistics.median(xs) if xs else None,
        "bootstrap_95_ci": bootstrap_ci(xs, seed=seed),
    }


def cluster_bootstrap_ci(rows: Sequence[Mapping[str, Any]],
                         value_fn: Callable[[Mapping[str, Any]], Optional[float]],
                         seed: int = DEFAULT_SEED,
                         n_boot: int = 4000) -> Optional[list[float]]:
    """以 vehicle 为独立抽样单位，抽中车辆时纳入它的全部 repetition。"""
    clusters: Dict[str, list[float]] = {}
    for row in rows:
        value = value_fn(row)
        if value is None or not math.isfinite(float(value)):
            continue
        clusters.setdefault(str(row.get("vehicle_id")), []).append(float(value))
    vehicle_ids = sorted(clusters)
    if not vehicle_ids:
        return None
    if len(vehicle_ids) == 1:
        mean = sum(clusters[vehicle_ids[0]]) / len(clusters[vehicle_ids[0]])
        return [mean, mean]
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sampled = [rng.choice(vehicle_ids) for _ in vehicle_ids]
        values = [value for vid in sampled for value in clusters[vid]]
        means.append(sum(values) / len(values))
    means.sort()
    return [
        means[int(0.025 * (len(means) - 1))],
        means[int(0.975 * (len(means) - 1))],
    ]


def cluster_stats(rows: Sequence[Mapping[str, Any]],
                  value_fn: Callable[[Mapping[str, Any]], Optional[float]],
                  seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    values = []
    vehicle_ids = set()
    repetitions = set()
    for row in rows:
        value = value_fn(row)
        if value is None or not math.isfinite(float(value)):
            continue
        values.append(float(value))
        vehicle_ids.add(str(row.get("vehicle_id")))
        repetitions.add(int(row.get("repetition", 0)))
    return {
        "n": len(values),
        "observation_n": len(values),
        "unique_vehicle_n": len(vehicle_ids),
        "repetitions": len(repetitions),
        "mean": (sum(values) / len(values)) if values else None,
        "median": statistics.median(values) if values else None,
        "bootstrap_95_ci": cluster_bootstrap_ci(
            rows, value_fn, seed=seed),
        "bootstrap_unit": "vehicle_id",
    }


def latest_case_rows(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """同一 case 可因 retry-failed 追加多行；统计只取最后一次。"""
    latest: Dict[tuple[str, int, str, str], Dict[str, Any]] = {}
    for row in rows:
        latest[case_key(row)] = row
    return list(latest.values())


def aggregate_case_costs(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """按 case 汇总追加重试产生的全部真实调用、token 与延迟。"""
    grouped: Dict[tuple[str, int, str, str], list[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(case_key(row), []).append(row)
    out = []
    for attempts in grouped.values():
        row = dict(attempts[-1])
        for name in (
            "llm_calls", "prompt_tokens", "completion_tokens",
            "retry_count", "provider_error_count",
        ):
            row[name] = sum(int(a.get(name) or 0) for a in attempts)
        for name in ("llm_elapsed_ms", "elapsed_ms"):
            row[name] = sum(float(a.get(name) or 0.0) for a in attempts)
        row["attempt_record_count"] = len(attempts)
        out.append(row)
    return out


def summarize(experiment_dir: Path) -> Dict[str, Any]:
    config = json.loads(
        (experiment_dir / "config.json").read_text(encoding="utf-8")
    )
    raw_rows = read_jsonl(experiment_dir / "raw_results.jsonl")
    rows = latest_case_rows(raw_rows)
    cost_rows = aggregate_case_costs(raw_rows)
    state = json.loads(
        (experiment_dir / "run_state.json").read_text(encoding="utf-8")
    )
    holdout = [r for r in rows if r.get("phase") == "holdout"]
    demo = [r for r in rows if r.get("phase") == "demo"]
    mode_summary: Dict[str, Any] = {}

    for idx, mode in enumerate(MODES):
        items = [r for r in holdout if r.get("mode") == mode]
        cost_items = [
            r for r in cost_rows
            if r.get("phase") == "holdout" and r.get("mode") == mode]
        ok = [r for r in items if not r.get("error") and r.get("agent_ok")]
        applicable = [r for r in ok if r.get("applicable")]
        delta_fn = lambda r: (
            float(r["proposal_score"]) - float(r["baseline_score"]))
        beaten_fn = lambda r: 1.0 if delta_fn(r) > 1e-9 else 0.0
        direction_rows = [
            r for r in applicable if r.get("direction_accuracy") is not None]
        json_rows = [
            r for r in items if MODE_SPECS[mode].get("use_llm")
        ]
        mode_summary[mode] = {
            "n_total": len(items),
            "n_success": len(ok),
            "n_applicable": len(applicable),
            "unique_vehicle_n": len({str(r["vehicle_id"]) for r in applicable}),
            "observation_n": len(applicable),
            "repetitions": len({int(r["repetition"]) for r in applicable}),
            "score_delta": cluster_stats(
                applicable, delta_fn, DEFAULT_SEED + idx),
            "baseline_beaten_rate": cluster_stats(
                applicable, beaten_fn, DEFAULT_SEED + 10 + idx),
            "constraint_violation_rate": (
                sum(not bool(r.get("constraint_ok")) for r in ok) / len(ok)
                if ok else None
            ),
            "clamp_rate": (
                sum(bool(r.get("clamped_params")) for r in ok) / len(ok)
                if ok else None
            ),
            "json_success_rate": (
                sum(bool(r.get("json_success")) for r in json_rows)
                / len(json_rows) if json_rows else None
            ),
            "direction_accuracy": cluster_stats(
                direction_rows,
                lambda r: float(r["direction_accuracy"]),
                DEFAULT_SEED + 20 + idx),
            "llm_calls": sum(int(r.get("llm_calls") or 0) for r in cost_items),
            "prompt_tokens": sum(
                int(r.get("prompt_tokens") or 0) for r in cost_items
            ),
            "completion_tokens": sum(
                int(r.get("completion_tokens") or 0) for r in cost_items
            ),
            "latency_ms": cluster_stats(
                cost_items, lambda r: float(r.get("elapsed_ms") or 0.0),
                DEFAULT_SEED + 30 + idx),
            "failure_rate": (
                sum(bool(r.get("error")) for r in items) / len(items)
                if items else None
            ),
            "retry_count": sum(
                int(r.get("retry_count") or 0) for r in cost_items
            ),
            "provider_error_count": sum(
                int(r.get("provider_error_count") or 0) for r in cost_items
            ),
        }

    per_stratum: Dict[str, Any] = {}
    for mode in MODES:
        per_stratum[mode] = {}
        for regime, timeline in STRATA:
            items = [
                r for r in holdout
                if r.get("mode") == mode
                and r.get("regime") == regime
                and r.get("timeline_quality") == timeline
                and not r.get("error")
                and r.get("applicable")
            ]
            per_stratum[mode][f"{regime}/{timeline}"] = cluster_stats(
                items,
                lambda r: float(r["proposal_score"]) - float(r["baseline_score"]),
                DEFAULT_SEED + len(per_stratum[mode]))

    per_repetition: Dict[str, Any] = {}
    for rep in range(1, int(config["repetitions"]) + 1):
        per_repetition[str(rep)] = {}
        for mode in MODES:
            items = [
                r for r in holdout
                if int(r.get("repetition", 0)) == rep
                and r.get("mode") == mode
                and not r.get("error")
                and r.get("applicable")
            ]
            per_repetition[str(rep)][mode] = cluster_stats(
                items,
                lambda r: float(r["proposal_score"]) - float(r["baseline_score"]),
                DEFAULT_SEED + rep)

    indexed = {
        (int(r["repetition"]), str(r["vehicle_id"]), str(r["mode"])): r
        for r in holdout
        if not r.get("error") and r.get("applicable")
    }
    paired_rows = []
    for rep in range(1, int(config["repetitions"]) + 1):
        for vid in STAGE1_HOLDOUT:
            no_mem = indexed.get((rep, vid, "llm+search-verifier"))
            mem = indexed.get((rep, vid, "llm+memory+search-verifier"))
            if not no_mem or not mem:
                continue
            diff = (
                float(mem["proposal_score"])
                - float(no_mem["proposal_score"])
            )
            paired_rows.append({
                "repetition": rep,
                "vehicle_id": vid,
                "score_difference_memory_minus_no_memory": diff,
            })
    by_vehicle: Dict[str, list[float]] = {}
    for row in paired_rows:
        by_vehicle.setdefault(str(row["vehicle_id"]), []).append(
            float(row["score_difference_memory_minus_no_memory"]))
    vehicle_means = [
        {
            "vehicle_id": vid,
            "repetition_n": len(values),
            "mean_score_difference": sum(values) / len(values),
        }
        for vid, values in sorted(by_vehicle.items())
    ]
    paired_vals = [r["mean_score_difference"] for r in vehicle_means]
    paired = {
        "definition": (
            "proposal_score(llm+memory+search-verifier) - "
            "proposal_score(llm+search-verifier)"
        ),
        "stats": stats(paired_vals, DEFAULT_SEED + 99),
        "bootstrap_unit": "vehicle_id after averaging repetitions",
        "unique_vehicle_n": len(vehicle_means),
        "observation_n": len(paired_rows),
        "repetitions": int(config["repetitions"]),
        "wins": sum(x > 1e-9 for x in paired_vals),
        "ties": sum(abs(x) <= 1e-9 for x in paired_vals),
        "losses": sum(x < -1e-9 for x in paired_vals),
        "pairs": paired_rows,
        "vehicle_means": vehicle_means,
    }

    calls = sum(int(r.get("llm_calls") or 0) for r in raw_rows)
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in raw_rows)
    completion_tokens = sum(
        int(r.get("completion_tokens") or 0) for r in raw_rows
    )
    total_latency = sum(float(r.get("elapsed_ms") or 0.0) for r in raw_rows)
    demo_llm = [r for r in demo if int(r.get("llm_calls") or 0) > 0]
    n_demo = len(demo_llm) or 1
    calls_per_demo = sum(
        int(r["llm_calls"]) for r in demo_llm
    ) / n_demo
    prompt_per_demo = sum(
        int(r["prompt_tokens"]) for r in demo_llm
    ) / n_demo
    completion_per_demo = sum(
        int(r["completion_tokens"]) for r in demo_llm
    ) / n_demo
    elapsed_per_demo = sum(
        float(r["elapsed_ms"]) for r in demo_llm
    ) / n_demo

    estimates = {}
    for size in (24, 48, 100, 200):
        multiplier = size * int(config["repetitions"])
        estimates[str(size)] = {
            "scope": (
                f"{size} demo cases x {config['repetitions']} repetitions; "
                "holdout excluded"
            ),
            "estimated_llm_calls": round(multiplier * calls_per_demo),
            "estimated_prompt_tokens": round(
                multiplier * prompt_per_demo
            ),
            "estimated_completion_tokens": round(
                multiplier * completion_per_demo
            ),
            "estimated_elapsed_s": round(
                multiplier * elapsed_per_demo / 1000.0, 1
            ),
        }

    memory_rows = [
        r for r in holdout if r.get("mode") == "llm+memory+search-verifier"
    ]
    neighbor_counts = [
        int(r.get("memory_neighbor_count") or 0) for r in memory_rows
    ]
    neighbor_distribution = {
        str(n): neighbor_counts.count(n) for n in sorted(set(neighbor_counts))
    }
    similarities = [
        float(value)
        for row in memory_rows
        for value in (row.get("memory_neighbor_similarities") or [])
    ]
    suggested_region_counts = [
        len(r.get("memory_suggested_regions") or []) for r in memory_rows
    ]
    all_regions = [
        region
        for diag in state.get("memory_diagnostics", {}).values()
        for region in diag.get("regions", [])
    ]
    memory_coverage = {
        "holdout_rows": len(memory_rows),
        "prior_available_rows": sum(
            bool(r.get("memory_prior_available")) for r in memory_rows
        ),
        "neighbor_count_distribution": neighbor_distribution,
        "neighbor_similarity": stats(similarities, DEFAULT_SEED + 120),
        "suggested_region_count": stats(
            suggested_region_counts, DEFAULT_SEED + 121
        ),
        "procedural_region_n_samples": {
            "n_regions": len(all_regions),
            "min": min((int(r["n_samples"]) for r in all_regions), default=None),
            "max": max((int(r["n_samples"]) for r in all_regions), default=None),
            "distribution": {
                str(n): sum(
                    int(r["n_samples"]) == n for r in all_regions
                )
                for n in sorted({int(r["n_samples"]) for r in all_regions})
            },
        },
    }

    proposal_vs_search: Dict[str, Any] = {}
    for idx, mode in enumerate(MODES):
        search_rows = [
            r for r in holdout
            if r.get("mode") == mode
            and not r.get("error")
            and r.get("applicable")
            and r.get("proposal_minus_search_score") is not None
            and MODE_SPECS[mode].get("use_search")
        ]
        proposal_vs_search[mode] = {
            "signed_gap": cluster_stats(
                search_rows,
                lambda r: float(r["proposal_minus_search_score"]),
                DEFAULT_SEED + 200 + idx),
            "proposal_beats_search_rate": cluster_stats(
                search_rows,
                lambda r: 1.0 if float(r["proposal_minus_search_score"]) > 1e-9 else 0.0,
                DEFAULT_SEED + 210 + idx),
            "best_observed_source": {
                source: sum(r.get("best_observed_source") == source for r in search_rows)
                for source in ("proposal", "deterministic-search")
            },
        }

    comparison: Dict[str, Any] = {
        "available": False,
        "note": "parent experiment summary unavailable",
    }
    parent_ref = config.get("parent_experiment")
    if parent_ref:
        parent_dir = (experiment_dir / str(parent_ref)).resolve()
        parent_summary_path = parent_dir / "summary.json"
        if parent_summary_path.exists():
            parent = json.loads(parent_summary_path.read_text(encoding="utf-8"))
            mode_map = {
                "llm-only": "llm-only",
                "search-only": "prior-only+search-verifier",
                "llm+search": "llm+search-verifier",
                "llm+memory+search": "llm+memory+search-verifier",
            }
            mode_changes = {}
            for old_mode, new_mode in mode_map.items():
                old = parent["mode_summary"][old_mode]
                new = mode_summary[new_mode]
                mode_changes[new_mode] = {
                    "pre_fix_mode": old_mode,
                    "pre_fix_score_delta_mean": old["score_delta"]["mean"],
                    "post_fix_score_delta_mean": new["score_delta"]["mean"],
                    "pre_fix_constraint_violation_rate": old[
                        "constraint_violation_rate"],
                    "post_fix_constraint_violation_rate": new[
                        "constraint_violation_rate"],
                    "pre_fix_prompt_tokens": old["prompt_tokens"],
                    "post_fix_prompt_tokens": new["prompt_tokens"],
                    "pre_fix_median_latency_ms": old["latency_ms"]["median"],
                    "post_fix_median_latency_ms": new["latency_ms"]["median"],
                }
            old_mem_stats = [
                d["stats"] for d in parent.get("memory_diagnostics", {}).values()]
            new_mem_stats = [
                d["stats"] for d in state.get("memory_diagnostics", {}).values()]
            comparison = {
                "available": True,
                "parent_experiment": str(parent_ref),
                "parent_head": config.get("parent_experiment_head"),
                "mode_changes": mode_changes,
                "memory": {
                    "pre_fix_admitted_total": sum(
                        int(x.get("n_admitted", 0)) for x in old_mem_stats),
                    "post_fix_admitted_total": sum(
                        int(x.get("n_admitted", 0)) for x in new_mem_stats),
                    "pre_fix_regions_total": sum(
                        int(x.get("n_procedural", 0)) for x in old_mem_stats),
                    "post_fix_regions_total": sum(
                        int(x.get("n_procedural", 0)) for x in new_mem_stats),
                },
                "paired_memory_effect": {
                    "pre_fix_mean": parent["paired_memory_effect"]["stats"]["mean"],
                    "pre_fix_ci_row_bootstrap": parent[
                        "paired_memory_effect"]["stats"]["bootstrap_95_ci"],
                    "post_fix_mean": paired["stats"]["mean"],
                    "post_fix_ci_vehicle_cluster": paired[
                        "stats"]["bootstrap_95_ci"],
                },
                "per_stratum": {
                    new_mode: {
                        stratum: {
                            "pre_fix_mean": parent["per_stratum"][old_mode][stratum]["mean"],
                            "post_fix_mean": per_stratum[new_mode][stratum]["mean"],
                            "post_fix_unique_vehicle_n": per_stratum[
                                new_mode][stratum]["unique_vehicle_n"],
                            "post_fix_observation_n": per_stratum[
                                new_mode][stratum]["observation_n"],
                        }
                        for stratum in per_stratum[new_mode]
                    }
                    for old_mode, new_mode in mode_map.items()
                },
                "warning": (
                    "修复前后目标函数、参数空间、模式语义和统计单位均已变化；"
                    "数值仅用于追踪修复影响，不能解释为同一实验条件下的因果差异。"
                ),
            }

    return {
        "experiment": {
            "git_commit_sha": config["git_commit_sha"],
            "date": config["date"],
            "provider": config["provider"],
            "model": config["model"],
            "base_url": config["base_url"],
            "repetitions": config["repetitions"],
            "demo_n_per_repetition": len(config["demo_vehicle_ids"]),
            "holdout_n_per_mode_per_repetition": len(
                config["holdout_vehicle_ids"]
            ),
        },
        "actual_usage": {
            "llm_calls": calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "case_elapsed_ms_sum": round(total_latency, 2),
            "case_elapsed_s_sum": round(total_latency / 1000.0, 2),
            "failed_cases": sum(bool(r.get("error")) for r in rows),
            "failed_attempt_records": sum(
                bool(r.get("error")) for r in raw_rows),
            "retry_count": sum(
                int(r.get("retry_count") or 0) for r in raw_rows
            ),
            "provider_error_count": sum(
                int(r.get("provider_error_count") or 0) for r in raw_rows
            ),
        },
        "mode_summary": mode_summary,
        "paired_memory_effect": paired,
        "proposal_vs_deterministic_search": proposal_vs_search,
        "pre_fix_comparison": comparison,
        "per_stratum": per_stratum,
        "per_repetition": per_repetition,
        "memory_diagnostics": state.get("memory_diagnostics", {}),
        "memory_coverage_summary": memory_coverage,
        "holdout_memory_retrieval": [
            {
                k: r.get(k) for k in (
                    "repetition", "vehicle_id", "regime",
                    "timeline_quality", "memory_prior_available",
                    "memory_neighbor_count",
                    "memory_neighbor_similarities",
                    "memory_suggested_regions",
                )
            }
            for r in holdout
            if r.get("mode") == "llm+memory+search-verifier"
        ],
        "expanded_demo_cost_estimates": estimates,
        "method_notes": [
            "跨模式不使用平均regret排名；主比较量为proposal_score-baseline_score。",
            "score统计只包含applicable=True且无运行错误的留出样本。",
            "Bootstrap CI以vehicle_id为cluster；同一车辆的repetition整组抽取。",
            "确定性搜索是与LLM候选分开的内部参考，不是现实道路真值。",
            "不同LLM模式采用独立请求；模式差异同时包含组件差异与模型生成波动。",
        ],
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def write_report(experiment_dir: Path, summary: Mapping[str, Any]) -> None:
    exp = summary["experiment"]
    usage = summary["actual_usage"]
    lines = [
        "# ECNU 真实模型轨迹清洗工作流组件消融实验（修复版）",
        "",
        "## 1. 实验设计与口径",
        "",
        f"- Provider：{exp['provider']}，模型：{exp['model']}，执行提交：{exp['git_commit_sha']}。",
        f"- 固定 12 条 demo、12 条 holdout、{exp['repetitions']} 次重复；demo 与 holdout 无重叠。",
        "- 每轮重建记忆库，holdout 只读；L2 参数区间要求至少 3 条 admitted demo。",
        "- 本实验比较工作流组件。带 `search-verifier` 的 LLM 模式中，Search 只提供内部参考，不修改 LLM proposal。",
        "- `det-search-output` 才把有界确定性搜索结果作为最终输出；该参考不代表现实真值或绝对最优。",
        "- 主质量分只含几何保真与压缩；runtime 单独报告，road 不可用时从分子和分母同时移除。",
        "",
        "## 2. 实际调用与可靠性",
        "",
        (f"共发起 {usage['llm_calls']} 次真实 LLM 请求，使用 {usage['prompt_tokens']} prompt tokens "
         f"和 {usage['completion_tokens']} completion tokens。累计 case 墙钟时间 "
         f"{usage['case_elapsed_s_sum']:.2f}s；失败 {usage['failed_cases']} 个，重试 "
         f"{usage['retry_count']} 次，provider 错误 {usage['provider_error_count']} 次。"
         f"JSONL 保留 {usage['failed_attempt_records']} 条失败尝试记录，"
         "最终统计按 case key 采用最后一次成功结果。"),
        "",
        "## 3. 主结果",
        "",
        "95% CI 使用 vehicle-cluster bootstrap：抽中一辆车时纳入该车全部 repetition。",
        "",
        "| 模式 | 独立车辆 | 观测行 | 重复 | 平均得分变化 | 95% CI | 超过默认参数比例 | 方向准确率 | LLM调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        item = summary["mode_summary"][mode]
        score = item["score_delta"]
        ci = score["bootstrap_95_ci"]
        ci_text = "N/A" if ci is None else f"[{fmt(ci[0])}, {fmt(ci[1])}]"
        lines.append(
            f"| {mode} | {score['unique_vehicle_n']} | {score['observation_n']} | "
            f"{score['repetitions']} | {fmt(score['mean'])} | {ci_text} | "
            f"{fmt(item['baseline_beaten_rate']['mean'], 3)} | "
            f"{fmt(item['direction_accuracy']['mean'], 3)} | {item['llm_calls']} |")

    lines += [
        "",
        "### 分层探索性结果",
        "",
        "| 模式 | 分层 | 独立车辆 | 观测行 | 重复 | 平均得分变化 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for stratum, item in summary["per_stratum"][mode].items():
            if not item["observation_n"]:
                continue
            lines.append(
                f"| {mode} | {stratum} | {item['unique_vehicle_n']} | "
                f"{item['observation_n']} | {item['repetitions']} | "
                f"{fmt(item['mean'])} |")

    lines += [
        "",
        "### LLM proposal 与纯确定性搜索参考",
        "",
        "| 模式 | 独立车辆 | proposal − search 平均值 | 95% CI | proposal 更高比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        item = summary["proposal_vs_deterministic_search"].get(mode)
        if not item or not item["signed_gap"]["observation_n"]:
            continue
        gap = item["signed_gap"]
        ci = gap["bootstrap_95_ci"]
        ci_text = "N/A" if ci is None else f"[{fmt(ci[0])}, {fmt(ci[1])}]"
        lines.append(
            f"| {mode} | {gap['unique_vehicle_n']} | {fmt(gap['mean'])} | "
            f"{ci_text} | {fmt(item['proposal_beats_search_rate']['mean'], 3)} |")
    lines += [
        "",
        "`det-search-output` 的较高内部得分不能解释为真实清洗质量提升：当前 objective 以去噪后轨迹作为 DP 参考，尚无人工漂移或道路真值独立核验清洗删除是否正确。",
    ]

    paired = summary["paired_memory_effect"]
    paired_ci = paired["stats"]["bootstrap_95_ci"]
    paired_ci_text = "N/A" if paired_ci is None else f"[{fmt(paired_ci[0])}, {fmt(paired_ci[1])}]"
    no_mem = summary["mode_summary"]["llm+search-verifier"]
    mem = summary["mode_summary"]["llm+memory+search-verifier"]
    lines += [
        "",
        "## 4. Memory 成本—收益联合评价",
        "",
        (f"按车辆先对三轮 paired difference 求平均，再以 {paired['unique_vehicle_n']} 辆独立车辆统计："
         f"Memory − no-memory 平均 {fmt(paired['stats']['mean'])}，95% CI {paired_ci_text}；"
         f"车辆级胜/平/负为 {paired['wins']}/{paired['ties']}/{paired['losses']}。"),
        "",
        "| 指标 | 无 Memory | 有 Memory |",
        "|---|---:|---:|",
        f"| LLM calls | {no_mem['llm_calls']} | {mem['llm_calls']} |",
        f"| prompt tokens | {no_mem['prompt_tokens']} | {mem['prompt_tokens']} |",
        f"| median case latency (s) | {fmt((no_mem['latency_ms']['median'] or 0)/1000, 2)} | {fmt((mem['latency_ms']['median'] or 0)/1000, 2)} |",
        f"| direction accuracy | {fmt(no_mem['direction_accuracy']['mean'], 3)} | {fmt(mem['direction_accuracy']['mean'], 3)} |",
        "",
        "当前结论需同时依据质量区间和成本变化；不同 LLM 模式采用独立请求，差异同时包含组件作用与模型生成波动。",
        "修复版没有检测到稳定的 Memory 质量增益；Memory prompt tokens 较高，调用次数和延迟受独立请求及失败重试影响，不能解释成稳定的成本下降。",
        "",
        "### 每辆车三轮平均 paired difference",
        "",
        "| vehicle | repetitions | mean difference |",
        "|---|---:|---:|",
    ]
    for row in paired["vehicle_means"]:
        lines.append(f"| {row['vehicle_id']} | {row['repetition_n']} | {fmt(row['mean_score_difference'], 6)} |")

    comparison = summary.get("pre_fix_comparison") or {}
    if comparison.get("available"):
        lines += [
            "",
            "## 5. 修复前后对比",
            "",
            comparison["warning"],
            "",
            "| 修复后模式 | 修复前模式 | score delta（前→后） | constraint violation（前→后） | prompt tokens（前→后） | median latency ms（前→后） |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for mode, row in comparison["mode_changes"].items():
            lines.append(
                f"| {mode} | {row['pre_fix_mode']} | "
                f"{fmt(row['pre_fix_score_delta_mean'])} → {fmt(row['post_fix_score_delta_mean'])} | "
                f"{fmt(row['pre_fix_constraint_violation_rate'], 3)} → {fmt(row['post_fix_constraint_violation_rate'], 3)} | "
                f"{row['pre_fix_prompt_tokens']} → {row['post_fix_prompt_tokens']} | "
                f"{fmt(row['pre_fix_median_latency_ms'], 1)} → {fmt(row['post_fix_median_latency_ms'], 1)} |")
        mem_cmp = comparison["memory"]
        pair_cmp = comparison["paired_memory_effect"]
        lines += [
            "",
            (f"Memory demo admitted 总数 {mem_cmp['pre_fix_admitted_total']} → "
             f"{mem_cmp['post_fix_admitted_total']}；L2 regions "
             f"{mem_cmp['pre_fix_regions_total']} → {mem_cmp['post_fix_regions_total']}。"),
            "",
            (f"Memory paired mean {fmt(pair_cmp['pre_fix_mean'])} → "
             f"{fmt(pair_cmp['post_fix_mean'])}；修复前为行级 bootstrap，修复后为车辆级统计。"),
            "",
            "### 两个 LLM+Verifier 模式的分层信号（修复前→修复后）",
            "",
            "| 模式 | 分层 | 独立车辆 | 观测行 | mean（前→后） |",
            "|---|---|---:|---:|---:|",
        ]
        for mode in ("llm+search-verifier", "llm+memory+search-verifier"):
            for stratum, row in comparison["per_stratum"][mode].items():
                if not row["post_fix_observation_n"]:
                    continue
                lines.append(
                    f"| {mode} | {stratum} | {row['post_fix_unique_vehicle_n']} | "
                    f"{row['post_fix_observation_n']} | "
                    f"{fmt(row['pre_fix_mean'])} → {fmt(row['post_fix_mean'])} |")

    lines += [
        "",
        "## 6. Memory 准入与覆盖",
        "",
        "不可适用的静止短轨迹只进入 L1 审计账本，不进入可检索 Memory 或 L2。",
        "",
        "| repetition | L1案例 | admitted | L2 region | stationary/ok | stationary/degraded | mixed/ok | mixed/degraded | moving/ok | moving/degraded |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rep, diag in summary["memory_diagnostics"].items():
        st = diag["stats"]
        a = diag["demo_admitted_by_stratum"]
        lines.append(
            f"| {rep} | {st['n_episodic']} | {st['n_admitted']} | {st['n_procedural']} | "
            f"{a['stationary/ok']} | {a['stationary/degraded']} | {a['mixed/ok']} | "
            f"{a['mixed/degraded']} | {a['moving/ok']} | {a['moving/degraded']} |")
    coverage = summary["memory_coverage_summary"]
    region = coverage["procedural_region_n_samples"]
    region_range = (
        "无（本轮没有满足 min_samples=3 的分层参数区间）"
        if region["n_regions"] == 0
        else f"{region['min']}–{region['max']}")
    lines += [
        "",
        (f"Memory 模式共有 {coverage['holdout_rows']} 条 holdout 记录，其中 "
         f"{coverage['prior_available_rows']} 条取得 episodic 先验；L2 region 共 "
         f"{region['n_regions']} 个，样本支持范围 {region_range}。"),
        "",
        "## 7. 分层结果的解释边界",
        "",
        "每个分层同时记录 unique_vehicle_n、observation_n 与 repetitions。当前每层独立车辆很少，分层差异只作为探索性信号。",
        "",
        "## 8. 产物与复现",
        "",
        "`raw_results.jsonl` 保留每条 case 的 raw / normalized / execution 参数、纯搜索参考与 best-observed 来源；"
        "`summary.json` 保存聚类统计；`run_state.json` 保存逐轮 Memory 诊断；`figures/` 全部由本次结果生成。",
        "",
    ]
    (experiment_dir / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")

def make_figures(experiment_dir: Path,
                 summary: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = experiment_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        r for r in latest_case_rows(
            read_jsonl(experiment_dir / "raw_results.jsonl"))
        if r.get("phase") == "holdout"
    ]
    colors = {
        "llm-only": "#4C78A8",
        "prior-only+search-verifier": "#72B7B2",
        "det-search-output": "#54A24B",
        "llm+search-verifier": "#F58518",
        "llm+memory+search-verifier": "#B279A2",
    }
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
    })

    proposal_modes = [
        "llm-only",
        "prior-only+search-verifier",
        "llm+search-verifier",
        "llm+memory+search-verifier",
    ]
    short_labels = {
        "llm-only": "LLM only",
        "prior-only+search-verifier": "Prior + verifier",
        "llm+search-verifier": "LLM + verifier",
        "llm+memory+search-verifier": "LLM + memory",
        "det-search-output": "Det. search",
    }

    def _score_deltas(mode: str) -> list[float]:
        return [
            float(r["proposal_score"]) - float(r["baseline_score"])
            for r in rows
            if r.get("mode") == mode
            and not r.get("error")
            and r.get("applicable")
        ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.0),
        gridspec_kw={"width_ratios": [4, 1.25]},
    )
    panels = [
        (axes[0], proposal_modes, "Proposal modes (detail)"),
        (axes[1], ["det-search-output"], "Search output\n(separate scale)"),
    ]
    for ax, modes, title in panels:
        bp = ax.boxplot(
            [_score_deltas(mode) for mode in modes],
            tick_labels=[short_labels[mode] for mode in modes],
            patch_artist=True,
            showmeans=True,
        )
        for patch, mode in zip(bp["boxes"], modes):
            patch.set_facecolor(colors[mode])
            patch.set_alpha(0.75)
        ax.axhline(0, color="#555", lw=1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("Score change")
    fig.suptitle("Score change relative to fixed default baseline")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "01_score_delta_by_mode.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    rates = [
        summary["mode_summary"][m]["baseline_beaten_rate"]["mean"] or 0
        for m in MODES
    ]
    ax.bar(MODES, rates, color=[colors[m] for m in MODES])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Baseline beaten rate (applicable cases)")
    ax.tick_params(axis="x", rotation=18)
    for i, value in enumerate(rates):
        ax.text(i, value + 0.025, f"{value:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(out / "02_baseline_beaten_rate.png", dpi=220)
    plt.close(fig)

    pairs = summary["paired_memory_effect"]["vehicle_means"]
    values = [
        p["mean_score_difference"] for p in pairs
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    point_colors = [
        "#2E8B57" if value > 0
        else "#C0392B" if value < 0
        else "#888888"
        for value in values
    ]
    ax.scatter(range(1, len(values) + 1), values,
               c=point_colors, s=28)
    ax.axhline(0, color="#555", lw=1)
    ax.set_xlabel("Vehicle (mean across repetitions)")
    ax.set_ylabel("Score difference")
    ax.set_title("Memory effect: vehicle-cluster paired difference")
    fig.tight_layout()
    fig.savefig(out / "03_paired_memory_effect.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    regimes = ["stationary", "mixed", "moving"]
    x = np.arange(len(regimes))
    width = 0.2
    for j, mode in enumerate(MODES):
        means = []
        for regime in regimes:
            values = [
                float(r["proposal_score"]) - float(r["baseline_score"])
                for r in rows
                if r.get("mode") == mode
                and r.get("regime") == regime
                and not r.get("error")
                and r.get("applicable")
            ]
            means.append(float(np.mean(values)) if values else 0.0)
        ax.bar(
            x + (j - 1.5) * width,
            means,
            width,
            label=mode,
            color=colors[mode],
        )
    ax.axhline(0, color="#555", lw=1)
    ax.set_xticks(x, regimes)
    ax.set_ylabel("Mean score change")
    ax.set_title("Score change by trajectory regime")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "04_by_regime.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    prompt = [
        summary["mode_summary"][m]["prompt_tokens"] for m in MODES
    ]
    completion = [
        summary["mode_summary"][m]["completion_tokens"] for m in MODES
    ]
    axes[0].bar(MODES, prompt, label="prompt", color="#4C78A8")
    axes[0].bar(
        MODES, completion, bottom=prompt,
        label="completion", color="#F58518",
    )
    axes[0].set_title("Token usage")
    axes[0].set_ylabel("Tokens")
    axes[0].legend()
    latency = [
        summary["mode_summary"][m]["latency_ms"]["median"] or 0
        for m in MODES
    ]
    axes[1].bar(
        MODES,
        np.array(latency) / 1000.0,
        color=[colors[m] for m in MODES],
    )
    axes[1].set_title("Median case latency")
    axes[1].set_ylabel("Seconds")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out / "05_token_latency.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = ["JSON failure", "Runtime error", "True bound violation"]
    x = np.arange(len(MODES))
    width = 0.23
    arrays = [
        [
            0 if summary["mode_summary"][m]["json_success_rate"] is None
            else 1 - summary["mode_summary"][m]["json_success_rate"]
            for m in MODES
        ],
        [
            summary["mode_summary"][m]["failure_rate"] or 0
            for m in MODES
        ],
        [
            summary["mode_summary"][m]["clamp_rate"] or 0
            for m in MODES
        ],
    ]
    for j, (label, values) in enumerate(zip(labels, arrays)):
        ax.bar(x + (j - 1) * width, values, width, label=label)
    ax.set_xticks(x, MODES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Structured output and execution reliability")
    ax.legend()
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(out / "06_reliability_rates.png", dpi=220)
    plt.close(fig)

    memory_diag = summary["memory_diagnostics"]
    stratum_labels = [f"{r}/{t}" for r, t in STRATA]
    admitted = [
        sum(
            int(
                memory_diag.get(str(rep), {})
                .get("demo_admitted_by_stratum", {})
                .get(label, 0)
            )
            for rep in range(
                1, int(summary["experiment"]["repetitions"]) + 1
            )
        )
        for label in stratum_labels
    ]
    memory_rows = summary["holdout_memory_retrieval"]
    neighbor_values = [
        int(r.get("memory_neighbor_count") or 0) for r in memory_rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].barh(stratum_labels, admitted, color="#B279A2")
    axes[0].set_xlabel("Admitted demo cases across repetitions")
    axes[0].set_title("L1 admitted evidence by stratum")
    axes[1].hist(
        neighbor_values,
        bins=range(0, 7),
        align="left",
        rwidth=0.8,
        color="#4C78A8",
    )
    axes[1].set_xlabel("Retrieved admitted neighbors")
    axes[1].set_ylabel("Holdout cases")
    axes[1].set_title("Memory retrieval coverage")
    fig.tight_layout()
    fig.savefig(out / "07_memory_coverage.png", dpi=220)
    plt.close(fig)


def analyze(experiment_dir: Path) -> Dict[str, Any]:
    summary = summarize(experiment_dir)
    atomic_json(experiment_dir / "summary.json", summary)
    make_figures(experiment_dir, summary)
    write_report(experiment_dir, summary)
    return summary
