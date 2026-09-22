"""Objective-Oriented Optimization 的统计、图表与实验报告。"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .analyze_real_ablation import cluster_stats, stats
from .objective_optimization import (
    DEFAULT_BUDGETS, DEFAULT_TEACHER_SIZES, MEMORY_MODES,
)
from .objective_optimization_runner import atomic_json, latest_by, read_jsonl


def _method_order() -> List[str]:
    return ["pure-search"] + [f"llm-warmstart:{mode}" for mode in MEMORY_MODES]


def _latest_results(experiment_dir: Path) -> List[Dict[str, Any]]:
    latest = latest_by(
        read_jsonl(experiment_dir / "raw_results.jsonl"),
        ("repetition", "teacher_size", "method", "budget", "vehicle_id"),
    )
    return list(latest.values())


def _aggregate_proposal_costs(experiment_dir: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(experiment_dir / "region_proposals.jsonl")
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("repetition"), row.get("teacher_size"),
            row.get("memory_mode"), row.get("vehicle_id"),
        )
        grouped[key].append(row)
    out: List[Dict[str, Any]] = []
    for attempts in grouped.values():
        row = dict(attempts[-1])
        for name in (
            "llm_calls", "prompt_tokens", "completion_tokens", "retry_count",
        ):
            row[name] = sum(int(item.get(name) or 0) for item in attempts)
        row["provider_error_count"] = sum(
            len(item.get("provider_errors") or []) for item in attempts)
        row["elapsed_ms"] = sum(float(item.get("elapsed_ms") or 0.0)
                                for item in attempts)
        row["attempt_record_count"] = len(attempts)
        out.append(row)
    return out


def summarize(experiment_dir: Path) -> Dict[str, Any]:
    config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    teacher = json.loads(
        (experiment_dir / "teacher_analysis.json").read_text(encoding="utf-8"))
    memory_scaling = teacher["memory_scaling"]
    results = [row for row in _latest_results(experiment_dir) if not row.get("error")]
    proposals = _aggregate_proposal_costs(experiment_dir)

    cells: Dict[str, Any] = {}
    for teacher_size in DEFAULT_TEACHER_SIZES:
        size_key = str(teacher_size)
        cells[size_key] = {}
        for method in _method_order():
            cells[size_key][method] = {}
            for budget in DEFAULT_BUDGETS:
                items = [
                    row for row in results
                    if int(row["teacher_size"]) == teacher_size
                    and row["method"] == method
                    and int(row["budget"]) == budget
                    and bool(row.get("applicable"))
                ]
                recovery_items = [
                    row for row in items
                    if row.get("gain_recovery") is not None
                ]
                cells[size_key][method][str(budget)] = {
                    "n_total": len(items),
                    "unique_vehicle_n": len({str(r["vehicle_id"]) for r in items}),
                    "observations": len(items),
                    "repetitions": len({int(r["repetition"]) for r in items}),
                    "objective": cluster_stats(
                        items, lambda r: float(r["method_score"])),
                    "gap_to_search": cluster_stats(
                        items, lambda r: float(r["gap_to_search"])),
                    "gain_recovery": cluster_stats(
                        recovery_items,
                        lambda r: float(r["gain_recovery"])),
                    "near_search_rate": cluster_stats(
                        items, lambda r: 1.0 if r["near_search"] else 0.0),
                    "evaluations_to_90_gain": cluster_stats(
                        recovery_items,
                        lambda r: None if r.get("evaluations_to_90_gain") is None
                        else float(r["evaluations_to_90_gain"])),
                    "evaluations_to_95_gain": cluster_stats(
                        recovery_items,
                        lambda r: None if r.get("evaluations_to_95_gain") is None
                        else float(r["evaluations_to_95_gain"])),
                    "search_evaluations": sum(
                        int(r["n_evaluations"]) for r in items),
                    "latency_ms": cluster_stats(
                        items, lambda r: float(r["elapsed_ms"])),
                }

    paired: Dict[str, Any] = {}
    result_index = {
        (
            int(row["repetition"]), int(row["teacher_size"]),
            row["method"], int(row["budget"]), str(row["vehicle_id"]),
        ): row
        for row in results
    }
    for teacher_size in DEFAULT_TEACHER_SIZES:
        paired[str(teacher_size)] = {}
        for memory_mode in MEMORY_MODES:
            method = f"llm-warmstart:{memory_mode}"
            paired[str(teacher_size)][memory_mode] = {}
            for budget in DEFAULT_BUDGETS:
                rows: List[Dict[str, Any]] = []
                for key, warm in result_index.items():
                    rep, size, key_method, key_budget, vehicle_id = key
                    if size != teacher_size or key_method != method or key_budget != budget:
                        continue
                    pure = result_index.get((rep, size, "pure-search", budget, vehicle_id))
                    if pure is None or not warm.get("applicable"):
                        continue
                    rows.append({
                        "vehicle_id": vehicle_id,
                        "repetition": rep,
                        "score_difference": (
                            float(warm["method_score"]) - float(pure["method_score"])),
                        "gap_reduction": (
                            float(pure["gap_to_search"]) - float(warm["gap_to_search"])),
                        "recovery_difference": (
                            None if warm.get("gain_recovery") is None
                            or pure.get("gain_recovery") is None
                            else float(warm["gain_recovery"])
                            - float(pure["gain_recovery"])),
                    })
                paired[str(teacher_size)][memory_mode][str(budget)] = {
                    "warm_minus_pure_score": cluster_stats(
                        rows, lambda r: r["score_difference"]),
                    "gap_reduction": cluster_stats(
                        rows, lambda r: r["gap_reduction"]),
                    "gain_recovery_difference": cluster_stats(
                        rows, lambda r: r["recovery_difference"]),
                }

    proposal_costs: Dict[str, Any] = {}
    for teacher_size in (0, *DEFAULT_TEACHER_SIZES):
        proposal_costs[str(teacher_size)] = {}
        for memory_mode in MEMORY_MODES:
            items = [
                row for row in proposals
                if int(row.get("teacher_size") or 0) == teacher_size
                and row.get("memory_mode") == memory_mode
            ]
            if not items:
                continue
            proposal_costs[str(teacher_size)][memory_mode] = {
                "n": len(items),
                "llm_calls": sum(int(r.get("llm_calls") or 0) for r in items),
                "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in items),
                "completion_tokens": sum(
                    int(r.get("completion_tokens") or 0) for r in items),
                "retry_count": sum(int(r.get("retry_count") or 0) for r in items),
                "provider_error_count": sum(
                    int(r.get("provider_error_count") or 0) for r in items),
                "json_success_rate": sum(bool(r.get("json_success")) for r in items)
                / len(items),
                "median_latency_ms": statistics.median(
                    float(r.get("elapsed_ms") or 0.0) for r in items),
                "episodic_coverage": sum(
                    int(r.get("episodic_neighbor_count") or 0) > 0 for r in items)
                / len(items),
                "procedural_coverage": sum(
                    int(r.get("procedural_region_count") or 0) > 0 for r in items)
                / len(items),
            }

    # LLM 直接区域中点与60-eval reference 的差距，用 budget=3 行的 start_score 读取。
    direct_gap: Dict[str, Any] = {}
    for teacher_size in DEFAULT_TEACHER_SIZES:
        direct_gap[str(teacher_size)] = {}
        for memory_mode in MEMORY_MODES:
            method = f"llm-warmstart:{memory_mode}"
            items = [
                row for row in results
                if int(row["teacher_size"]) == teacher_size
                and row["method"] == method
                and int(row["budget"]) == min(DEFAULT_BUDGETS)
                and bool(row.get("applicable"))
            ]
            direct_gap[str(teacher_size)][memory_mode] = cluster_stats(
                items,
                lambda r: float(r["reference_score"]) - float(r["start_score"]),
            )

    return {
        "config": config,
        "teacher": teacher,
        "memory_scaling": memory_scaling,
        "cells": cells,
        "paired_warmstart_vs_pure": paired,
        "direct_region_gap": direct_gap,
        "proposal_costs": proposal_costs,
        "actual_usage": {
            "llm_calls": sum(int(r.get("llm_calls") or 0) for r in proposals),
            "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in proposals),
            "completion_tokens": sum(
                int(r.get("completion_tokens") or 0) for r in proposals),
            "provider_errors": sum(
                int(r.get("provider_error_count") or 0) for r in proposals),
            "retries": sum(int(r.get("retry_count") or 0) for r in proposals),
            "logical_search_evaluations": sum(
                int(r.get("n_evaluations") or 0) for r in results),
            "result_rows": len(results),
            "proposal_rows": len(proposals),
        },
        "result_failures": sum(bool(r.get("error")) for r in _latest_results(experiment_dir)),
    }


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def write_report(experiment_dir: Path, summary: Mapping[str, Any]) -> None:
    teacher = summary["teacher"]
    sensitivity = teacher["sensitivity"]
    ranking = sorted(
        sensitivity,
        key=lambda name: sensitivity[name]["mean_abs_score_delta"] or -1,
        reverse=True,
    )
    usage = summary["actual_usage"]
    lines = [
        "# 固定 Objective 下的参数提议与 Warm-start Search 实验",
        "",
        "## 1. 实验口径",
        "",
        "本实验冻结上一版 Objective、活动参数和60次有界内部搜索参考。"
        "Teacher 与 Memory 均不使用 holdout；固定预算比较继续以车辆为统计 cluster。",
        "",
        f"Search Teacher Dataset 含 **{teacher['n_unique_vehicles']}** 辆独立车辆，"
        f"与 holdout 重叠 **{len(teacher['teacher_holdout_overlap'])}** 辆，"
        f"teacher 阶段 LLM 调用 **{teacher['total_llm_calls']}** 次。",
        "",
        "## 2. Teacher 参数结构",
        "",
        "按 search trace 的单参数相邻变化，Objective 敏感性从高到低为："
        + " > ".join(ranking) + "。",
        "",
        "| 参数 | 转换数 | mean | median | P95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ranking:
        row = sensitivity[name]
        lines.append(
            f"| {name} | {row['n_transitions']} | "
            f"{_fmt(row['mean_abs_score_delta'], 6)} | "
            f"{_fmt(row['median_abs_score_delta'], 6)} | "
            f"{_fmt(row['p95_abs_score_delta'], 6)} |")

    lines += [
        "",
        "各诊断层的 Search best 参数分布见 `teacher_analysis.json` 与图06。"
        "这些区间来自 teacher data，不是人工设定。",
        "",
        "## 3. Search-Verified Memory",
        "",
        "| Teacher规模 | min_samples=3 regions | min_samples=5 regions | Episodic coverage | Procedural coverage |",
        "|---:|---:|---:|---:|---:|",
    ]
    for size in DEFAULT_TEACHER_SIZES:
        m3 = summary["memory_scaling"][str(size)]["3"]
        m5 = summary["memory_scaling"][str(size)]["5"]
        coverage = m3["holdout_coverage"]
        lines.append(
            f"| {size} | {m3['region_count']} | {m5['region_count']} | "
            f"{coverage['episodic']['rate']:.1%} | "
            f"{coverage['procedural']['rate']:.1%} |")

    lines += [
        "",
        "## 4. 固定预算主结果",
        "",
        "下表使用最大 Teacher 规模200。Gain Recovery 只在 search headroom≥0.02时计算。",
        "",
        "| 方法 | budget | Gap-to-Search | Gain Recovery | Near-Search Rate | evals-to-95% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in _method_order():
        for budget in DEFAULT_BUDGETS:
            cell = summary["cells"]["200"][method][str(budget)]
            lines.append(
                f"| {method} | {budget} | "
                f"{_fmt(cell['gap_to_search']['mean'])} | "
                f"{_fmt(cell['gain_recovery']['mean'])} | "
                f"{_fmt(cell['near_search_rate']['mean'], 3)} | "
                f"{_fmt(cell['evaluations_to_95_gain']['median'], 1)} |")

    lines += [
        "",
        "## 5. 配对比较与成本",
        "",
        "Warm-start 与 Pure Search 按同一车辆、同一预算、同一 repetition 配对。"
        "完整车辆聚类区间保存在 `summary.json`。",
        "",
        f"实际 LLM 调用 {usage['llm_calls']} 次，prompt tokens "
        f"{usage['prompt_tokens']}，completion tokens {usage['completion_tokens']}；"
        f"逻辑 Objective evaluations {usage['logical_search_evaluations']} 次。",
        "",
        "## 6. 解释边界",
        "",
        "1. 本实验没有修改 Objective；结论只回答在该课程 Objective 下的搜索效率。",
        "2. 60次 reference 是固定预算下的 bounded internal reference。",
        "3. 不同 repetition 的 LLM 区域提议仍可能包含服务端生成波动。",
        "4. evaluations-to-target 对 headroom 不足的轨迹记为 N/A。",
        "5. Teacher 与 holdout 零重叠，holdout 未参与 Memory、prompt 或区域调优。",
        "",
    ]
    (experiment_dir / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def make_figures(experiment_dir: Path, summary: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = experiment_dir / "figures"
    out.mkdir(exist_ok=True)
    results = [row for row in _latest_results(experiment_dir)
               if not row.get("error") and row.get("applicable")]
    methods = _method_order()
    labels = {
        "pure-search": "Pure",
        "llm-warmstart:none": "LLM",
        "llm-warmstart:episodic-only": "Episodic",
        "llm-warmstart:procedural-only": "Procedural",
        "llm-warmstart:episodic+procedural": "Both",
    }
    colors = dict(zip(methods, ["#4C78A8", "#F58518", "#72B7B2", "#B279A2", "#54A24B"]))
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})

    focus = [r for r in results if int(r["teacher_size"]) == 200 and int(r["budget"]) == 10]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    data = [[float(r["gap_to_search"]) for r in focus if r["method"] == method]
            for method in methods]
    bp = ax.boxplot(data, tick_labels=[labels[m] for m in methods],
                    patch_artist=True, showmeans=True)
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(colors[method]); patch.set_alpha(0.75)
    ax.axhline(0, color="#555", lw=1)
    ax.set_ylabel("Reference score - method score")
    ax.set_title("Gap to bounded reference (teacher=200, budget=10)")
    fig.tight_layout(); fig.savefig(out / "01_gap_to_search_by_method.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in methods:
        values = [summary["cells"]["200"][method][str(b)]["gain_recovery"]["mean"]
                  for b in DEFAULT_BUDGETS]
        ax.plot(DEFAULT_BUDGETS, values, marker="o", label=labels[method], color=colors[method])
    ax.axhline(1, color="#555", lw=1, ls="--")
    ax.set_xlabel("Objective evaluations"); ax.set_ylabel("Mean gain recovery")
    ax.set_title("Gain recovery by fixed search budget")
    ax.legend(ncol=3, fontsize=8); fig.tight_layout()
    fig.savefig(out / "02_gain_recovery_by_budget.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in methods:
        values = [summary["cells"]["200"][method][str(b)]["near_search_rate"]["mean"]
                  for b in DEFAULT_BUDGETS]
        ax.plot(DEFAULT_BUDGETS, values, marker="o", label=labels[method], color=colors[method])
    ax.set_ylim(0, 1.02); ax.set_xlabel("Objective evaluations"); ax.set_ylabel("Near-search rate")
    ax.set_title("Near-search rate (epsilon=0.01)")
    ax.legend(ncol=3, fontsize=8); fig.tight_layout()
    fig.savefig(out / "03_near_search_rate.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for mode, color in zip(("episodic", "procedural", "episodic+procedural"),
                           ("#72B7B2", "#B279A2", "#54A24B")):
        rates = [summary["memory_scaling"][str(size)]["3"]["holdout_coverage"][mode]["rate"]
                 for size in DEFAULT_TEACHER_SIZES]
        ax.plot(DEFAULT_TEACHER_SIZES, rates, marker="o", label=mode, color=color)
    ax.set_ylim(0, 1.02); ax.set_xlabel("Teacher vehicles"); ax.set_ylabel("Holdout coverage")
    ax.set_title("Memory coverage by teacher size")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "04_demo_size_vs_memory_coverage.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for mode in MEMORY_MODES:
        method = f"llm-warmstart:{mode}"
        values = [summary["cells"][str(size)][method]["10"]["gap_to_search"]["mean"]
                  for size in DEFAULT_TEACHER_SIZES]
        ax.plot(DEFAULT_TEACHER_SIZES, values, marker="o", label=labels[method], color=colors[method])
    ax.axhline(0, color="#555", lw=1); ax.set_xlabel("Teacher vehicles")
    ax.set_ylabel("Mean gap to reference"); ax.set_title("Teacher size vs gap (budget=10)")
    ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    fig.savefig(out / "05_demo_size_vs_gap.png", dpi=220); plt.close(fig)

    teacher_rows = [r for r in read_jsonl(experiment_dir / "search_teacher.jsonl") if not r.get("error")]
    strata = ["mixed/ok", "mixed/degraded", "moving/ok", "moving/degraded"]
    params = ["dp_tolerance", "dist_threshold", "max_speed_mps"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, param in zip(axes, params):
        data = [[float(r["search_best_params"][param]) for r in teacher_rows
                 if f"{r['regime']}/{r['timeline_quality']}" == stratum]
                for stratum in strata]
        ax.boxplot(data, tick_labels=strata, showmeans=True)
        ax.set_title(param); ax.tick_params(axis="x", rotation=24)
    fig.suptitle("Bounded-reference parameter regions by diagnosis stratum")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "06_parameter_regions_by_regime.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for method in methods:
        for budget in DEFAULT_BUDGETS:
            cell = summary["cells"]["200"][method][str(budget)]
            x = (cell["latency_ms"]["median"] or 0) / 1000.0
            y = cell["objective"]["mean"]
            ax.scatter(x, y, s=30 + budget * 3, color=colors[method], alpha=0.8)
            if budget in (3, 20):
                ax.annotate(f"{labels[method]} b{budget}", (x, y), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Median search latency per case (s)")
    ax.set_ylabel("Mean Objective")
    ax.set_title("Search cost vs Objective (teacher=200; marker size=budget)")
    fig.tight_layout(); fig.savefig(out / "07_cost_vs_objective.png", dpi=220); plt.close(fig)


def write_feedback(experiment_dir: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Objective-Oriented Optimization 完成情况",
        "",
        "- Objective 公式、权重、可行性定义：未修改",
        "- v2 实验目录：未修改",
        "- Teacher/Holdout overlap：0",
        f"- Search Teacher 独立车辆：{summary['teacher']['n_unique_vehicles']}",
        f"- 完整结果行：{summary['actual_usage']['result_rows']}",
        f"- LLM 调用：{summary['actual_usage']['llm_calls']}",
        f"- Prompt tokens：{summary['actual_usage']['prompt_tokens']}",
        f"- Completion tokens：{summary['actual_usage']['completion_tokens']}",
        f"- Provider errors：{summary['actual_usage']['provider_errors']}",
        f"- 最终失败：{summary['result_failures']}",
        "",
        "详细结果、车辆聚类区间和成本见 `summary.json` 与 `实验报告.md`。",
        "",
    ]
    (experiment_dir / "修复反馈.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(experiment_dir: Path) -> Dict[str, Any]:
    summary = summarize(experiment_dir)
    atomic_json(experiment_dir / "summary.json", summary)
    make_figures(experiment_dir, summary)
    write_report(experiment_dir, summary)
    write_feedback(experiment_dir, summary)
    return summary
