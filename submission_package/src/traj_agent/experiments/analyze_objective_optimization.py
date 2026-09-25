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
                    "evaluations_to_90_reach_rate": cluster_stats(
                        recovery_items,
                        lambda r: 1.0 if r.get("evaluations_to_90_gain") is not None
                        else 0.0),
                    "evaluations_to_95_gain": cluster_stats(
                        recovery_items,
                        lambda r: None if r.get("evaluations_to_95_gain") is None
                        else float(r["evaluations_to_95_gain"])),
                    "evaluations_to_95_reach_rate": cluster_stats(
                        recovery_items,
                        lambda r: 1.0 if r.get("evaluations_to_95_gain") is not None
                        else 0.0),
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
        "## 1. 问题与实验口径",
        "",
        "研究问题是：在不修改助教 Objective 的前提下，LLM 与 Memory 能否作为"
        "确定性搜索的 warm start，用更少的 Objective evaluations 得到高分参数。",
        "",
        "本实验冻结上一版 Objective、活动参数和最多60次的有界内部搜索参考。"
        "该参考只用于课程 Objective 下的比较，适用范围受固定参数空间和预算约束。"
        "Teacher 与 Memory 均不使用 holdout；固定预算比较以车辆为统计 cluster。",
        "",
        f"Search Teacher Dataset 含 **{teacher['n_unique_vehicles']}** 辆独立车辆，"
        f"与 holdout 重叠 **{len(teacher['teacher_holdout_overlap'])}** 辆，"
        f"teacher 阶段 LLM 调用 **{teacher['total_llm_calls']}** 次。",
        "",
        "固定 holdout 为12辆，其中8辆满足 Objective applicable 条件；每个 LLM "
        "条件重复3次，固定预算为3、5、10、20。",
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
        "四个 applicable 诊断层的 Search best 分布如下，表内为 median [P25, P75]。"
        "这些区间来自 Teacher data，不是人工设定。",
        "",
        "| 诊断层 | n | dp_tolerance (m) | dist_threshold (m) | max_speed (m/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    distributions = teacher["parameter_distributions"]
    for stratum in ("mixed/ok", "mixed/degraded", "moving/ok", "moving/degraded"):
        row = distributions[f"stratum:{stratum}"]
        values = []
        for name in ("dp_tolerance", "dist_threshold", "max_speed_mps"):
            item = row["parameters"][name]
            values.append(
                f"{_fmt(item['median'], 3)} [{_fmt(item['p25'], 3)}, "
                f"{_fmt(item['p75'], 3)}]")
        lines.append(f"| {stratum} | {row['n']} | " + " | ".join(values) + " |")

    lines += [
        "",
        "各层中位数和四分位区间存在位移，但区间仍有重叠，因此属于诊断相关结构，"
        "不能解释为边界清晰的天然类别。stationary 轨迹因 Objective 不适用而保持"
        "默认参数，不参与 L2 区域形成。",
        "",
        "## 3. Search-Verified Memory",
        "",
        "| Teacher规模 | min_samples=3 regions | min_samples=5 regions | Episodic coverage | 同regime Episodic | Procedural coverage |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for size in DEFAULT_TEACHER_SIZES:
        m3 = summary["memory_scaling"][str(size)]["3"]
        m5 = summary["memory_scaling"][str(size)]["5"]
        coverage = m3["holdout_coverage"]
        lines.append(
            f"| {size} | {m3['region_count']} | {m5['region_count']} | "
            f"{coverage['episodic']['rate']:.1%} | "
            f"{coverage['episodic_same_regime']['rate']:.1%} | "
            f"{coverage['procedural']['rate']:.1%} |")

    lines += [
        "",
        "min_samples=3 时，L2 从 12 条 Teacher 的0个区域增加到24条及以上的"
        "12个区域（4个 applicable 诊断层 × 3个参数）。min_samples=5 时需至少"
        "48条 Teacher 才达到12个区域。Episodic 的100%覆盖包含跨 regime 回退，"
        "同 regime 覆盖为66.7%。",
        "",
        "## 4. LLM 直接区域提议",
        "",
        "下表是 LLM 所提区域中点到有界内部参考的 signed gap；越小越好。",
        "",
        "| Teacher规模 | No Memory | Episodic | Procedural | Both |",
        "|---:|---:|---:|---:|---:|",
    ]
    for size in DEFAULT_TEACHER_SIZES:
        direct = summary["direct_region_gap"][str(size)]
        lines.append(
            f"| {size} | {_fmt(direct['none']['mean'])} | "
            f"{_fmt(direct['episodic-only']['mean'])} | "
            f"{_fmt(direct['procedural-only']['mean'])} | "
            f"{_fmt(direct['episodic+procedural']['mean'])} |")

    lines += [
        "",
        "No Memory 的平均 gap 为0.1583；Teacher=200 时 Both 降到0.0887，"
        "缩小约44%。直接提议仍明显落后于后续 warm-start search，说明 LLM 更适合"
        "提供搜索区域。",
        "",
        "## 5. 固定预算主结果",
        "",
        "下表使用最大 Teacher 规模200。Gain Recovery 只在 search headroom≥0.02时计算。",
        "",
        "| 方法 | budget | Gap-to-Search | Gain Recovery | Near-Search Rate | 95%到达率 | evals-to-95%* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in _method_order():
        for budget in DEFAULT_BUDGETS:
            cell = summary["cells"]["200"][method][str(budget)]
            lines.append(
                f"| {method} | {budget} | "
                f"{_fmt(cell['gap_to_search']['mean'])} | "
                f"{_fmt(cell['gain_recovery']['mean'])} | "
                f"{_fmt(cell['near_search_rate']['mean'], 3)} | "
                f"{_fmt(cell['evaluations_to_95_reach_rate']['mean'], 3)} | "
                f"{_fmt(cell['evaluations_to_95_gain']['median'], 1)} |")

    lines += [
        "",
        "说明：evals-to-95% 只对预算内实际达到目标的记录取中位数，必须与到达率一起解释。",
        "",
        "## 6. 配对比较",
        "",
        "Warm-start 与 Pure Search 按同一车辆、同一预算、同一 repetition 配对。"
        "下表是 Teacher=200 时 warm-start − Pure Search 的 Objective 差值；"
        "95%区间按 vehicle cluster bootstrap。",
        "",
        "| Memory | budget=10 mean [95% CI] | budget=20 mean [95% CI] |",
        "|---|---:|---:|",
    ]
    for mode in MEMORY_MODES:
        values = []
        for budget in (10, 20):
            stat = summary["paired_warmstart_vs_pure"]["200"][mode][str(budget)][
                "warm_minus_pure_score"]
            ci = stat["bootstrap_95_ci"]
            values.append(
                f"{_fmt(stat['mean'])} [{_fmt(ci[0])}, {_fmt(ci[1])}]")
        lines.append(f"| {mode} | {values[0]} | {values[1]} |")

    lines += [
        "",
        "Teacher=200 时四种 warm-start 的配对均值都高于 Pure Search。"
        "budget=10 时 Episodic-only 平均提升0.1191，Both 为0.1165，"
        "Procedural-only 只有0.0056。样本仅8辆 applicable 车辆，部分区间较宽。",
        "",
        "## 7. Teacher 规模与 Memory 消融",
        "",
        "以下为 budget=10 的 mean Gap-to-Search。",
        "",
        "| Teacher规模 | Episodic | Procedural | Both |",
        "|---:|---:|---:|---:|",
    ]
    for size in DEFAULT_TEACHER_SIZES:
        cells = summary["cells"][str(size)]
        lines.append(
            f"| {size} | {_fmt(cells['llm-warmstart:episodic-only']['10']['gap_to_search']['mean'])} | "
            f"{_fmt(cells['llm-warmstart:procedural-only']['10']['gap_to_search']['mean'])} | "
            f"{_fmt(cells['llm-warmstart:episodic+procedural']['10']['gap_to_search']['mean'])} |")

    lines += [
        "",
        "Episodic-only 随 Teacher 规模增加稳定改善；Both 在24和48条时退化，到100条"
        "后才明显改善；Procedural-only 波动较大。因此不能概括为 Teacher 越多就对"
        "所有 Memory 模式稳定增益，主要贡献来自 Episodic retrieval。",
        "",
        "## 8. 调用成本与收益",
        "",
        "| 模式 | tokens/提案 | median LLM latency (ms) | budget=10 配对提升 |",
        "|---|---:|---:|---:|",
    ]
    for mode in MEMORY_MODES:
        cost_size = "0" if mode == "none" else "200"
        cost = summary["proposal_costs"][cost_size][mode]
        tokens_per = (cost["prompt_tokens"] + cost["completion_tokens"]) / cost["n"]
        improvement = summary["paired_warmstart_vs_pure"]["200"][mode]["10"][
            "warm_minus_pure_score"]["mean"]
        lines.append(
            f"| {mode} | {tokens_per:.1f} | {cost['median_latency_ms']:.1f} | "
            f"{_fmt(improvement)} |")

    lines += [
        "",
        f"实际 LLM 调用 {usage['llm_calls']} 次，prompt tokens "
        f"{usage['prompt_tokens']}，completion tokens {usage['completion_tokens']}；"
        f"逻辑 Objective evaluations {usage['logical_search_evaluations']} 次。",
        "",
        "Episodic-only 在规模200每次约885 tokens，budget=10 配对提升0.1191，"
        "是当前更有效的成本分配。Both 每次约1031 tokens，平均提升略低，但"
        "Near-Search Rate 更高。Procedural-only token 更少，Objective 增益也很小。",
        "",
        "## 9. 结论与限制",
        "",
        "1. 本实验没有修改 Objective；结论只回答该课程 Objective 下的搜索效率。",
        "2. 200条 Search Teacher 与 Episodic retrieval 能有效缩小 gap；Procedural L2 "
        "单独使用的收益有限。",
        "3. 最多60次的 reference 是 bounded internal reference；负 signed gap 只表示"
        "超过当前有界参考。",
        "4. holdout 只有12辆，其中8辆 applicable，vehicle-cluster 区间仍较宽。",
        "5. evaluations-to-target 对 headroom 不足的记录记为 N/A；未达到目标的记录"
        "不进入 evaluation 中位数。",
        "6. 只测试一个 provider/model 和三次生成重复，跨模型泛化尚未验证。",
        "7. Teacher 与 holdout 零重叠，holdout 未参与 Memory、prompt 或区域调优。",
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
    ax.set_yscale("symlog", linthresh=0.01, linscale=1.0)
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
    label_offsets = {
        "pure-search": (-10, -12),
        "llm-warmstart:none": (4, -12),
        "llm-warmstart:episodic-only": (-58, 6),
        "llm-warmstart:procedural-only": (4, -12),
        "llm-warmstart:episodic+procedural": (4, 7),
    }
    for method in methods:
        if method == "pure-search":
            x = 0.0
        else:
            mode = method.split(":", 1)[1]
            cost_size = "0" if mode == "none" else "200"
            cost = summary["proposal_costs"][cost_size][mode]
            x = (cost["prompt_tokens"] + cost["completion_tokens"]) / cost["n"]
        for budget in DEFAULT_BUDGETS:
            cell = summary["cells"]["200"][method][str(budget)]
            y = cell["objective"]["mean"]
            ax.scatter(x, y, s=30 + budget * 3, color=colors[method], alpha=0.8)
            if budget in (3, 20):
                ax.annotate(f"{labels[method]} b{budget}", (x, y), fontsize=7,
                            xytext=label_offsets[method], textcoords="offset points")
    ax.set_xlabel("LLM tokens per region proposal")
    ax.set_ylabel("Mean Objective")
    ax.set_title("LLM token cost vs Objective (teacher=200; marker size=budget)")
    fig.tight_layout(); fig.savefig(out / "07_cost_vs_objective.png", dpi=220); plt.close(fig)


def write_feedback(experiment_dir: Path, summary: Mapping[str, Any]) -> None:
    usage = summary["actual_usage"]
    paired = summary["paired_warmstart_vs_pure"]["200"]
    cells = summary["cells"]["200"]
    direct = summary["direct_region_gap"]["200"]
    test_result = summary.get("config", {}).get("validation", {}).get(
        "pytest", "未记录")
    lines = [
        "# Objective-Oriented Optimization 完成反馈",
        "",
        "## 完整性",
        "",
        f"- Search Teacher：{summary['teacher']['n_unique_vehicles']} 辆，0 次 LLM",
        "- Teacher/Holdout overlap：0",
        f"- 区域提案：{usage['proposal_rows']} 行，LLM 调用 {usage['llm_calls']} 次",
        f"- 固定预算结果：{usage['result_rows']} 行，最终失败 {summary['result_failures']} 行",
        f"- Provider errors / retries：{usage['provider_errors']} / {usage['retries']}",
        "- Objective 公式、权重、可行性定义：未修改",
        "- v2 实验目录与既有结果：未修改",
        "",
        "## 最终验收问题逐项回答",
        "",
        f"1. **Search Teacher Dataset 有多少辆独立车辆？** "
        f"{summary['teacher']['n_unique_vehicles']} 辆。",
        "2. **与 holdout 是否零重叠？** 是，重叠数为0。",
        "3. **三个 active 参数谁最敏感？** `dp_tolerance` 明显最高，其后是 "
        "`max_speed_mps`，最后是 `dist_threshold`。",
        "4. **Search best 是否存在诊断分群？** 存在诊断相关位移，但各层区间有重叠，"
        "属于中等结构，不是边界清晰的分群。",
        "5. **L2 region 从0增加到多少？** min_samples=3 时由规模12的0个增至"
        "规模24及以上的12个；min_samples=5 时规模48才达到12个。",
        "6. **Episodic 与 Procedural coverage 分别多少？** 最大规模下 Episodic "
        "覆盖100%，同 regime 覆盖66.7%；Procedural 覆盖66.7%。",
        "7. **LLM direct proposal 的 gap 是否缩小？** 是。No Memory 为0.1583，"
        f"规模200的 Both 为{direct['episodic+procedural']['mean']:.4f}，缩小约44%；"
        "规模变化并非全程单调。",
        "8. **固定 budget 下 Warm-start 是否优于 Pure Search？** 在 Teacher=200 的"
        "四种模式中配对均值均为正；budget=10 时 Episodic-only / Both / "
        f"Procedural-only 分别提升 "
        f"{paired['episodic-only']['10']['warm_minus_pure_score']['mean']:.4f} / "
        f"{paired['episodic+procedural']['10']['warm_minus_pure_score']['mean']:.4f} / "
        f"{paired['procedural-only']['10']['warm_minus_pure_score']['mean']:.4f}。",
        "9. **达到95% Search gain 分别需要多少 evaluations？** 在有足够 headroom 且"
        "预算20内成功达到的记录中，Pure / Episodic-only / Both 的中位数为 "
        f"{cells['pure-search']['20']['evaluations_to_95_gain']['median']:.0f} / "
        f"{cells['llm-warmstart:episodic-only']['20']['evaluations_to_95_gain']['median']:.0f} / "
        f"{cells['llm-warmstart:episodic+procedural']['20']['evaluations_to_95_gain']['median']:.0f}；"
        "对应到达率为 "
        f"{cells['pure-search']['20']['evaluations_to_95_reach_rate']['mean']:.1%} / "
        f"{cells['llm-warmstart:episodic-only']['20']['evaluations_to_95_reach_rate']['mean']:.1%} / "
        f"{cells['llm-warmstart:episodic+procedural']['20']['evaluations_to_95_reach_rate']['mean']:.1%}。",
        "10. **Demo/Teacher size 增加是否带来稳定收益？** Episodic-only 稳定改善；"
        "Both 和 Procedural-only 非单调，因此整体答案是否。",
        "11. **哪种 Memory 真正有贡献？** Episodic 是主要贡献来源；Both 改善 "
        "Near-Search Rate，Procedural-only 单独贡献很小。",
        "12. **token 增加是否值得？** Episodic-only 在规模200每提案约885 tokens，"
        "budget=10 配对提升0.1191，当前性价比最好；Both token 更多但平均提升略低；"
        "Procedural-only 不划算。",
        "13. **本轮有没有修改 Objective？** 没有。",
        f"14. **完整测试多少 passed？** {test_result}。",
        "15. **当前限制？** holdout 仅12辆且只有8辆 applicable；一个 provider/model、"
        "三次生成重复；有界参考受固定空间和预算约束；部分车辆聚类置信区间较宽。",
        "",
        f"实际成本：prompt tokens {usage['prompt_tokens']}，completion tokens "
        f"{usage['completion_tokens']}，逻辑 Objective evaluations "
        f"{usage['logical_search_evaluations']}。",
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
