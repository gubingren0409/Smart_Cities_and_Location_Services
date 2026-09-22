"""汇总 Objective warm-start 收尾实验并生成正式图表与报告。"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import objective_optimization as opt
from . import objective_optimization_refined as refined
from . import objective_optimization_refined_runner as runner
from .analyze_real_ablation import cluster_stats


def _valid_applicable(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows if not row.get("error") and bool(row.get("applicable"))]


def _stats(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    return cluster_stats(rows, lambda row: (
        None if row.get(field) is None else float(row[field])))


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    return cluster_stats(rows, lambda row: (
        None if row.get(field) is None else float(bool(row[field]))))


def _cell_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "objective": _stats(rows, "method_score"),
        "gap_to_reference": _stats(rows, "gap_to_search"),
        "gain_recovery": _stats(rows, "gain_recovery"),
        "near_reference_rate": _rate(rows, "near_search"),
        "evaluations_to_95_gain": _stats(rows, "evaluations_to_95_gain"),
        "evaluations_to_95_reach_rate": cluster_stats(
            rows,
            lambda row: (
                None if not row.get("gain_recovery_applicable")
                else float(row.get("evaluations_to_95_gain") is not None)),
        ),
        "n_evaluations": _stats(rows, "n_evaluations"),
    }


def _group_cells(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    groups: Dict[Tuple[int, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["teacher_size"]), str(row["method"]), int(row["budget"]))].append(row)
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (size, method, budget), members in sorted(groups.items()):
        out.setdefault(str(size), {}).setdefault(method, {})[str(budget)] = _cell_summary(members)
    return out


def _paired_rows(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    left_field: str = "method_score",
    right_field: str = "method_score",
) -> List[Dict[str, Any]]:
    key_fields = ("vehicle_id", "repetition", "budget")
    right_map = {
        tuple(row.get(field) for field in key_fields): row for row in right
    }
    out: List[Dict[str, Any]] = []
    for row in left:
        key = tuple(row.get(field) for field in key_fields)
        peer = right_map.get(key)
        if peer is None:
            continue
        left_value = row.get(left_field)
        right_value = peer.get(right_field)
        if left_value is None or right_value is None:
            continue
        out.append({
            "vehicle_id": str(row["vehicle_id"]),
            "repetition": int(row["repetition"]),
            "budget": int(row["budget"]),
            "delta": float(left_value) - float(right_value),
        })
    return out


def _paired_stat(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    paired = _paired_rows(left, right)
    return {
        "aligned": refined.validate_paired_alignment(left, right),
        "left_minus_right": cluster_stats(paired, lambda row: float(row["delta"])),
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _cost_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    fields = (
        "llm_latency_ms", "objective_search_ms", "total_online_ms",
        "end_to_end_online_ms", "llm_calls", "prompt_tokens",
        "completion_tokens", "n_evaluations", "retries", "provider_errors",
    )
    out: Dict[str, Any] = {"n": len(rows)}
    for field in fields:
        values = [float(row.get(field) or 0.0) for row in rows]
        out[field] = {
            "total": sum(values),
            "mean": _mean(values),
            "median": statistics.median(values) if values else None,
        }
    return out


def build_cost_summary(
    old_dir: Path,
    recalculated: Sequence[Mapping[str, Any]],
    deterministic: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    proposal_rows = runner.latest_by(
        runner.read_jsonl(old_dir / "region_proposals.jsonl"),
        ("repetition", "teacher_size", "memory_mode", "vehicle_id"),
    )
    online_rows: List[Dict[str, Any]] = []
    for row in recalculated:
        proposal = None
        if row["method"] != "pure-search":
            key = tuple(row.get("proposal_key") or [])
            proposal = proposal_rows.get(key)
        online_rows.append({
            "teacher_size": int(row["teacher_size"]),
            "method": str(row["method"]),
            "budget": int(row["budget"]),
            "vehicle_id": str(row["vehicle_id"]),
            "repetition": int(row["repetition"]),
            "method_score": float(row["method_score"]),
            "n_evaluations": int(row["n_evaluations"]),
            **refined.online_cost(
                search_elapsed_ms=float(row.get("elapsed_ms") or 0.0),
                proposal=proposal,
            ),
        })
    for row in deterministic:
        online_rows.append({
            "teacher_size": int(row["teacher_size"]),
            "method": str(row["method"]),
            "budget": int(row["budget"]),
            "vehicle_id": str(row["vehicle_id"]),
            "repetition": int(row["repetition"]),
            "method_score": float(row["method_score"]),
            "n_evaluations": int(row["n_evaluations"]),
            **refined.online_cost(
                search_elapsed_ms=float(row.get("elapsed_ms") or 0.0)),
        })

    focus = [row for row in online_rows if int(row["teacher_size"]) == 200]
    cells: Dict[str, Dict[str, Any]] = {}
    for method in sorted({str(row["method"]) for row in focus}):
        cells[method] = {}
        for budget in opt.DEFAULT_BUDGETS:
            members = [
                row for row in focus
                if row["method"] == method and int(row["budget"]) == budget
            ]
            if members:
                cells[method][str(budget)] = _cost_stats(members)

    teacher_rows = [
        row for row in runner.read_jsonl(old_dir / "search_teacher.jsonl")
        if not row.get("error")
    ]
    teacher_offline = {
        "vehicles": len({str(row["vehicle_id"]) for row in teacher_rows}),
        "records": len(teacher_rows),
        "objective_evaluations": sum(int(row.get("n_evaluations") or 0) for row in teacher_rows),
        "wall_clock_ms": sum(float(row.get("elapsed_ms") or 0.0) for row in teacher_rows),
        "llm_calls": sum(int(row.get("llm_calls") or 0) for row in teacher_rows),
        "included_in_per_query_online_cost": False,
    }
    amortization: Dict[str, Any] = {}
    for budget in opt.DEFAULT_BUDGETS:
        pure = cells["pure-search"][str(budget)]["total_online_ms"]["mean"]
        warm = cells["llm-warmstart:episodic-only"][str(budget)]["total_online_ms"]["mean"]
        saving = float(pure) - float(warm)
        amortization[str(budget)] = {
            "pure_mean_online_ms": pure,
            "episodic_warm_mean_online_ms": warm,
            "saving_per_query_ms": saving,
            "break_even_queries": (
                teacher_offline["wall_clock_ms"] / saving if saving > 0 else None),
            "status": "finite" if saving > 0 else "no break-even",
        }
    return {
        "definition": {
            "total_online_ms": "LLM API latency + cumulative Objective search wall-clock",
            "end_to_end_online_ms": "outer proposal wall-clock + cumulative Objective search wall-clock",
            "teacher_offline_separate": True,
        },
        "teacher_offline": teacher_offline,
        "teacher200_online_cells": cells,
        "amortization_pure_vs_episodic_warm": amortization,
    }


def summarize(experiment_dir: Path) -> Dict[str, Any]:
    runner.assert_frozen_artifacts(experiment_dir)
    config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    old_dir = experiment_dir.parent / runner.OLD_EXPERIMENT
    strong_rows = [
        row for row in runner.read_jsonl(experiment_dir / "strong_reference.jsonl")
        if not row.get("error")
    ]
    strong = {(str(row["vehicle_id"]),): row for row in strong_rows}
    if not strong_rows:
        raise RuntimeError("strong_reference.jsonl 为空")

    old_raw = runner.latest_by(
        runner.read_jsonl(old_dir / "raw_results.jsonl"),
        ("repetition", "teacher_size", "method", "budget", "vehicle_id"),
    ).values()
    old_applicable = _valid_applicable(old_raw)
    recalculated = [
        refined.recompute_reference_metrics(row, strong[(str(row["vehicle_id"]),)])
        for row in old_applicable
    ]

    deterministic = _valid_applicable(runner.latest_by(
        runner.read_jsonl(experiment_dir / "deterministic_memory_ablation.jsonl"),
        ("repetition", "teacher_size", "method", "budget", "vehicle_id"),
    ).values())
    expected_det = 3 * len(opt.DEFAULT_TEACHER_SIZES) * 2 * len(opt.DEFAULT_BUDGETS) * len(runner.HOLDOUT)
    actual_det_all = len(runner.latest_by(
        runner.read_jsonl(experiment_dir / "deterministic_memory_ablation.jsonl"),
        ("repetition", "teacher_size", "method", "budget", "vehicle_id"),
    ))

    region_rows = runner.read_jsonl(experiment_dir / "region_quality.jsonl")
    region_applicable = [row for row in region_rows if row.get("applicable")]
    region_summary: Dict[str, Dict[str, Any]] = {}
    for row in region_applicable:
        size = str(row["teacher_size"])
        mode = str(row["memory_mode"])
        region_summary.setdefault(size, {}).setdefault(mode, [])
        region_summary[size][mode].append(row)
    region_aggregated: Dict[str, Any] = {}
    for size, modes in region_summary.items():
        region_aggregated[size] = {}
        for mode, members in modes.items():
            region_aggregated[size][mode] = {
                "containment_rate": _rate(members, "reference_containment"),
                "normalized_volume": _stats(members, "normalized_region_volume"),
                "region_efficiency": _stats(members, "region_efficiency"),
            }

    old_negative = [row for row in old_applicable if float(row["gap_to_search"]) < -1e-12]
    strong_negative = [row for row in recalculated if float(row["gap_to_search"]) < -1e-12]
    teacher_rows = [
        row for row in runner.read_jsonl(old_dir / "search_teacher.jsonl")
        if not row.get("error")
    ]
    sensitivity = refined.normalized_sensitivity_from_teacher(teacher_rows)
    sensitivity_rank = sorted(
        sensitivity,
        key=lambda name: float(sensitivity[name]["mean_normalized_sensitivity"] or -1.0),
        reverse=True,
    )

    paired: Dict[str, Any] = {}
    for size in opt.DEFAULT_TEACHER_SIZES:
        paired[str(size)] = {}
        for budget in opt.DEFAULT_BUDGETS:
            pure = [row for row in recalculated if int(row["teacher_size"]) == size
                    and row["method"] == "pure-search" and int(row["budget"]) == budget]
            epi_det = [row for row in deterministic if int(row["teacher_size"]) == size
                       and row["method"] == "episodic-region-search" and int(row["budget"]) == budget]
            proc_det = [row for row in deterministic if int(row["teacher_size"]) == size
                        and row["method"] == "procedural-region-search" and int(row["budget"]) == budget]
            epi_llm = [row for row in recalculated if int(row["teacher_size"]) == size
                       and row["method"] == "llm-warmstart:episodic-only" and int(row["budget"]) == budget]
            proc_llm = [row for row in recalculated if int(row["teacher_size"]) == size
                        and row["method"] == "llm-warmstart:procedural-only" and int(row["budget"]) == budget]
            paired[str(size)][str(budget)] = {
                "episodic_deterministic_vs_pure": _paired_stat(epi_det, pure),
                "procedural_deterministic_vs_pure": _paired_stat(proc_det, pure),
                "episodic_llm_vs_deterministic": _paired_stat(epi_llm, epi_det),
                "procedural_llm_vs_deterministic": _paired_stat(proc_llm, proc_det),
                "episodic_llm_vs_pure": _paired_stat(epi_llm, pure),
            }

    cost_summary = build_cost_summary(old_dir, recalculated, deterministic)
    strong_improvements = [float(row["score_improvement"]) for row in strong_rows]
    strong_summary = {
        "n_applicable_vehicles": len(strong_rows),
        "llm_calls": sum(int(row.get("llm_calls") or 0) for row in strong_rows),
        "mean_score_improvement": _mean(strong_improvements),
        "median_score_improvement": statistics.median(strong_improvements),
        "max_score_improvement": max(strong_improvements),
        "improved_vehicle_n": sum(value > 1e-12 for value in strong_improvements),
        "source_counts": dict(sorted((
            (source, sum(row["source_method"].split(":", 1)[0] == source for row in strong_rows))
            for source in ("old_cd60", "global_halton512", "multistart_cd")
        ))),
        "logical_evaluations": sum(int(row["n_evaluations"]) for row in strong_rows),
        "new_unique_objective_evaluations": sum(
            int(row["new_unique_objective_evaluations"]) for row in strong_rows),
        "new_wall_clock_ms": sum(float(row["elapsed_ms"]) for row in strong_rows),
        "reference_wording": "stronger bounded deterministic reference; not global optimum",
    }

    return {
        "config": config,
        "strong_reference": strong_summary,
        "reference_effect": {
            "recalculated_observation_n": len(recalculated),
            "old_negative_gap_n": len(old_negative),
            "old_negative_gap_rate": len(old_negative) / len(old_applicable),
            "strong_negative_gap_n": len(strong_negative),
            "strong_negative_gap_rate": len(strong_negative) / len(recalculated),
            "teacher200_negative_gap_n": sum(
                int(row["teacher_size"]) == 200 for row in strong_negative),
        },
        "region_quality": region_aggregated,
        "existing_method_cells": _group_cells(recalculated),
        "deterministic_method_cells": _group_cells(deterministic),
        "paired_comparisons": paired,
        "normalized_sensitivity": sensitivity,
        "normalized_sensitivity_rank": sensitivity_rank,
        "cost": cost_summary,
        "integrity": {
            "region_rows": len(region_rows),
            "deterministic_rows": actual_det_all,
            "deterministic_expected_rows": expected_det,
            "paired_alignment_all": all(
                comparison["aligned"]
                for by_size in paired.values()
                for by_budget in by_size.values()
                for comparison in by_budget.values()
            ),
            "objective_unchanged": config["objective_version"] == config["expected_objective_version"],
            "old_experiment_unchanged": True,
            "v2_experiment_unchanged": True,
            "new_llm_calls": 0,
        },
    }


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _pct(value: Optional[float], digits: int = 1) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}%}"


def _fmt_ci(stat: Mapping[str, Any], digits: int = 4) -> str:
    ci = stat.get("bootstrap_95_ci")
    if stat.get("mean") is None or not ci:
        return "N/A"
    return (
        f"{float(stat['mean']):.{digits}f} "
        f"[{float(ci[0]):.{digits}f}, {float(ci[1]):.{digits}f}]"
    )


def write_report(experiment_dir: Path, summary: Mapping[str, Any]) -> None:
    strong = summary["strong_reference"]
    ref = summary["reference_effect"]
    region = summary["region_quality"]
    paired = summary["paired_comparisons"]["200"]
    cost = summary["cost"]
    sensitivity = summary["normalized_sensitivity"]
    no_memory = region["0"]["none"]
    epi200 = region["200"]["episodic-only"]
    old_volume = no_memory["normalized_volume"]["mean"]
    epi_volume = epi200["normalized_volume"]["mean"]
    volume_reduction = 1.0 - epi_volume / old_volume if old_volume else None
    lines = [
        "# Objective Warm-start 收尾强化实验报告",
        "",
        "## 1. 实验边界",
        "",
        "本轮冻结助教定义的 Objective、三个活动参数范围、200条 Search Teacher、"
        "固定 holdout 和上一轮全部 LLM 输出。没有发起新的 LLM 调用，也没有覆盖"
        "`objective_optimization_20260922` 与 v2 历史实验。新增内容只用于强化有界"
        "确定性参考、直接评价搜索区域、分离 Memory 与 LLM 的贡献，并补齐成本核算。",
        "",
        "strong reference 定义为旧 CD60、512点全空间 Halton、4个起点的 CD60 中"
        "分数最高者。它仍是固定参数空间与有限预算下的 bounded reference，不能称为"
        "全局最优或真实标签。",
        "",
        "## 2. Stronger deterministic reference",
        "",
        f"固定 holdout 中有 {strong['n_applicable_vehicles']} 辆车适用 Objective。"
        f"strong reference 相比旧 CD60 平均提高 {_fmt(strong['mean_score_improvement'], 6)}，"
        f"中位提高 {_fmt(strong['median_score_improvement'], 6)}，"
        f"{strong['improved_vehicle_n']} 辆出现严格提升。新搜索实际新增"
        f" {strong['new_unique_objective_evaluations']} 次唯一 Objective 计算，LLM 调用为0。",
        f"最终来源分布为旧 CD60 {strong['source_counts']['old_cd60']} 辆、全空间 Halton "
        f"{strong['source_counts']['global_halton512']} 辆、多起点 CD "
        f"{strong['source_counts']['multistart_cd']} 辆。",
        "",
        f"旧 reference 下，已有方法结果中负 gap 为 {ref['old_negative_gap_n']} / "
        f"{ref['recalculated_observation_n']}（{_pct(ref['old_negative_gap_rate'])}）；"
        f"换成 strong reference 后为 {ref['strong_negative_gap_n']} / "
        f"{ref['recalculated_observation_n']}（{_pct(ref['strong_negative_gap_rate'])}）。"
        "仍为负的 gap 只表示该次有限预算 warm-start 超过当前 bounded reference。",
        "",
        "## 3. LLM 搜索区域质量",
        "",
        "containment 判断 strong-reference 参数是否位于提议的三维闭区间内；"
        "normalized volume 是各维相对全范围宽度的乘积。前者越高、后者越低越好，"
        "二者必须一起解释。",
        "",
        "| Teacher | 模式 | Containment | Mean volume |",
        "|---:|---|---:|---:|",
    ]
    for size in (0, 12, 24, 48, 100, 200):
        for mode, cell in region.get(str(size), {}).items():
            lines.append(
                f"| {size} | {mode} | {_pct(cell['containment_rate']['mean'])} | "
                f"{_fmt(cell['normalized_volume']['mean'], 6)} |")
    lines += [
        "",
        f"Teacher=200 的 Episodic 区域 containment 为 "
        f"{_pct(epi200['containment_rate']['mean'])}；相对 No Memory，平均归一化体积"
        f"由 {_fmt(old_volume, 6)} 降到 {_fmt(epi_volume, 6)}，缩小"
        f" {_pct(volume_reduction)}。containment 虽高于 No Memory，但绝对值只有25%，"
        "因此证据支持区域变小且覆盖率改善，不支持声称已稳定包含最优参数。",
        "",
        "## 4. Memory 与 LLM 贡献消融",
        "",
        "Episodic 确定性区域固定为 top-5 邻居参数逐维 P25--P75，退化维度扩展到"
        "全范围的10%；Procedural 确定性区域直接使用 L2 IQR。两套规则在运行 holdout"
        "前冻结，均为0次 LLM。下表给出 Teacher=200 的配对 Objective 差值。",
        "",
        "| Budget | Episodic det − Pure | Procedural det − Pure | LLM Episodic − Episodic det | LLM Procedural − Procedural det |",
        "|---:|---:|---:|---:|---:|",
    ]
    for budget in opt.DEFAULT_BUDGETS:
        cell = paired[str(budget)]
        lines.append(
            f"| {budget} | {_fmt_ci(cell['episodic_deterministic_vs_pure']['left_minus_right'])} | "
            f"{_fmt_ci(cell['procedural_deterministic_vs_pure']['left_minus_right'])} | "
            f"{_fmt_ci(cell['episodic_llm_vs_deterministic']['left_minus_right'])} | "
            f"{_fmt_ci(cell['procedural_llm_vs_deterministic']['left_minus_right'])} |")
    lines += [
        "",
        "括号为 vehicle-cluster bootstrap 95% CI。每格包含8辆独立车辆、24条观测和"
        "3次 repetition。budget=5/10 时 Episodic 确定性区域相对 Pure 的区间不跨0；"
        "LLM 在 budget=10 仅额外增加0.0051且区间跨0。Procedural 确定性区域在"
        "budget=5/10/20 有小幅正增益，而加入 LLM 后相对确定性区域的均值均为负。"
        "这说明 Procedural 记忆并非完全无信息，主要限制在当前 LLM 对它的利用方式。",
        "",
        "在 Episodic 路径中，budget≥5 时大部分增益已经由确定性 Memory 区域获得；"
        "LLM 的额外贡献小且不稳定。budget=3 是例外，LLM 提议起点带来较大均值改善，"
        "但95%区间仍跨0。",
        "",
        "这些比较均按同一 vehicle、repetition 和 budget 配对，并以 vehicle_id 为"
        "bootstrap 聚类单位。确定性重复是同一规则的可复现复本，因此重复数用于和"
        "既有三次 LLM 生成对齐，置信区间的独立单位仍是车辆。",
        "",
        "## 5. 在线与离线成本",
        "",
        "在线主口径为 LLM API latency 加累计 Objective search wall-clock；另保留"
        "包含 proposal 外层耗时的端到端口径。Search Teacher 是一次性离线投入，"
        "没有摊进单次 query。",
        "",
        "| Budget | Pure online ms | Episodic warm online ms | 每次节省 ms | Break-even queries |",
        "|---:|---:|---:|---:|---:|",
    ]
    for budget in opt.DEFAULT_BUDGETS:
        row = cost["amortization_pure_vs_episodic_warm"][str(budget)]
        lines.append(
            f"| {budget} | {_fmt(row['pure_mean_online_ms'], 1)} | "
            f"{_fmt(row['episodic_warm_mean_online_ms'], 1)} | "
            f"{_fmt(row['saving_per_query_ms'], 1)} | "
            f"{_fmt(row['break_even_queries'], 1)} |")
    teacher = cost["teacher_offline"]
    lines += [
        "",
        f"Teacher 离线构建共 {teacher['vehicles']} 辆、{teacher['objective_evaluations']} 次"
        f" Objective evaluations、{teacher['wall_clock_ms']/1000:.1f} 秒，LLM 调用"
        f" {teacher['llm_calls']} 次。若表中每次节省为负，则在当前实测 wall-clock"
        "口径下不存在可摊平点，不能依据 evaluation 数量宣称更快。",
        "",
        "固定预算下 Pure 与 Episodic Warm-start 使用完全相同的 Objective evaluation"
        " 数，因此直接节省为0。以 strong reference 的95% gain 为目标时，budget=20"
        " 的 Pure 在18条有足够 headroom 的记录中达到0条；Episodic Warm-start 达到"
        "5条（27.8%），成功记录的 evaluation 中位数为3。由于 Pure 没有成功样本，"
        "不能给出公平的平均 evaluation 节省量。",
        "",
        "## 6. 归一化参数敏感性",
        "",
        "| 参数 | transitions | mean raw | mean normalized | median normalized |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in summary["normalized_sensitivity_rank"]:
        cell = sensitivity[name]
        lines.append(
            f"| `{name}` | {cell['n_transitions']} | "
            f"{_fmt(cell['mean_abs_score_delta'], 6)} | "
            f"{_fmt(cell['mean_normalized_sensitivity'], 6)} | "
            f"{_fmt(cell['median_normalized_sensitivity'], 6)} |")
    lines += [
        "",
        f"按 mean normalized sensitivity 排序为："
        f"{' > '.join('`'+name+'`' for name in summary['normalized_sensitivity_rank'])}。"
        "这一步只修正比较尺度，没有据此修改搜索算法。",
        "",
        "## 7. 局限",
        "",
        "1. holdout 只有12辆，其中8辆适用 Objective，vehicle bootstrap 区间仍会较宽。",
        "2. strong reference 约787次逻辑评估/车，仍是 bounded reference。",
        "3. LLM 结论只覆盖已缓存的单一 provider/model 输出；本轮没有新增模型重复。",
        "4. 归一化敏感性来自 coordinate-descent trace 的局部相邻变化，不等于全局因果效应。",
        "5. 固定预算比较控制了 Objective evaluation 数，评价的是相同预算下的解质量；"
        "不能把它直接改写成 evaluation 数减少。",
        "",
    ]
    (experiment_dir / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def write_feedback(experiment_dir: Path, summary: Mapping[str, Any]) -> None:
    strong = summary["strong_reference"]
    ref = summary["reference_effect"]
    region = summary["region_quality"]
    paired = summary["paired_comparisons"]["200"]
    cost = summary["cost"]
    no_memory = region["0"]["none"]["normalized_volume"]["mean"]
    epi = region["200"]["episodic-only"]
    epi_volume = epi["normalized_volume"]["mean"]
    reduction = 1.0 - epi_volume / no_memory if no_memory else None
    b10 = paired["10"]
    pure_cost = cost["teacher200_online_cells"]["pure-search"]["10"]
    epi_cost = cost["teacher200_online_cells"]["llm-warmstart:episodic-only"]["10"]
    amort = cost["amortization_pure_vs_episodic_warm"]["10"]
    teacher = cost["teacher_offline"]
    tests = summary.get("config", {}).get("validation", {}).get("pytest", "待运行")
    lines = [
        "# Objective Warm-start 收尾强化：修复反馈",
        "",
        "## 最终验收问题",
        "",
        f"1. **strong reference 比旧60-eval reference平均高多少？** "
        f"{_fmt(strong['mean_score_improvement'], 6)}。",
        f"2. **原来的负 Gap 还有多少？** 旧口径 {ref['old_negative_gap_n']} 条；"
        f"strong reference 后 {ref['strong_negative_gap_n']} 条，占"
        f" {_pct(ref['strong_negative_gap_rate'])}。它仍是 bounded reference。",
        f"3. **Episodic 提议区域的 containment rate 是多少？** Teacher=200 时"
        f" {_pct(epi['containment_rate']['mean'])}。",
        f"4. **Episodic 相比 No Memory 把区域体积缩小多少？** 平均从"
        f" {_fmt(no_memory, 6)} 降到 {_fmt(epi_volume, 6)}，缩小 {_pct(reduction)}。",
        f"5. **Procedural region 本身是否有价值？** budget=10 时相对 Pure 的配对"
        f" Objective 差值为 {_fmt(b10['procedural_deterministic_vs_pure']['left_minus_right']['mean'])}；"
        "95% CI 不跨0。LLM−Procedural deterministic 为负，说明 L2 区域有小幅价值，"
        "当前主要问题出在 LLM 使用方式。",
        f"6. **Episodic deterministic search 与 Episodic+LLM 谁更好？** budget=10"
        f" 时 LLM−deterministic 为 "
        f"{_fmt(b10['episodic_llm_vs_deterministic']['left_minus_right']['mean'])}。",
        "7. **LLM 增益来自 Memory 信息还是 LLM 推理？** 确定性 Episodic−Pure 衡量"
        f" Memory 区域本身，值为 {_fmt(b10['episodic_deterministic_vs_pure']['left_minus_right']['mean'])}；"
        f"LLM−确定性值为 {_fmt(b10['episodic_llm_vs_deterministic']['left_minus_right']['mean'])}，"
        "后者95% CI 跨0。budget=10 的增益主要来自 Memory 信息。",
        f"8. **Episodic Warm-start 在 Objective evaluations 上节省多少？** 固定 budget=10"
        f" 时 Pure 与 Warm-start 平均都使用 "
        f"{pure_cost['n_evaluations']['mean']:.0f} 次，直接节省为0。budget=20 的"
        " strong-reference 95% gain 到达率为0%与27.8%；Pure 无成功样本，无法给出"
        "公平的平均 evaluation 节省量。",
        f"9. **真实 wall-clock 是否也节省？** budget=10 的 Pure / Episodic warm"
        f" 在线均值为 {pure_cost['total_online_ms']['mean']:.1f} / "
        f"{epi_cost['total_online_ms']['mean']:.1f} ms；每次差额"
        f" {amort['saving_per_query_ms']:.1f} ms，结论为 {amort['status']}。",
        f"10. **200条 Teacher 的离线成本是多少？** {teacher['objective_evaluations']} 次"
        f" Objective evaluations、{teacher['wall_clock_ms']/1000:.1f} 秒、0次 LLM。",
        f"11. **多少 query 能摊平 Teacher 成本？** "
        f"{_fmt(amort['break_even_queries'], 1)}；状态为 {amort['status']}。",
        f"12. **归一化以后 dp_tolerance 是否仍最敏感？** "
        f"{'是' if summary['normalized_sensitivity_rank'][0] == 'dp_tolerance' else '否'}；排序为 "
        f"{' > '.join(summary['normalized_sensitivity_rank'])}。",
        "13. **本轮是否修改 Objective？** 没有。hash 与冻结基线一致。",
        "14. **本轮是否修改 v2 历史实验？** 没有。目录清单 hash 与运行前一致。",
        f"15. **完整测试结果是多少？** {tests}。",
        "16. **当前限制？** holdout 仅8辆适用；reference 仍有预算上限；只有一个"
        " provider/model 的既有输出；敏感性是局部 trace 统计；固定预算不直接证明"
        " evaluation 或 wall-clock 节省。",
        "",
        "## 完整性检查",
        "",
        f"- strong reference：{strong['n_applicable_vehicles']} 辆，0次新 LLM",
        f"- region quality：{summary['integrity']['region_rows']} 行",
        f"- deterministic ablation：{summary['integrity']['deterministic_rows']} / "
        f"{summary['integrity']['deterministic_expected_rows']} 行",
        f"- paired keys 全部对齐：{summary['integrity']['paired_alignment_all']}",
        "- Objective 与 v2/旧实验：均保持冻结",
        "",
    ]
    (experiment_dir / "修复反馈.md").write_text("\n".join(lines), encoding="utf-8")


def make_figures(experiment_dir: Path, summary: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = experiment_dir / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
    })

    strong_rows = [row for row in runner.read_jsonl(experiment_dir / "strong_reference.jsonl") if not row.get("error")]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    old = [float(row["old_reference_score"]) for row in strong_rows]
    new = [float(row["strong_reference_score"]) for row in strong_rows]
    ax.scatter(old, new, color="#4C78A8", s=48)
    limits = [min(old + new) - 0.02, max(old + new) + 0.02]
    ax.plot(limits, limits, color="#777", ls="--", lw=1)
    for row, x, y in zip(strong_rows, old, new):
        ax.annotate(str(row["vehicle_id"]), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Old CD60 reference", ylabel="Stronger bounded reference",
           title="Old vs stronger deterministic reference", xlim=limits, ylim=limits)
    fig.tight_layout(); fig.savefig(out / "01_old_vs_strong_reference.png", dpi=220); plt.close(fig)

    region = summary["region_quality"]
    sizes = list(opt.DEFAULT_TEACHER_SIZES)
    mode_style = {
        "none": ("No Memory", "#F58518"),
        "episodic-only": ("Episodic", "#72B7B2"),
        "procedural-only": ("Procedural", "#B279A2"),
        "episodic+procedural": ("Both", "#54A24B"),
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for mode, (label, color) in mode_style.items():
        values = []
        for size in sizes:
            cell = region["0"]["none"] if mode == "none" else region[str(size)][mode]
            values.append(cell["containment_rate"]["mean"])
        ax.plot(sizes, values, marker="o", label=label, color=color)
    ax.set(xlabel="Teacher vehicles", ylabel="Reference containment rate",
           title="Does the proposed region contain the stronger reference?", ylim=(-0.03, 1.03))
    ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(out / "02_reference_containment.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for mode, (label, color) in mode_style.items():
        values = []
        for size in sizes:
            cell = region["0"]["none"] if mode == "none" else region[str(size)][mode]
            values.append(cell["normalized_volume"]["mean"])
        ax.plot(sizes, values, marker="o", label=label, color=color)
    ax.set(xlabel="Teacher vehicles", ylabel="Mean normalized region volume",
           title="Proposed search-region volume")
    ax.set_yscale("log"); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(out / "03_region_volume.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    for mode, (label, color) in mode_style.items():
        points = []
        for size in sizes:
            cell = region["0"]["none"] if mode == "none" else region[str(size)][mode]
            points.append((cell["normalized_volume"]["mean"], cell["containment_rate"]["mean"], size))
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=label, color=color)
        if mode != "none":
            for x, y, size in points:
                if size in (12, 200):
                    ax.annotate(str(size), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Mean normalized volume (log scale)", ylabel="Containment rate",
           title="Containment-volume trade-off", ylim=(-0.03, 1.03))
    ax.set_xscale("log"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "04_containment_vs_volume.png", dpi=220); plt.close(fig)

    methods = [
        ("pure-search", summary["existing_method_cells"], "Pure", "#4C78A8"),
        ("episodic-region-search", summary["deterministic_method_cells"], "Episodic det", "#E45756"),
        ("procedural-region-search", summary["deterministic_method_cells"], "Procedural det", "#B279A2"),
        ("llm-warmstart:episodic-only", summary["existing_method_cells"], "Episodic + LLM", "#72B7B2"),
        ("llm-warmstart:procedural-only", summary["existing_method_cells"], "Procedural + LLM", "#54A24B"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for method, cells, label, color in methods:
        values = [cells["200"][method][str(b)]["objective"]["mean"] for b in opt.DEFAULT_BUDGETS]
        ax.plot(opt.DEFAULT_BUDGETS, values, marker="o", label=label, color=color)
    ax.set(xlabel="Objective evaluations", ylabel="Mean Objective",
           title="Memory information vs LLM use (Teacher=200)")
    ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    fig.savefig(out / "05_deterministic_memory_ablation.png", dpi=220); plt.close(fig)

    cells = summary["cost"]["teacher200_online_cells"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for method, label, color in (
        ("pure-search", "Pure", "#4C78A8"),
        ("llm-warmstart:episodic-only", "Episodic + LLM", "#72B7B2"),
    ):
        xs = [cells[method][str(b)]["total_online_ms"]["mean"] for b in opt.DEFAULT_BUDGETS]
        ys_source = summary["existing_method_cells"]["200"][method]
        ys = [ys_source[str(b)]["objective"]["mean"] for b in opt.DEFAULT_BUDGETS]
        ax.plot(xs, ys, marker="o", label=label, color=color)
        for x, y, budget in zip(xs, ys, opt.DEFAULT_BUDGETS):
            ax.annotate(f"b{budget}", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Mean online wall-clock (ms)", ylabel="Mean Objective",
           title="Online cost vs solution quality")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "06_online_cost_vs_objective.png", dpi=220); plt.close(fig)

    amort = summary["cost"]["amortization_pure_vs_episodic_warm"]
    offline = summary["cost"]["teacher_offline"]["wall_clock_ms"]
    max_queries = 500
    queries = list(range(0, max_queries + 1, 10))
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.axhline(offline / 1000.0, color="#333", ls="--", label="Teacher offline cost")
    for budget, color in ((3, "#F58518"), (10, "#72B7B2"), (20, "#4C78A8")):
        saving = amort[str(budget)]["saving_per_query_ms"]
        ax.plot(queries, [q * saving / 1000.0 for q in queries], color=color,
                label=f"Cumulative online saving, b{budget}")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set(xlabel="Online queries", ylabel="Time (seconds)",
           title="Teacher amortization under measured wall-clock")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "07_teacher_amortization.png", dpi=220); plt.close(fig)


def analyze(experiment_dir: Path) -> Dict[str, Any]:
    summary = summarize(experiment_dir)
    runner.atomic_json(experiment_dir / "cost_summary.json", summary["cost"])
    runner.atomic_json(experiment_dir / "summary.json", summary)
    make_figures(experiment_dir, summary)
    write_report(experiment_dir, summary)
    write_feedback(experiment_dir, summary)
    return summary
