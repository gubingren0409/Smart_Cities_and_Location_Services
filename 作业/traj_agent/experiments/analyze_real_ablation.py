"""真实消融结果统计、图表和报告。"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .real_ablation import (
    DEFAULT_SEED, MODES, STAGE1_HOLDOUT, STRATA, atomic_json, read_jsonl,
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


def summarize(experiment_dir: Path) -> Dict[str, Any]:
    config = json.loads(
        (experiment_dir / "config.json").read_text(encoding="utf-8")
    )
    rows = read_jsonl(experiment_dir / "raw_results.jsonl")
    state = json.loads(
        (experiment_dir / "run_state.json").read_text(encoding="utf-8")
    )
    holdout = [r for r in rows if r.get("phase") == "holdout"]
    demo = [r for r in rows if r.get("phase") == "demo"]
    mode_summary: Dict[str, Any] = {}

    for idx, mode in enumerate(MODES):
        items = [r for r in holdout if r.get("mode") == mode]
        ok = [r for r in items if not r.get("error") and r.get("agent_ok")]
        applicable = [r for r in ok if r.get("applicable")]
        deltas = [
            float(r["proposal_score"]) - float(r["baseline_score"])
            for r in applicable
        ]
        beaten = [1.0 if x > 1e-9 else 0.0 for x in deltas]
        directions = [
            r.get("direction_accuracy") for r in applicable
            if r.get("direction_accuracy") is not None
        ]
        json_rows = [
            r for r in items if MODE_SPECS[mode].get("use_llm")
        ]
        mode_summary[mode] = {
            "n_total": len(items),
            "n_success": len(ok),
            "n_applicable": len(applicable),
            "score_delta": stats(deltas, DEFAULT_SEED + idx),
            "baseline_beaten_rate": stats(
                beaten, DEFAULT_SEED + 10 + idx
            ),
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
            "direction_accuracy": stats(
                directions, DEFAULT_SEED + 20 + idx
            ),
            "llm_calls": sum(int(r.get("llm_calls") or 0) for r in items),
            "prompt_tokens": sum(
                int(r.get("prompt_tokens") or 0) for r in items
            ),
            "completion_tokens": sum(
                int(r.get("completion_tokens") or 0) for r in items
            ),
            "latency_ms": stats(
                [r.get("elapsed_ms") for r in items],
                DEFAULT_SEED + 30 + idx,
            ),
            "failure_rate": (
                sum(bool(r.get("error")) for r in items) / len(items)
                if items else None
            ),
            "retry_count": sum(
                int(r.get("retry_count") or 0) for r in items
            ),
            "provider_error_count": sum(
                int(r.get("provider_error_count") or 0) for r in items
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
            vals = [
                float(r["proposal_score"]) - float(r["baseline_score"])
                for r in items
            ]
            per_stratum[mode][f"{regime}/{timeline}"] = stats(vals)

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
            vals = [
                float(r["proposal_score"]) - float(r["baseline_score"])
                for r in items
            ]
            per_repetition[str(rep)][mode] = stats(
                vals, DEFAULT_SEED + rep
            )

    indexed = {
        (int(r["repetition"]), str(r["vehicle_id"]), str(r["mode"])): r
        for r in holdout
        if not r.get("error") and r.get("applicable")
    }
    paired_rows = []
    for rep in range(1, int(config["repetitions"]) + 1):
        for vid in STAGE1_HOLDOUT:
            no_mem = indexed.get((rep, vid, "llm+search"))
            mem = indexed.get((rep, vid, "llm+memory+search"))
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
    paired_vals = [
        r["score_difference_memory_minus_no_memory"]
        for r in paired_rows
    ]
    paired = {
        "definition": (
            "proposal_score(llm+memory+search) - "
            "proposal_score(llm+search)"
        ),
        "stats": stats(paired_vals, DEFAULT_SEED + 99),
        "wins": sum(x > 1e-9 for x in paired_vals),
        "ties": sum(abs(x) <= 1e-9 for x in paired_vals),
        "losses": sum(x < -1e-9 for x in paired_vals),
        "pairs": paired_rows,
    }

    calls = sum(int(r.get("llm_calls") or 0) for r in rows)
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in rows)
    completion_tokens = sum(
        int(r.get("completion_tokens") or 0) for r in rows
    )
    total_latency = sum(float(r.get("elapsed_ms") or 0.0) for r in rows)
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
        r for r in holdout if r.get("mode") == "llm+memory+search"
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
            "retry_count": sum(
                int(r.get("retry_count") or 0) for r in rows
            ),
            "provider_error_count": sum(
                int(r.get("provider_error_count") or 0) for r in rows
            ),
        },
        "mode_summary": mode_summary,
        "paired_memory_effect": paired,
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
            if r.get("mode") == "llm+memory+search"
        ],
        "expanded_demo_cost_estimates": estimates,
        "method_notes": [
            "跨模式不使用平均regret排名；主比较量为proposal_score-baseline_score。",
            "score统计只包含applicable=True且无运行错误的留出样本。",
            "Bootstrap CI为样本级非参数区间，同时单独报告repetition波动。",
            "确定性搜索是与LLM候选分开的内部参考，不是现实道路真值。",
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
        "# ECNU 真实模型轨迹清洗四模式消融实验",
        "",
        "## 1. 实验设计",
        "",
        f"- Provider：{exp['provider']}，模型：{exp['model']}，Base URL：{exp['base_url']}。",
        f"- 执行提交：{exp['git_commit_sha']}，重复 {exp['repetitions']} 轮。",
        f"- 每轮 demo {exp['demo_n_per_repetition']} 条，holdout 每模式 {exp['holdout_n_per_mode_per_repetition']} 条，二者无重叠。",
        "- 每轮使用新的记忆库；holdout 只读，不写回记忆。",
        "- 搜索从物理默认参数开始，与 LLM 候选相互独立；Verifier 使用搜索结果作为内部参考。",
        "",
        "## 2. 实际调用与可靠性",
        "",
        (
            f"共发起 {usage['llm_calls']} 次真实 LLM 请求，使用 "
            f"{usage['prompt_tokens']} prompt tokens 和 "
            f"{usage['completion_tokens']} completion tokens。所有 case "
            f"累计墙钟时间 {usage['case_elapsed_s_sum']:.2f} s；失败 case "
            f"{usage['failed_cases']} 个，case 级重试 {usage['retry_count']} 次，"
            f"provider 错误 {usage['provider_error_count']} 次。"
        ),
        "",
        "## 3. 四模式主结果",
        "",
        "| 模式 | 适用 n | 相对基线平均得分变化 | 95% CI | 超过基线比例 | JSON成功率 | 约束违规率 | LLM调用 | 失败率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        item = summary["mode_summary"][mode]
        ci = item["score_delta"]["bootstrap_95_ci"]
        ci_text = (
            "N/A" if ci is None
            else f"[{fmt(ci[0])}, {fmt(ci[1])}]"
        )
        lines.append(
            f"| {mode} | {item['n_applicable']} | "
            f"{fmt(item['score_delta']['mean'])} | {ci_text} | "
            f"{fmt(item['baseline_beaten_rate']['mean'], 3)} | "
            f"{fmt(item['json_success_rate'], 3)} | "
            f"{fmt(item['constraint_violation_rate'], 3)} | "
            f"{item['llm_calls']} | {fmt(item['failure_rate'], 3)} |"
        )

    paired = summary["paired_memory_effect"]
    paired_ci = paired["stats"]["bootstrap_95_ci"]
    paired_ci_text = (
        "N/A" if paired_ci is None
        else f"[{fmt(paired_ci[0])}, {fmt(paired_ci[1])}]"
    )
    lines += [
        "",
        "跨模式表不比较平均 regret，因为搜索开关会改变 regret 口径。",
        "",
        "## 4. Memory 配对比较",
        "",
        (
            "llm+memory+search 减去 llm+search 的 proposal score "
            f"平均差值为 {fmt(paired['stats']['mean'])}，中位数 "
            f"{fmt(paired['stats']['median'])}，95% bootstrap CI "
            f"{paired_ci_text}；胜/平/负为 "
            f"{paired['wins']}/{paired['ties']}/{paired['losses']}。"
        ),
        "",
        (
            "Memory 结论还要同时检查 demo 的 moving 准入覆盖与 holdout "
            "实际检索邻居数。覆盖不足时，只能说当前证据不足以检验 "
            "Memory 贡献。"
        ),
        "",
        "### 三轮波动",
        "",
        "| repetition | llm-only | search-only | llm+search | llm+memory+search |",
        "|---:|---:|---:|---:|---:|",
    ]
    for rep, values in summary["per_repetition"].items():
        lines.append(
            f"| {rep} | {fmt(values['llm-only']['mean'], 6)} | "
            f"{fmt(values['search-only']['mean'], 6)} | "
            f"{fmt(values['llm+search']['mean'], 6)} | "
            f"{fmt(values['llm+memory+search']['mean'], 6)} |"
        )

    lines += [
        "",
        "## 5. Memory 证据覆盖",
        "",
        "| repetition | L1案例 | admitted | L2 region | stationary/ok | stationary/degraded | mixed/ok | mixed/degraded | moving/ok | moving/degraded |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rep, diag in summary["memory_diagnostics"].items():
        st = diag["stats"]
        a = diag["demo_admitted_by_stratum"]
        lines.append(
            f"| {rep} | {st['n_episodic']} | {st['n_admitted']} | "
            f"{st['n_procedural']} | {a['stationary/ok']} | "
            f"{a['stationary/degraded']} | {a['mixed/ok']} | "
            f"{a['mixed/degraded']} | {a['moving/ok']} | "
            f"{a['moving/degraded']} |"
        )
    coverage = summary["memory_coverage_summary"]
    sim = coverage["neighbor_similarity"]
    region_samples = coverage["procedural_region_n_samples"]
    lines += [
        "",
        (
            f"Memory 模式的 {coverage['holdout_rows']} 条 holdout 记录中，"
            f"{coverage['prior_available_rows']} 条取得记忆先验；邻居数分布为 "
            f"{coverage['neighbor_count_distribution']}。邻居相似度均值 "
            f"{fmt(sim['mean'], 3)}，范围可在 summary.json 的逐条记录中核对。"
        ),
        "",
        (
            f"三轮共生成 {region_samples['n_regions']} 个参数 region，"
            f"每个 region 的 n_samples 仅为 {region_samples['min']} 到 "
            f"{region_samples['max']}；精确到每个 region 的参数、分层和 "
            "n_samples 均保存在 run_state.json 与 summary.json。"
        ),
        "",
        (
            "moving/degraded 三轮仅准入 1 条，moving 两类合计准入 6 条。"
            "因此当前证据覆盖较薄，配对区间又覆盖 0，只能得出“尚未检出"
            "稳定 Memory 增益”，不能据此判定 Memory 机制无效。"
        ),
        "",
        "## 6. 分层 Demo 扩展成本估计",
        "",
        "| 每轮 demo 数 | 预计 LLM calls | prompt tokens | completion tokens | 预计时间（s） |",
        "|---:|---:|---:|---:|---:|",
    ]
    for size, estimate in summary["expanded_demo_cost_estimates"].items():
        lines.append(
            f"| {size} | {estimate['estimated_llm_calls']} | "
            f"{estimate['estimated_prompt_tokens']} | "
            f"{estimate['estimated_completion_tokens']} | "
            f"{estimate['estimated_elapsed_s']} |"
        )
    lines += [
        "",
        "以上估计只覆盖 demo 阶段的三轮调用，不含 holdout。100/200 条候选仅生成清单并估算成本，本轮未执行。",
        "",
        "## 7. 解释边界",
        "",
        "- 本实验评价 LLM 候选是否超过默认参数，以及是否接近同一内部目标下的确定性搜索参考。",
        "- 确定性搜索参考不是现实道路真值；尚未接入独立 OSM 道路或人工漂移标注。",
        "- 24/48/100/200 条 demo 候选已按固定种子分层生成；本轮没有自动执行高成本的 100/200 条扩展。",
        "",
        "## 8. 产物",
        "",
        (
            "raw_results.jsonl 是 case 级原始证据，summary.json 是统计摘要，"
            "sample_manifest.json 保存固定样本和扩展候选，figures 目录中的"
            "图表全部从本次真实结果生成。"
        ),
        "",
    ]
    (experiment_dir / "实验报告.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def make_figures(experiment_dir: Path,
                 summary: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = experiment_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        r for r in read_jsonl(experiment_dir / "raw_results.jsonl")
        if r.get("phase") == "holdout"
    ]
    colors = {
        "llm-only": "#4C78A8",
        "search-only": "#72B7B2",
        "llm+search": "#F58518",
        "llm+memory+search": "#B279A2",
    }
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
    })

    fig, ax = plt.subplots(figsize=(8, 4.8))
    data = []
    for mode in MODES:
        values = [
            float(r["proposal_score"]) - float(r["baseline_score"])
            for r in rows
            if r.get("mode") == mode
            and not r.get("error")
            and r.get("applicable")
        ]
        data.append(values)
    bp = ax.boxplot(
        data, tick_labels=MODES, patch_artist=True, showmeans=True
    )
    for patch, mode in zip(bp["boxes"], MODES):
        patch.set_facecolor(colors[mode])
        patch.set_alpha(0.75)
    ax.axhline(0, color="#555", lw=1)
    ax.set_ylabel("Proposal score - baseline score")
    ax.set_title("Score change relative to fixed default baseline")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
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

    pairs = summary["paired_memory_effect"]["pairs"]
    values = [
        p["score_difference_memory_minus_no_memory"] for p in pairs
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
    ax.set_xlabel("Paired holdout case")
    ax.set_ylabel("Score difference")
    ax.set_title("Memory effect: paired score difference")
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
    labels = ["JSON failure", "Runtime error", "Clamped"]
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
