"""端到端演示：一次跑出全部报告产物。

用法：
    MPLCONFIGDIR=./.mplcache python3 examples/run_demo.py

产出（默认写到 ./figures 与 ./demo_out）：
    - 六类图
    - 单条轨迹完整闭环的 trace
    - 消融实验表
    - 分层抽样的 regime / timeline 分布
    - 汇总 JSON，便于写进报告
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".mplcache"))

from traj_agent.agent.loop import TrajCleaningAgent
from traj_agent.agent import provider as prov
from traj_agent.core import (clean, diagnosis, params as params_mod, segment,
                             simplify, traj as tm)
from traj_agent.memory.store import MemoryStore
from traj_agent.report import figures
from traj_agent.tools.playbook import vault_status
from traj_agent.tools.registry import ToolRegistry, attach_dataset
from traj_agent.verifier import ablation as ablation_mod
from traj_agent.verifier.verify import ablation_summary

DEMO_VEHICLES = ["246", "256", "306", "209"]

# 按 (regime, timeline) 分层挑选，demo 与 holdout **完全不重叠**。
# 为什么必须分两组：若在评测轨迹上边跑边攒记忆，
# agent 会把「这条轨迹自己的上一次结果」检索回来当先验——属于信息泄漏，
# 会让消融表得出完全错误的结论。
ABLATION_DEMO = ["0", "1", "3", "7", "18", "62", "68", "129", "101", "149", "170", "187"]
ABLATION_HOLDOUT = ["2", "4", "10", "22", "111", "137", "153", "154", "165", "194", "201", "208"]
ALL_MODES = list(ablation_mod.MODE_SPECS)


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="traj_dict.json")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--vehicles", nargs="*", default=DEMO_VEHICLES)
    ap.add_argument("--sample", type=int, default=200,
                    help="分层抽样规模，用于数据集统计")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.figures, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    t_start = time.perf_counter()
    summary = {}

    section("0. 环境")
    print("provider 配置：", json.dumps(prov.provider_status(), ensure_ascii=False))
    print("Obsidian vault：", json.dumps(vault_status(), ensure_ascii=False))
    raw = tm.load_raw(args.data)
    print(f"数据集：{len(raw)} 辆车")

    section("1. 数据集 regime 分布（分层抽样）")
    ids = sorted(raw.keys(), key=lambda k: (len(k), k))[:args.sample]
    cards = [diagnosis.diagnose(tm.traj_from_raw(v, *raw[v])) for v in ids]
    dist = diagnosis.dataset_regime_summary(cards)
    print(json.dumps(dist, ensure_ascii=False, indent=1))
    summary["dataset"] = {"n_vehicles": len(raw), "sample": len(ids), **dist}

    section("2. 单条轨迹完整闭环")
    mem = MemoryStore(os.path.join(args.out, "memory.sqlite"))
    agent = TrajCleaningAgent(llm=prov.MockProvider(), memory=mem,
                              regret_threshold=0.05).attach_dataset(raw)
    vid = args.vehicles[0]
    r = agent.run(vid)
    d = r.to_dict(with_trace=False)
    v = d["verification"]
    print(f"车辆 {vid}：ok={r.ok} 来源={r.proposal_source}")
    print(f"诊断 regime={d['regime']} timeline={d['timeline_quality']}")
    print(f"提议参数：{json.dumps(d['proposal_params'], ensure_ascii=False)}")
    print(f"依据：{r.proposal_raw.get('rationale', '')}")
    print(f"得分 提议={v['proposal_score']} 基线={v['baseline_score']} 最优={v['best_score']}")
    print(f"regret={v['regret']:.4f}（{v['regret_basis']}） 准入={v['admitted']}")
    print(f"理由：{v['admit_reason']}")
    print(f"实测：{json.dumps({k: x for k, x in d['measured'].items() if k != 'clean_report'}, ensure_ascii=False)}")
    print(f"工具调用：LLM 发起 {d['llm_tool_calls']} 次，总计 {d['tool_calls']} 次，"
          f"LLM 轮次 {d['llm_turns']}")
    print("\ntrace：")
    for s in r.trace:
        print(f"  [{s.index:2d}] {s.kind:6s} {s.name:22s} "
              f"{json.dumps(s.detail, ensure_ascii=False)[:96]}")
    summary["single_case"] = d

    section("3. 出图")
    t_raw = tm.traj_from_raw(vid, *raw[vid])
    t_clean, clean_report = clean.denoise_trajectory(t_raw)
    tol = d["proposal_params"].get("dp_tolerance", 5.0)
    t_simp = simplify.simplify_trajectory(t_clean, tol).traj
    print("清洗账本：", json.dumps(clean_report.to_dict(), ensure_ascii=False))

    reg = ToolRegistry()
    attach_dataset(reg.ctx, raw)
    h = reg.call("load_trajectory", vehicle_id=vid)["handle"]
    hc = reg.call("clean_trajectory", handle=h)["handle"]
    curve = reg.call("run_search", handle=hc, param="dp_tolerance", n=8,
                     reference_handle=hc)["curve"]
    knee = reg.call("find_knee", curve_json=json.dumps(curve))["knee"]

    figs = {
        "overlay": figures.plot_clean_overlay(t_simp, reference=t_raw,
                                             out_dir=args.figures,
                                             filename="report_clean_overlay.png"),
        "anomaly": figures.plot_anomaly_scatter(t_raw, out_dir=args.figures,
                                                filename="report_anomaly_scatter.png"),
        "heatmap": figures.plot_heatmap(t_simp, reference=t_raw,
                                        out_dir=args.figures,
                                        filename="report_heatmap.png"),
        "curve": figures.plot_quality_compression(curve, knee, out_dir=args.figures,
                                                  filename="report_quality_compression.png"),
        "sensitivity": figures.plot_sensitivity_heatmap(
            segment.sweep_split_thresholds(t_clean, [15, 30, 60, 120],
                                           [100, 200, 400, 800, 1600]),
            out_dir=args.figures, filename="report_sensitivity.png"),
    }
    for k, p in figs.items():
        print(f"  {k:12s} -> {p}  ({os.path.getsize(p) / 1024:.1f} KB)")
    summary["figures"] = figs
    summary["knee"] = knee
    summary["curve"] = curve

    section("4. 异常原因直方图（跨抽样）")
    from traj_agent.core import anomalies
    hist: dict = {}
    for c in cards:
        for k, n in c.anomaly_counts.items():
            hist[k] = hist.get(k, 0) + n
    hist = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
    print(f"{'原因':28s} {'总次数':>8} {'中文':>20}")
    for k, n in hist.items():
        print(f"{k:28s} {n:8d} {anomalies.REASON_ZH.get(k, '-'):>20}")
    summary["anomaly_histogram"] = hist

    section("5. 消融实验（留出集评测）")
    if args.skip_ablation:
        print("（已跳过）")
    else:
        print(f"demo 集  ({len(ABLATION_DEMO)} 条，用于积累记忆): {ABLATION_DEMO}")
        print(f"holdout  ({len(ABLATION_HOLDOUT)} 条，评测用，与 demo 无重叠): "
              f"{ABLATION_HOLDOUT}")
        abl = ablation_mod.run_ablation(
            raw, ABLATION_DEMO, ABLATION_HOLDOUT,
            memory_path=os.path.join(args.out, "ablation_memory.sqlite"))
        rows = abl.rows
        print("\n记忆积累结果：", json.dumps(abl.memory_stats, ensure_ascii=False))
        print()
        hdr = f"{'模式':22s} {'n':>3} {'平均regret':>11} {'平均分':>9} " \
              f"{'平均LLM调用':>11} {'方向准确':>9} {'准入率':>8}"
        print(hdr)
        print("-" * len(hdr))
        for row in rows:
            print(f"{row['mode']:22s} {row['n']:3d} {str(row['mean_regret']):>11} "
                  f"{str(row['mean_score']):>9} {str(row['mean_evals']):>11} "
                  f"{str(row['mean_direction_accuracy']):>9} {str(row['admit_rate']):>8}")
        print()
        for c in abl.caveats:
            print("  *", c)
        summary["ablation"] = rows
        summary["ablation_cases"] = abl.cases
        summary["ablation_caveats"] = abl.caveats
        summary["ablation_sets"] = {"demo": ABLATION_DEMO, "holdout": ABLATION_HOLDOUT}
        summary["ablation_memory"] = abl.memory_stats
        p = figures.plot_ablation(rows, out_dir=args.figures,
                                  filename="report_ablation.png")
        print(f"\n消融图 -> {p}")
        summary["figures"]["ablation"] = p

    section("6. 记忆状态")
    mem.rebuild_procedural(min_samples=1)
    print(json.dumps(mem.stats(), ensure_ascii=False, indent=1))
    regions = [r.to_dict() for r in mem.query_regions()]
    print("\nL2 参数区间：")
    for rr in regions[:10]:
        print("  ", json.dumps(rr, ensure_ascii=False))
    summary["memory"] = mem.stats()
    summary["regions"] = regions
    mem.close()

    summary["elapsed_s"] = round(time.perf_counter() - t_start, 2)
    out_path = os.path.join(args.out, "summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    section("完成")
    print(f"耗时 {summary['elapsed_s']}s")
    print(f"汇总 -> {out_path}")
    print(f"图件 -> {os.path.abspath(args.figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
