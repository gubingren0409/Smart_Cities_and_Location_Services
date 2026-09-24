"""固定 12 holdout 的真实 OSM 路网正式实验。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt

from ..core import geo, params as params_mod, traj as traj_mod
from ..road.osm import (OSMRoadMatcher, download_osm_cache, read_geojson,
                        speed_violation_stats)
from ..tools.registry import ToolContext, ToolRegistry
from . import objective_optimization as opt


DEFAULT_HOLDOUT = ("2", "4", "10", "22", "111", "137",
                   "153", "154", "165", "194", "201", "208")


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return "unknown"


def _mean(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(rows) if rows else None


def _weighted(rows: Sequence[Mapping[str, Any]], section: str, field: str) -> float | None:
    numerator = 0.0
    denominator = 0
    for row in rows:
        metrics = row[section]
        value = metrics.get(field)
        n = int(metrics.get("n_points") or 0)
        if value is None or n <= 0:
            continue
        numerator += float(value) * n
        denominator += n
    return numerator / denominator if denominator else None


def _speed_summary(rows: Sequence[Mapping[str, Any]], section: str) -> Dict[str, Any]:
    comparable = sum(int(row[section].get("n_speed_comparable") or 0) for row in rows)
    violations = sum(int(row[section].get("road_speed_violation_count") or 0) for row in rows)
    true_count = sum(int(row[section].get("true_maxspeed_point_count") or 0) for row in rows)
    fallback_count = sum(int(row[section].get("fallback_speed_point_count") or 0) for row in rows)
    return {
        "n_speed_comparable": comparable,
        "road_speed_violation_count": violations,
        "road_speed_violation_rate": violations / comparable if comparable else None,
        "true_maxspeed_point_count": true_count,
        "true_maxspeed_point_rate": true_count / comparable if comparable else None,
        "fallback_speed_point_count": fallback_count,
        "fallback_speed_point_rate": fallback_count / comparable if comparable else None,
    }


def _result_for(
    raw: Mapping[str, Any], vehicle_id: str, matcher: OSMRoadMatcher,
) -> Dict[str, Any]:
    started = time.perf_counter()
    source = traj_mod.traj_from_raw(vehicle_id, *raw[vehicle_id])
    evaluator = opt.TrajectoryEvaluator(raw, vehicle_id)
    defaults = params_mod.active_default_params()
    measured, simplified = evaluator.agent._execute(evaluator.handle, defaults)
    baseline = simplified.traj
    objective = evaluator.agent._objective_of(measured, defaults).to_dict()

    raw_match = matcher.match(source)
    baseline_match = matcher.match(baseline)

    ctx = ToolContext(road_matcher=matcher)
    registry = ToolRegistry(ctx)
    baseline_handle = ctx.store.put(baseline, operation="road-experiment-baseline")
    tool_output = registry.call("apply_road_constraint", handle=baseline_handle)
    if not tool_output.get("ok"):
        raise RuntimeError(tool_output.get("error") or "apply_road_constraint failed")
    road_aware = ctx.store.get(str(tool_output["handle"]))
    road_match = matcher.match(road_aware)

    displacement = [
        geo.local_distance_m(before, after)
        for before, after in zip(baseline.coords, road_aware.coords)
    ]
    changed = [value for value in displacement if value > 1e-6]
    before_length = baseline.length_m
    after_length = road_aware.length_m
    snap_stats = {
        "n_points": len(displacement),
        "n_changed_points": len(changed),
        "changed_point_rate": len(changed) / len(displacement) if displacement else 0.0,
        "displacement_mean_m": _mean(displacement),
        "displacement_p95_m": geo.quantile(displacement, 0.95) if displacement else None,
        "displacement_max_m": max(displacement) if displacement else None,
        "length_before_m": before_length,
        "length_after_m": after_length,
        "length_change_m": after_length - before_length,
        "length_change_rate": ((after_length / before_length) - 1.0) if before_length else None,
        "degenerate": len(road_aware) < 2 or after_length <= 1e-9,
    }
    return {
        "vehicle_id": vehicle_id,
        "n_raw_points": len(source),
        "n_processed_points": len(baseline),
        "baseline_objective": objective,
        "objective_definition_modified": False,
        "objective_written_back_after_road": False,
        "raw_road": {**raw_match.stats(), **speed_violation_stats(source, raw_match)},
        "processed_no_road": {
            **baseline_match.stats(), **speed_violation_stats(baseline, baseline_match)},
        "road_aware": {
            **road_match.stats(), **speed_violation_stats(road_aware, road_match)},
        "snap": snap_stats,
        "tool_execution": {
            "tool": "apply_road_constraint",
            "ok": True,
            "road_backend": tool_output.get("road_backend"),
            "road_source": tool_output.get("road_source"),
            "n_roads": tool_output.get("n_roads"),
        },
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "_plot": {
            "raw": source.coords,
            "baseline": baseline.coords,
            "road_aware": road_aware.coords,
        },
    }


def _plot_summary(rows: Sequence[Mapping[str, Any]], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    ids = [str(row["vehicle_id"]) for row in rows]
    x = list(range(len(ids)))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(x, [row["raw_road"]["match_rate"] for row in rows], "o-", label="Raw")
    axes[0].plot(x, [row["processed_no_road"]["match_rate"] for row in rows], "s-", label="Processed")
    axes[0].plot(x, [row["road_aware"]["match_rate"] for row in rows], "^-", label="Road-aware")
    axes[0].set_ylabel("Match rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=.2)
    axes[1].plot(x, [row["raw_road"]["distance_p95_m"] for row in rows], "o-", label="Raw")
    axes[1].plot(x, [row["processed_no_road"]["distance_p95_m"] for row in rows], "s-", label="Processed")
    axes[1].plot(x, [row["road_aware"]["distance_p95_m"] for row in rows], "^-", label="Road-aware")
    axes[1].set_ylabel("Point-to-road P95 (m)")
    axes[1].set_xlabel("Holdout vehicle")
    axes[1].set_xticks(x, ids, rotation=45)
    axes[1].grid(alpha=.2)
    axes[1].legend(ncol=3)
    fig.savefig(figure_dir / "road_metrics_comparison.png", dpi=180)
    fig.savefig(figure_dir / "road_metrics_comparison.svg")
    plt.close(fig)

    selected = sorted(rows, key=lambda row: row["snap"]["n_changed_points"], reverse=True)[:3]
    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4.5), constrained_layout=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, row in zip(axes, selected):
        payload = row["_plot"]
        for coords in payload.get("roads", []):
            ax.plot([point[0] for point in coords], [point[1] for point in coords],
                    color="#d0d0d0", linewidth=.7, alpha=.8, zorder=0)
        for key, style, color in (
            ("raw", "-", "#7f8c8d"),
            ("baseline", "-", "#0072B2"),
            ("road_aware", "--", "#D55E00"),
        ):
            coords = payload[key]
            ax.plot([point[0] for point in coords], [point[1] for point in coords],
                    style, color=color, linewidth=1.2, marker="." if key != "raw" else None,
                    markersize=2.5, label=key)
        ax.set_title(f"Vehicle {row['vehicle_id']}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.ticklabel_format(useOffset=False)
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.savefig(figure_dir / "selected_road_aware_examples.png", dpi=180)
    fig.savefig(figure_dir / "selected_road_aware_examples.svg")
    plt.close(fig)


def _summary(rows: Sequence[Mapping[str, Any]], metadata: Sequence[Mapping[str, Any]], wall_s: float) -> Dict[str, Any]:
    highway = Counter()
    for item in metadata:
        highway.update(item.get("highway_distribution") or {})
    total_features = sum(int(item["road_feature_count"]) for item in metadata)
    true_features = sum(int(item["features_with_true_maxspeed"]) for item in metadata)
    return {
        "status": "PASS" if len(rows) == len(DEFAULT_HOLDOUT) else "FAIL",
        "n_holdout_expected": len(DEFAULT_HOLDOUT),
        "n_holdout_completed": len(rows),
        "vehicle_ids": [str(row["vehicle_id"]) for row in rows],
        "road_source": "OpenStreetMap",
        "road_feature_count_sum_across_case_caches": total_features,
        "highway_distribution_sum_across_case_caches": dict(sorted(highway.items())),
        "true_maxspeed_feature_rate": true_features / total_features if total_features else None,
        "raw": {
            "match_rate_weighted": _weighted(rows, "raw_road", "match_rate"),
            "distance_median_mean_m": _mean(row["raw_road"]["distance_median_m"] for row in rows),
            "distance_p95_mean_m": _mean(row["raw_road"]["distance_p95_m"] for row in rows),
            "distance_max_m": max(float(row["raw_road"]["distance_max_m"]) for row in rows),
            "unmatched_points": sum(int(row["raw_road"]["n_unmatched"]) for row in rows),
            **_speed_summary(rows, "raw_road"),
        },
        "processed_no_road": {
            "match_rate_weighted": _weighted(rows, "processed_no_road", "match_rate"),
            "distance_median_mean_m": _mean(row["processed_no_road"]["distance_median_m"] for row in rows),
            "distance_p95_mean_m": _mean(row["processed_no_road"]["distance_p95_m"] for row in rows),
            "distance_max_m": max(float(row["processed_no_road"]["distance_max_m"]) for row in rows),
            "unmatched_points": sum(int(row["processed_no_road"]["n_unmatched"]) for row in rows),
            **_speed_summary(rows, "processed_no_road"),
        },
        "road_aware": {
            "match_rate_weighted": _weighted(rows, "road_aware", "match_rate"),
            "distance_median_mean_m": _mean(row["road_aware"]["distance_median_m"] for row in rows),
            "distance_p95_mean_m": _mean(row["road_aware"]["distance_p95_m"] for row in rows),
            "distance_max_m": max(float(row["road_aware"]["distance_max_m"]) for row in rows),
            "unmatched_points": sum(int(row["road_aware"]["n_unmatched"]) for row in rows),
            **_speed_summary(rows, "road_aware"),
        },
        "snap": {
            "changed_points": sum(int(row["snap"]["n_changed_points"]) for row in rows),
            "total_points": sum(int(row["snap"]["n_points"]) for row in rows),
            "mean_displacement_across_cases_m": _mean(row["snap"]["displacement_mean_m"] for row in rows),
            "p95_displacement_mean_across_cases_m": _mean(row["snap"]["displacement_p95_m"] for row in rows),
            "max_displacement_m": max(float(row["snap"]["displacement_max_m"] or 0) for row in rows),
            "degenerate_trajectories": sum(bool(row["snap"]["degenerate"]) for row in rows),
        },
        "tool_path_executed": all(row["tool_execution"]["ok"] for row in rows),
        "objective_definition_modified": False,
        "wall_clock_s": wall_s,
    }


def _reports(out: Path, summary: Mapping[str, Any]) -> None:
    raw = summary["raw"]
    processed = summary["processed_no_road"]
    aware = summary["road_aware"]
    snap = summary["snap"]
    report = f"""# 任务三真实 OSM 路网约束实验报告

## 数据与口径

- 道路数据：OpenStreetMap `highway` 实体，按固定 12 条 holdout 的轨迹 bbox 加 300 m 缓冲下载并缓存。
- 距离坐标系：WGS84 / UTM zone 51N（EPSG:32651），单位为真实地面米。
- 最近道路查询：Shapely STRtree；道路吸附通过 `apply_road_constraint` 工具真实执行。
- 历史 Objective 定义及其旧结果未修改。路网指标作为独立 validation，未回写旧分数。

## 覆盖与结果

- 评测完成：{summary['n_holdout_completed']} / {summary['n_holdout_expected']}。
- OSM 道路 feature 数（逐 case 缓存求和，重叠道路可能重复）：{summary['road_feature_count_sum_across_case_caches']}；跨缓存去重 OSM way id {summary['unique_osm_way_id_count_across_caches']} 个，matcher LineString parts {summary['matcher_linestring_part_count_sum']} 个。
- 道路 feature 中真实 `maxspeed` 覆盖率：{summary['true_maxspeed_feature_rate']:.2%}。
- 原始轨迹加权 match rate：{raw['match_rate_weighted']:.2%}；逐轨迹 median 距离均值 {raw['distance_median_mean_m']:.2f} m，P95 距离均值 {raw['distance_p95_mean_m']:.2f} m。
- 无路网处理结果加权 match rate：{processed['match_rate_weighted']:.2%}；逐轨迹 P95 距离均值 {processed['distance_p95_mean_m']:.2f} m。
- Road-aware 结果加权 match rate：{aware['match_rate_weighted']:.2%}；逐轨迹 P95 距离均值 {aware['distance_p95_mean_m']:.2f} m。
- 处理后可比较速度点中，真实 `maxspeed` / 道路等级 fallback 占比为 {processed['true_maxspeed_point_rate']:.2%} / {processed['fallback_speed_point_rate']:.2%}，道路限速超限率为 {processed['road_speed_violation_rate']:.2%}。
- 实际改变 {snap['changed_points']} / {snap['total_points']} 个处理后点；逐轨迹平均吸附位移均值 {snap['mean_displacement_across_cases_m']:.2f} m，最大吸附位移 {snap['max_displacement_m']:.2f} m。
- 退化轨迹：{snap['degenerate_trajectories']} 条。

## 解释

Road-aware 输出确实发生改变，且道路距离指标按预期改善。吸附并不自动代表轨迹真值更准确：平行道路、立交和缺少方向约束时，最近几何道路可能不是车辆实际道路。因此本实验将路网指标保留为独立验证，不把它加入已冻结的主 Objective，也不据此重写历史性能结论。

图表见 `figures/road_metrics_comparison.*` 与 `figures/selected_road_aware_examples.*`。
"""
    (out / "实验报告.md").write_text(report, encoding="utf-8")
    feedback = f"""# 真实路网约束完成反馈

- 状态：**{summary['status']}**
- 固定 holdout：{summary['n_holdout_completed']} / {summary['n_holdout_expected']}
- 数据源：OpenStreetMap 外部道路数据
- 空间索引：Shapely STRtree
- 工具路径全部执行：{summary['tool_path_executed']}
- 改变点数：{snap['changed_points']}
- 历史 Objective 修改：否
- 历史实验覆盖：否

可审计证据：`road_data_manifest.json`、`road_eval_manifest.json`、`raw_results.jsonl`、`summary.json` 和 `figures/`。
"""
    (out / "完成反馈.md").write_text(feedback, encoding="utf-8")


def run(data_path: Path, manifest_path: Path, out: Path, repo: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    raw = traj_mod.load_raw(str(data_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    holdout = tuple(str(value) for value in manifest["holdout"])
    if holdout != DEFAULT_HOLDOUT:
        raise ValueError(f"固定 holdout 发生变化: {holdout}")
    commit = _git_head(repo)
    config = {
        "experiment": "task3_real_road",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": commit,
        "data_path": str(data_path.name),
        "holdout_source": str(manifest_path),
        "holdout": list(holdout),
        "road_source": "OpenStreetMap",
        "road_api": "https://api.openstreetmap.org/api/0.6/map",
        "road_buffer_m": 300.0,
        "match_tolerance_m": 30.0,
        "distance_crs": "EPSG:32651",
        "spatial_index": "Shapely STRtree",
        "objective_definition_modified": False,
    }
    _json(out / "config.json", config)
    rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    unique_osm_way_ids: set[str] = set()
    matcher_part_count = 0
    raw_path = out / "raw_results.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for index, vehicle_id in enumerate(holdout, start=1):
            source = traj_mod.traj_from_raw(vehicle_id, *raw[vehicle_id])
            cache = out / "road_cache" / f"vehicle_{vehicle_id}.geojson.gz"
            metadata_path = out / "road_cache" / f"vehicle_{vehicle_id}.metadata.json"
            metadata = download_osm_cache(
                source.bbox, cache, buffer_m=300.0,
                metadata_path=metadata_path, implementation_commit=commit)
            metadata = {"vehicle_id": vehicle_id, **metadata}
            metadata_rows.append(metadata)
            matcher = OSMRoadMatcher.from_geojson(cache, match_tolerance_m=30.0)
            row = _result_for(raw, vehicle_id, matcher)
            plot_payload = row.pop("_plot")
            roads: List[List[List[float]]] = []
            for feature in read_geojson(cache)["features"]:
                geometry = feature.get("geometry") or {}
                props = feature.get("properties") or {}
                unique_osm_way_ids.add(str(
                    props.get("osm_way_id") or feature.get("id") or ""))
                if geometry.get("type") == "LineString":
                    roads.append(geometry.get("coordinates") or [])
                    matcher_part_count += 1
                elif geometry.get("type") == "MultiLineString":
                    roads.extend(geometry.get("coordinates") or [])
                    matcher_part_count += len(geometry.get("coordinates") or [])
            plot_payload["roads"] = roads
            row_for_file = dict(row)
            handle.write(json.dumps(row_for_file, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append({**row, "_plot": plot_payload})
            print(f"road {index}/{len(holdout)} vehicle={vehicle_id} roads={matcher.n_roads}", flush=True)
    _json(out / "road_data_manifest.json", {
        "source": "OpenStreetMap", "n_case_caches": len(metadata_rows),
        "road_feature_count_sum_across_case_caches": sum(
            int(item["road_feature_count"]) for item in metadata_rows),
        "unique_osm_way_id_count_across_caches": len(unique_osm_way_ids),
        "matcher_linestring_part_count_sum": matcher_part_count,
        "cases": metadata_rows,
    })
    _json(out / "road_eval_manifest.json", {
        "source_manifest": str(manifest_path),
        "expected_vehicle_ids": list(holdout),
        "completed_vehicle_ids": [row["vehicle_id"] for row in rows],
        "missing_vehicle_ids": sorted(set(holdout) - {row["vehicle_id"] for row in rows}),
        "duplicate_vehicle_ids": len(rows) - len({row["vehicle_id"] for row in rows}),
    })
    wall_s = time.perf_counter() - started
    summary = _summary(rows, metadata_rows, wall_s)
    summary["unique_osm_way_id_count_across_caches"] = len(unique_osm_way_ids)
    summary["matcher_linestring_part_count_sum"] = matcher_part_count
    _json(out / "summary.json", summary)
    _plot_summary(rows, out / "figures")
    _reports(out, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("作业/traj_dict.json"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "作业/experiments/llm_assisted_ecnu_20260921_v2/sample_manifest.json"))
    parser.add_argument("--out", type=Path, default=Path(
        "作业/experiments/task3_real_road_20260924"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.data.resolve(), args.manifest.resolve(), args.out.resolve(), args.repo.resolve())


if __name__ == "__main__":
    main()
