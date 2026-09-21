"""轨迹分段：按车辆 ID、时间间隔、空间跳跃阈值切分。

对应学生必做任务①的「轨迹分段函数」。
分段的两条规则互相独立，一次切分可以同时命中，故记录 trigger 集合。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import geo
from .traj import Traj

# 切分触发原因
TRIGGER_TIME_GAP = "time_gap"
TRIGGER_SPACE_JUMP = "space_jump"
TRIGGER_ILLEGAL_TIME = "illegal_time"


@dataclass
class SegmentResult:
    """分段结果 + 切分点诊断信息。"""

    segments: List[Traj] = field(default_factory=list)
    # 每个切分点：(原始索引, 触发原因集合)
    cut_points: List[Tuple[int, Tuple[str, ...]]] = field(default_factory=list)
    # 被丢弃的碎片（点数不足或长度不足）
    dropped: List[Traj] = field(default_factory=list)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_points_kept(self) -> int:
        return int(sum(len(s) for s in self.segments))

    def trigger_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, triggers in self.cut_points:
            for t in triggers:
                counts[t] = counts.get(t, 0) + 1
        return counts

    def summary(self) -> Dict[str, object]:
        lens = [len(s) for s in self.segments]
        return {
            "n_segments": self.n_segments,
            "n_points_kept": self.n_points_kept,
            "n_points_dropped": int(sum(len(s) for s in self.dropped)),
            "seg_len_min": min(lens) if lens else 0,
            "seg_len_median": geo.median([float(x) for x in lens]) if lens else 0.0,
            "seg_len_max": max(lens) if lens else 0,
            "trigger_counts": self.trigger_counts(),
        }


def split_trajectory(traj: Traj,
                     dt_threshold: float = 30.0,
                     dist_threshold: float = 400.0,
                     min_points: int = 5,
                     min_length_m: float = 65.0) -> SegmentResult:
    """按时间间隔与空间跳跃切分单条轨迹。

    参数
    ----
    dt_threshold : 相邻点时间间隔超过该值（秒）即在此处断开。
    dist_threshold : 相邻点平面距离超过该值（米）即在此处断开。
    min_points : 分段最少点数，不足者进 dropped。
    min_length_m : 分段最短长度（米），不足者进 dropped。

    注意 `min_points` / `min_length_m` 只影响「是否保留」，不影响切分位置，
    这样调这两个参数不会反过来改变切分结构，便于做敏感性实验。
    """
    if dt_threshold < 0 or dist_threshold < 0:
        raise ValueError("阈值不能为负")
    if len(traj) < 2:
        return SegmentResult(segments=[], dropped=[traj] if len(traj) else [],
                             cut_points=[])

    ts = [int(t) for t in traj.timestamps]
    coords = list(traj.coords)

    cuts: List[int] = []          # 切分位置：新段从该索引开始
    cut_points: List[Tuple[int, Tuple[str, ...]]] = []

    for i in range(1, len(ts)):
        triggers: List[str] = []
        dt = ts[i] - ts[i - 1]
        if dt < 0:
            triggers.append(TRIGGER_ILLEGAL_TIME)
        elif dt > dt_threshold:
            triggers.append(TRIGGER_TIME_GAP)

        # 空间跳跃：任一坐标非法时不判断（避免 NaN 传播），
        # 这类点由 anomalies 的 illegal_coord 规则负责。
        if geo.is_legal_lonlat(*coords[i]) and geo.is_legal_lonlat(*coords[i - 1]):
            if geo.local_distance_m(coords[i - 1], coords[i]) > dist_threshold:
                triggers.append(TRIGGER_SPACE_JUMP)

        if triggers:
            cuts.append(i)
            cut_points.append((i, tuple(triggers)))

    bounds = [0] + cuts + [len(coords)]
    result = SegmentResult(cut_points=cut_points)
    for seg_i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        sub = traj.positions(range(start, end))
        sub.segment_index = seg_i
        if len(sub) < max(2, int(min_points)) or sub.length_m < min_length_m:
            result.dropped.append(sub)
        else:
            result.segments.append(sub)
    return result


def split_all(raw_trajs: Sequence[Traj],
              dt_threshold: float = 30.0,
              dist_threshold: float = 400.0,
              min_points: int = 5,
              min_length_m: float = 65.0) -> Dict[str, List[Traj]]:
    """批量切分，返回 {vehicle_id: [segments]}。"""
    out: Dict[str, List[Traj]] = {}
    for traj in raw_trajs:
        res = split_trajectory(traj, dt_threshold, dist_threshold, min_points, min_length_m)
        out[traj.vehicle_id] = res.segments
    return out


def sweep_split_thresholds(traj: Traj,
                           dt_values: Sequence[float],
                           dist_values: Sequence[float],
                           min_points: int = 5,
                           min_length_m: float = 65.0) -> List[Dict[str, float]]:
    """二维阈值扫描，供任务②的敏感性实验与任务③的目标函数地形图复用。

    返回每个 (dt, dist) 组合下的分段数与保留点数。
    """
    rows: List[Dict[str, float]] = []
    for dt in dt_values:
        for dist in dist_values:
            res = split_trajectory(traj, dt, dist, min_points, min_length_m)
            rows.append({
                "dt_threshold": float(dt),
                "dist_threshold": float(dist),
                "n_segments": float(res.n_segments),
                "n_points_kept": float(res.n_points_kept),
                "n_points_dropped": float(sum(len(s) for s in res.dropped)),
            })
    return rows
