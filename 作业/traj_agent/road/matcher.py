"""路网匹配与道路约束接口。

对应任务①的新需求「加入地图匹配或道路约束」。
本轮只提供**协议 + 可运行的离线实现**，真实路网数据由后续工作接入。

设计要点：路网能力对上层是**可选**的。
    未接入时 road_match_rate 为 None，诊断卡与评分函数都能正常工作；
    接入后只影响评分里的路网项，不改变其它逻辑。
这样任务③的智能体与核验器可以先在无路网数据的情况下完整跑通。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from ..core import geo
from ..core.traj import Traj


@dataclass
class MatchResult:
    """一条轨迹的匹配结果。"""

    matched_coords: List[geo.LonLat] = field(default_factory=list)
    # 每个点到最近路段的距离（米）
    distances_m: List[float] = field(default_factory=list)
    matched: List[bool] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return len(self.distances_m)

    @property
    def match_rate(self) -> float:
        if not self.distances_m:
            return 0.0
        return sum(1 for m in self.matched if m) / len(self.matched)

    def stats(self) -> Dict[str, float]:
        if not self.distances_m:
            return {"n_points": 0, "match_rate": 0.0}
        return {
            "n_points": self.n_points,
            "match_rate": round(self.match_rate, 4),
            "distance_median_m": round(geo.median(self.distances_m), 2),
            "distance_p95_m": round(geo.quantile(self.distances_m, 0.95), 2),
            "distance_max_m": round(max(self.distances_m), 2),
        }


@runtime_checkable
class RoadMatcher(Protocol):
    """路网匹配器协议。真实实现应接入 OSM / 高德等路网数据。"""

    def match(self, traj: Traj) -> MatchResult:
        ...

    def snap(self, point: geo.LonLat) -> geo.LonLat:
        ...

    def distance_to_road_m(self, point: geo.LonLat) -> float:
        ...


class NoRoadMatcher:
    """空实现：不做任何匹配，road_match_rate 返回 None。

    这是默认实现，使整条流水线在无路网数据时也能完整运行。
    """

    def match(self, traj: Traj) -> MatchResult:
        return MatchResult(
            matched_coords=list(traj.coords),
            distances_m=[],
            matched=[],
        )

    def snap(self, point: geo.LonLat) -> geo.LonLat:
        return point

    def distance_to_road_m(self, point: geo.LonLat) -> float:
        return float("nan")

    @property
    def available(self) -> bool:
        return False


@dataclass
class SegmentRoad:
    """一条路段：由若干折线点构成的连通折线。"""

    road_id: str
    points: List[geo.LonLat]
    highway: str = "unclassified"          # 道路等级，如 motorway/primary/residential
    speed_limit_mps: Optional[float] = None

    def distance_m(self, point: geo.LonLat) -> Tuple[float, int]:
        """点到本路段的最短距离（米）及所在子段索引。"""
        if len(self.points) < 2:
            if not self.points:
                return float("inf"), -1
            return geo.local_distance_m(point, self.points[0]), 0
        best = float("inf")
        best_k = -1
        for k in range(len(self.points) - 1):
            d = geo.segment_distance_m(point, self.points[k], self.points[k + 1])
            if d < best:
                best = d
                best_k = k
        return best, best_k

    def project(self, point: geo.LonLat) -> geo.LonLat:
        """把点投影到本路段上（吸附）。"""
        d, k = self.distance_m(point)
        if k < 0:
            return point
        ax, ay = geo.local_xy(self.points[k])
        bx, by = geo.local_xy(self.points[k + 1])
        px, py = geo.local_xy(point)
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return self.points[k]
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
        return geo.unproject([(ax + t * dx, ay + t * dy)])[0]


# 道路等级 → 限速（m/s），用于速度阈值的路网约束
HIGHWAY_SPEED_LIMIT_MPS: Dict[str, float] = {
    "motorway": 33.3,        # 120 km/h
    "trunk": 27.8,           # 100 km/h
    "primary": 22.2,         # 80 km/h
    "secondary": 16.7,       # 60 km/h
    "tertiary": 13.9,        # 50 km/h
    "residential": 8.3,      # 30 km/h
    "service": 5.6,          # 20 km/h
    "unclassified": 13.9,
}


class PolylineRoadMatcher:
    """基于折线路段的离线匹配器。

    没有真实路网时，可以用**轨迹自身的几何**构造一个近似的「路网」：
    对高频采样的城市数据，把主要通行路径当作道路约束是合理的近似。
    这使「仅几何规则 vs 路网约束」的对照实验在无外部数据时也能做。

    真实接入方式：实现同一个 RoadMatcher 协议，把 segments 换成
    从 OSM 等地读取的路段即可，上层代码无需改动。
    """

    def __init__(self, roads: Sequence[SegmentRoad],
                 match_tolerance_m: float = 30.0,
                 speed_bounds: bool = True):
        self.roads = list(roads)
        self.match_tolerance_m = float(match_tolerance_m)
        self.speed_bounds = speed_bounds

    @property
    def available(self) -> bool:
        return len(self.roads) > 0

    def nearest(self, point: geo.LonLat) -> Tuple[Optional[SegmentRoad], float, int]:
        best_road = None
        best_d = float("inf")
        best_k = -1
        for road in self.roads:
            d, k = road.distance_m(point)
            if d < best_d:
                best_d, best_road, best_k = d, road, k
        return best_road, best_d, best_k

    def match(self, traj: Traj) -> MatchResult:
        res = MatchResult()
        for p in traj.coords:
            if not geo.is_legal_lonlat(p[0], p[1]):
                res.matched_coords.append(p)
                res.distances_m.append(float("inf"))
                res.matched.append(False)
                continue
            road, d, _ = self.nearest(p)
            if road is None:
                res.matched_coords.append(p)
                res.distances_m.append(float("inf"))
                res.matched.append(False)
                continue
            res.matched_coords.append(road.project(p))
            res.distances_m.append(d)
            res.matched.append(d <= self.match_tolerance_m)
        return res

    def snap(self, point: geo.LonLat) -> geo.LonLat:
        road, _, _ = self.nearest(point)
        return road.project(point) if road is not None else point

    def distance_to_road_m(self, point: geo.LonLat) -> float:
        _, d, _ = self.nearest(point)
        return d

    def speed_limit_at(self, point: geo.LonLat) -> Optional[float]:
        """该点所在道路的限速（m/s）；无路网或等级未知时返回 None。"""
        road, d, _ = self.nearest(point)
        if road is None or d > self.match_tolerance_m:
            return None
        if road.speed_limit_mps is not None:
            return road.speed_limit_mps
        return HIGHWAY_SPEED_LIMIT_MPS.get(road.highway)


# 退化路网的长度下限：低于此值说明简化后只剩一个点或几个重合点，
# 这样的「道路」无法提供有意义的约束。
MIN_ROAD_LENGTH_M = 50.0


def roads_from_trajectory(traj: Traj,
                          highway: str = "primary",
                          simplify_tolerance_m: float = 15.0) -> List[SegmentRoad]:
    """把一条（已清洗的）轨迹简化后当作「道路折线」。

    仅用于无真实路网时的对照实验：得到一个与轨迹自身几何一致的「道路约束」，
    从而可以量化「路网约束」相对「纯几何规则」到底改变了什么。

    **退化情形必须显式拒绝**：若简化后只剩 1 个点、或折线总长不足
    MIN_ROAD_LENGTH_M，则返回空列表。
    否则会构造出一条长度为零的「道路」，任何点投影到它上面距离都是 0，
    于是 match_rate 报出 1.0 —— 这个数字看起来完美，实际上毫无信息量。
    宁可让调用方拿到空列表并显式报错。
    """
    from ..core import simplify as simplify_mod
    idx = simplify_mod.douglas_peucker(traj.coords, simplify_tolerance_m)
    pts = [traj.coords[i] for i in idx]
    if len(pts) < 2:
        return []
    if geo.path_length_m(pts) < MIN_ROAD_LENGTH_M:
        return []
    return [SegmentRoad(road_id=f"traj-{traj.seg_id}", points=pts, highway=highway)]


def road_constraint_report(traj: Traj, matcher: Optional[RoadMatcher]) -> Dict[str, object]:
    """路网约束摘要，供诊断卡与报告使用。"""
    if matcher is None or not getattr(matcher, "available", False):
        return {"available": False, "match_rate": None}
    m = matcher.match(traj)
    stats = m.stats()
    stats["available"] = True
    return stats
