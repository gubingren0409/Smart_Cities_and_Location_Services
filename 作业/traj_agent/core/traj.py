"""轨迹数据模型与载入。

原始数据格式（traj_dict.json）：
    { "<vehicle_id>": [ [ts, ts, ...], [[lon, lat], ...] ] }
两个列表等长，按时间升序。缺时间戳或坐标不齐的记录视为坏记录。

坐标约定：(lon, lat)，见 geo.py。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import geo

Point = Tuple[float, float]


@dataclass
class Traj:
    """一条（或一段）轨迹。时间戳与坐标严格等长且按时间升序。"""

    vehicle_id: str
    timestamps: List[int]
    coords: List[Point]
    segment_index: int = 0
    # 该点是由哪个原因保留下来的：'' 表示原始点，其余为清洗动作标签
    reasons: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if len(self.coords) != n:
            raise ValueError(
                f"时间戳({n})与坐标({len(self.coords)})长度不一致，vehicle={self.vehicle_id}"
            )
        if not self.reasons:
            self.reasons = [""] * n
        elif len(self.reasons) != n:
            raise ValueError("reasons 长度必须与点数量一致")

    # ---- 基本属性 -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.coords)

    @property
    def seg_id(self) -> str:
        return f"{self.vehicle_id}#{self.segment_index}"

    @property
    def duration_s(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return geo.bbox_of(self.coords)

    @property
    def length_m(self) -> float:
        return geo.path_length_m(self.coords)

    def positions(self, index: Sequence[int]) -> "Traj":
        """按索引取子集，返回新 Traj（不复制共享数据之外的东西）。"""
        idx = list(index)
        return Traj(
            vehicle_id=self.vehicle_id,
            timestamps=[self.timestamps[i] for i in idx],
            coords=[self.coords[i] for i in idx],
            segment_index=self.segment_index,
            reasons=[self.reasons[i] for i in idx],
        )

    def time_deltas(self) -> List[float]:
        """相邻点时间间隔（秒），长度与点数一致，首元素 0.0。"""
        return geo.consecutive_differences([float(t) for t in self.timestamps])

    def space_deltas_m(self) -> List[float]:
        """相邻点平面距离（米），长度与点数一致，首元素 0.0。"""
        n = len(self.coords)
        out = [0.0] * n
        for i in range(1, n):
            out[i] = geo.local_distance_m(self.coords[i - 1], self.coords[i])
        return out

    def speeds_mps(self, max_gap_s: Optional[float] = None) -> List[float]:
        """逐点速度（m/s），长度与点数一致，首元素 0.0。

        dt <= 0 或 dt > max_gap_s 的位置返回 0.0（无法定义瞬时速度），
        这些位置由 anomalies 的动作规则单独处理，不要在这里伪造数值。
        """
        dt = self.time_deltas()
        ds = self.space_deltas_m()
        out = [0.0] * len(self.coords)
        for i in range(1, len(self.coords)):
            if dt[i] <= 0.0:
                continue
            if max_gap_s is not None and dt[i] > max_gap_s:
                continue
            out[i] = ds[i] / dt[i]
        return out

    def accelerations_mps2(self, max_gap_s: Optional[float] = None) -> List[float]:
        """逐点加速度（m/s^2），长度与点数一致，首元素 0.0。"""
        v = self.speeds_mps(max_gap_s=max_gap_s)
        dt = self.time_deltas()
        out = [0.0] * len(self.coords)
        for i in range(1, len(v)):
            if dt[i] <= 0.0:
                continue
            if max_gap_s is not None and dt[i] > max_gap_s:
                continue
            out[i] = (v[i] - v[i - 1]) / dt[i]
        return out

    @property
    def is_stationary(self, threshold_m: float = 200.0) -> bool:
        """是否为静止轨迹（总位移低于阈值）。

        本项目 400 条抽样中约 63% 的车辆全程 20 分钟静止，
        它们的「漂移」「转向角」都是 GPS 抖动而非行为，必须区别对待。
        """
        return self.length_m < threshold_m

    def to_points(self) -> List[Point]:
        return list(self.coords)

    def summary(self) -> Dict[str, object]:
        """紧凑摘要，可安全序列化给 LLM 或写入报告。"""
        return {
            "seg_id": self.seg_id,
            "vehicle_id": self.vehicle_id,
            "segment_index": self.segment_index,
            "n_points": len(self.coords),
            "duration_s": self.duration_s,
            "length_m": round(self.length_m, 2),
        }


def load_raw(path: str) -> Dict[str, Tuple[List[int], List[Point]]]:
    """读取原始 JSON，转成 {vehicle_id: (timestamps, coords)}。

    只做结构校验，不做清洗；坏记录被跳过并计入 load_json_report。
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return normalize_raw(raw)


def normalize_raw(raw: Dict[str, object]) -> Dict[str, Tuple[List[int], List[Point]]]:
    """把原始 dict 规范化；跳过结构损坏的记录。"""
    out: Dict[str, Tuple[List[int], List[Point]]] = {}
    for vid, value in raw.items():
        pair = _as_pairs(value)
        if pair is None:
            continue
        ts, coords = pair
        out[str(vid)] = (ts, coords)
    return out


def _as_pairs(value: object) -> Optional[Tuple[List[int], List[Point]]]:
    """接受 [ts, coords] 或 {'timestamps':..., 'coords':...} 两种写法。"""
    if isinstance(value, dict):
        ts = value.get("timestamps") or value.get("time") or value.get("ts")
        coords = value.get("coords") or value.get("coordinates") or value.get("points")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        ts, coords = value[0], value[1]
    else:
        return None
    if not isinstance(ts, (list, tuple)) or not isinstance(coords, (list, tuple)):
        return None
    if len(ts) != len(coords) or len(ts) < 2:
        return None

    out_ts: List[int] = []
    out_coords: List[Point] = []
    for t, p in zip(ts, coords):
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        # 保留原始数值（含 NaN/None），非法坐标由 anomalies 规则负责识别，
        # 不要在载入阶段静默删除——那会让「非法坐标计数」永远为 0。
        lon = geo.finite_or_none(p[0])
        lat = geo.finite_or_none(p[1])
        out_coords.append((lon if lon is not None else float("nan"),
                           lat if lat is not None else float("nan")))
        try:
            out_ts.append(int(t))
        except (TypeError, ValueError):
            return None

    # 时间升序校验：乱序则按下标排序
    if any(out_ts[i] < out_ts[i - 1] for i in range(1, len(out_ts))):
        order = sorted(range(len(out_ts)), key=lambda i: out_ts[i])
        out_ts = [out_ts[i] for i in order]
        out_coords = [out_coords[i] for i in order]
    return out_ts, out_coords


def traj_from_raw(vehicle_id: str, ts: Sequence[int], coords: Sequence[Point]) -> Traj:
    return Traj(vehicle_id=str(vehicle_id), timestamps=[int(t) for t in ts],
                coords=[(float(p[0]), float(p[1])) for p in coords])


def iter_trajs(raw: Dict[str, Tuple[List[int], List[Point]]],
               limit: Optional[int] = None) -> Iterable[Traj]:
    """按 vehicle_id 的字典序稳定遍历，便于复现。"""
    for i, vid in enumerate(sorted(raw.keys(), key=lambda k: (len(k), k))):
        if limit is not None and i >= limit:
            return
        ts, coords = raw[vid]
        yield traj_from_raw(vid, ts, coords)


def dataset_stats(path: str) -> Dict[str, object]:
    """数据集级统计，用于给参数先验定锚（也便于写进报告）。"""
    raw = load_raw(path)
    n_pts: List[int] = []
    dts: List[float] = []
    dup_ratio: List[float] = []
    for vid, (ts, coords) in raw.items():
        n_pts.append(len(coords))
        if len(ts) > 1:
            dts.extend(float(ts[i] - ts[i - 1]) for i in range(1, len(ts)))
        dup = sum(1 for i in range(1, len(coords)) if coords[i] == coords[i - 1])
        dup_ratio.append(dup / max(1, len(coords) - 1))
    return {
        "n_vehicles": len(raw),
        "n_points_total": int(sum(n_pts)),
        "points_per_vehicle": {
            "min": min(n_pts) if n_pts else 0,
            "median": geo.median([float(x) for x in n_pts]) if n_pts else 0.0,
            "max": max(n_pts) if n_pts else 0,
        },
        "dt_s": {
            "min": min(dts) if dts else 0.0,
            "median": geo.median(dts) if dts else 0.0,
            "p95": geo.quantile(dts, 0.95) if dts else 0.0,
            "max": max(dts) if dts else 0.0,
            "mode": _mode(dts),
            "bimodal": geo.is_bimodal(dts) if dts else False,
        },
        "consecutive_dup_ratio": {
            "median": geo.median(dup_ratio) if dup_ratio else 0.0,
            "p90": geo.quantile(dup_ratio, 0.9) if dup_ratio else 0.0,
        },
    }


def _mode(values: Sequence[float]):
    if not values:
        return None
    counts: Dict[float, int] = {}
    for v in values:
        counts[float(v)] = counts.get(float(v), 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]
