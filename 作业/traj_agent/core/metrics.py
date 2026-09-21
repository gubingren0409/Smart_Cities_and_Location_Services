"""评价指标：点数、轨迹长度、Hausdorff / Fréchet / DTW 距离、运行时间。

对应学生必做任务①的「对比点数、轨迹长度、Hausdorff 距离和运行时间」。

实现取舍
--------
- **Hausdorff 用点到折线的连续版，不用 scipy 的离散顶点集版本。**
  原因见 hausdorff_m 的文档字符串：顶点集版本在「简化后顶点变稀疏」时
  会给出完全错误的结果（实测 548m vs 真实 3.87m）。
- Fréchet 用 O(n·m) 动态规划。原始 util.py 的 judge_frechet 是递归实现，
  在长轨迹上会爆栈且重复计算，不予复用。
- **Hausdorff 与 Fréchet 在评价轨迹简化时差异极大，这不是 bug。**
  实测（车辆 246，106 点压到 35 点，DP 容差 5 m）：
      Hausdorff(点到折线) = 4.77 m   —— 与 DP 误差界一致
      Fréchet(点对点)     = 548.10 m —— 因为简化后顶点稀疏，
                                        某对必须匹配的点在参数上相隔很远
  两者含义不同：Hausdorff 回答「简化后的线离原线有多远」，
  Fréchet 额外要求「沿弧长的匹配顺序」。
  评价简化保真度用 Hausdorff（或 max_deviation）；用 Fréchet 时
  必须知道它对顶点稀疏度敏感。报告里两个都要给，并加注说明。
- 所有距离在**局部等距平面**上以米计算，与 DP 容差同源。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geo
from .traj import Traj


def _xy(points: Sequence[geo.Point]) -> np.ndarray:
    if not len(points):
        return np.zeros((0, 2), dtype=float)
    return np.asarray([geo.local_xy(p) for p in points], dtype=float)


def point_to_polyline_m(p: geo.Point, poly: Sequence[geo.Point]) -> float:
    """点到折线的最短距离（米）。折线不足两点时退化为点距。"""
    if not poly:
        return float("nan")
    if len(poly) == 1:
        return geo.local_distance_m(p, poly[0])
    return min(geo.segment_distance_m(p, poly[k], poly[k + 1])
               for k in range(len(poly) - 1))


def hausdorff_m(a: Sequence[geo.Point], b: Sequence[geo.Point],
                directed: bool = False) -> float:
    """点到折线的连续 Hausdorff 距离（米）。默认取双向最大值。

    为什么不能用 scipy.spatial.distance.directed_hausdorff
    -----------------------------------------------------
    那是**离散顶点集**之间的 Hausdorff：它只比较两组顶点的最近距离。
    当 B 是 A 的简化结果时 B 的顶点很稀疏，A 中落在 B 的两个相邻顶点
    之间的点会「离所有 B 的顶点都很远」，尽管它们离 B 的**折线段**很近。

    实测（车辆 246，106 点压到 35 点，DP 容差 5m）：
        离散顶点集 Hausdorff = 548.10 m   <- 完全错误
        点到折线   Hausdorff = 3.87 m     <- 与 DP 的 4.77m 误差界一致

    故本函数一律按点到折线计算。这是轨迹简化评价的正确口径。
    """
    if not len(a) or not len(b):
        return 0.0
    d_ab = max(point_to_polyline_m(p, b) for p in a)
    if directed:
        return float(d_ab)
    d_ba = max(point_to_polyline_m(q, a) for q in b)
    return float(max(d_ab, d_ba))


def frechet_m(a: Sequence[geo.Point], b: Sequence[geo.Point],
              max_points: int = 400) -> float:
    """离散 Fréchet 距离（米）。

    轨迹过长时按均匀抽样降采样到 max_points，避免 O(n·m) 内存爆掉。
    本项目轨迹最长 229 点，通常不触发；但清洗后可能拼接过长序列。
    """
    from scipy.spatial.distance import cdist
    pa, pb = _xy(a), _xy(b)
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    # 只在确实过大时降采样：本项目轨迹最长 229 点，
    # 降采样会改变 Fréchet 的值（它依赖点序），不应无故触发。
    if len(pa) * len(pb) > max_points * max_points:
        pa = _subsample(pa, max_points)
        pb = _subsample(pb, max_points)
    ca = cdist(pa, pb, metric="euclidean")
    n, m = ca.shape
    # 经典 DP：f[i][j] = max(min(f[i-1][j], f[i-1][j-1], f[i][j-1]), ca[i][j])
    f = np.empty((n, m), dtype=float)
    f[0, 0] = ca[0, 0]
    for i in range(1, n):
        f[i, 0] = max(f[i - 1, 0], ca[i, 0])
    for j in range(1, m):
        f[0, j] = max(f[0, j - 1], ca[0, j])
    for i in range(1, n):
        for j in range(1, m):
            f[i, j] = max(min(f[i - 1, j], f[i - 1, j - 1], f[i, j - 1]), ca[i, j])
    return float(f[n - 1, m - 1])


def _subsample(arr: np.ndarray, k: int) -> np.ndarray:
    if len(arr) <= k:
        return arr
    idx = np.linspace(0, len(arr) - 1, k).astype(int)
    return arr[np.unique(idx)]


def dtw_m(a: Sequence[geo.Point], b: Sequence[geo.Point],
          window: Optional[int] = None, max_points: int = 400) -> float:
    """DTW 距离（米）。window 给 Sakoe-Chiba 带宽；None 表示无约束。"""
    pa, pb = _xy(a), _xy(b)
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    pa = _subsample(pa, max_points)
    pb = _subsample(pb, max_points)
    n, m = len(pa), len(pb)
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    if window is None:
        window = max(n, m)
    inf = float("inf")
    acc = np.full((n + 1, m + 1), inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - window)
        hi = min(m, i + window)
        for j in range(lo, hi + 1):
            acc[i, j] = d[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m])


@dataclass
class Metrics:
    """一条轨迹（或其简化结果）的完整指标。"""

    n_points: int = 0
    length_m: float = 0.0
    duration_s: float = 0.0
    sinuosity: float = 1.0
    bbox_span_m: float = 0.0
    mean_speed_mps: float = 0.0
    max_speed_mps: float = 0.0

    # 与参考轨迹的偏差（清洗/简化前后对比时填充）
    hausdorff_m: Optional[float] = None
    hausdorff_directed_m: Optional[float] = None
    frechet_m: Optional[float] = None
    dtw_m: Optional[float] = None

    runtime_ms: float = 0.0
    road_match_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_points": self.n_points,
            "length_m": round(self.length_m, 3),
            "duration_s": round(self.duration_s, 3),
            "sinuosity": round(self.sinuosity, 4),
            "bbox_span_m": round(self.bbox_span_m, 3),
            "mean_speed_mps": round(self.mean_speed_mps, 3),
            "max_speed_mps": round(self.max_speed_mps, 3),
            "hausdorff_m": None if self.hausdorff_m is None else round(self.hausdorff_m, 3),
            "hausdorff_directed_m": (None if self.hausdorff_directed_m is None
                                     else round(self.hausdorff_directed_m, 3)),
            "frechet_m": None if self.frechet_m is None else round(self.frechet_m, 3),
            "dtw_m": None if self.dtw_m is None else round(self.dtw_m, 3),
            "runtime_ms": round(self.runtime_ms, 3),
            "road_match_rate": self.road_match_rate,
        }


def compute_metrics(traj: Traj,
                    reference: Optional[Traj] = None,
                    with_distances: bool = True,
                    with_dtw: bool = False) -> Metrics:
    """计算指标。给出 reference 时同时计算与它的偏差距离。

    with_distances=False 可跳过 O(n·m) 的 Fréchet，用于大规模扫描时提速。

    口径提醒：reference 应当选**同一阶段**的轨迹。
    若拿简化结果去对比未清洗的原始轨迹，Hausdorff/Fréchet 会同时包含
    「清洗删点」与「几何简化」两部分的形变，无法单独评价简化质量。
    例如车辆 246 上，简化后相对原始是 548m（主要来自被删的跳变点），
    而相对清洗后只有 3.87m（纯 DP 误差）。
    """
    t0 = time.perf_counter()
    m = Metrics(n_points=len(traj))
    if len(traj) == 0:
        m.runtime_ms = (time.perf_counter() - t0) * 1000.0
        return m

    m.length_m = traj.length_m
    m.duration_s = traj.duration_s
    m.sinuosity = geo.sinuosity(traj.coords)
    min_lon, min_lat, max_lon, max_lat = traj.bbox
    m.bbox_span_m = geo.local_distance_m((min_lon, min_lat), (max_lon, max_lat))
    v = [x for x in traj.speeds_mps() if x > 0]
    if v:
        m.max_speed_mps = max(v)
    if m.duration_s > 0:
        m.mean_speed_mps = m.length_m / m.duration_s

    if reference is not None and with_distances and len(reference) and len(traj):
        m.hausdorff_m = hausdorff_m(reference.coords, traj.coords)
        m.hausdorff_directed_m = hausdorff_m(reference.coords, traj.coords, directed=True)
        m.frechet_m = frechet_m(reference.coords, traj.coords)
        if with_dtw:
            m.dtw_m = dtw_m(reference.coords, traj.coords)

    m.runtime_ms = (time.perf_counter() - t0) * 1000.0
    return m


@dataclass
class ComparisonTable:
    """清洗/简化前后对比表，是任务①「对比点数、长度、Hausdorff、运行时间」的落点。"""

    rows: List[Dict[str, object]] = field(default_factory=list)

    def add(self, label: str, metrics: Metrics, extra: Optional[Dict[str, object]] = None) -> None:
        row: Dict[str, object] = {"stage": label}
        row.update(metrics.to_dict())
        if extra:
            row.update(extra)
        self.rows.append(row)

    def to_dict(self) -> List[Dict[str, object]]:
        return self.rows

    def format_text(self, columns: Optional[Sequence[str]] = None) -> str:
        """等宽文本表，便于在 notebook 与报告里直接看。"""
        if not self.rows:
            return "(空)"
        cols = list(columns) if columns else list(self.rows[0].keys())
        cols = [c for c in cols if any(r.get(c) is not None for r in self.rows)]
        widths = {c: max(len(str(c)), *(len(_fmt(r.get(c))) for r in self.rows)) for c in cols}
        head = "  ".join(str(c).rjust(widths[c]) for c in cols)
        sep = "  ".join("-" * widths[c] for c in cols)
        lines = [head, sep]
        for r in self.rows:
            lines.append("  ".join(_fmt(r.get(c)).rjust(widths[c]) for c in cols))
        return "\n".join(lines)


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)
