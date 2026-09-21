"""轨迹简化：Douglas-Peucker。

对应学生必做任务①的「完成去噪与 DP 压缩」。

关于原始的 douglas_peucker.py
---------------------------
仓库里已有一份实现，但有两个必须知道的缺陷，本项目**不复用它**：

1. `point2LineDistance` 在 `point_b[0] == point_c[0]`（垂直线段）时
   直接返回 9999999，使 DP 在垂直路段上永不收敛于该阈值；
2. 它算的是到**无限长直线**的距离，而非到**线段**的距离，
   端点外的点会被错误地判为「距离很小」。

`douglas_peucker_broken` 保留了原实现的语义，仅用于课堂对照实验
（tests/test_simplify.py 里有回归断言）。生产路径走 `douglas_peucker`。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import geo
from .traj import Traj


# ---------------------------------------------------------------------------
# 正确的 DP
# ---------------------------------------------------------------------------
def douglas_peucker_indices(points: Sequence[geo.Point],
                            tolerance_m: float) -> Tuple[List[int], float]:
    """返回 (保留点索引升序, 实际最大偏差米)。

    实际最大偏差从递归结构里直接得到，无需二次计算：
    每次递归处理区间 [start, end] 时，若最大距离 d <= tolerance 则该区间
    所有中间点都被丢弃，它们到简化折线的真实偏差就是 d（DP 的构造保证
    这些点到线段 (start, end) 的距离即为 d，而非到别的线段）。
    取所有「被丢弃区间」里最大的 d，即得真实误差上界，且是精确值。

    比事后用最近线段搜索便宜且不会有实现错误——本项目最初的事后版本
    在 tol=2 时报出 8349m 的偏差（容差 2m 不可能产生 8km 误差）。
    """
    n = len(points)
    if n <= 2:
        return list(range(n)), 0.0
    if tolerance_m <= 0:
        return list(range(n)), 0.0

    keep = [False] * n
    keep[0] = keep[n - 1] = True
    stack: List[Tuple[int, int]] = [(0, n - 1)]
    max_dev = 0.0

    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        max_dist = -1.0
        max_idx = -1
        for i in range(start + 1, end):
            d = geo.segment_distance_m(points[i], points[start], points[end])
            if d > max_dist:
                max_dist = d
                max_idx = i
        if max_idx <= 0:
            continue
        if max_dist <= tolerance_m:
            # 整段中间点被丢弃，其真实偏差即 max_dist
            max_dev = max(max_dev, max_dist)
        else:
            keep[max_idx] = True
            stack.append((start, max_idx))
            stack.append((max_idx, end))

    return [i for i, k in enumerate(keep) if k], max_dev


def douglas_peucker(points: Sequence[geo.Point], tolerance_m: float) -> List[int]:
    """返回被保留点的索引（升序，含首尾）。

    用迭代栈而非递归：本项目轨迹最长 229 点尚可递归，但学生数据可能更长，
    且递归在退化情形（如大量重复点）会触发 RecursionError。

    tolerance_m 单位为米，距离由 geo.segment_distance_m 在局部等距平面给出，
    与 DP 容差预算同源（不要换成墨卡托：那会放大 17%，见 geo 模块说明）。
    """
    return douglas_peucker_indices(points, tolerance_m)[0]


def douglas_peucker_broken(points: Sequence[geo.Point], tolerance_m: float) -> List[int]:
    """对照用的**错误**实现，复刻 douglas_peucker.py 的两个 bug。

    仅用于教学：让学生看到垂直线段上 DP 失效、以及端点外点被判为近点。
    绝不用于生产。
    """
    n = len(points)
    if n <= 2:
        return list(range(n))

    def broken_point_line_distance(p, a, b) -> float:
        # bug 1: 垂直线段硬编码返回巨大值
        if abs(a[0] - b[0]) < 1e-12:
            return 9999999.0
        ax, ay = geo.local_xy(a)
        bx, by = geo.local_xy(b)
        px, py = geo.local_xy(p)
        slope = (by - ay) / (bx - ax)
        intercept = by - slope * bx
        # bug 2: 到无限长直线的距离
        return abs(slope * px - py + intercept) / math.sqrt(1.0 + slope * slope)

    keep = [False] * n
    keep[0] = keep[n - 1] = True
    stack: List[Tuple[int, int]] = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        max_dist = -1.0
        max_idx = -1
        for i in range(start + 1, end):
            d = broken_point_line_distance(points[i], points[start], points[end])
            if d > max_dist:
                max_dist = d
                max_idx = i
        if max_idx > 0 and max_dist > tolerance_m:
            keep[max_idx] = True
            stack.append((start, max_idx))
            stack.append((max_idx, end))
    return [i for i, k in enumerate(keep) if k]


# ---------------------------------------------------------------------------
# 其它简化算法（用于任务②的算法对比）
# ---------------------------------------------------------------------------
def uniform_sample_indices(n: int, ratio: float) -> List[int]:
    """均匀抽样索引，按给定保留比例。作为 DP 的对照基线。"""
    if n <= 2:
        return list(range(n))
    k = max(2, min(n, int(round(n * max(0.0, min(1.0, ratio))))))
    if k >= n:
        return list(range(n))
    step = (n - 1) / (k - 1)
    idx = sorted({int(round(i * step)) for i in range(k)})
    if idx[0] != 0:
        idx.insert(0, 0)
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def perp_distance_indices(points: Sequence[geo.Point], tolerance_m: float) -> List[int]:
    """垂距限值法（Perpendicular Distance）：在线简化里常与 DP 对比。

    逐点判断到「上一保留点 → 当前点」的垂距，超限则保留当前点。
    对噪声敏感、且不保证全局误差界，正好用来衬托 DP 的优势。
    """
    n = len(points)
    if n <= 2:
        return list(range(n))
    keep = [0]
    anchor = 0
    for i in range(1, n - 1):
        if geo.segment_distance_m(points[i], points[anchor], points[i + 1]) > tolerance_m:
            keep.append(i)
            anchor = i
    keep.append(n - 1)
    return sorted(set(keep))


ALGORITHMS: Dict[str, Callable[[Sequence[geo.Point], float], List[int]]] = {
    "dp": douglas_peucker,
    "dp_broken": douglas_peucker_broken,
    "perp": perp_distance_indices,
}


@dataclass
class SimplifyResult:
    traj: Traj
    kept_indices: List[int]
    algorithm: str = "dp"
    tolerance_m: float = 0.0
    elapsed_ms: float = 0.0
    # 简化造成的最大偏差（即 DP 的误差上界）
    max_deviation_m: float = 0.0

    @property
    def n_before(self) -> int:
        return self._n_before

    _n_before: int = field(default=0, repr=False)

    @property
    def compression_ratio(self) -> float:
        if self._n_before <= 0:
            return 0.0
        return 1.0 - len(self.kept_indices) / self._n_before

    def to_dict(self) -> Dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "tolerance_m": self.tolerance_m,
            "n_before": self._n_before,
            "n_after": len(self.kept_indices),
            "compression_ratio": round(self.compression_ratio, 4),
            "max_deviation_m": round(self.max_deviation_m, 3),
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


def simplify_trajectory(traj: Traj,
                        tolerance_m: float = 5.0,
                        algorithm: str = "dp") -> SimplifyResult:
    """对轨迹做简化，返回简化结果与耗时。"""
    import time
    if algorithm not in ALGORITHMS:
        raise KeyError(f"未知算法 {algorithm!r}；可用 {sorted(ALGORITHMS)}")
    t0 = time.perf_counter()
    if algorithm == "dp":
        idx, dev = douglas_peucker_indices(traj.coords, float(tolerance_m))
    else:
        idx = ALGORITHMS[algorithm](traj.coords, float(tolerance_m))
        dev = max_deviation_m(traj.coords, [traj.coords[i] for i in idx]) if idx else 0.0
    elapsed = (time.perf_counter() - t0) * 1000.0

    if not idx:
        idx = [0] if len(traj) else []
    out = traj.positions(idx)
    # 保留式标签：被保留的点若原本带清洗标签，则延续
    res = SimplifyResult(
        traj=out,
        kept_indices=idx,
        algorithm=algorithm,
        tolerance_m=float(tolerance_m),
        elapsed_ms=elapsed,
    )
    res._n_before = len(traj)
    res.max_deviation_m = dev
    return res


def max_deviation_m(original: Sequence[geo.Point],
                    simplified: Sequence[geo.Point]) -> float:
    """被丢弃点到简化折线的最大垂距（米）。

    对每个原始点**全局**搜索最近的简化线段。早先的版本只在滑动窗口内搜索
    且窗口指针不推进，会把真实偏差高估数千倍；在 n<=500、m<=200 的规模下
    直接全局搜索完全可接受，正确性优先。

    注意这与 metrics.hausdorff_m 的一致性：后者同样是点到折线口径，
    故对 DP 结果有 max_deviation_m ≈ hausdorff_m(原始, 简化)。
    """
    if len(original) < 3 or len(simplified) < 2:
        return 0.0
    segs = [(simplified[k], simplified[k + 1]) for k in range(len(simplified) - 1)]
    worst = 0.0
    for p in original:
        best = min(geo.segment_distance_m(p, a, b) for a, b in segs)
        if best > worst:
            worst = best
    return worst


def sweep_dp_tolerance(traj: Traj,
                       tolerances: Sequence[float],
                       algorithm: str = "dp") -> List[SimplifyResult]:
    """扫描 DP 容差，产出「质量—压缩率曲线」的原始数据。

    这是任务②的核心实验，也是任务③里 verifier 计算 regret 的 ground truth 来源。
    """
    return [simplify_trajectory(traj, tol, algorithm) for tol in tolerances]
