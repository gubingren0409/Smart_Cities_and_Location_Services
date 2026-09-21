"""目标函数与 knee point：把「清洗得好不好」变成一个可比较的标量。

这是任务③「用定量指标核验建议」的判据来源，
也是任务②「质量—压缩率曲线」的曲线定义。

设计原则
--------
1. **显式多目标**。质量、压缩率、路网一致性、运行时间各自可解释，
   不做隐式加权。
2. **knee point 优先于加权和**。加权和的权重本身会变成新的玄学参数；
   knee point 从实测曲线上取，无需人为定权。
3. **归一化可复现**。所有归一化基准由 ground-truth 扫描确定，
   不依赖提议本身，否则 regret 不可比。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ObjectiveWeights:
    """目标函数权重。仅在需要单一标量时使用（knee point 不需要）。"""

    quality: float = 1.0          # 保真度（越小偏差越好，取 1 - 归一化偏差）
    compression: float = 1.0      # 压缩率（越大越好）
    road: float = 0.5             # 路网匹配率（越大越好）
    runtime: float = 0.2          # 运行时间（越小越好，取负贡献）


@dataclass
class ObjectiveResult:
    """一次评估的原始量与归一化后的分数。"""

    # 原始量
    n_points: int = 0
    n_points_ref: int = 0
    tolerance_m: float = 0.0
    max_deviation_m: float = 0.0
    length_ratio: float = 1.0          # 简化后长度 / 原始长度
    hausdorff_m: float = 0.0
    frechet_m: float = 0.0
    road_match_rate: Optional[float] = None
    runtime_ms: float = 0.0

    # 归一化分量（均落在 [0,1]，越大越好）
    compression: float = 0.0
    fidelity: float = 0.0
    road_term: float = 0.0
    runtime_term: float = 0.0

    score: float = 0.0
    feasible: bool = True
    violations: List[str] = field(default_factory=list)
    # applicable=False 表示这条轨迹上「压缩率/保真度」这套目标函数没有意义，
    # 此时 score 不应被用于比较或 regret 计算。
    applicable: bool = True
    inapplicable_reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_points": self.n_points,
            "n_points_ref": self.n_points_ref,
            "tolerance_m": round(self.tolerance_m, 3),
            "max_deviation_m": round(self.max_deviation_m, 3),
            "length_ratio": round(self.length_ratio, 4),
            "hausdorff_m": round(self.hausdorff_m, 3),
            "frechet_m": round(self.frechet_m, 3),
            "road_match_rate": self.road_match_rate,
            "runtime_ms": round(self.runtime_ms, 3),
            "compression": round(self.compression, 4),
            "fidelity": round(self.fidelity, 4),
            "road_term": round(self.road_term, 4),
            "runtime_term": round(self.runtime_term, 4),
            "score": round(self.score, 4),
            "feasible": self.feasible,
            "violations": list(self.violations),
            "applicable": self.applicable,
            "inapplicable_reason": self.inapplicable_reason,
        }


def compression_ratio(n_after: int, n_before: int) -> float:
    if n_before <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - n_after / n_before))


def fidelity_from_deviation(deviation_m: float, tolerance_m: float,
                            scale_m: float = 30.0) -> float:
    """把偏差折成 [0,1] 的保真度。偏差 0 -> 1；偏差达到 scale_m -> 0。

    scale_m 取 GPS 精度上限（30 m ≈ 3-4 倍 CEP），
    含义是「偏差超过它就等于把点扔到了另一条路上」。
    """
    if scale_m <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - deviation_m / scale_m))


# 低于该长度的轨迹不适用「压缩率—保真度」目标函数。
# 取值与 diagnosis.classify_regime 的静止阈值一致，概念上同源。
MIN_APPLICABLE_LENGTH_M = 200.0


def compute_objective(n_after: int,
                      n_before: int,
                      max_deviation_m: float,
                      tolerance_m: float,
                      length_after_m: float,
                      length_before_m: float,
                      runtime_ms: float,
                      runtime_budget_ms: float = 50.0,
                      road_match_rate: Optional[float] = None,
                      weights: Optional[ObjectiveWeights] = None,
                      length_ratio_floor: float = 0.98,
                      min_applicable_length_m: float = MIN_APPLICABLE_LENGTH_M
                      ) -> ObjectiveResult:
    """计算目标函数。

    length_ratio_floor 是**可行性**约束：简化后长度相对原长不能掉太多。
    过长丢失说明把真实绕行删掉了，此时无论压缩率多高都不可接受。

    短轨迹（长度 < min_applicable_length_m）标记 applicable=False：
    实测车辆 0 是静止轨迹，清洗后仅 21 点、总长 12.29 m，全是厘米级抖动。
    此时任何容差（0.5 m 已接近整条轨迹的尺度）都会让长度比崩到 40–86%，
    目标函数必然判为不可行——但这不是「清洗方案不好」，
    而是这个指标在这类轨迹上没有意义。
    对不可适用的轨迹，核验应按 regime 正确性判定，而不是按 regret。
    """
    if length_before_m < min_applicable_length_m:
        return ObjectiveResult(
            n_points=n_after, n_points_ref=n_before, tolerance_m=tolerance_m,
            max_deviation_m=max_deviation_m,
            length_ratio=(length_after_m / length_before_m) if length_before_m > 0 else 1.0,
            runtime_ms=runtime_ms, road_match_rate=road_match_rate,
            applicable=False,
            inapplicable_reason=(
                f"轨迹总长 {length_before_m:.1f} m 低于 {min_applicable_length_m:.0f} m，"
                "压缩率与保真度在该尺度上没有物理含义"),
        )
    w = weights or ObjectiveWeights()
    r = ObjectiveResult(
        n_points=n_after,
        n_points_ref=n_before,
        tolerance_m=tolerance_m,
        max_deviation_m=max_deviation_m,
        length_ratio=(length_after_m / length_before_m) if length_before_m > 0 else 1.0,
        runtime_ms=runtime_ms,
        road_match_rate=road_match_rate,
    )
    r.compression = compression_ratio(n_after, n_before)
    r.fidelity = fidelity_from_deviation(max_deviation_m, tolerance_m)
    r.road_term = float(road_match_rate) if road_match_rate is not None else 0.0
    r.runtime_term = max(0.0, 1.0 - runtime_ms / runtime_budget_ms) if runtime_budget_ms > 0 else 0.0

    if r.length_ratio < length_ratio_floor:
        r.feasible = False
        r.violations.append(
            f"简化后长度仅为原长的 {r.length_ratio:.1%}，低于下限 {length_ratio_floor:.0%}，"
            "说明删掉了真实几何")
    if max_deviation_m > tolerance_m * 1.5 + 1e-9:
        r.feasible = False
        r.violations.append(
            f"最大偏差 {max_deviation_m:.2f}m 超出容差 {tolerance_m:.2f}m 的 1.5 倍")

    total_w = w.quality + w.compression + w.road + w.runtime
    if total_w <= 0:
        r.score = 0.0
    else:
        r.score = (w.quality * r.fidelity
                   + w.compression * r.compression
                   + w.road * r.road_term
                   + w.runtime * r.runtime_term) / total_w
    if not r.feasible:
        r.score -= 1.0     # 不可行解一律排在可行解之后
    return r


# ---------------------------------------------------------------------------
# knee point
# ---------------------------------------------------------------------------
@dataclass
class KneePoint:
    index: int = -1
    tolerance_m: float = 0.0
    n_points: int = 0
    compression: float = 0.0
    fidelity: float = 0.0
    method: str = "curvature"

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "tolerance_m": round(self.tolerance_m, 3),
            "n_points": self.n_points,
            "compression": round(self.compression, 4),
            "fidelity": round(self.fidelity, 4),
            "method": self.method,
        }


def find_knee_point(curve: Sequence[Tuple[float, float]],
                    method: str = "curvature") -> KneePoint:
    """从「质量—压缩率曲线」上找拐点。

    曲线输入为 [(compression, fidelity), ...]，按 compression 升序。
    返回拐点位置。

    method="curvature"：对曲线做 min-max 归一化后，找离首尾连线最远的点。
        这等价于最大曲率法，且不需要拟合，数值稳定。
    method="max_score"：直接取 fidelity + compression 最大处（加权和的特例）。
    """
    pts = [(float(c), float(f)) for c, f in curve]
    if not pts:
        return KneePoint()
    if len(pts) == 1:
        return KneePoint(index=0, compression=pts[0][0], fidelity=pts[0][1], method=method)

    order = sorted(range(len(pts)), key=lambda i: pts[i][0])
    sorted_pts = [pts[i] for i in order]

    if method == "max_score":
        best = max(range(len(sorted_pts)), key=lambda i: sorted_pts[i][0] + sorted_pts[i][1])
        pos = best
    else:
        # 归一化到单位方格，再找离首尾连线最远的点
        cs = [p[0] for p in sorted_pts]
        fs = [p[1] for p in sorted_pts]
        c_lo, c_hi = min(cs), max(cs)
        f_lo, f_hi = min(fs), max(fs)
        c_span = (c_hi - c_lo) or 1.0
        f_span = (f_hi - f_lo) or 1.0
        norm = [((c - c_lo) / c_span, (f - f_lo) / f_span) for c, f in sorted_pts]
        x0, y0 = norm[0]
        x1, y1 = norm[-1]
        dx, dy = x1 - x0, y1 - y0
        denom = math.hypot(dx, dy)
        best_dist = -1.0
        pos = 0
        for i, (x, y) in enumerate(norm):
            if denom <= 1e-12:
                d = math.hypot(x - x0, y - y0)
            else:
                # 到首尾连线的垂距
                d = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / denom
            if d > best_dist:
                best_dist = d
                pos = i

    orig_index = order[pos]
    return KneePoint(index=orig_index, compression=pts[orig_index][0],
                     fidelity=pts[orig_index][1], method=method)


def knee_from_results(results: Sequence[ObjectiveResult],
                      method: str = "curvature") -> Tuple[KneePoint, Optional[ObjectiveResult]]:
    """从一组目标函数结果里找 knee point 对应的那条结果。"""
    curve = [(r.compression, r.fidelity) for r in results]
    kp = find_knee_point(curve, method=method)
    if kp.index < 0:
        return kp, None
    return kp, results[kp.index]


def pareto_front(results: Sequence[ObjectiveResult]) -> List[int]:
    """返回 Pareto 前沿上的索引（压缩率与保真度都不劣于他人）。

    用于报告：前沿上的点才是「值得讨论的取舍」，
    被支配的点不提供额外信息。
    """
    front: List[int] = []
    for i, a in enumerate(results):
        dominated = False
        for j, b in enumerate(results):
            if i == j:
                continue
            if (b.compression >= a.compression and b.fidelity >= a.fidelity
                    and (b.compression > a.compression or b.fidelity > a.fidelity)):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front
