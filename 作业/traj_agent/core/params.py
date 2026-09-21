"""参数空间：物理先验、合法区间、默认值。

设计原则（对应任务③的「三段式调参」第一段）：
    每个参数都必须能从**数据分布**或**物理量**反推，不允许拍脑袋。
    这里给出的是 (下界, 上界, 默认值, 先验来源说明)，LLM 只能在界内提议。

标定依据（本项目实测，见 dataset_stats）：
    dt 主频 10 s，p95 = 20 s，max 60 s，且 10/20 双峰
    → 全局用 3×中位数 = 30 s 会切掉正常的 20 s 采样，这是任务③里
      LLM 应当发现的第一个问题。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import geo


@dataclass(frozen=True)
class ParamSpec:
    """一个可调参数的物理先验。"""

    name: str
    low: float
    high: float
    default: float
    unit: str
    rationale: str
    integer: bool = False

    def clamp(self, value: float) -> float:
        v = max(self.low, min(self.high, float(value)))
        return float(round(v)) if self.integer else float(v)

    def in_range(self, value: float) -> bool:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        return self.low <= v <= self.high

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "default": self.default,
            "unit": self.unit,
            "integer": self.integer,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# 城市道路物理锚点（上海）
# ---------------------------------------------------------------------------
SPEED_LIMIT_CITY_MPS = 13.9       # 50 km/h 城区主干道
SPEED_LIMIT_ARTERIAL_MPS = 22.2   # 80 km/h 快速路
SPEED_LIMIT_HIGHWAY_MPS = 33.3    # 120 km/h 高速
GPS_CEP_M = 8.0                   # 民用 GPS 圆概率误差典型值
JUMP_SAFETY_FACTOR = 1.5


PARAM_SPECS: Dict[str, ParamSpec] = {
    "dt_threshold": ParamSpec(
        name="dt_threshold", low=15.0, high=180.0, default=30.0, unit="s", integer=True,
        rationale=(
            "切分用的时间间隔上界。数据 dt 主频 10s、p95 20s 且 10/20 双峰，"
            "故下界取 15s（保住正常 20s 采样），上界 180s（超过即为真实停车/失联）。"
            "默认 30s 是经验值但在此数据集上偏大，属于应当被实验推翻的起点。"
        ),
    ),
    "dist_threshold": ParamSpec(
        name="dist_threshold", low=50.0, high=3000.0, default=400.0, unit="m",
        rationale=(
            "切分用的空间跳跃上界。物理推导 = dt × 限速 × 安全系数，"
            "dt=20s 时 20×33.3×1.5≈1000m；下界 50m 用于拦住静止漂移。"
            "默认 400m 对应 20m/s × 20s，与讲义一致。"
        ),
    ),
    "max_speed_mps": ParamSpec(
        name="max_speed_mps", low=15.0, high=55.0, default=38.0, unit="m/s",
        rationale=(
            "速度突变上界。下界 15m/s=54km/h 低于城区限速会误杀正常行驶；"
            "上界 55m/s≈198km/h 覆盖 GPS 噪声下的高速路段。"
            "默认 38m/s≈137km/h，高于任何合法城市行驶速度。"
        ),
    ),
    "max_accel_mps2": ParamSpec(
        name="max_accel_mps2", low=1.0, high=15.0, default=6.0, unit="m/s^2",
        rationale=(
            "加速度突变上界。乘用车紧急制动约 -8m/s^2、起步约 3m/s^2；"
            "取 6m/s^2 可容忍急刹而不放过 GPS 跳变造成的假加速度。"
        ),
    ),
    "max_turn_deg": ParamSpec(
        name="max_turn_deg", low=45.0, high=175.0, default=150.0, unit="deg",
        rationale=(
            "单点转向角上界，用于识别漂移毛刺。真实路口掉头接近 180°但会持续多点；"
            "单点突变超过 150° 更可能是定位漂移。上界 175° 保留极限掉头。"
        ),
    ),
    "dp_tolerance": ParamSpec(
        name="dp_tolerance", low=0.5, high=30.0, default=5.0, unit="m",
        rationale=(
            "DP 压缩容差。理论上限约为 GPS 定位精度 CEP（8m）——超过它就是在删真实几何。"
            "下界 0.5m 避免退化为不压缩；默认 5m 与讲义一致。"
        ),
    ),
    "min_seg_points": ParamSpec(
        name="min_seg_points", low=2.0, high=50.0, default=5.0, unit="pt", integer=True,
        rationale="一个分段至少含有的点数。少于 2 点无法构成轨迹；默认 5 点与讲义一致。",
    ),
    "min_seg_length_m": ParamSpec(
        name="min_seg_length_m", low=0.0, high=500.0, default=65.0, unit="m",
        rationale="分段最小长度。默认 65m 与讲义一致，用于滤除静止抖动碎片。",
    ),
    "smooth_window": ParamSpec(
        name="smooth_window", low=2.0, high=15.0, default=5.0, unit="pt", integer=True,
        rationale=(
            "中值平滑窗口点数（奇数）。窗口过小无法压住孤立毛刺，"
            "过大（>15 点 ≈ 150s）会削平真实转弯。"
        ),
    ),
}


def get_spec(name: str) -> ParamSpec:
    if name not in PARAM_SPECS:
        raise KeyError(f"未知参数 {name!r}；可用：{sorted(PARAM_SPECS)}")
    return PARAM_SPECS[name]


def default_params() -> Dict[str, float]:
    return {k: v.default for k, v in PARAM_SPECS.items()}


def clamp_params(params: Dict[str, float],
                 only: Optional[Sequence[str]] = None) -> Tuple[Dict[str, float], List[str]]:
    """把参数夹到合法区间，返回 (结果, 被夹紧的参数名列表)。

    这是 verifier 的「约束满足率」判据来源，也是防止 LLM 给出
    dp_tolerance=500 这类建议的第一道闸门。
    """
    out = dict(params)
    clamped: List[str] = []
    names = list(only) if only is not None else list(params.keys())
    for name in names:
        if name not in params:
            continue
        spec = PARAM_SPECS.get(name)
        if spec is None:
            continue
        val = spec.clamp(params[name])
        if float(params[name]) != float(val):
            clamped.append(name)
        out[name] = val
    return out, sorted(set(clamped))


def validate_params(params: Dict[str, float]) -> List[str]:
    """返回违规说明列表；空列表表示全部合规。"""
    problems: List[str] = []
    for name, value in params.items():
        spec = PARAM_SPECS.get(name)
        if spec is None:
            problems.append(f"未知参数 {name!r}")
            continue
        if not spec.in_range(value):
            problems.append(
                f"{name}={value} 超出物理先验区间 [{spec.low}, {spec.high}] {spec.unit}"
            )
    return problems


def param_bounds_table() -> List[Dict[str, object]]:
    """给 LLM 看的参数区间表。"""
    return [spec.to_dict() for spec in PARAM_SPECS.values()]


def data_driven_priors(dt_values: Sequence[float],
                       dist_values: Optional[Sequence[float]] = None,
                       speed_values: Optional[Sequence[float]] = None) -> Dict[str, object]:
    """从单条轨迹的实测分布给出「建议起点」，供 LLM 参考而非替代它决策。

    这里刻意不做成自动调参：输出的是观测事实 + 一个保守建议，
    真正的决策权留给 LLM，并由 verifier 打分。

    关于 dist_threshold 的重要修正
    ----------------------------
    早先版本取 `p95(单步位移) × 3`。这在行驶轨迹上会给出荒谬的值：
    实测车辆 246 的单步 p95 达 250 m，于是建议阈值 773 m，**等于完全不切分**。
    纯确定性基线因此表现异常，污染了消融实验的对比。

    正确的物理推导是：空间跳跃阈值描述的是「dt 内以限速能走多远」，
    与单步位移的分布无关：
        dist_threshold ≈ dt_threshold × 限速 × 安全系数
    故这里先由 Δt 定出 dt_threshold，再用它推导距离阈值，
    并用 `jump_lower_bound` 兜住静止抖动。
    """
    obs: Dict[str, object] = {}
    suggestion: Dict[str, float] = {}

    if dt_values:
        dt_med = geo.median(dt_values)
        dt_p95 = geo.quantile(dt_values, 0.95)
        obs["dt_median_s"] = round(dt_med, 2)
        obs["dt_p95_s"] = round(dt_p95, 2)
        obs["dt_bimodal"] = geo.is_bimodal(dt_values)
        # 双峰时不能用「中位数×3」这种全局规则，取 p95 与物理下界的较大者
        dt_th = max(jump_lower_bound(dt_med) / (SPEED_LIMIT_CITY_MPS * JUMP_SAFETY_FACTOR),
                    dt_p95 * 1.2)
        dt_th = get_spec("dt_threshold").clamp(dt_th)
        suggestion["dt_threshold"] = dt_th

        # 距离阈值由 dt_threshold 与限速推导，不由位移分布推导
        implied_dt = max(dt_th, dt_p95)
        phys = physical_jump_threshold(implied_dt, SPEED_LIMIT_ARTERIAL_MPS)
        suggestion["dist_threshold"] = get_spec("dist_threshold").clamp(
            max(jump_lower_bound(dt_med), phys))
        obs["dist_threshold_from_physics_m"] = round(phys, 1)

    if speed_values:
        positive = [v for v in speed_values if v > 0]
        if positive:
            sp_med = geo.median(positive)
            sp_p99 = geo.quantile(positive, 0.99)
            obs["speed_median_mps"] = round(sp_med, 2)
            obs["speed_p99_mps"] = round(sp_p99, 2)
            obs["speed_p99_over_limit"] = round(sp_p99 / SPEED_LIMIT_ARTERIAL_MPS, 2)
            # 上界取物理上限与「p99 的 1.5 倍」中的较小者：
            # p99 本身可能已被时间戳伪影污染（实测最高 253 m/s），
            # 直接乘系数会把阈值推到荒谬的高度。
            suggestion["max_speed_mps"] = get_spec("max_speed_mps").clamp(
                min(max(SPEED_LIMIT_ARTERIAL_MPS * 1.3, sp_p99 * 1.5),
                    SPEED_LIMIT_HIGHWAY_MPS * 1.2))

    if dist_values:
        obs["dist_p95_m"] = round(geo.quantile(dist_values, 0.95), 2)

    return {"observations": obs, "suggested_start": suggestion}


def jump_lower_bound(dt_s: float) -> float:
    """空间跳跃阈值的物理下界：dt × 城区限速 × 安全系数。"""
    return float(dt_s) * SPEED_LIMIT_CITY_MPS * JUMP_SAFETY_FACTOR


def physical_jump_threshold(dt_s: float, speed_mps: float = SPEED_LIMIT_ARTERIAL_MPS) -> float:
    """按物理推导给出跳跃阈值：dt × 限速 × 安全系数。"""
    return float(dt_s) * float(speed_mps) * JUMP_SAFETY_FACTOR
