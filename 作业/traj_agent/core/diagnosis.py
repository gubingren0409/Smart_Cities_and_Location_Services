"""诊断卡：喂给 LLM 的唯一数据视图。

设计约束（对应「坐标永不进上下文」）：
    诊断卡只含**标量统计与计数**，不含任何坐标、不含原始序列。
    一条轨迹的诊断卡序列化后约 400-600 字节，11386 条全量也只有几 MB。

诊断卡的核心价值是让 LLM 看出「这条轨迹属于哪个 regime」——
本项目实测存在两个截然不同的 regime：
    - 静止轨迹（约 63%）：全程 20 分钟位移 < 200m，重复点占绝大多数
    - 行驶轨迹（约 37%）：位移数公里至 20km，含 GPS 跳变与时间戳伪影
对这两类用同一套参数是不可能的，这正是任务③中 LLM 应当发现的第一件事。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import anomalies, geo
from .anomalies import RuleParams
from .traj import Traj

# 时间轴质量等级
TIMELINE_OK = "ok"
TIMELINE_DEGRADED = "degraded"      # 存在少量重复时间戳，可修
TIMELINE_UNUSABLE = "unusable"      # 时间跨度短到无法支撑速度分析
TIMELINE_NOT_MONOTONIC = "not_monotonic"  # 存在负的 dt，顺序不可信


@dataclass
class DiagnosisCard:
    """单条轨迹诊断卡。所有字段可安全 JSON 序列化。"""

    seg_id: str
    vehicle_id: str
    n_points: int
    duration_s: float

    # regime
    regime: str = "unknown"                 # stationary | moving | mixed
    is_stationary: bool = False

    # 时间轴
    dt_median_s: float = 0.0
    dt_p95_s: float = 0.0
    dt_max_s: float = 0.0
    dt_zero_ratio: float = 0.0
    dt_zero_burst: int = 0                   # 最长连续 dt=0 的点数
    dt_bimodal: bool = False
    timeline_quality: str = TIMELINE_OK

    # 空间
    length_m: float = 0.0
    displacement_m: float = 0.0              # 首尾直线距离
    bbox_span_m: float = 0.0
    sinuosity: float = 1.0

    # 重复与速度
    consecutive_dup_ratio: float = 0.0
    dup_run_count: int = 0
    speed_median_mps: float = 0.0
    speed_p95_mps: float = 0.0
    speed_max_mps: float = 0.0

    # 异常计数（原因 → 点数）
    anomaly_counts: Dict[str, int] = field(default_factory=dict)
    n_anomalous_points: int = 0
    preserved_behavior: Dict[str, int] = field(default_factory=dict)

    # 路网（任务①接入后填充；未接入为 None）
    road_match_rate: Optional[float] = None

    # 给 LLM 的观测建议（不是决策，决策权在 LLM）
    observations: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        d = {
            "seg_id": self.seg_id,
            "vehicle_id": self.vehicle_id,
            "n_points": self.n_points,
            "duration_s": self.duration_s,
            "regime": self.regime,
            "is_stationary": self.is_stationary,
            "timeline": {
                "quality": self.timeline_quality,
                "dt_median_s": self.dt_median_s,
                "dt_p95_s": self.dt_p95_s,
                "dt_max_s": self.dt_max_s,
                "dt_zero_ratio": self.dt_zero_ratio,
                "dt_zero_burst": self.dt_zero_burst,
                "dt_bimodal": self.dt_bimodal,
            },
            "space": {
                "length_m": self.length_m,
                "displacement_m": self.displacement_m,
                "bbox_span_m": self.bbox_span_m,
                "sinuosity": self.sinuosity,
            },
            "duplicates": {
                "consecutive_dup_ratio": self.consecutive_dup_ratio,
                "dup_run_count": self.dup_run_count,
            },
            "speed": {
                "median_mps": self.speed_median_mps,
                "p95_mps": self.speed_p95_mps,
                "max_mps": self.speed_max_mps,
            },
            "anomalies": {
                "counts": dict(sorted(self.anomaly_counts.items())),
                "n_anomalous_points": self.n_anomalous_points,
            },
            "preserved_behavior": dict(sorted(self.preserved_behavior.items())),
            "road_match_rate": self.road_match_rate,
            "observations": self.observations,
        }
        return d

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def classify_timeline(traj: Traj,
                      unusable_span_s: float = 30.0,
                      degraded_zero_ratio: float = 0.05) -> str:
    """判定时间轴质量。

    本项目实测轨迹 352 的 170 个点全部落在 3 秒内（ts 仅 4 个不同值）。
    这类轨迹的坐标序列也许仍然有效，但**任何基于 dt 的量（速度、加速度）
    都失去意义**，不能靠删点来「修复」——删掉 68% 的点只会掩盖问题。
    """
    n = len(traj)
    if n < 2:
        return TIMELINE_UNUSABLE
    dt = traj.time_deltas()
    if any(dt[i] < 0 for i in range(1, n)):
        return TIMELINE_NOT_MONOTONIC
    span = float(traj.timestamps[-1] - traj.timestamps[0])
    if span < unusable_span_s:
        return TIMELINE_UNUSABLE
    zeros = sum(1 for i in range(1, n) if dt[i] == 0)
    if zeros / max(1, n - 1) > degraded_zero_ratio:
        return TIMELINE_DEGRADED
    return TIMELINE_OK


def classify_regime(traj: Traj, stationary_length_m: float = 200.0) -> str:
    """静止 / 行驶 / 混合。

    用「位移 + 运动点占比」双判据：只看长度会把「停车 19 分钟然后开走」
    误判为静止；只看速度会在时间轴损坏时失效。
    """
    n = len(traj)
    if n < 2:
        return "stationary"
    v = traj.speeds_mps()
    moving_ratio = sum(1 for x in v if x > 1.0) / n
    length = traj.length_m
    if length < stationary_length_m and moving_ratio < 0.1:
        return "stationary"
    if moving_ratio > 0.6:
        return "moving"
    return "mixed"


def diagnose(traj: Traj,
             rules: Optional[RuleParams] = None,
             with_anomalies: bool = True) -> DiagnosisCard:
    """生成诊断卡。"""
    r = rules or RuleParams()
    n = len(traj)
    dt = traj.time_deltas()
    positive_dt = [dt[i] for i in range(1, n) if dt[i] > 0]
    v_all = traj.speeds_mps()
    v_pos = [v for v in v_all if v > 0]
    dup = sum(1 for i in range(1, n) if traj.coords[i] == traj.coords[i - 1])
    zeros = sum(1 for i in range(1, n) if dt[i] == 0)
    zero_burst = 0
    run = 0
    for i in range(1, n):
        if dt[i] == 0:
            run += 1
            zero_burst = max(zero_burst, run)
        else:
            run = 0

    min_lon, min_lat, max_lon, max_lat = traj.bbox
    span = 0.0
    if n:
        span = geo.local_distance_m((min_lon, min_lat), (max_lon, max_lat))

    card = DiagnosisCard(
        seg_id=traj.seg_id,
        vehicle_id=traj.vehicle_id,
        n_points=n,
        duration_s=round(traj.duration_s, 1),
        regime=classify_regime(traj),
        is_stationary=traj.is_stationary,
        dt_median_s=round(geo.median(positive_dt), 2) if positive_dt else 0.0,
        dt_p95_s=round(geo.quantile(positive_dt, 0.95), 2) if positive_dt else 0.0,
        dt_max_s=round(max(positive_dt), 2) if positive_dt else 0.0,
        dt_zero_ratio=round(zeros / max(1, n - 1), 4),
        dt_zero_burst=zero_burst,
        dt_bimodal=geo.is_bimodal(positive_dt) if positive_dt else False,
        timeline_quality=classify_timeline(traj),
        length_m=round(traj.length_m, 2),
        displacement_m=round(geo.local_distance_m(traj.coords[0], traj.coords[-1]), 2) if n else 0.0,
        bbox_span_m=round(span, 2),
        sinuosity=round(geo.sinuosity(traj.coords), 4),
        consecutive_dup_ratio=round(dup / max(1, n - 1), 4),
        speed_median_mps=round(geo.median(v_pos), 2) if v_pos else 0.0,
        speed_p95_mps=round(geo.quantile(v_pos, 0.95), 2) if v_pos else 0.0,
        speed_max_mps=round(max(v_pos), 2) if v_pos else 0.0,
    )

    if with_anomalies:
        res = anomalies.detect_anomalies(traj, r)
        card.anomaly_counts = res.counts()
        card.n_anomalous_points = int(sum(1 for rs in res.reasons if rs))
        card.dup_run_count = len(res.window_hits.get("duplicate_runs", []))
        card.preserved_behavior = {
            "u_turn": len(res.window_hits.get("u_turns", [])),
            anomalies.REASON_SPEED_OVER_LIMIT: len(
                res.flags.get(anomalies.REASON_SPEED_OVER_LIMIT, [])),
        }

    card.observations = _observations(card)
    return card


def _observations(card: DiagnosisCard) -> Dict[str, object]:
    """把数字翻成「值得注意的事实」，供 LLM 决策时引用。

    刻意只陈述观测与风险，不给结论参数——参数由 LLM 提，由 verifier 判。
    """
    obs: Dict[str, object] = {}
    t = card.timeline_quality
    if t == TIMELINE_UNUSABLE:
        obs["timeline"] = (
            f"时间跨度仅 {card.duration_s}s，时间轴不可用："
            "速度、加速度类指标全部无效，应改用纯几何规则并跳过速度阈值调参。"
        )
    elif t == TIMELINE_NOT_MONOTONIC:
        obs["timeline"] = "存在负的时间间隔，点顺序不可信，需先按时间排序或剔除。"
    elif t == TIMELINE_DEGRADED:
        if card.dt_zero_burst >= 10:
            obs["timeline"] = (
                f"重复时间戳占比 {card.dt_zero_ratio:.1%}，且最长连续 {card.dt_zero_burst} 个点"
                "落在同一时刻（突发式）：这是采集端时间戳故障，"
                "删点会损失大量有效坐标，应先修时间戳或整段隔离。"
            )
        else:
            obs["timeline"] = (
                f"重复时间戳占比 {card.dt_zero_ratio:.1%}（散布式），速度会被虚假放大，"
                "建议先修正时间戳再评估速度类规则。"
            )

    if card.is_stationary:
        obs["regime"] = (
            f"静止轨迹（位移 {card.displacement_m:.0f}m）："
            "重复点占比高属正常，勿按行驶轨迹的阈值处理。"
        )
    elif card.dt_bimodal:
        obs["regime"] = (
            f"时间间隔呈双峰（中位 {card.dt_median_s}s，p95 {card.dt_p95_s}s）："
            "全局 3×中位数 的切分规则会误切正常采样，建议按 p95 设定。"
        )

    if card.consecutive_dup_ratio > 0.5:
        obs["duplicates"] = (
            f"连续重复点占 {card.consecutive_dup_ratio:.1%}（{card.dup_run_count} 段）："
            "折叠重复点是收益最高的一步，应先做它再压缩。"
        )

    if card.speed_max_mps > 50:
        obs["speed"] = (
            f"最大速度 {card.speed_max_mps:.0f} m/s 明显超出城市物理上限："
            "优先怀疑时间戳伪影而非坐标错误。"
        )

    if card.anomaly_counts.get(anomalies.REASON_DRIFT, 0) > 0.3 * card.n_points:
        obs["drift"] = (
            f"漂移点数 {card.anomaly_counts[anomalies.REASON_DRIFT]} 占比过高，"
            "可能是阈值未做尺度归一化的误判。"
        )
    return obs


def diagnose_many(trajs: Sequence[Traj],
                  rules: Optional[RuleParams] = None) -> List[DiagnosisCard]:
    return [diagnose(t, rules) for t in trajs]


def dataset_regime_summary(cards: Sequence[DiagnosisCard]) -> Dict[str, object]:
    """数据集级 regime 汇总，用于分层抽样与成本估算。"""
    regimes: Dict[str, int] = {}
    timelines: Dict[str, int] = {}
    for c in cards:
        regimes[c.regime] = regimes.get(c.regime, 0) + 1
        timelines[c.timeline_quality] = timelines.get(c.timeline_quality, 0) + 1
    return {
        "n_cards": len(cards),
        "regimes": dict(sorted(regimes.items(), key=lambda kv: -kv[1])),
        "timelines": dict(sorted(timelines.items(), key=lambda kv: -kv[1])),
    }
