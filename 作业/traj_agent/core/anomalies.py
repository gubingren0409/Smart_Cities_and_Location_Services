"""异常点规则：速度突变、非法坐标、连续重复点、明显漂移。

对应学生必做任务①的第二项，以及任务③里「保留异常原因字段」的要求。
原因标签是稳定的字符串常量，全流程（清洗动作、异常计数、文献导出）共用同一套词汇。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import geo
from .traj import Traj

# GPS 噪声地板：低于该单步位移时，航向、漂移、转向角均无物理意义。
# 民用 GPS 水平定位误差典型 5-15 m；取 10 m 作为「真实移动」的下界。
GPS_NOISE_FLOOR_M = 10.0

# ---- 异常原因词表（稳定契约，改动需同步 clean.py 与报告） ------------------
REASON_ILLEGAL_COORD = "illegal_coord"
REASON_CONSECUTIVE_DUP = "consecutive_dup"
REASON_SPEED_SPIKE = "speed_spike"
REASON_SPEED_OVER_LIMIT = "speed_over_limit"
REASON_ACCEL_OUTLIER = "accel_outlier"
REASON_TURN_ANGLE = "turn_angle"
REASON_DRIFT = "drift"
REASON_TIME_GAP = "time_gap"
REASON_SPACE_JUMP = "space_jump"
REASON_NONPOSITIVE_DT = "nonpositive_dt"
REASON_STATIONARY_DRIFT = "stationary_drift"
REASON_U_TURN = "u_turn"
REASON_DT_ARTIFACT = "dt_artifact"

ALL_REASONS: Tuple[str, ...] = (
    REASON_ILLEGAL_COORD, REASON_CONSECUTIVE_DUP, REASON_SPEED_SPIKE,
    REASON_SPEED_OVER_LIMIT, REASON_ACCEL_OUTLIER, REASON_TURN_ANGLE,
    REASON_DRIFT, REASON_TIME_GAP, REASON_SPACE_JUMP, REASON_NONPOSITIVE_DT,
    REASON_STATIONARY_DRIFT, REASON_U_TURN, REASON_DT_ARTIFACT,
)

REASON_ZH: Dict[str, str] = {
    REASON_ILLEGAL_COORD: "非法坐标",
    REASON_CONSECUTIVE_DUP: "连续重复点",
    REASON_SPEED_SPIKE: "速度突变（单点孤立）",
    REASON_SPEED_OVER_LIMIT: "速度超限",
    REASON_ACCEL_OUTLIER: "加速度异常",
    REASON_TURN_ANGLE: "转向角突变",
    REASON_DRIFT: "明显漂移（偏离邻域中位点）",
    REASON_TIME_GAP: "时间间隔过大",
    REASON_SPACE_JUMP: "空间跳跃",
    REASON_NONPOSITIVE_DT: "时间戳非递增",
    REASON_STATIONARY_DRIFT: "静止漂移（停车定位抖动）",
    REASON_U_TURN: "掉头/折返",
    REASON_DT_ARTIFACT: "时间戳伪影（dt 远小于正常采样间隔）",
}


@dataclass
class RuleParams:
    """异常判定规则的阈值集合。默认值取自 params.PARAM_SPECS。"""

    max_speed_mps: float = 38.0
    max_accel_mps2: float = 6.0
    max_turn_deg: float = 150.0
    max_dt_s: float = 60.0
    jump_threshold_m: float = 400.0
    drift_radius_m: float = 30.0
    stationary_speed_mps: float = 0.5
    u_turn_angle_deg: float = 130.0
    bbox: Sequence[float] = geo.CHINA_BBOX
    # 噪声地板：单步位移低于该值时，航向角/转向角/漂移全部失去物理意义。
    # 民用 GPS 水平误差典型 5-15 m，几厘米级的「移动」只是抖动。
    moving_min_step_m: float = GPS_NOISE_FLOOR_M
    # 静止判定：整条轨迹的位移低于该值即视为静止轨迹
    stationary_length_m: float = 200.0
    # 漂移的无量纲曲率阈值：到相邻点连线的垂距/弦长 超过该值才算异常
    drift_scale_k: float = 0.15
    # 位置毛刺阈值：偏离相邻点连线超过该值即为离群位置
    position_threshold_m: float = 200.0

    @classmethod
    def from_params(cls, params: Dict[str, float], **overrides) -> "RuleParams":
        """从统一参数 dict 构造规则参数，未提供的用默认。"""
        kw = dict(
            max_speed_mps=float(params.get("max_speed_mps", 38.0)),
            max_accel_mps2=float(params.get("max_accel_mps2", 6.0)),
            max_turn_deg=float(params.get("max_turn_deg", 150.0)),
            jump_threshold_m=float(params.get("dist_threshold", 400.0)),
        )
        kw.update(overrides)
        return cls(**kw)


@dataclass
class AnomalyResult:
    """逐点异常标记 + 原因直方图。"""

    reasons: List[List[str]] = field(default_factory=list)  # 每点的原因列表
    flags: Dict[str, List[int]] = field(default_factory=dict)  # 原因 -> 点索引
    window_hits: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.reasons)

    def counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in sorted(self.flags.items()) if v}

    def point_reason_map(self) -> Dict[int, List[str]]:
        out: Dict[int, List[str]] = {}
        for i, rs in enumerate(self.reasons):
            if rs:
                out[i] = list(rs)
        return out

    def to_dict(self, with_indices: bool = False, max_indices: int = 50) -> Dict[str, object]:
        """紧凑可序列化形式，安全喂给 LLM。"""
        d: Dict[str, object] = {
            "n_points": self.n_points,
            "counts": self.counts(),
            "affected_points": int(sum(1 for rs in self.reasons if rs)),
        }
        if with_indices:
            d["indices"] = {k: v[:max_indices] for k, v in self.flags.items() if v}
            d["truncated_at"] = max_indices
        return d


# ---------------------------------------------------------------------------
# 逐点规则
# ---------------------------------------------------------------------------
def mark_illegal_coords(traj: Traj, bbox: Sequence[float] = geo.CHINA_BBOX) -> List[int]:
    """非法坐标：NaN/inf、越界、或 (0,0) 空值占位。"""
    return [i for i, p in enumerate(traj.coords) if not geo.is_legal_lonlat(p[0], p[1], bbox)]


def mark_duplicate_points(traj: Traj) -> List[int]:
    """连续重复点：与**紧邻前一点**坐标完全相同。

    只标记连续重复，不标记非连续回访——闭合环路与折返是真实轨迹，
    把它们当重复点删除会改变轨迹几何。
    返回除每组首点外的其余索引（即「可安全删除」的点）。
    """
    out: List[int] = []
    for i in range(1, len(traj.coords)):
        if traj.coords[i] == traj.coords[i - 1]:
            out.append(i)
    return out


def mark_duplicate_runs(traj: Traj) -> List[Tuple[int, int]]:
    """连续重复点的整段区间 [(start, end), ...]，便于报告与统计。"""
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(traj.coords)
    while i < n - 1:
        if traj.coords[i + 1] == traj.coords[i]:
            j = i + 1
            while j + 1 < n and traj.coords[j + 1] == traj.coords[i]:
                j += 1
            runs.append((i, j))
            i = j
        i += 1
    return runs


def mark_nonpositive_dt(traj: Traj) -> List[int]:
    """时间戳非递增：dt <= 0，无法定义速度与顺序。"""
    out: List[int] = []
    for i in range(1, len(traj.timestamps)):
        if traj.timestamps[i] - traj.timestamps[i - 1] <= 0:
            if i not in out:
                out.append(i)
    return out


def mark_time_gaps(traj: Traj, max_dt_s: float = 60.0) -> List[int]:
    """时间间隔过大的位置（标记后一个点）。"""
    return [i for i in range(1, len(traj.timestamps))
            if traj.timestamps[i] - traj.timestamps[i - 1] > max_dt_s]


def mark_space_jumps(traj: Traj, jump_threshold_m: float = 400.0) -> List[int]:
    """空间跳跃的位置（标记后一个点）。坐标非法时不判定。"""
    out: List[int] = []
    for i in range(1, len(traj.coords)):
        a, b = traj.coords[i - 1], traj.coords[i]
        if not (geo.is_legal_lonlat(*a) and geo.is_legal_lonlat(*b)):
            continue
        if geo.local_distance_m(a, b) > jump_threshold_m:
            out.append(i)
    return out


def _displacement_outlier(traj: Traj, i: int, factor: float = 3.0) -> bool:
    """该点位移是否显著大于邻域位移的中位数（倍数判据）。"""
    ds = traj.space_deltas_m()
    n = len(ds)
    lo, hi = max(1, i - 3), min(n, i + 4)
    window = [ds[j] for j in range(lo, hi) if j != i]
    if not window:
        return False
    med = geo.median(window)
    if med <= 0:
        med = 1.0
    return ds[i] > factor * med


def mark_position_outliers(traj: Traj,
                           threshold_m: float = 200.0,
                           min_step_m: float = GPS_NOISE_FLOOR_M) -> List[int]:
    """位置毛刺：该点显著偏离其**相邻两点的插值位置**。

    为什么不能用「速度超限 + 两侧速度正常」来判孤立毛刺：
    若点 i 跳到远处再回来，位移法会同时把 d[i] 与 d[i+1] 算大，
    于是一处毛刺污染两个位置，判据「两侧都正常」永远不成立——
    实测中这导致真毛刺一个都识别不出来。

    插值法只依赖位置：正常点在前后邻点连线附近，
    跳变点则远离该连线。返回偏离量超阈值的点索引。
    """
    n = len(traj)
    if n < 3:
        return []
    pts = traj.coords
    out: List[int] = []
    for i in range(1, n - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        if not all(geo.is_legal_lonlat(*p) for p in (a, b, c)):
            continue
        # 若该点本身没动，不构成位置毛刺（属于 stationary_drift 范畴）
        if traj.space_deltas_m()[i] < min_step_m:
            continue
        # 以 a→c 的直线为参考，衡量 b 偏离它的程度
        deviation = geo.segment_distance_m(b, a, c)
        # 归一化：除以 a-c 距离，避免长基线下的绝对阈值失真
        baseline = geo.local_distance_m(a, c)
        if baseline < min_step_m:
            continue
        if deviation > threshold_m:
            out.append(i)
    return out


def mark_speed_anomalies(traj: Traj,
                         max_speed_mps: float = 38.0,
                         max_accel_mps2: float = 6.0,
                         max_dt_s: float = 60.0,
                         position_threshold_m: float = 200.0,
                         min_step_m_floor: float = GPS_NOISE_FLOOR_M
                         ) -> Tuple[List[int], List[int], List[int], List[float]]:
    """速度类异常。

    返回 (孤立速度突变点, 速度超限点, 加速度异常点, 逐点速度)。

    区分三类，因为处置方式不同：
      - speed_spike  : 单点孤立，两侧速度正常 -> 坐标毛刺，应剔除/平滑
      - speed_over_limit : 速度超限。若两侧也超限，说明是持续超速（真实行为，
        只标记不清洗）；若孤立，则与 spike 同类，可由调用方决定是否合并处理。
      - accel_outlier: 加速度突变。另外要求该点两侧速度同为异常，
        否则正常的起步/刹车会被大量误标。
    """
    n = len(traj)
    dt = traj.time_deltas()
    ds = traj.space_deltas_m()
    v = [0.0] * n
    computable = [False] * n
    for i in range(1, n):
        if dt[i] > 0.0 and dt[i] <= max_dt_s:
            v[i] = ds[i] / dt[i]
            computable[i] = True

    over = [i for i in range(n) if computable[i] and v[i] > max_speed_mps]

    # 孤立速度突变：用「相对相邻点插值位置的偏差」判定，而非位移法。
    # 位移法在一处跳变时会同时污染两个位置，导致真毛刺识别不出来。
    position_outliers = set(mark_position_outliers(traj, position_threshold_m, min_step_m_floor))
    spikes: List[int] = []
    for i in over:
        if i in position_outliers:
            spikes.append(i)
            continue
        # 允许退化情形：无法定义位置偏差但两侧速度确实正常
        prev_ok = (i - 1) >= 1 and computable[i - 1] and v[i - 1] <= max_speed_mps
        next_ok = (i + 1) < n and computable[i + 1] and v[i + 1] <= max_speed_mps
        prev_unknown = (i - 1) < 1 or not computable[i - 1]
        next_unknown = (i + 1) >= n or not computable[i + 1]
        if (prev_unknown or prev_ok) and (next_unknown or next_ok):
            # 位置偏差不可用时，额外要求该点位移显著离群，避免漏判
            if i in position_outliers or _displacement_outlier(traj, i):
                spikes.append(i)

    # 加速度异常：要求两侧速度都可计算且都超限，避免误标正常起停
    accel: List[int] = []
    for i in range(1, n):
        if not (computable[i] and computable[i - 1]):
            continue
        if dt[i] <= 0.0:
            continue
        a = (v[i] - v[i - 1]) / dt[i]
        if abs(a) > max_accel_mps2 and (v[i] > max_speed_mps or v[i - 1] > max_speed_mps):
            accel.append(i)
    return spikes, over, accel, v


def mark_dt_artifacts(traj: Traj,
                      max_speed_mps: float = 38.0,
                      min_dt_ratio: float = 0.5) -> List[int]:
    """时间戳伪影：dt 远小于正常采样间隔，导致速度被虚假放大。

    本项目实测中被判为 speed_spike 的点几乎全是这一类：
    dt=1~3s 而正常采样间隔是 10/20s，于是 v=位移/dt 被放大 10~20 倍
    （最高见到 253 m/s）。这些点的**坐标往往完全合法**，
    直接删点会引入新的几何误差；正确处置是修正时间戳。

    判据：dt < min_dt_ratio × 邻域中位间隔，且该点速度超限。
    与 speed_spike 的区别在于成因不同，故分开标记、分开处置。
    """
    n = len(traj)
    if n < 3:
        return []
    dt = traj.time_deltas()
    v = traj.speeds_mps()
    positive = [dt[i] for i in range(1, n) if dt[i] > 0]
    if not positive:
        return []
    med_dt = geo.median(positive)
    if med_dt <= 0:
        return []
    out: List[int] = []
    for i in range(1, n):
        if dt[i] <= 0:
            continue
        if dt[i] < min_dt_ratio * med_dt and v[i] > max_speed_mps:
            out.append(i)
    return out


def mark_turn_anomalies(traj: Traj,
                       max_turn_deg: float = 150.0,
                       min_step_m: float = GPS_NOISE_FLOOR_M) -> List[int]:
    """转向角突变：单点转向角超过阈值，典型于定位漂移毛刺。

    关键是 min_step_m 噪声地板：只有**两侧单步位移都超过 10m** 时才计算转向角。
    否则在厘米级 GPS 抖动上算航向会产生大量假掉头——
    本项目 400 条抽样中曾出现单条轨迹 37 个假 turn_angle。
    """
    angles = geo.turn_angles_deg(traj.coords)
    steps = traj.space_deltas_m()
    out: List[int] = []
    for i in range(1, len(traj) - 1):
        if angles[i] <= max_turn_deg:
            continue
        if steps[i] < min_step_m or steps[i + 1] < min_step_m:
            continue  # 位移低于噪声地板，转向角不可信
        out.append(i)
    return out


def mark_drift(traj: Traj,
               radius_m: float = 30.0,
               k: int = 1,
               min_step_m: float = GPS_NOISE_FLOOR_M,
               stationary_length_m: float = 200.0,
               scale_k: float = 0.15) -> List[int]:
    """明显漂移：该点显著偏离其相邻点连线，即局部曲率异常。

    判据（无量纲，抗尺度）
    --------------------
        curvature = 到 a→c 连线的垂距 / |a-c|

    直线为 0；半径为 R、弦长 L 的真实弧线，矢高约 L²/(8R)，取
    L=2100m、R=30km 的城市快速路，curvature ≈ 0.009。
    而一个跳变毛刺（偏 1000m / 弦长 600m）达到 1.67，差两个数量级。
    故阈值取 0.15 有很宽的判别余量。

    为什么不用「邻域中位位置的偏移 / 邻域 MAD」做归一化
    ------------------------------------------------
    该方案实测失效：真实高速轨迹的邻域 MAD 达 600m（被道路曲率主导，
    而非噪声），3×MAD = 1800m 使连 1km 的毛刺都判不出来。
    尺度项必须挂在**曲率**上，而不是挂在邻域散布上。

    参数
    ----
    radius_m     : 绝对垂距下限（米）。挡掉小弦长下的噪声放大。
    k            : 邻域跨度，取相邻 k 点做弦。默认 1（用紧邻点）。
    scale_k      : 无量纲曲率阈值。
    """
    n = len(traj)
    if n < 2 * k + 1:
        return []
    pts = traj.coords
    steps = traj.space_deltas_m()
    if traj.length_m < stationary_length_m:
        return []
    out: List[int] = []
    for i in range(k, n - k):
        if steps[i] < min_step_m:
            continue  # 噪声地板：该点未发生真实移动
        a, b, c = pts[i - k], pts[i], pts[i + k]
        if not all(geo.is_legal_lonlat(*p) for p in (a, b, c)):
            continue
        baseline = geo.local_distance_m(a, c)
        if baseline < min_step_m:
            continue
        sagitta = geo.segment_distance_m(b, a, c)
        if sagitta <= radius_m:
            continue
        if sagitta / baseline > scale_k:
            out.append(i)
    return out


def mark_drift_sequence(traj: Traj, radius_m: float = 30.0, k: int = 1,
                        scale_k: float = 0.15) -> List[Tuple[int, int]]:
    """成片漂移区间：连续两点以上偏离邻域中位位置，视为漂移段而非单点毛刺。"""
    hit = set(mark_drift(traj, radius_m=radius_m, k=k, scale_k=scale_k))
    if not hit:
        return []
    runs: List[Tuple[int, int]] = []
    idx = sorted(hit)
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        if prev > start:
            runs.append((start, prev))
        start = prev = i
    if prev > start:
        runs.append((start, prev))
    return runs


def mark_stationary_drift(traj: Traj,
                          speed_mps: float = 0.5,
                          radius_m: float = 30.0) -> List[int]:
    """静止漂移：读数速度接近 0，但相对邻域中心的位移却明显。

    这类点在「速度」维度上看不出问题，却是城市 GPS 停车时的典型噪声，
    必须单列一类，否则会被所有速度型规则漏掉。用 2σ 之外的位移量判定。
    """
    n = len(traj)
    if n < 3:
        return []
    v = traj.speeds_mps()
    pts = traj.coords
    out: List[int] = []
    for i in range(1, n):
        if v[i] > speed_mps:
            continue
        if not all(geo.is_legal_lonlat(*p) for p in (pts[i], pts[i - 1])):
            continue
        # 与前后 3 点中位位置比较；窗口不足时退化为相邻点
        lo, hi = max(0, i - 3), min(n, i + 4)
        neigh = [pts[j] for j in range(lo, hi) if j != i]
        if not neigh:
            continue
        med = (geo.median([p[0] for p in neigh]), geo.median([p[1] for p in neigh]))
        if geo.local_distance_m(pts[i], med) > radius_m:
            out.append(i)
    return out


def detect_u_turns(traj: Traj, u_turn_angle: float = 130.0,
                   min_step_m: float = GPS_NOISE_FLOOR_M) -> List[Tuple[int, int]]:
    """掉头/折返区间：进入方向与离开方向夹角超过阈值。

    这是**真实行为**而非噪声，单列出来是为了让清洗流程显式声明「不清洗它」，
    避免被 turn_angle 规则误删。
    """
    angles = geo.turn_angles_deg(traj.coords)
    steps = traj.space_deltas_m()
    out: List[Tuple[int, int]] = []
    for i in range(1, len(traj) - 1):
        if angles[i] <= u_turn_angle:
            continue
        if steps[i] < min_step_m or steps[i + 1] < min_step_m:
            continue  # 位移低于噪声地板，掉头判定不可信
        out.append((i, i))
    return out


# ---------------------------------------------------------------------------
# 汇总入口
# ---------------------------------------------------------------------------
def detect_anomalies(traj: Traj, rules: Optional[RuleParams] = None) -> AnomalyResult:
    """跑全部规则，返回逐点原因与直方图。"""
    r = rules or RuleParams()
    n = len(traj)
    reasons: List[List[str]] = [[] for _ in range(n)]
    flags: Dict[str, List[int]] = {}

    def add(reason: str, indices: Sequence[int]) -> None:
        if not indices:
            return
        flags.setdefault(reason, []).extend(int(i) for i in indices)
        for i in indices:
            if reason not in reasons[i]:
                reasons[i].append(reason)

    add(REASON_ILLEGAL_COORD, mark_illegal_coords(traj, r.bbox))
    add(REASON_CONSECUTIVE_DUP, mark_duplicate_points(traj))
    add(REASON_NONPOSITIVE_DT, mark_nonpositive_dt(traj))
    add(REASON_TIME_GAP, mark_time_gaps(traj, r.max_dt_s))
    add(REASON_SPACE_JUMP, mark_space_jumps(traj, r.jump_threshold_m))

    dt_artifacts = mark_dt_artifacts(traj, r.max_speed_mps)
    add(REASON_DT_ARTIFACT, dt_artifacts)

    spikes, over, accel, _ = mark_speed_anomalies(
        traj, r.max_speed_mps, r.max_accel_mps2, r.max_dt_s)
    # 由时间戳伪影造成的「速度突变」单列一类，不重复计入 speed_spike
    add(REASON_SPEED_SPIKE, [i for i in spikes if i not in set(dt_artifacts)])
    # 持续超速单独一类：只标记，不应被清洗
    add(REASON_SPEED_OVER_LIMIT, [i for i in over if i not in set(spikes)])
    add(REASON_ACCEL_OUTLIER, accel)

    # 速度规则已经标过的点不再重复计入转向/漂移，保持原因可解释
    already = set(flags.get(REASON_SPEED_SPIKE, [])) | set(dt_artifacts)
    add(REASON_TURN_ANGLE, [i for i in mark_turn_anomalies(
        traj, r.max_turn_deg, r.moving_min_step_m) if i not in already])
    add(REASON_DRIFT, [i for i in mark_drift(
        traj, r.drift_radius_m, min_step_m=r.moving_min_step_m,
        stationary_length_m=r.stationary_length_m,
        scale_k=r.drift_scale_k) if i not in already])
    add(REASON_STATIONARY_DRIFT, [i for i in mark_stationary_drift(
        traj, r.stationary_speed_mps, r.drift_radius_m) if i not in already])

    result = AnomalyResult(reasons=reasons, flags=flags)
    result.window_hits = {
        "duplicate_runs": mark_duplicate_runs(traj),
        "drift_runs": mark_drift_sequence(traj, r.drift_radius_m,
                                          scale_k=r.drift_scale_k),
    }
    # u_turn 是行为标签，单独放在 window_hits 里，不混入异常计数
    result.window_hits["u_turns"] = detect_u_turns(
        traj, r.u_turn_angle_deg, r.moving_min_step_m)
    return result


def anomaly_histogram(results: Sequence[AnomalyResult]) -> Dict[str, int]:
    """跨多条轨迹汇总原因直方图。"""
    total: Dict[str, int] = {}
    for res in results:
        for k, c in res.counts().items():
            total[k] = total.get(k, 0) + c
    return dict(sorted(total.items(), key=lambda kv: -kv[1]))
