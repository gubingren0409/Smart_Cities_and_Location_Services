"""去噪：把异常点规则的结果转成实际动作，并保留原因字段。

对应学生必做任务①的「完成去噪」与「保留异常原因字段」。

设计要点
--------
1. **先分段再清洗**。整条原始轨迹往往混合了静止与行驶两种 regime，
   在混合轨迹上算速度/漂移会两边都不准。故 clean 的入口是单个分段。
2. **迭代收敛**。删点会改变邻居的速度与位移，故删除与平滑交替进行，
   直到没有新的可删除点或达到 max_iter。
3. **坐标合法但时间戳伪造的点不删**。dt_artifact 由修时间戳处置（见 fix_dt_artifacts）。
4. **不清洗行为**。掉头、持续超速是真实行为，只标记不删除。
   这条边界必须显式声明的理由是：把真实行为当噪声删掉是不可逆的信息损失。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import anomalies, geo
from .anomalies import RuleParams
from .traj import Traj

Action = str

# 清洗动作标签（与 anomalies 的原因词表对齐）
ACTION_DROP_DUPLICATE = "drop_duplicate"
ACTION_DROP_ILLEGAL = "drop_illegal"
ACTION_DROP_NONPOSITIVE_DT = "drop_nonpositive_dt"
ACTION_DROP_SPEED_SPIKE = "drop_speed_spike"
ACTION_DROP_JUMP_ARTIFACT = "drop_jump_artifact"
ACTION_FIX_DT_ARTIFACT = "fix_dt_artifact"
ACTION_MEDIAN_SMOOTH = "median_smooth"
ACTION_FILL_SMALL_GAP = "fill_small_gap"


@dataclass
class CleanConfig:
    """清洗流程开关与参数。"""

    rules: RuleParams = field(default_factory=RuleParams)
    do_drop_duplicates: bool = True
    do_drop_illegal: bool = True
    do_fix_dt_artifacts: bool = True
    do_drop_spikes: bool = True
    do_drop_jump_artifacts: bool = True
    do_median_smooth: bool = True
    smooth_window: int = 5
    max_iter: int = 3
    # 平滑后位移仍需超限才保留；低于此值的平滑结果视为无改善
    preserve_behavior: bool = True

    @classmethod
    def from_params(cls, params: Dict[str, float]) -> "CleanConfig":
        return cls(
            rules=RuleParams.from_params(params),
            smooth_window=int(params.get("smooth_window", 5)),
        )


@dataclass
class CleanReport:
    """清洗账本：每个动作删/改了多少点，以及原因构成。"""

    n_before: int = 0
    n_after: int = 0
    actions: Dict[str, int] = field(default_factory=dict)
    # 原因 → 出现次数。注意一个点可能同时带多个原因（如既有 space_jump
    # 又有 dt_artifact），故 sum(dropped_reasons.values()) 通常**大于**
    # 实际删除点数；要校验删除总数请用 dropped_with_reason。
    dropped_reasons: Dict[str, int] = field(default_factory=dict)
    # 至少带一个原因的被删点数。与 n_before - n_after 恒等。
    dropped_with_reason: int = 0
    n_iterations: int = 0
    preserved_behavior: Dict[str, int] = field(default_factory=dict)

    @property
    def drop_ratio(self) -> float:
        if self.n_before <= 0:
            return 0.0
        return (self.n_before - self.n_after) / self.n_before

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_before": self.n_before,
            "n_after": self.n_after,
            "dropped": self.n_before - self.n_after,
            "drop_ratio": round(self.drop_ratio, 4),
            "actions": dict(sorted(self.actions.items())),
            "dropped_reasons": dict(sorted(self.dropped_reasons.items())),
            "dropped_with_reason": self.dropped_with_reason,
            "preserved_behavior": dict(sorted(self.preserved_behavior.items())),
            "n_iterations": self.n_iterations,
        }

    def merge(self, other: "CleanReport") -> None:
        self.n_before += other.n_before
        self.n_after += other.n_after
        for k, v in other.actions.items():
            self.actions[k] = self.actions.get(k, 0) + v
        for k, v in other.dropped_reasons.items():
            self.dropped_reasons[k] = self.dropped_reasons.get(k, 0) + v
        self.dropped_with_reason += other.dropped_with_reason
        for k, v in other.preserved_behavior.items():
            self.preserved_behavior[k] = self.preserved_behavior.get(k, 0) + v
        self.n_iterations += other.n_iterations


def _point_keys(traj: Traj) -> List[Tuple[int, float, float]]:
    """点的身份标识 (时间戳, 经度, 纬度)，用于删除前后差分。"""
    return [(int(t), float(p[0]), float(p[1]))
            for t, p in zip(traj.timestamps, traj.coords)]


def _diff_dropped(before: Traj, after: Traj) -> List[int]:
    """返回被删除的点在 before 中的索引。

    通过对身份标识做多重集合差分，能正确处理重复点
    （多次出现的同一 (t, lon, lat) 会被逐次抵销）。
    """
    from collections import Counter
    cnt_before = Counter(_point_keys(before))
    cnt_after = Counter(_point_keys(after))
    remaining = cnt_before - cnt_after
    total = sum(remaining.values())
    if total == 0:
        return []
    dropped: List[int] = []
    for i, key in enumerate(_point_keys(before)):
        if remaining.get(key, 0) > 0:
            remaining[key] -= 1
            dropped.append(i)
    return sorted(dropped)


def _select(traj: Traj, keep_mask: Sequence[bool], reasons: List[str]) -> Traj:
    """按掩码保留点，并把动作标签写入 reasons。"""
    idx = [i for i, k in enumerate(keep_mask) if k]
    out = traj.positions(idx)
    out.reasons = [reasons[i] for i in idx]
    return out


def drop_consecutive_duplicates(traj: Traj) -> Tuple[Traj, int]:
    """折叠连续重复点，保留每组的最后一个。

    保留最后一个而非第一个：该组末尾是车辆离开该位置前的最新观测，
    与后续点的连续性更好。
    非连续回访（闭合环路、折返）不在此列，那属于真实几何。
    """
    n = len(traj)
    if n < 2:
        return traj, 0
    keep = [True] * n
    dropped = 0
    for i in range(1, n):
        if traj.coords[i] == traj.coords[i - 1] and traj.coords[i] == traj.coords[i - 1]:
            # 同一组内：保留最后一个
            if keep[i - 1] and traj.coords[i] == traj.coords[i - 1]:
                keep[i - 1] = False
                dropped += 1
    reasons = list(traj.reasons)
    for i in range(n):
        if not keep[i]:
            reasons[i] = ACTION_DROP_DUPLICATE
    return _select(traj, keep, reasons), dropped


def drop_illegal_coords(traj: Traj, bbox: Sequence[float] = geo.CHINA_BBOX) -> Tuple[Traj, int]:
    """删除非法坐标点（NaN / 越界 / 空值占位）。"""
    bad = set(anomalies.mark_illegal_coords(traj, bbox))
    if not bad:
        return traj, 0
    reasons = [ACTION_DROP_ILLEGAL if i in bad else r for i, r in enumerate(traj.reasons)]
    keep = [i not in bad for i in range(len(traj))]
    return _select(traj, keep, reasons), len(bad)


def drop_nonpositive_dt(traj: Traj) -> Tuple[Traj, int]:
    """删除时间戳非递增的点。

    这些点的顺序不可信，且使速度无法定义。注意本项目实测中有轨迹
    68% 的点 dt=0，此时该轨迹的时间轴已经不可用。
    """
    bad = set(anomalies.mark_nonpositive_dt(traj))
    if not bad:
        return traj, 0
    reasons = [ACTION_DROP_NONPOSITIVE_DT if i in bad else r
               for i, r in enumerate(traj.reasons)]
    keep = [i not in bad for i in range(len(traj))]
    return _select(traj, keep, reasons), len(bad)


def drop_jump_artifacts(traj: Traj, jump_threshold_m: float = 400.0,
                       max_speed_mps: float = 38.0) -> Tuple[Traj, int]:
    """删除空间跳跃伪影点。

    只删除「既跳跃又速度超限」的点：单纯的空间大跳跃可能是真实的
    长间隔行驶（本项目 max dt 达 1037s），那应由分段处理而不是删点。
    """
    jump = set(anomalies.mark_space_jumps(traj, jump_threshold_m))
    if not jump:
        return traj, 0
    v = traj.speeds_mps()
    bad = {i for i in jump if v[i] > max_speed_mps}
    if not bad:
        return traj, 0
    reasons = [ACTION_DROP_JUMP_ARTIFACT if i in bad else r
               for i, r in enumerate(traj.reasons)]
    keep = [i not in bad for i in range(len(traj))]
    return _select(traj, keep, reasons), len(bad)


def fix_dt_artifacts(traj: Traj, max_speed_mps: float = 38.0,
                     min_dt_ratio: float = 0.5) -> Tuple[Traj, List[int]]:
    """修正时间戳伪影：把异常小的 dt 替换为邻域中位采样间隔。

    这是与「删点」并列的另一种处置。理由：这类点的坐标通常合法，
    问题只在于时间轴，删点会丢失有效位置信息，修时间戳才是对的。
    返回 (新轨迹, 被修正的点索引)。
    """
    bad = anomalies.mark_dt_artifacts(traj, max_speed_mps, min_dt_ratio)
    if not bad:
        return traj, []
    dt = traj.time_deltas()
    positive = [dt[i] for i in range(1, len(dt)) if dt[i] > 0]
    if not positive:
        return traj, []
    med = geo.median(positive)

    ts = [int(t) for t in traj.timestamps]
    fixed: List[int] = []
    # 从后往前改，避免改动影响后续 dt 判断
    for i in sorted(bad, reverse=True):
        if i <= 0 or i >= len(ts):
            continue
        ts[i] = ts[i - 1] + int(round(med))
        fixed.append(i)
        # 后续时间戳需同步平移，否则会出现新的非递增
        for j in range(i + 1, len(ts)):
            if ts[j] <= ts[j - 1]:
                ts[j] = ts[j - 1] + int(round(med))
    reasons = list(traj.reasons)
    for i in fixed:
        reasons[i] = ACTION_FIX_DT_ARTIFACT
    out = Traj(traj.vehicle_id, ts, list(traj.coords), traj.segment_index, reasons)
    return out, sorted(fixed)


def median_smooth(traj: Traj, window: int = 5,
                  apply_to: Optional[Sequence[int]] = None) -> Tuple[Traj, List[int]]:
    """对坐标做滑动中值平滑，返回 (新轨迹, 被修改的点索引)。

    apply_to 给出待平滑的点；None 表示对全部点平滑。
    中值而非均值：中值不被单个离群点拉动，适合脉冲型噪声。

    注意：只对**坐标**平滑，不动时间戳。平滑可能把点移出合法范围，
    故对结果做合法性回退（非法则保留原值）。
    """
    n = len(traj)
    if n < 3:
        return traj, []
    w = max(3, int(window) | 1)  # 保证为 >=3 的奇数
    half = w // 2
    targets = set(range(n)) if apply_to is None else {int(i) for i in apply_to}
    coords = list(traj.coords)
    new_coords = list(coords)
    changed: List[int] = []
    for i in range(n):
        if i not in targets:
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window_pts = [coords[j] for j in range(lo, hi)
                      if geo.is_legal_lonlat(coords[j][0], coords[j][1])]
        if len(window_pts) < 3:
            continue
        lon = geo.median([p[0] for p in window_pts])
        lat = geo.median([p[1] for p in window_pts])
        if not geo.is_legal_lonlat(lon, lat):
            continue
        if abs(lon - coords[i][0]) < 1e-12 and abs(lat - coords[i][1]) < 1e-12:
            continue
        new_coords[i] = (lon, lat)
        changed.append(i)
    if not changed:
        return traj, []
    reasons = list(traj.reasons)
    for i in changed:
        if not reasons[i]:
            reasons[i] = ACTION_MEDIAN_SMOOTH
    out = Traj(traj.vehicle_id, list(traj.timestamps), new_coords,
               traj.segment_index, reasons)
    return out, changed


def denoise_trajectory(traj: Traj, config: Optional[CleanConfig] = None
                       ) -> Tuple[Traj, CleanReport]:
    """完整去噪流程：迭代执行「结构性删除 → 时间戳修正 → 平滑」。

    迭代的原因：删点会改变邻居的 dt 与位移，从而暴露新的异常。
    收敛条件是某一轮没有任何动作，或达到 max_iter。
    """
    cfg = config or CleanConfig()
    report = CleanReport(n_before=len(traj))
    cur = traj
    reasons_seen: Counter = Counter()

    # 原因只记录**确实被删掉**的点。
    # 早期版本在删除前就按预判掩码记录，导致报告里出现
    # 「原因计数 3 但实际只删 2 个点」的口径不一致——判据在修正时间戳后
    # 可能不再命中，预判掩码必然与实际删除集合有出入。
    def record_dropped(before: Traj, after: Traj, det=None) -> int:
        dropped = _diff_dropped(before, after)
        if not dropped:
            return 0
        if det is None:
            det = anomalies.detect_anomalies(before, cfg.rules)
        # 重新计算被删点在 before 中的位置对应的原因
        keys = _point_keys(before)
        after_keys = set(_point_keys(after))
        # 用差分得到的 before 索引直接取原因
        for i in dropped:
            rs = det.reasons[i] if i < len(det.reasons) else []
            if rs:
                for reason in rs:
                    reasons_seen[reason] += 1
            else:
                reasons_seen["unspecified"] += 1
        report.dropped_with_reason += len(dropped)
        return len(dropped)

    def apply_drop(before: Traj, fn, *args, action: str, **kwargs) -> Tuple[Traj, int]:
        """执行一个删除动作，按差分记录真实原因，并累计账本。"""
        nonlocal acted
        after, _ = fn(before, *args, **kwargs)
        n = record_dropped(before, after)
        if n:
            report.actions[action] = report.actions.get(action, 0) + n
            acted = True
        return after, n

    for it in range(max(1, cfg.max_iter)):
        report.n_iterations = it + 1
        acted = False

        if cfg.do_drop_illegal:
            cur, _ = apply_drop(cur, drop_illegal_coords, cfg.rules.bbox,
                                action=ACTION_DROP_ILLEGAL)

        if cfg.do_drop_duplicates:
            cur, _ = apply_drop(cur, drop_consecutive_duplicates,
                                action=ACTION_DROP_DUPLICATE)

        # 时间戳非递增：一律删除（顺序不可信，无法修）
        cur, _ = apply_drop(cur, drop_nonpositive_dt,
                            action=ACTION_DROP_NONPOSITIVE_DT)

        if cfg.do_fix_dt_artifacts:
            cur, fixed = fix_dt_artifacts(cur, cfg.rules.max_speed_mps)
            if fixed:
                report.actions[ACTION_FIX_DT_ARTIFACT] = (
                    report.actions.get(ACTION_FIX_DT_ARTIFACT, 0) + len(fixed))
                acted = True

        if cfg.do_drop_jump_artifacts:
            cur, _ = apply_drop(cur, drop_jump_artifacts,
                                cfg.rules.jump_threshold_m, cfg.rules.max_speed_mps,
                                action=ACTION_DROP_JUMP_ARTIFACT)

        if cfg.do_median_smooth or cfg.do_drop_spikes:
            spikes, _, _, _ = anomalies.mark_speed_anomalies(
                cur, cfg.rules.max_speed_mps, cfg.rules.max_accel_mps2, cfg.rules.max_dt_s)
            targets = sorted(set(spikes))

            if cfg.do_median_smooth and targets:
                cur, changed = median_smooth(cur, cfg.smooth_window, targets)
                if changed:
                    report.actions[ACTION_MEDIAN_SMOOTH] = (
                        report.actions.get(ACTION_MEDIAN_SMOOTH, 0) + len(changed))
                    acted = True

            # 平滑后重新检测；仍判为毛刺的点才删除
            if cfg.do_drop_spikes:
                still = sorted(set(anomalies.mark_speed_anomalies(
                    cur, cfg.rules.max_speed_mps, cfg.rules.max_accel_mps2,
                    cfg.rules.max_dt_s)[0]))
                if still:
                    before = cur
                    still_set = set(still)
                    reasons = [ACTION_DROP_SPEED_SPIKE if i in still_set else r
                               for i, r in enumerate(cur.reasons)]
                    keep = [i not in still_set for i in range(len(cur))]
                    cur = _select(cur, keep, reasons)
                    n = record_dropped(before, cur)
                    if n:
                        report.actions[ACTION_DROP_SPEED_SPIKE] = (
                            report.actions.get(ACTION_DROP_SPEED_SPIKE, 0) + n)
                        acted = True

        if not acted:
            break

    # 记录被显式保留的真实行为，向调用方声明「这些不是漏网之鱼」
    if cfg.preserve_behavior:
        res = anomalies.detect_anomalies(cur, cfg.rules)
        report.preserved_behavior["u_turn"] = len(res.window_hits.get("u_turns", []))
        over = res.flags.get(anomalies.REASON_SPEED_OVER_LIMIT, [])
        report.preserved_behavior[anomalies.REASON_SPEED_OVER_LIMIT] = len(over)

    report.n_after = len(cur)
    report.dropped_reasons = dict(reasons_seen)
    return cur, report
