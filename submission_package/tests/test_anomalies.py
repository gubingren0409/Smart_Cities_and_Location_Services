"""异常检测：每条规则都用手造轨迹断言到精确点索引。"""
from __future__ import annotations

import math

import pytest

from traj_agent.core import anomalies
from traj_agent.core.anomalies import RuleParams
from traj_agent.core.traj import Traj

LON0, LAT0 = 121.4700, 31.2300


def pt(east_m: float, north_m: float):
    """以 (LON0, LAT0) 为原点，按米生成坐标。口径与 core.geo 一致。"""
    return (LON0 + east_m / 95274.27, LAT0 + north_m / 110873.43)


# ---- 非法坐标 -------------------------------------------------------------
def test_illegal_coords_exact_indices(illegal_traj):
    assert anomalies.mark_illegal_coords(illegal_traj) == [2, 5]


def test_illegal_coords_accepts_custom_bbox():
    t = Traj("bb", [0, 10, 20], [(121.47, 31.23), (121.60, 31.40), (121.48, 31.23)])
    assert anomalies.mark_illegal_coords(t) == []
    tight = (121.45, 31.20, 121.50, 31.25)
    assert anomalies.mark_illegal_coords(t, tight) == [1]


# ---- 连续重复点 -----------------------------------------------------------
def test_duplicate_points_exact_indices(dup_traj):
    """索引 1,2 与 0 相同；4 与 3 相同。保留每段最后一个，故可删 {1,2,4}。"""
    assert anomalies.mark_duplicate_points(dup_traj) == [1, 2, 4]


def test_duplicate_runs_span_exact(dup_traj):
    assert anomalies.mark_duplicate_runs(dup_traj) == [(0, 2), (3, 4)]


def test_nonconsecutive_revisit_is_not_duplicate():
    """闭合环路/折返是真实几何，不能被当作重复点。"""
    t = Traj("loop", [0, 10, 20, 30],
             [(121.4700, 31.2300), (121.4750, 31.2300),
              (121.4750, 31.2350), (121.4700, 31.2300)])
    assert anomalies.mark_duplicate_points(t) == []


def test_duplicate_detection_on_stationary_real_data(real_raw):
    """真实静止轨迹（车辆 0）应有 78 个连续重复点。"""
    from traj_agent.core import traj as traj_mod
    t = traj_mod.traj_from_raw("0", *real_raw["0"])
    assert len(anomalies.mark_duplicate_points(t)) == 78


# ---- 时间 ----------------------------------------------------------------
def test_nonpositive_dt_exact_indices(nonmonotonic_traj):
    """ts = [0,10,20,20,15,30,40] -> dt 为 [_,10,10,0,-5,15,10]，
    非递增位置是索引 3 (dt=0) 与 4 (dt=-5)。"""
    assert anomalies.mark_nonpositive_dt(nonmonotonic_traj) == [3, 4]


def test_time_gaps_exact_index(timegap_traj):
    assert anomalies.mark_time_gaps(timegap_traj, max_dt_s=60) == [4]


# ---- 空间跳跃 -------------------------------------------------------------
def test_space_jumps_exact_indices(jump_traj):
    assert anomalies.mark_space_jumps(jump_traj, jump_threshold_m=400) == [3, 4]


def test_space_jump_skips_illegal_coords(illegal_traj):
    assert anomalies.mark_space_jumps(illegal_traj, jump_threshold_m=200) == []


# ---- 速度类 ---------------------------------------------------------------
def test_speed_spike_detected_on_isolated_outlier(speed_spike_traj):
    spikes, over, accel, v = anomalies.mark_speed_anomalies(
        speed_spike_traj, max_speed_mps=38, max_accel_mps2=6, max_dt_s=60)
    assert 5 in spikes, f"孤立毛刺应命中索引 5，实际 {spikes}"
    assert over == spikes or set(spikes) <= set(over)


def test_clean_straight_trajectory_has_no_speed_anomaly(straight_traj):
    """10 m/s 匀速直线不应触发任何速度异常。"""
    spikes, over, accel, v = anomalies.mark_speed_anomalies(
        straight_traj, max_speed_mps=38, max_accel_mps2=6, max_dt_s=60)
    assert spikes == [] and over == [] and accel == []
    assert all(abs(x - 10.0) < 1e-6 for x in v[1:])


def test_sustained_over_limit_is_not_spike():
    """持续超速是真实行为，不应被判为孤立突变。"""
    ts = [10 * i for i in range(8)]
    coords = [(121.4700 + 0.005 * i, 31.2300) for i in range(8)]  # 每步约 476m/10s=47.6m/s
    t = Traj("fast", ts, coords)
    spikes, over, accel, v = anomalies.mark_speed_anomalies(t, 38.0, 6.0, 60.0)
    assert len(over) >= 5, "应识别为持续超速"
    assert spikes == [], "持续超速不是孤立突变"


def test_dt_artifact_distinguishes_from_coord_glitch():
    """dt 远小于正常采样间隔时，应判为时间戳伪影而非坐标错误。"""
    ts = [0, 10, 20, 21, 30, 40]           # 索引 3 处 dt=1s，明显异常
    coords = [(121.4700, 31.2300), (121.4710, 31.2300), (121.4720, 31.2300),
              (121.4745, 31.2300),          # 1s 内走了约 238m -> v=238 m/s
              (121.4750, 31.2300), (121.4760, 31.2300)]
    t = Traj("dtart", ts, coords)
    assert anomalies.mark_dt_artifacts(t, max_speed_mps=38.0) == [3]


# ---- 转向角与噪声地板 -----------------------------------------------------
def test_u_turn_detected_on_real_fold_back(u_turn_traj):
    """U 型折返应被识别为行为标签。"""
    turns = anomalies.detect_u_turns(u_turn_traj, u_turn_angle=130)
    assert len(turns) >= 1


def test_turn_angle_noise_floor_blocks_jitter_false_positive():
    """回归：在厘米级抖动上算航向角会产生假掉头。

    300 个点在 1m 范围内抖动，任意两点的"方向"都是纯噪声。
    没有噪声地板时这里会报出大量 turn_angle；有了它必须为 0。
    """
    n = 300
    ts = [10 * i for i in range(n)]
    coords = [(121.4700 + (i % 7) * 1e-6, 31.2300 + (i % 5) * 1e-6) for i in range(n)]
    t = Traj("jitter", ts, coords)
    assert anomalies.mark_turn_anomalies(t, max_turn_deg=150, min_step_m=10.0) == []
    # 关掉地板（min_step_m=0）时才会出现误判，证明这条规则确实在起作用
    assert len(anomalies.mark_turn_anomalies(t, max_turn_deg=150, min_step_m=0.0)) > 0


def test_stationary_trajectory_has_no_turn_or_u_turn(stationary_traj):
    a = anomalies.detect_anomalies(stationary_traj)
    assert a.counts().get(anomalies.REASON_TURN_ANGLE, 0) == 0
    assert len(a.window_hits["u_turns"]) == 0


# ---- 漂移：尺度归一化 -----------------------------------------------------
def test_drift_curvature_criterion_separates_spike_from_road_curvature():
    """曲率判据必须把「真毛刺」从「道路曲率 + GPS 抖动」里分出来。

    构造：单步 300m 的弯曲行驶轨迹，叠加 ±35m 抖动，注入 2 个 900m 毛刺。
    实测对比：
        不做尺度归一化（scale_k=0）：56 个命中（几乎全是抖动误判）
        曲率判据（scale_k=0.15）：6 个命中，且完整覆盖两个真毛刺
    真毛刺会连带其邻居一起判出（邻居的弦被毛刺拉偏），这是预期的。
    """
    import math

    n = 80
    ts = [10 * i for i in range(n)]
    coords = [pt(300.0 * i, 40.0 * math.sin(i * 0.25) + 35.0 * math.sin(i * 2.7))
              for i in range(n)]
    coords[30] = (coords[30][0], coords[30][1] + 900.0 / 110873.43)
    coords[55] = (coords[55][0], coords[55][1] - 900.0 / 110873.43)
    t = Traj("noisy_highway", ts, coords)

    hits_no_scale = anomalies.mark_drift(t, radius_m=30.0, k=1, scale_k=0.0)
    hits_curv = anomalies.mark_drift(t, radius_m=30.0, k=1, scale_k=0.15)

    assert len(hits_no_scale) > 40, f"无尺度项应大面积误判，实际 {len(hits_no_scale)}"
    assert len(hits_curv) <= 10, f"曲率判据应收紧到个位数，实际 {len(hits_curv)}"
    assert 30 in hits_curv, f"真毛刺 30 被漏掉：{hits_curv}"
    assert 55 in hits_curv, f"真毛刺 55 被漏掉：{hits_curv}"


def test_drift_criterion_is_scale_free_with_respect_to_speed():
    """同样形状的轨迹，整体缩放速度后命中集合应基本不变（尺度无关性）。

    这是用曲率而非绝对偏移做判据的主要理由：
    城市道路与高速公路的合理参数不应差一个数量级。
    """
    import math

    def build(step_m):
        n = 60
        ts = [10 * i for i in range(n)]
        coords = [pt(step_m * i, 0.02 * step_m * math.sin(i * 0.3)) for i in range(n)]
        coords[30] = (coords[30][0], coords[30][1] + 3.0 * step_m / 110873.43)
        return Traj("scaled", ts, coords), step_m

    slow, _ = build(30.0)     # 32 km/h 级别
    fast, _ = build(300.0)    # 108 km/h 级别
    # 两条轨迹的「相对毛刺幅度」相同，故命中比例应相近
    r_slow = len(anomalies.mark_drift(slow, radius_m=30.0, k=1, scale_k=0.15)) / len(slow)
    r_fast = len(anomalies.mark_drift(fast, radius_m=30.0, k=1, scale_k=0.15)) / len(fast)
    assert abs(r_slow - r_fast) < 0.15, f"速度缩放后命中率差异过大: {r_slow:.3f} vs {r_fast:.3f}"


def test_drift_detects_genuine_outlier_spike():
    """真离群点应被抓到：直线轨迹里插入一个横向偏移 950m 的点。

    直线段上正常点的曲率为 0，故任何超限垂距都是真异常，不应漏判。
    """
    n = 40
    ts = [10 * i for i in range(n)]
    coords = [pt(100.0 * i, 0.0) for i in range(n)]
    coords[20] = (coords[20][0], coords[20][1] + 950.0 / 110873.43)  # 横向 950m
    t = Traj("spike_line", ts, coords)
    hits = anomalies.mark_drift(t, radius_m=30.0, k=1, scale_k=0.15)
    assert 20 in hits, f"真离群点 20 未命中，实际 {hits}"


def test_drift_regression_on_real_trajectory_306(real_raw):
    """回归：真实驾驶轨迹 306 曾被漂移规则误判 56/111 个点（50%）。

    那些点的单步位移中位 235m、速度中位 21.6 m/s，全部是正常行驶。
    加入尺度归一化后必须降到极低水平。
    """
    from traj_agent.core import traj as traj_mod
    if "306" not in real_raw:
        pytest.skip("需要车辆 306")
    t = traj_mod.traj_from_raw("306", *real_raw["306"])
    # 曲率判据下实测 3/111 = 2.7%，且命中的是真实毛刺
    hits = anomalies.mark_drift(t, radius_m=30.0, k=1, scale_k=0.15)
    assert len(hits) <= 0.05 * len(t), f"误判仍过多: {len(hits)}/{len(t)}"


def test_drift_skips_stationary_trajectory(stationary_traj):
    """静止轨迹不适用漂移概念（那是 stationary_drift 的职责）。"""
    assert anomalies.mark_drift(stationary_traj) == []


# ---- 汇总 -----------------------------------------------------------------
def test_detect_anomalies_flags_align_with_reasons(straight_traj):
    a = anomalies.detect_anomalies(straight_traj)
    assert a.n_points == len(straight_traj)
    for reason, idxs in a.flags.items():
        for i in idxs:
            assert reason in a.reasons[i], f"{reason} 未写入点 {i} 的原因列表"


def test_detect_anomalies_no_duplicate_reason_per_point(dup_traj):
    a = anomalies.detect_anomalies(dup_traj)
    for rs in a.reasons:
        assert len(rs) == len(set(rs))


def test_anomaly_counts_and_histogram(dup_traj, illegal_traj):
    a1 = anomalies.detect_anomalies(dup_traj)
    a2 = anomalies.detect_anomalies(illegal_traj)
    hist = anomalies.anomaly_histogram([a1, a2])
    assert hist.get(anomalies.REASON_CONSECUTIVE_DUP, 0) == 3
    assert hist.get(anomalies.REASON_ILLEGAL_COORD, 0) == 2


def test_all_reasons_are_documented():
    """每个原因常量都必须有中文说明，否则报告会出现裸英文 key。"""
    for r in anomalies.ALL_REASONS:
        assert r in anomalies.REASON_ZH


def test_rule_params_from_params_maps_dist_threshold():
    rp = RuleParams.from_params({"max_speed_mps": 20.0, "dist_threshold": 123.0})
    assert rp.max_speed_mps == 20.0
    assert rp.jump_threshold_m == 123.0


def test_to_dict_is_json_safe(speed_spike_traj):
    import json
    a = anomalies.detect_anomalies(speed_spike_traj)
    payload = a.to_dict(with_indices=True)
    json.dumps(payload)  # 不抛异常即通过
    assert payload["n_points"] == len(speed_spike_traj)


# ---- 真实数据回归 ---------------------------------------------------------
@pytest.mark.parametrize("vid,expect_regime", [("0", "stationary"), ("246", "moving")])
def test_real_data_regimes(real_raw, vid, expect_regime):
    from traj_agent.core import diagnosis, traj as traj_mod
    t = traj_mod.traj_from_raw(vid, *real_raw[vid])
    assert diagnosis.classify_regime(t) == expect_regime


def test_real_data_fast_driving_no_drift_false_positive(real_raw):
    """真实行驶轨迹不应被漂移规则大面积误判（回归：曾误判 50% 的点）。"""
    from traj_agent.core import traj as traj_mod
    for vid in ("246", "256", "306"):
        if vid not in real_raw:
            continue
        t = traj_mod.traj_from_raw(vid, *real_raw[vid])
        hits = anomalies.mark_drift(t, radius_m=30.0, k=3, scale_k=3.0)
        assert len(hits) <= 0.05 * len(t), f"{vid} 漂移误判过多: {len(hits)}/{len(t)}"
