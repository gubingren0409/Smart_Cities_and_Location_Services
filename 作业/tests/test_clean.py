"""去噪：动作精确性、原因字段完整性、以及「不清洗真实行为」的边界。"""
from __future__ import annotations

import pytest

from traj_agent.core import anomalies, clean
from traj_agent.core.clean import CleanConfig, CleanReport
from traj_agent.core.traj import Traj


def test_drop_duplicates_keeps_last_of_each_run(dup_traj):
    """每段重复保留最后一个。

    夹具：ts=[0,10,20,30,40,50,60]，
    coords 索引 0,1,2 相同；索引 3,4 相同；索引 5,6 各自独立。
    保留组内最后一个 => 删 {1,2,4}，保留 {0,3,5,6}，
    对应时间戳 [0, 30, 50, 60]。
    """
    out, n = clean.drop_consecutive_duplicates(dup_traj)
    assert n == 3
    assert len(out) == 4
    # 期望值从夹具推导，不硬编码：每段重复的最后一个索引分别是 2、4、5、6
    expected_idx = [2, 4, 5, 6]
    assert out.timestamps == [dup_traj.timestamps[i] for i in expected_idx]
    assert out.coords == [dup_traj.coords[i] for i in expected_idx]


def test_drop_duplicates_on_stationary_real_data(real_raw):
    """真实静止轨迹 0：99 点 -> 21 点，删除 78 个重复点。"""
    from traj_agent.core import traj as traj_mod
    t = traj_mod.traj_from_raw("0", *real_raw["0"])
    out, n = clean.drop_consecutive_duplicates(t)
    assert n == 78
    assert len(out) == 21


def test_drop_illegal_removes_exact_points(illegal_traj):
    out, n = clean.drop_illegal_coords(illegal_traj)
    assert n == 2
    assert len(out) == 5
    for p in out.coords:
        assert p[0] == p[0]  # 无 NaN


def test_drop_nonpositive_dt(nonmonotonic_traj):
    out, n = clean.drop_nonpositive_dt(nonmonotonic_traj)
    assert n == 2
    dts = out.time_deltas()
    assert all(d > 0 for d in dts[1:])


def test_fix_dt_artifact_does_not_delete_points():
    """时间戳伪影应通过修时间戳处置，点一个都不能少。"""
    ts = [0, 10, 20, 21, 30, 40, 50, 60]
    coords = [(121.4700 + 0.001 * i, 31.2300) for i in range(8)]
    coords[3] = (121.4745, 31.2300)   # 1s 内走 238m
    t = Traj("dtart", ts, coords)
    before = len(t)
    out, fixed = clean.fix_dt_artifacts(t, max_speed_mps=38.0)
    assert fixed == [3]
    assert len(out) == before, "修时间戳不应删除任何点"
    # 修正后该点速度应回落到合理范围
    assert out.speeds_mps()[3] < 38.0
    # 时间戳仍需严格递增
    dts = out.time_deltas()
    assert all(d > 0 for d in dts[1:])


def test_median_smooth_pulls_in_isolated_spike(speed_spike_traj):
    out, changed = clean.median_smooth(speed_spike_traj, window=5, apply_to=[5])
    assert changed == [5]
    # 平滑后该点应回到正常轨迹附近
    assert abs(out.coords[5][1] - speed_spike_traj.coords[4][1]) < 200.0


def test_median_smooth_leaves_untargeted_points_alone(straight_traj):
    out, changed = clean.median_smooth(straight_traj, window=5, apply_to=[5])
    assert changed == []          # 直线上的点本就在窗口中位位置
    assert out.coords == straight_traj.coords


def test_median_smooth_window_forced_odd_and_min_3(straight_traj):
    """窗口参数应被规范化为 >=3 的奇数，否则中位位置有歧义。"""
    for w in (2, 4, 6, 1, 0):
        out, _ = clean.median_smooth(straight_traj, window=w, apply_to=[4])
        assert len(out) == len(straight_traj)   # 不崩溃即通过


def test_denoise_records_reasons_for_every_dropped_point(real_raw):
    """核心契约：每个被删的点都必须有原因记录。

    注意两个计数的语义差异：
      dropped_with_reason = 带原因的被删点数，恒等于实际删除数；
      sum(dropped_reasons.values()) = 原因出现次数，一个点多因时会更大。
    """
    from traj_agent.core import traj as traj_mod
    t = traj_mod.traj_from_raw("246", *real_raw["246"])
    out, rep = clean.denoise_trajectory(t)
    dropped = rep.n_before - rep.n_after
    assert dropped > 0
    assert rep.dropped_reasons, "必须记录删除原因"
    assert rep.dropped_with_reason == dropped, (
        f"带原因的被删点数 {rep.dropped_with_reason} != 实际删除数 {dropped}")
    # 原因次数不少于被删点数（多因点使其更大）
    assert sum(rep.dropped_reasons.values()) >= dropped


def test_denoise_report_arithmetic_is_consistent(real_raw):
    from traj_agent.core import traj as traj_mod
    for vid in ("246", "306", "256"):
        if vid not in real_raw:
            continue
        t = traj_mod.traj_from_raw(vid, *real_raw[vid])
        out, rep = clean.denoise_trajectory(t)
        assert rep.n_before == len(t)
        assert rep.n_after == len(out)
        assert rep.n_before - rep.n_after == rep.dropped_with_reason
        assert 0.0 <= rep.drop_ratio <= 1.0


def test_denoise_preserves_behavior_declaration(real_raw):
    """掉头与持续超速必须出现在 preserved_behavior 里，声明「不是漏网之鱼」。"""
    from traj_agent.core import traj as traj_mod
    t = traj_mod.traj_from_raw("352", *real_raw["352"])
    out, rep = clean.denoise_trajectory(t)
    assert "u_turn" in rep.preserved_behavior


def test_denoise_does_not_delete_genuine_u_turn(u_turn_traj):
    """U 型折返是真实行为，去噪不应把它抹掉。"""
    out, rep = clean.denoise_trajectory(u_turn_traj)
    assert len(out) >= 0.7 * len(u_turn_traj), (
        f"折返轨迹被过度删除: {len(out)}/{len(u_turn_traj)}")


def test_denoise_is_idempotent_on_clean_trajectory(straight_traj):
    out1, rep1 = clean.denoise_trajectory(straight_traj)
    out2, rep2 = clean.denoise_trajectory(out1)
    assert rep1.n_after == len(straight_traj)
    assert rep2.n_after == rep1.n_after


def test_denoise_empty_trajectory_does_not_crash():
    t = Traj("empty", [], [])
    out, rep = clean.denoise_trajectory(t)
    assert len(out) == 0
    assert rep.n_before == 0
    assert rep.drop_ratio == 0.0


def test_denoise_single_point_does_not_crash():
    t = Traj("one", [0], [(121.47, 31.23)])
    out, rep = clean.denoise_trajectory(t)
    assert len(out) == 1


def test_clean_config_from_params():
    cfg = CleanConfig.from_params({"smooth_window": 7, "dist_threshold": 250.0,
                                  "max_speed_mps": 30.0})
    assert cfg.smooth_window == 7
    assert cfg.rules.jump_threshold_m == 250.0
    assert cfg.rules.max_speed_mps == 30.0


def test_custom_config_can_disable_actions(dup_traj):
    cfg = CleanConfig(do_drop_duplicates=False, do_median_smooth=False)
    out, rep = clean.denoise_trajectory(dup_traj, cfg)
    assert rep.actions.get(clean.ACTION_DROP_DUPLICATE, 0) == 0


def test_report_merge():
    a = CleanReport(n_before=10, n_after=8, actions={"x": 2}, dropped_reasons={"r": 2})
    b = CleanReport(n_before=5, n_after=5, actions={"y": 0}, dropped_reasons={})
    a.merge(b)
    assert a.n_before == 15 and a.n_after == 13
    assert a.actions == {"x": 2, "y": 0}
    assert a.dropped_reasons == {"r": 2}


def test_reasons_field_length_always_matches_points(real_raw):
    """reasons 与点必须始终一一对应，否则下游按索引取原因会错位。"""
    from traj_agent.core import traj as traj_mod
    for vid in ("246", "0"):
        t = traj_mod.traj_from_raw(vid, *real_raw[vid])
        out, _ = clean.denoise_trajectory(t)
        assert len(out.reasons) == len(out.coords) == len(out.timestamps)
