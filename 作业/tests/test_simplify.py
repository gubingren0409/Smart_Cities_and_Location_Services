"""DP 简化：正确性、误差界、以及与原始有 bug 实现的对照。"""
from __future__ import annotations

import math

import pytest

from traj_agent.core import geo, simplify
from traj_agent.core.traj import Traj


def test_straight_line_collapses_to_endpoints(straight_traj):
    r = simplify.simplify_trajectory(straight_traj, tolerance_m=5.0)
    assert r.kept_indices == [0, len(straight_traj) - 1]
    assert r.compression_ratio > 0.7


def test_endpoints_always_kept(straight_traj):
    for tol in (0.1, 1, 10, 100, 10000):
        idx = simplify.douglas_peucker(straight_traj.coords, tol)
        assert idx[0] == 0
        assert idx[-1] == len(straight_traj) - 1
        assert idx == sorted(idx)


def test_indices_are_strictly_increasing_and_unique(straight_traj):
    idx = simplify.douglas_peucker(straight_traj.coords, 3.0)
    assert len(idx) == len(set(idx))
    assert all(b > a for a, b in zip(idx, idx[1:]))


def test_deviation_never_exceeds_tolerance(real_raw):
    """DP 的数学保证：任何被丢弃点到简化折线的距离 <= 容差。

    这是本项目最初出错的地方：事后用滑动窗口估计偏差时报出 8349m
    （tol=2m 不可能产生这种误差）。
    """
    from traj_agent.core import traj as traj_mod
    for vid in ("246", "256", "306"):
        if vid not in real_raw:
            continue
        t = traj_mod.traj_from_raw(vid, *real_raw[vid])
        for tol in (1.0, 2.0, 5.0, 10.0):
            r = simplify.simplify_trajectory(t, tol)
            assert r.max_deviation_m <= tol + 1e-6, (
                f"{vid} tol={tol} 偏差 {r.max_deviation_m} 超界")


def test_reported_deviation_matches_independent_global_search(real_raw):
    """DP 自报偏差必须与独立的全局最近线段搜索一致。"""
    from traj_agent.core import traj as traj_mod
    t = traj_mod.traj_from_raw("246", *real_raw["246"])
    for tol in (2.0, 5.0, 10.0):
        r = simplify.simplify_trajectory(t, tol)
        independent = simplify.max_deviation_m(t.coords, r.traj.coords)
        assert r.max_deviation_m == pytest.approx(independent, abs=1e-6)


def test_zero_tolerance_keeps_everything(straight_traj):
    idx = simplify.douglas_peucker(straight_traj.coords, 0.0)
    assert idx == list(range(len(straight_traj)))


def test_two_point_trajectory_unchanged():
    t = Traj("two", [0, 10], [(121.47, 31.23), (121.48, 31.23)])
    r = simplify.simplify_trajectory(t, 5.0)
    assert r.kept_indices == [0, 1]
    assert r.max_deviation_m == 0.0


def test_vertical_segment_bug_regression():
    """回归：原始 point2LineDistance 在垂直线段上返回 9999999，
    导致 DP 几乎不压缩。正确实现应把 5 点垂直线压成 2 点。"""
    pts = [(121.4700, 31.2300 + 0.001 * i) for i in range(5)]
    correct = simplify.douglas_peucker(pts, 10.0)
    broken = simplify.douglas_peucker_broken(pts, 10.0)
    assert correct == [0, 4], f"正确实现应压缩到 2 点，得到 {correct}"
    assert len(broken) == 5, f"有 bug 的实现不压缩，得到 {broken}"


def test_distance_metric_differs_from_infinite_line():
    """量化两个距离度量在「端点外」情形下的差异。

    共线点无法用来区分这两种度量（三点共线时两种距离都是 0，
    DP 必然压成两点）。真正有区别的是**偏离直线但位于端点外侧**的点：
      到线段距离：必须绕到最近端点，故很大
      到无限长直线距离：可以很小
    这里断言几何量本身，并确认正确实现因此保留该点。
    """
    a = (121.4700, 31.2300)
    b = (121.4750, 31.2300)
    beyond_off = (121.4800, 31.2320)     # 在 ab 延长线外侧，且偏离 222m

    seg_d = geo.segment_distance_m(beyond_off, a, b)
    assert seg_d > 400, f"到线段距离应很大，得到 {seg_d:.1f}"

    pts = [a, b, beyond_off]
    # 正确实现保留中间点 b（因为 beyond_off 离线段 a-b 很远，
    # 于是 a-b 段内没有点被丢；b 因是 a 与 beyond_off 的拐点被保留）
    assert simplify.douglas_peucker(pts, 10.0) == [0, 1, 2]


def test_perp_algorithm_runs_and_is_sane(straight_traj):
    idx = simplify.perp_distance_indices(straight_traj.coords, 5.0)
    assert idx[0] == 0 and idx[-1] == len(straight_traj) - 1
    assert idx == sorted(set(idx))


def test_unknown_algorithm_raises(straight_traj):
    with pytest.raises(KeyError):
        simplify.simplify_trajectory(straight_traj, 5.0, algorithm="nope")


def test_all_registered_algorithms_are_callable(straight_traj):
    for name in simplify.ALGORITHMS:
        idx = simplify.ALGORITHMS[name](straight_traj.coords, 5.0)
        assert isinstance(idx, list) and len(idx) >= 2


def test_uniform_sample_indices_respects_ratio():
    idx = simplify.uniform_sample_indices(100, 0.2)
    assert idx[0] == 0 and idx[-1] == 99
    assert 15 <= len(idx) <= 25


def test_compression_monotone_in_tolerance(real_raw):
    """容差越大，保留点越少（单调性）——这是质量—压缩率曲线的基本前提。"""
    from traj_agent.core import traj as traj_mod
    t = traj_mod.traj_from_raw("246", *real_raw["246"])
    counts = [len(simplify.simplify_trajectory(t, tol).kept_indices)
              for tol in (1, 2, 5, 10, 20)]
    assert counts == sorted(counts, reverse=True), f"非单调: {counts}"


def test_sweep_returns_one_result_per_tolerance(real_raw):
    from traj_agent.core import traj as traj_mod
    t = traj_mod.traj_from_raw("246", *real_raw["246"])
    res = simplify.sweep_dp_tolerance(t, [1, 5, 10])
    assert len(res) == 3
    assert [r.tolerance_m for r in res] == [1, 5, 10]
