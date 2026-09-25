"""指标：已知答案的距离值、退化输入、以及对比表输出。"""
from __future__ import annotations

import math

import pytest

from traj_agent.core import geo, metrics
from traj_agent.core.traj import Traj


def test_identical_trajectories_have_zero_distance(straight_traj):
    assert metrics.hausdorff_m(straight_traj.coords, straight_traj.coords) == 0.0
    assert metrics.frechet_m(straight_traj.coords, straight_traj.coords) == 0.0
    assert metrics.dtw_m(straight_traj.coords, straight_traj.coords) == 0.0


def test_hausdorff_known_value():
    """一条线段到其上方 100m 处的平行线段：Hausdorff 应为 100m。

    因为每条线段的端点到另一线段的最短距离都是 100m（垂足在段内），
    但更严格地说端点到线段端点的距离是 sqrt(100²+...)。
    这里取两条长度相同、端点对齐的平行线段，故最短距离就是 100m。
    """
    a = [(121.4700 + 0.001 * i, 31.2300) for i in range(5)]
    b = [(lon, lat + 100.0 / 110873.43) for lon, lat in a]
    d = metrics.hausdorff_m(a, b)
    assert d == pytest.approx(100.0, rel=1e-3)


def test_hausdorff_is_symmetric():
    a = [(121.4700, 31.2300), (121.4750, 31.2300), (121.4800, 31.2300)]
    b = [(121.4700, 31.2300), (121.4790, 31.2350)]
    assert metrics.hausdorff_m(a, b) == pytest.approx(metrics.hausdorff_m(b, a), rel=1e-9)


def test_hausdorff_directed_can_differ():
    """单向 Hausdorff 不必对称：一条长线到一条短线的单向距离可能很小。"""
    a = [(121.4700 + 0.001 * i, 31.2300) for i in range(10)]
    b = [(121.4700, 31.2300), (121.4701, 31.2300)]
    d_ab = metrics.hausdorff_m(a, b, directed=True)
    d_ba = metrics.hausdorff_m(b, a, directed=True)
    assert d_ab > d_ba


def test_frechet_detects_ordering_difference():
    """Fréchet 对点的「顺序」敏感，Hausdorff 不敏感——这是两者的关键差异。"""
    a = [(121.4700, 31.2300), (121.4750, 31.2300), (121.4800, 31.2300)]
    reordered = [a[0], a[2], a[1]]
    assert metrics.frechet_m(a, reordered) > 50
    # Hausdorff 对同一集合的顺序不敏感
    assert metrics.hausdorff_m(a, reordered) == 0.0


def test_frechet_upper_bounds_hausdorff_for_same_ordering():
    a = [(121.4700 + 0.001 * i, 31.2300) for i in range(6)]
    b = [(lon, lat + 0.0005) for lon, lat in a]
    assert metrics.frechet_m(a, b) >= metrics.hausdorff_m(a, b) - 1e-6


def test_dtw_zero_for_identical_and_positive_for_shifted():
    a = [(121.4700 + 0.001 * i, 31.2300) for i in range(5)]
    b = [(lon, lat + 0.001) for lon, lat in a]
    assert metrics.dtw_m(a, a) == 0.0
    assert metrics.dtw_m(a, b) > 0


def test_empty_inputs_return_zero():
    a = [(121.47, 31.23)]
    assert metrics.hausdorff_m([], a) == 0.0
    assert metrics.hausdorff_m(a, []) == 0.0
    assert metrics.frechet_m([], a) == 0.0
    assert metrics.dtw_m([], []) == 0.0


def test_subsample_keeps_endpoints():
    import numpy as np
    arr = np.arange(1000).reshape(-1, 2).astype(float)
    out = metrics._subsample(arr, 50)
    assert len(out) <= 50
    assert out[0][0] == arr[0][0]
    assert out[-1][0] == arr[-1][0]


def test_long_trajectory_does_not_explode():
    """超长轨迹应被降采样而不是让 Fréchet 的 O(n·m) 内存爆掉。"""
    a = [(121.4700 + 1e-5 * i, 31.2300) for i in range(3000)]
    b = [(lon, lat + 1e-5) for lon, lat in a]
    d = metrics.frechet_m(a, b, max_points=200)
    assert d > 0 and math.isfinite(d)


def test_compute_metrics_basic_fields(straight_traj):
    m = metrics.compute_metrics(straight_traj)
    assert m.n_points == 10
    assert m.duration_s == 90.0
    assert m.length_m == pytest.approx(900.0, rel=1e-3)
    assert m.mean_speed_mps == pytest.approx(10.0, rel=1e-2)
    assert m.sinuosity == pytest.approx(1.0, abs=1e-6)
    assert m.hausdorff_m is None


def test_compute_metrics_with_reference(straight_traj):
    shorter = straight_traj.positions([0, 1, 2, 3, 4])
    m = metrics.compute_metrics(shorter, reference=straight_traj)
    assert m.hausdorff_m is not None
    assert m.hausdorff_m > 0
    assert m.frechet_m is not None


def test_compute_metrics_can_skip_distances(straight_traj):
    m = metrics.compute_metrics(straight_traj, reference=straight_traj, with_distances=False)
    assert m.hausdorff_m is None and m.frechet_m is None


def test_compute_metrics_records_runtime(straight_traj):
    m = metrics.compute_metrics(straight_traj)
    assert m.runtime_ms >= 0.0


def test_metrics_to_dict_is_json_safe(straight_traj):
    import json
    json.dumps(metrics.compute_metrics(straight_traj).to_dict())


def test_comparison_table_accumulates_rows(straight_traj):
    tbl = metrics.ComparisonTable()
    tbl.add("raw", metrics.compute_metrics(straight_traj))
    tbl.add("half", metrics.compute_metrics(straight_traj.positions(range(0, 10, 2))))
    assert len(tbl.to_dict()) == 2
    text = tbl.format_text(["stage", "n_points", "length_m"])
    assert "raw" in text and "half" in text
    assert "stage" in text


def test_comparison_table_empty_format():
    assert metrics.ComparisonTable().format_text() == "(空)"


def test_comparison_table_renders_none_as_dash(straight_traj):
    tbl = metrics.ComparisonTable()
    tbl.add("raw", metrics.compute_metrics(straight_traj))
    assert "-" in tbl.format_text(["stage", "hausdorff_m"])


# ---------------------------------------------------------------------------
# 口径回归：这两个测试锁定的是本项目最容易误读的两处设计
# ---------------------------------------------------------------------------
def test_hausdorff_matches_dp_deviation_bound():
    """回归：Hausdorff 必须与 DP 的误差界一致。

    早期实现用了 scipy 的**离散顶点集** Hausdorff，在简化后顶点稀疏时
    给出 548m（真实值 4.77m）。改为点到折线口径后两者应逐位相等。
    """
    from traj_agent.core import clean, simplify, traj as traj_mod
    import os
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "traj_dict.json")
    if not os.path.exists(data):
        pytest.skip("需要 traj_dict.json")
    raw = traj_mod.load_raw(data)
    t = traj_mod.traj_from_raw("246", *raw["246"])
    cs, _ = clean.denoise_trajectory(t)
    for tol in (1.0, 2.0, 5.0, 10.0):
        r = simplify.simplify_trajectory(cs, tol)
        hd = metrics.hausdorff_m(cs.coords, r.traj.coords)
        assert hd == pytest.approx(r.max_deviation_m, abs=1e-6), (
            f"tol={tol}: Hausdorff {hd} != DP 偏差 {r.max_deviation_m}")
        assert hd <= tol + 1e-6


def test_frechet_exceeds_hausdorff_for_sparse_simplification():
    """回归：点对点 Fréchet 远大于点到折线 Hausdorff，这是定义差异而非 bug。

    简化后顶点变稀疏，Fréchet 强制保持弧长匹配顺序，
    于是必须把相距较远的顶点配成一对。报告里必须两个都给。
    """
    from traj_agent.core import clean, simplify, traj as traj_mod
    import os
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "traj_dict.json")
    if not os.path.exists(data):
        pytest.skip("需要 traj_dict.json")
    raw = traj_mod.load_raw(data)
    t = traj_mod.traj_from_raw("246", *raw["246"])
    cs, _ = clean.denoise_trajectory(t)
    r = simplify.simplify_trajectory(cs, 5.0)
    hd = metrics.hausdorff_m(cs.coords, r.traj.coords)
    fd = metrics.frechet_m(cs.coords, r.traj.coords)
    assert hd < 10.0
    assert fd > 100.0, f"Frecbet 应因顶点稀疏而显著更大，得到 {fd}"
    assert fd > hd * 10


def test_point_to_polyline_known_value():
    """点到折线的已知答案：点在折线中点附近 100m。"""
    poly = [(121.4700, 31.2300), (121.4800, 31.2300)]
    p = (121.4750, 31.2300 + 100.0 / 110873.43)
    assert metrics.point_to_polyline_m(p, poly) == pytest.approx(100.0, rel=1e-3)


def test_point_to_polyline_degenerate_inputs():
    assert metrics.point_to_polyline_m((121.47, 31.23), []) != \
        metrics.point_to_polyline_m((121.47, 31.23), [])  # nan != nan
    single = [(121.4700, 31.2300)]
    p = (121.4750, 31.2300)
    assert metrics.point_to_polyline_m(p, single) == pytest.approx(
        geo.local_distance_m(p, single[0]))
