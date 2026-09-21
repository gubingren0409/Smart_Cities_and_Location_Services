"""geo 模块的已知答案测试。

包含对 douglas_peucker.point2LineDistance 两个 bug 的回归对照，
以及对「球面 haversine 与椭球局部投影混算」这一隐患的同源性护栏。
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traj_agent.core import geo  # noqa: E402


# --------------------------------------------------------------------------
# 球面距离（仅作交叉验证口径）
# --------------------------------------------------------------------------
def test_haversine_known_city_pair():
    """上海人民广场 -> 北京天安门，公开大地线距离约 1067 km。"""
    d = geo.haversine_m((121.4737, 31.2304), (116.3975, 39.9087))
    assert 1060_000 < d < 1075_000


def test_haversine_identical_points_is_zero():
    assert geo.haversine_m((121.47, 31.23), (121.47, 31.23)) == 0.0


def test_haversine_symmetric():
    a, b = (121.47, 31.23), (121.52, 31.28)
    assert geo.haversine_m(a, b) == pytest.approx(geo.haversine_m(b, a), abs=1e-9)


def test_haversine_one_degree_latitude():
    """1 度纬度在 WGS84 下约 110.6-111.7 km。"""
    d = geo.haversine_m((121.0, 31.0), (121.0, 32.0))
    assert 110_000 < d < 112_000


# --------------------------------------------------------------------------
# 局部平面投影：本项目唯一距离度量来源
# --------------------------------------------------------------------------
def test_local_projection_is_self_consistent_across_spans():
    """平面与球面的偏差必须**有界且不随跨度漂移**。

    固定约 2e-3 的偏差来自「椭球局部展开 vs 平均球半径」的口径差异
    （6371008.8 vs 6383884），是已知且可接受的。
    若这个值随跨度显著增长，才说明投影实现有问题。
    """
    for half in (0.01, 0.05, 0.1, 0.2):
        errs = []
        for dlon in (-half, 0.0, half):
            for dlat in (-half, 0.0, half):
                if dlon == 0.0 and dlat == 0.0:
                    continue
                errs.append(geo.equirect_relative_error(121.47, 31.23, (121.47 + dlon, 31.23 + dlat)))
        assert max(errs) < 5e-3, f"跨度 ±{half} 偏差 {max(errs):.3e} 过大"


def test_local_projection_matches_ellipsoidal_meridian_arc():
    """南北方向的平面距离必须与 WGS84 子午圈弧长一致（相对误差 < 1e-4）。

    这条断言把投影钉在椭球上：若有人把 _m_per_deg 换成 110540 之类的球体常数，
    它会立刻失败（那会带来约 3e-3 的偏差）。

    注意积分变量：h 已是「度」，故弧长 = ∫ M(φ) dφ 需再乘 pi/180 一次，
    仅一次。这里显式用 radians 转 phi，最后统一乘一次 pi/180。
    """
    lat0, lat1 = 31.23, 31.33
    planar = abs(geo.local_xy((121.47, lat1))[1] - geo.local_xy((121.47, lat0))[1])

    n = 64
    a, f = geo._WGS84_A, geo._WGS84_F
    e2 = f * (2.0 - f)
    h_deg = (lat1 - lat0) / n
    total = 0.0
    for i in range(n + 1):
        phi = math.radians(lat0 + i * h_deg)
        m_rad = a * (1.0 - e2) / (1.0 - e2 * math.sin(phi) ** 2) ** 1.5
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        total += w * m_rad
    arc = total * (h_deg * math.pi / 180.0) / 3.0  # 辛普森系数 /3，且只转一次弧度
    assert planar == pytest.approx(arc, rel=1e-4), f"平面 {planar:.3f} vs 椭球弧长 {arc:.3f}"


def test_local_projection_is_not_web_mercator():
    """回归护栏：Web Mercator 会把地面距离放大 1/cos(lat) ≈ 1.17（31°N）。

    若有人把 project() 换回墨卡托，这个断言会失败——有意设置。
    """
    p1, p2 = (121.47, 31.23), (121.52, 31.23)
    planar = geo.euclid(geo.local_xy(p1), geo.local_xy(p2))
    spherical = geo.haversine_m(p1, p2)
    web_merc = geo.euclid(geo.lonlat_to_mercator(*p1), geo.lonlat_to_mercator(*p2))
    assert planar == pytest.approx(spherical, rel=5e-3)
    assert web_merc / spherical > 1.1, "墨卡托应当明显偏大，否则说明投影根本不是它"


# --------------------------------------------------------------------------
# 点到线段距离：对照原始实现的 bug
# --------------------------------------------------------------------------
def test_segment_distance_perpendicular_is_finite():
    """回归 bug 1：到垂直/近垂直线段的距离不能被硬编码成 9999999。"""
    a, b, p = (121.4700, 31.2300), (121.4700, 31.2400), (121.4750, 31.2310)
    d = geo.segment_distance_m(p, a, b)
    assert 400 < d < 550, f"期望约 476 m，得到 {d}"


def test_segment_distance_uses_segment_not_infinite_line():
    """回归 bug 2：必须是到线段而非无限长直线的距离，端点外的点要绕到端点。"""
    a, b = (121.4700, 31.2300), (121.4700, 31.2400)
    beyond = (121.4700, 31.2405)
    d = geo.segment_distance_m(beyond, a, b)
    assert d == pytest.approx(geo.local_distance_m(beyond, b), rel=1e-6)
    assert d < 100, "若走无限长直线，该距离会被算成 ~0"


def test_segment_distance_on_segment_is_zero():
    a, b = (121.4700, 31.2300), (121.4800, 31.2300)
    assert geo.segment_distance_m((121.4750, 31.2300), a, b) < 1.0


def test_segment_distance_degenerate_segment():
    a = p = (121.47, 31.23)
    q = (121.48, 31.23)
    assert geo.segment_distance_m(q, a, p) == pytest.approx(geo.local_distance_m(a, q), rel=1e-9)


# --------------------------------------------------------------------------
# 坐标合法性
# --------------------------------------------------------------------------
def test_illegal_lonlat_detects_null_island_and_nan():
    assert not geo.is_legal_lonlat(0.0, 0.0)
    assert not geo.is_legal_lonlat(float("nan"), 31.0)
    assert not geo.is_legal_lonlat(121.0, float("inf"))
    assert not geo.is_legal_lonlat(200.0, 31.0)
    assert not geo.is_legal_lonlat(121.0, 95.0)
    assert not geo.is_legal_lonlat(None, 31.0)
    assert geo.is_legal_lonlat(121.4737, 31.2304)


def test_illegal_lonlat_outside_china_bbox():
    assert not geo.is_legal_lonlat(-74.0, 40.7)  # 纽约
    assert not geo.is_legal_lonlat(139.7, 35.7)  # 东京


# --------------------------------------------------------------------------
# 长度 / 角度 / 统计
# --------------------------------------------------------------------------
def test_path_length_matches_local_distance_sum():
    pts = [(121.47, 31.23), (121.48, 31.23), (121.49, 31.24)]
    expected = geo.local_distance_m(pts[0], pts[1]) + geo.local_distance_m(pts[1], pts[2])
    assert geo.path_length_m(pts) == pytest.approx(expected, rel=1e-12)


def test_path_length_single_point_is_zero():
    assert geo.path_length_m([(121.47, 31.23)]) == 0.0


def test_bearing_cardinal_directions():
    """南北向航向精确；东西向有一条已知的球面偏移。

    沿纬圈向东的大圆起点方位角并不精确等于 90°：球面几何上它偏向极点，
    偏移量约 cos(lat)*dlon/2。由于本项目关心的转向角量级是几十度，
    0.03° 的偏移可忽略，但必须显式记录，避免以后被误判成 bug。
    """
    o = (121.47, 31.23)
    assert geo.bearing_deg(o, (121.47, 31.33)) == pytest.approx(0.0, abs=1e-9)
    assert geo.bearing_deg(o, (121.47, 31.13)) == pytest.approx(180.0, abs=1e-9)
    # 东西向：允许 cos(lat)*dlon/2 量级的球面偏移
    assert geo.bearing_deg(o, (121.57, 31.23)) == pytest.approx(90.0, abs=0.1)
    assert geo.bearing_deg(o, (121.37, 31.23)) == pytest.approx(270.0, abs=0.1)


def test_angle_diff_wraps_around_360():
    assert geo.angle_diff_deg(350.0, 10.0) == pytest.approx(20.0)
    assert geo.angle_diff_deg(10.0, 350.0) == pytest.approx(20.0)
    assert geo.angle_diff_deg(0.0, 180.0) == pytest.approx(180.0)
    assert geo.angle_diff_deg(90.0, 90.0) == pytest.approx(0.0)


def test_turn_angles_u_turn_detected():
    u = [(121.47, 31.23), (121.47, 31.24), (121.47, 31.23)]
    assert geo.turn_angles_deg(u)[1] == pytest.approx(180.0, abs=1.0)
    straight = [(121.47, 31.23), (121.48, 31.23), (121.49, 31.23)]
    assert geo.turn_angles_deg(straight)[1] == pytest.approx(0.0, abs=1e-6)


def test_turn_angles_length_matches_points():
    pts = [(121.47, 31.23), (121.48, 31.23), (121.49, 31.24), (121.50, 31.25)]
    assert len(geo.turn_angles_deg(pts)) == len(pts)


def test_sinuosity_straight_line():
    line = [(121.47, 31.23), (121.48, 31.23), (121.49, 31.23)]
    assert geo.sinuosity(line) == pytest.approx(1.0, abs=1e-6)


def test_sinuosity_go_and_return_returns_one_by_contract():
    """去而复返时首尾重合，直线距离为 0，按契约返回 1.0 而非 2.0。

    这是有意定义：sinuosity 用于识别「绕行」而非「折返」，
    折返由 anomalies.detect_u_turns 负责。
    """
    detour = [(121.47, 31.23), (121.49, 31.23), (121.47, 31.23)]
    assert geo.sinuosity(detour) == 1.0


def test_sinuosity_l_shaped_detour_exceeds_one():
    """L 形绕行：长度 / 直线距离 = 2/sqrt(2) ≈ 1.414。"""
    l_shape = [(121.47, 31.23), (121.49, 31.23), (121.49, 31.25)]
    assert geo.sinuosity(l_shape) == pytest.approx(1.414, rel=5e-3)


def test_sinuosity_single_point():
    assert geo.sinuosity([(121.47, 31.23)]) == 1.0


def test_quantile_matches_known_values():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert geo.quantile(vals, 0.0) == 1.0
    assert geo.quantile(vals, 1.0) == 4.0
    assert geo.quantile(vals, 0.5) == pytest.approx(2.5)
    assert geo.quantile(vals, 0.25) == pytest.approx(1.75)
    assert geo.quantile([], 0.5) == 0.0


def test_is_bimodal_on_realistic_dt_distribution():
    """本项目真实数据 Δt 主频 10s 与 20s，必须被判为双峰。"""
    vals = [10] * 70 + [20] * 18 + [11, 19, 9, 21, 1, 8]
    assert geo.is_bimodal(vals)
    assert not geo.is_bimodal([10] * 60 + [10, 11, 9, 10, 11, 9])


def test_is_bimodal_needs_enough_samples():
    assert not geo.is_bimodal([10, 20, 10])


def test_consecutive_differences_alignment():
    """差分长度必须等于原序列长度，首个补 0，以便与点索引对齐。"""
    assert geo.consecutive_differences([1, 3, 6, 10]) == [0.0, 2.0, 3.0, 4.0]
    assert geo.consecutive_differences([]) == []
    assert geo.consecutive_differences([5]) == [0.0]


def test_bbox_of():
    pts = [(121.47, 31.23), (121.49, 31.20), (121.45, 31.25)]
    assert geo.bbox_of(pts) == (121.45, 31.20, 121.49, 31.25)
    assert geo.bbox_of([]) == (0.0, 0.0, 0.0, 0.0)


def test_finite_or_none():
    assert geo.finite_or_none(1.5) == 1.5
    assert geo.finite_or_none("2.5") == 2.5
    assert geo.finite_or_none(float("nan")) is None
    assert geo.finite_or_none(None) is None
    assert geo.finite_or_none("abc") is None
