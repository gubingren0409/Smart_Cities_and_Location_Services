"""几何与坐标工具。

坐标约定（全项目唯一约定，勿改）：
    point = (lon, lat)，单位：度 (WGS84)

距离约定：
    所有对外距离函数返回 **米**。
    平面计算一律走 local_xy()（局部等距圆柱投影），**不要用 Web Mercator 算距离**：
    墨卡托在 31°N 会把距离系统性放大 1/cos(31°) ≈ 1.17 倍，
    而 DP 容差、跳跃阈值都是「米」为单位的，用错投影会让容差预算凭空偏 17%。
    需要权威球面距离时用 haversine_m。

本项目不 import geopy（环境缺失），haversine 自行实现。
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Point = Sequence[float]
LonLat = Tuple[float, float]

# WGS84 平均地球半径
EARTH_RADIUS_M = 6371008.8
# Web Mercator 半周长
MERCATOR_MAX = 20037508.342789244

# 中国大陆粗略包围盒 (min_lon, min_lat, max_lon, max_lat)，用于「非法坐标」规则
CHINA_BBOX = (73.0, 3.0, 136.0, 54.0)
# 上海实验区粗略包围盒（本项目数据采集地）
SHANGHAI_BBOX = (120.8, 30.6, 122.2, 31.9)


def haversine_m(p1: Point, p2: Point) -> float:
    """两点**球面**大圆距离，单位米，点格式 (lon, lat)。

    仅用于独立交叉验证与已知答案测试（如城市间距离）。
    与 local_xy 给出的平面距离不是同一度量：在 31°N 附近前者系统性小约 2e-3。
    任何进入指标或阈值的距离都必须取自 local_xy / segment_distance_m / path_length_m。
    """
    lon1, lat1 = float(p1[0]), float(p1[1])
    lon2, lat2 = float(p2[0]), float(p2[1])
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def lonlat_to_mercator(lon: float, lat: float) -> Tuple[float, float]:
    """WGS84 经纬度 -> Web Mercator 米。lat 夹紧避免 log 溢出。"""
    lat = max(-85.05112878, min(85.05112878, float(lat)))
    x = float(lon) * MERCATOR_MAX / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * MERCATOR_MAX / 180.0
    return x, y


def mercator_to_lonlat(x: float, y: float) -> Tuple[float, float]:
    """Web Mercator 米 -> WGS84 经纬度。"""
    lon = float(x) / MERCATOR_MAX * 180.0
    lat = float(y) / MERCATOR_MAX * 180.0
    lat = 180.0 / math.pi * (2.0 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lon, lat


# 参考纬度 31.23°N（上海）。每度对应的地面米数由 WGS84 曲率半径给出，不要用
# 110540/111320 这类球体近似常数——它们会带来约 0.3% 的系统偏差。
#   M = 子午圈曲率半径 (6352579 m) -> 每度纬度 110873.43 m
#   N = 卯酉圈曲率半径 (6383884 m) -> 每度经度 N*cos(lat)*pi/180 = 95274.27 m
# 本投影为局部等距圆柱（在切平面上对椭球的一阶展开），是本项目**唯一**的
# 距离度量来源：DP 容差、跳跃阈值、长度统计全部由它给出。
# 与 haversine_m（球面）在同纬度会系统性相差约 2e-3 —— 因为平均球半径
# 6371008.8 m 比该纬度卯酉圈半径 6383884 m 小 0.2%。这是有意的取舍：
# 投影是椭球的局部展开，比平均球更接近真实地面距离。
# 需要球面交叉验证时才用 haversine_m，不要把它和 local_xy 混算同一指标。
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_REF_LAT_DEG = 31.23


def _m_per_deg(lat_deg: float) -> Tuple[float, float]:
    """给定纬度处的 (每度经度米数, 每度纬度米数)，基于 WGS84 曲率半径。"""
    e2 = _WGS84_F * (2.0 - _WGS84_F)
    lat = math.radians(float(lat_deg))
    sin2 = math.sin(lat) ** 2
    m_rad = _WGS84_A * (1.0 - e2) / (1.0 - e2 * sin2) ** 1.5   # 子午圈
    n_rad = _WGS84_A / math.sqrt(1.0 - e2 * sin2)               # 卯酉圈
    return (n_rad * math.cos(lat) * math.pi / 180.0,
            m_rad * math.pi / 180.0)


_M_PER_DEG_LON, _M_PER_DEG_LAT = _m_per_deg(_REF_LAT_DEG)


def local_xy(p: Point) -> Tuple[float, float]:
    """经纬度 -> 局部平面米坐标（原点在参考点，非线性等距）。"""
    return (float(p[0]) * _M_PER_DEG_LON, float(p[1]) * _M_PER_DEG_LAT)


def project(points: Sequence[Point]) -> List[Tuple[float, float]]:
    """整条轨迹投影到局部等距平面，返回 [(x_m, y_m), ...]。"""
    return [local_xy(p) for p in points]


def equirect_relative_error(lon: float, lat: float, p2: Point) -> float:
    """局部平面距离与球面距离的相对偏差。

    该值约 2e-3 且随跨度基本不变 —— 说明平面投影本身是自洽的，
    差异来自「椭球局部展开」与「平均球」的口径不同，而非投影精度不足。
    若这个值随跨度显著漂移，才说明投影实现有问题。
    """
    a = (float(lon), float(lat))
    plan = euclid(local_xy(a), local_xy(p2))
    sph = haversine_m(a, p2)
    if sph <= 1e-9:
        return 0.0
    return abs(plan - sph) / sph


def unproject(xy_points: Sequence[Sequence[float]]) -> List[LonLat]:
    """墨卡托平面 -> 经纬度。"""
    return [mercator_to_lonlat(p[0], p[1]) for p in xy_points]


def euclid(a: Sequence[float], b: Sequence[float]) -> float:
    """平面欧氏距离（作用于墨卡托坐标时即为米）。"""
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def is_legal_lonlat(lon: float, lat: float, bbox: Sequence[float] = CHINA_BBOX) -> bool:
    """坐标是否合法：数值有效、范围合法、且不是 (0,0) 空值占位。

    bbox = (min_lon, min_lat, max_lon, max_lat)。
    """
    try:
        lon = float(lon)
        lat = float(lat)
    except (TypeError, ValueError):
        return False
    if math.isnan(lon) or math.isnan(lat) or math.isinf(lon) or math.isinf(lat):
        return False
    if abs(lon) < 1e-9 and abs(lat) < 1e-9:
        return False  # 常见的「缺失值占位」
    min_lon, min_lat, max_lon, max_lat = bbox
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)


def segment_distance_m(p: Point, a: Point, b: Point) -> float:
    """点 p 到**线段** ab 的最短距离（米）。

    在局部等距平面内计算（单位即米），规避 douglas_peucker.py 里 point2LineDistance 的两个坑：
      1. 垂直/近垂直线段被硬编码返回 9999999；
      2. 算的是到「无限长直线」的距离，而非到「线段」的距离。
    """
    ax, ay = local_xy(a)
    bx, by = local_xy(b)
    px, py = local_xy(p)
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def local_distance_m(p1: Point, p2: Point) -> float:
    """局部平面上的两点距离（米）。这是指标口径，不随跨度退化。"""
    return euclid(local_xy(p1), local_xy(p2))


def path_length_m(points: Sequence[Point]) -> float:
    """轨迹累计长度（米）。逐段用局部平面距离，与 DP 容差同源。"""
    if len(points) < 2:
        return 0.0
    return sum(local_distance_m(points[i - 1], points[i]) for i in range(1, len(points)))


def bearing_deg(p1: Point, p2: Point) -> float:
    """p1 -> p2 的航向角，正北为 0，顺时针 [0, 360)。"""
    lon1, lat1 = math.radians(float(p1[0])), math.radians(float(p1[1]))
    lon2, lat2 = math.radians(float(p2[0])), math.radians(float(p2[1]))
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """两个航向角的最小夹角，落在 [0, 180]。"""
    d = abs((float(a) - float(b)) % 360.0)
    return 360.0 - d if d > 180.0 else d


def turn_angles_deg(points: Sequence[Point]) -> List[float]:
    """每个点的转向角（度），长度为 len(points)，首尾补 0.0。

    转向角 = 进入方向与离开方向的夹角，用于识别 U 型折返与漂移毛刺。
    """
    n = len(points)
    if n < 3:
        return [0.0] * n
    out = [0.0] * n
    for i in range(1, n - 1):
        inc = bearing_deg(points[i - 1], points[i])
        outg = bearing_deg(points[i], points[i + 1])
        out[i] = angle_diff_deg(inc, outg)
    return out


def bbox_of(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    """返回 (min_lon, min_lat, max_lon, max_lat)。空序列返回全 0。"""
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    lons = [float(p[0]) for p in points]
    lats = [float(p[1]) for p in points]
    return (min(lons), min(lats), max(lons), max(lats))


def sinuosity(points: Sequence[Point]) -> float:
    """曲折度 = 轨迹长度 / 首尾直线距离。

    接近 1 表示近似直线；明显大于 1 表示绕行。
    首尾重合（静止轨迹）时返回 1.0 以避免除零。
    """
    if len(points) < 2:
        return 1.0
    straight = local_distance_m(points[0], points[-1])
    total = path_length_m(points)
    if straight < 1e-6:
        return 1.0
    return total / straight


def quantile(values: Sequence[float], q: float) -> float:
    """线性插值分位数。core 不依赖 numpy，故自行实现。"""
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def median(values: Sequence[float]) -> float:
    return quantile(values, 0.5)


def is_bimodal(values: Sequence[float], min_gap_ratio: float = 0.5, min_support: float = 0.15) -> bool:
    """粗判一维样本是否双峰，用于 Δt 分布诊断。

    取出现最多的两个不同取值，若间隔显著且各自占比足够则判为双峰。
    这是启发式，不求严谨——它只用来提示 LLM「别套用全局阈值」。
    """
    vals = [float(v) for v in values]
    if len(vals) < 8:
        return False
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
    if len(top) < 2:
        return False
    (v1, c1), (v2, c2) = top
    spread = max(vals) - min(vals)
    if spread <= 0:
        return False
    if abs(v1 - v2) < max(1.0, min_gap_ratio * spread * 0.5):
        return False
    n = len(vals)
    return (c1 / n) >= min_support and (c2 / n) >= min_support


def consecutive_differences(values: Sequence[float]) -> List[float]:
    """相邻差分，长度 = len(values)，首元素补 0.0 以便与点一一对应。"""
    n = len(values)
    if n == 0:
        return []
    out = [0.0] * n
    for i in range(1, n):
        out[i] = float(values[i]) - float(values[i - 1])
    return out


def finite_or_none(x):
    """转 float；NaN/inf 返回 None。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v
