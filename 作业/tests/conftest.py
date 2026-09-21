"""共享测试夹具：手造小轨迹（已知答案）+ 真实数据抽样。

手造轨迹刻意包含：重复点、空间跳跃、非法坐标、速度毛刺、
时间戳非递增、U 型折返、静止段——每一种异常都有确定位置，
以便断言精确到点索引。
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from traj_agent.core.traj import Traj  # noqa: E402

DATA_JSON = os.path.join(ROOT, "traj_dict.json")

# 基准点：上海人民广场附近
LON0, LAT0 = 121.4700, 31.2300
# 0.001 度纬度 ≈ 110.87 m；0.001 度经度 ≈ 95.27 m
D_LAT_100M = 0.000902   # ≈100 m
D_LON_100M = 0.001050   # ≈100 m


# 每度经度 / 纬度的地面米数（上海纬度，与 traj_agent.core.geo 口径一致）
M_PER_DEG_LON = 95274.27
M_PER_DEG_LAT = 110873.43


def pt(east_m: float, north_m: float):
    """以 (LON0, LAT0) 为原点，按米生成坐标。测试模块可直接 import 使用。"""
    return (LON0 + east_m / M_PER_DEG_LON, LAT0 + north_m / M_PER_DEG_LAT)


@pytest.fixture
def straight_traj():
    """10 点正东直线，每点间隔 10s、100m，速度恒为 10 m/s。无任何异常。"""
    ts = [1523293200 + 10 * i for i in range(10)]
    coords = [pt(100.0 * i, 0.0) for i in range(10)]
    return Traj(vehicle_id="straight", timestamps=ts, coords=coords)


@pytest.fixture
def dup_traj():
    """重复点：索引 1,2 与 0 相同；4 与 3 相同。真实可删点为 {1,2,4}。"""
    ts = [0, 10, 20, 30, 40, 50, 60]
    coords = [pt(0, 0), pt(0, 0), pt(0, 0), pt(100, 0), pt(100, 0), pt(200, 0), pt(300, 0)]
    return Traj(vehicle_id="dup", timestamps=ts, coords=coords)


@pytest.fixture
def jump_traj():
    """空间跳跃：索引 3 突然出现在 5km 外，索引 4 又回来。"""
    ts = [0, 10, 20, 30, 40, 50, 60]
    coords = [pt(0, 0), pt(100, 0), pt(200, 0), pt(5000, 0), pt(300, 0), pt(400, 0), pt(500, 0)]
    return Traj(vehicle_id="jump", timestamps=ts, coords=coords)


@pytest.fixture
def illegal_traj():
    """非法坐标：索引 2 为 NaN，索引 5 为越界坐标 (200, 100)。

    刻意**不用** (0,0) 作哨兵：它在球面上是个真实位置，与其余点相距 1.3 万公里，
    会同时触发 space_jump，把「非法坐标」这个测试目标污染掉。
    用越界坐标（经度 200 超出 ±180）则只命中非法性判定。
    """
    ts = [0, 10, 20, 30, 40, 50, 60]
    coords = [pt(0, 0), pt(100, 0), (float("nan"), float("nan")),
              pt(300, 0), pt(400, 0), (200.0, 100.0), pt(600, 0)]
    return Traj(vehicle_id="illegal", timestamps=ts, coords=coords)


@pytest.fixture
def timegap_traj():
    """时间间隔：索引 4 处间隔 500s，超过 30s 阈值。"""
    ts = [0, 10, 20, 30, 530, 540, 550, 560]
    coords = [pt(100.0 * i, 0.0) for i in range(8)]
    return Traj(vehicle_id="gap", timestamps=ts, coords=coords)


@pytest.fixture
def nonmonotonic_traj():
    """时间戳非递增：索引 3 与 4 的时间戳与前面相同/倒退。"""
    ts = [0, 10, 20, 20, 15, 30, 40]
    coords = [pt(100.0 * i, 0.0) for i in range(7)]
    return Traj(vehicle_id="nonmono", timestamps=ts, coords=coords)


@pytest.fixture
def u_turn_traj():
    """U 型折返：向东 1km 后原路返回，索引 5 处转向约 180 度。"""
    ts = [0, 10, 20, 30, 40, 50, 60, 70]
    coords = [pt(0, 0), pt(200, 0), pt(400, 0), pt(600, 0), pt(800, 0),
              pt(600, 0), pt(400, 0), pt(200, 0)]
    return Traj(vehicle_id="uturn", timestamps=ts, coords=coords)


@pytest.fixture
def stationary_traj():
    """静止轨迹：30 点，坐标在 3m 内抖动。"""
    ts = [10 * i for i in range(30)]
    jitter = [(i % 3 - 1) * 1.0 for i in range(30)]
    coords = [pt(j, j * 0.5) for j in jitter]
    return Traj(vehicle_id="stationary", timestamps=ts, coords=coords)


@pytest.fixture
def speed_spike_traj():
    """速度毛刺：索引 5 处单点跳到 1km 外又回来。"""
    ts = [10 * i for i in range(11)]
    coords = [pt(100.0 * i, 0.0) for i in range(11)]
    coords[5] = pt(1100.0, 0.0)   # 单点离群 ~1km
    return Traj(vehicle_id="spike", timestamps=ts, coords=coords)


@pytest.fixture(scope="session")
def real_raw():
    """真实数据（session 级缓存，44MB 只读一次）。"""
    if not os.path.exists(DATA_JSON):
        pytest.skip("traj_dict.json 不存在")
    from traj_agent.core import traj as traj_mod
    return traj_mod.load_raw(DATA_JSON)


@pytest.fixture(scope="session")
def real_ids(real_raw):
    """真实数据里挑选有代表性的车辆：
    0 = 静止轨迹；246/306/256 = 行驶轨迹；352 = 时间轴故障轨迹。
    """
    want = ["0", "246", "256", "306", "352"]
    return [v for v in want if v in real_raw]
