"""轨迹分段：已知答案测试。"""
from __future__ import annotations

import pytest

from traj_agent.core import segment
from traj_agent.core.traj import Traj


def test_no_cut_on_clean_straight_trajectory(straight_traj):
    res = segment.split_trajectory(straight_traj, dt_threshold=30, dist_threshold=400,
                                  min_points=2, min_length_m=0)
    assert res.n_segments == 1
    assert res.cut_points == []
    assert res.n_points_kept == len(straight_traj)


def test_time_gap_cuts_at_exact_index(timegap_traj):
    """间隔 500s 出现在索引 4，切分点必须是 4，且原因是 time_gap。"""
    res = segment.split_trajectory(timegap_traj, dt_threshold=30, dist_threshold=1e9,
                                  min_points=2, min_length_m=0)
    assert res.n_segments == 2
    assert [i for i, _ in res.cut_points] == [4]
    assert segment.TRIGGER_TIME_GAP in res.cut_points[0][1]
    assert len(res.segments[0]) == 4
    assert len(res.segments[1]) == 4


def test_space_jump_cuts_at_exact_index(jump_traj):
    """索引 3 跳到 5km 外、索引 4 又跳回来。

    两处距离分别是 4800m 与 4700m，都超过 400m 阈值，
    故切分点应为 [3, 4]，孤立跳跃点被切成独占一段，共 3 段。
    """
    res = segment.split_trajectory(jump_traj, dt_threshold=1e9, dist_threshold=400,
                                  min_points=2, min_length_m=0)
    assert [i for i, _ in res.cut_points] == [3, 4]
    for _, triggers in res.cut_points:
        assert segment.TRIGGER_SPACE_JUMP in triggers
    # 中间那段只有一个点（孤立跳跃点），因 min_points>=2 进 dropped，
    # 故保留段为 2 段、丢弃 1 段；三段之和仍等于原点数。
    assert res.n_segments == 2
    assert [len(s) for s in res.segments] == [3, 3]
    assert [len(s) for s in res.dropped] == [1]
    assert res.n_points_kept + sum(len(s) for s in res.dropped) == len(jump_traj)


def test_both_triggers_recorded_on_same_cut():
    """同时命中时间与空间规则时，两个原因都要记录，不能只留一个。"""
    ts = [0, 10, 20, 1020, 1030]              # 索引 3 处 dt=1000s
    coords = [(121.4700, 31.2300), (121.4710, 31.2300), (121.4720, 31.2300),
              (121.5700, 31.3300), (121.5710, 31.3300)]  # 且跳跃 >10km
    t = Traj("both", ts, coords)
    res = segment.split_trajectory(t, dt_threshold=30, dist_threshold=400,
                                  min_points=2, min_length_m=0)
    idx = [i for i, _ in res.cut_points]
    assert idx == [3]
    triggers = res.cut_points[0][1]
    assert segment.TRIGGER_TIME_GAP in triggers
    assert segment.TRIGGER_SPACE_JUMP in triggers


def test_nonpositive_dt_cuts(nonmonotonic_traj):
    res = segment.split_trajectory(nonmonotonic_traj, dt_threshold=30, dist_threshold=1e9,
                                  min_points=2, min_length_m=0)
    assert segment.TRIGGER_ILLEGAL_TIME in res.trigger_counts() or \
           segment.TRIGGER_TIME_GAP in res.trigger_counts()


def test_min_points_moves_short_seg_to_dropped(jump_traj):
    """min_points 只影响保留与否，不影响切分位置。"""
    res_keep = segment.split_trajectory(jump_traj, dt_threshold=1e9, dist_threshold=400,
                                       min_points=2, min_length_m=0)
    res_drop = segment.split_trajectory(jump_traj, dt_threshold=1e9, dist_threshold=400,
                                       min_points=5, min_length_m=0)
    assert [i for i, _ in res_keep.cut_points] == [i for i, _ in res_drop.cut_points]
    assert res_drop.n_segments < res_keep.n_segments
    assert len(res_drop.dropped) > 0


def test_min_length_moves_short_seg_to_dropped():
    """长度不足的分段应进 dropped。"""
    ts = [0, 10, 20, 30, 40, 50]
    coords = [(121.4700, 31.2300), (121.4700, 31.2301), (121.4700, 31.2302),
              (121.4700, 31.2303), (121.4700, 31.2304), (121.4700, 31.2305)]  # 总长约 55m
    t = Traj("short", ts, coords)
    res_keep = segment.split_trajectory(t, min_points=2, min_length_m=0)
    res_drop = segment.split_trajectory(t, min_points=2, min_length_m=65)
    assert res_keep.n_segments == 1
    assert res_drop.n_segments == 0
    assert len(res_drop.dropped) == 1


def test_illegal_coords_do_not_trigger_space_jump(illegal_traj):
    """非法坐标不能参与空间跳跃判断。

    夹具里正常点间距是 100m，故阈值取 200m：此时**不该有任何**空间跳跃。
    若非法点参与了距离计算（尤其是越界点 (200,100) 距上海约 7500km），
    就会凭空产生 space_jump —— 那才是 bug。
    注意不用 50m 这类低于正常间距的阈值：那会每步都切分，
    反而掩盖了「非法点被拦下」这个真正要检验的性质。
    """
    res = segment.split_trajectory(illegal_traj, dt_threshold=1e9, dist_threshold=200,
                                  min_points=2, min_length_m=0)
    assert segment.TRIGGER_SPACE_JUMP not in res.trigger_counts()
    assert res.cut_points == []

    # 反向对照：把阈值压到 50m（低于正常间距），说明该夹具确实能触发跳跃规则，
    # 从而证明上一条断言的「无跳跃」不是因为规则整体失效。
    res_low = segment.split_trajectory(illegal_traj, dt_threshold=1e9, dist_threshold=50,
                                      min_points=2, min_length_m=0)
    assert segment.TRIGGER_SPACE_JUMP in res_low.trigger_counts()
    # 且低阈值下的切分点只出现在两个正常点之间（索引 1 与 4），
    # 从不出现于含非法点的相邻对（索引 2、3、5）
    assert [i for i, _ in res_low.cut_points] == [1, 4]


def test_thresholds_zero_cut_everywhere(straight_traj):
    res = segment.split_trajectory(straight_traj, dt_threshold=0, dist_threshold=0,
                                  min_points=2, min_length_m=0)
    assert len(res.cut_points) == len(straight_traj) - 1


def test_huge_thresholds_never_cut(straight_traj):
    res = segment.split_trajectory(straight_traj, dt_threshold=1e9, dist_threshold=1e9,
                                  min_points=2, min_length_m=0)
    assert res.n_segments == 1
    assert res.cut_points == []


def test_segment_index_assigned_sequentially(timegap_traj):
    res = segment.split_trajectory(timegap_traj, dt_threshold=30, dist_threshold=1e9,
                                  min_points=2, min_length_m=0)
    assert [s.segment_index for s in res.segments] == list(range(res.n_segments))


def test_segment_points_partition_original(timegap_traj):
    """分段 + dropped 必须恰好是原轨迹的一个划分，不重不漏。"""
    res = segment.split_trajectory(timegap_traj, dt_threshold=30, dist_threshold=1e9,
                                  min_points=2, min_length_m=0)
    total = res.n_points_kept + sum(len(s) for s in res.dropped)
    assert total == len(timegap_traj)


def test_negative_threshold_raises(straight_traj):
    with pytest.raises(ValueError):
        segment.split_trajectory(straight_traj, dt_threshold=-1)


def test_tiny_trajectory_does_not_crash():
    """退化输入不应崩溃：单点轨迹、以及两点但被阈值切开的轨迹。"""
    t = Traj("one", [0], [(121.47, 31.23)])
    res = segment.split_trajectory(t)
    assert res.n_segments == 0

    # 两点相距约 950m > 默认 400m 阈值，必然被切成两个单点段，
    # 而单点段因 min_points>=2 进 dropped —— 这是正确行为。
    t2 = Traj("two", [0, 10], [(121.47, 31.23), (121.48, 31.23)])
    res2 = segment.split_trajectory(t2, min_points=2, min_length_m=0)
    assert res2.n_segments == 0
    assert len(res2.dropped) == 2

    # 把阈值放宽，两点应构成一段
    res3 = segment.split_trajectory(t2, dist_threshold=1e9, min_points=2, min_length_m=0)
    assert res3.n_segments == 1


def test_trigger_counts_aggregate(timegap_traj):
    res = segment.split_trajectory(timegap_traj, dt_threshold=30, dist_threshold=1e9,
                                  min_points=2, min_length_m=0)
    counts = res.trigger_counts()
    assert counts.get(segment.TRIGGER_TIME_GAP, 0) == 1


def test_sweep_returns_full_grid(straight_traj):
    rows = segment.sweep_split_thresholds(straight_traj, [10, 30, 60], [100, 400])
    assert len(rows) == 6
    assert {"dt_threshold", "dist_threshold", "n_segments", "n_points_kept"} <= set(rows[0])


def test_real_data_stationary_trajectory_is_all_dropped(real_raw):
    """真实数据：车辆 0 是静止轨迹（位移 12m），按默认规则应全部丢弃。"""
    from traj_agent.core import traj as traj_mod
    t = traj_mod.traj_from_raw("0", *real_raw["0"])
    res = segment.split_trajectory(t)
    assert res.n_segments == 0
    assert res.n_points_kept == 0


def test_real_data_moving_trajectory_is_kept(real_raw):
    """真实数据：车辆 246 是行驶轨迹（11km），应保留绝大部分点。"""
    from traj_agent.core import traj as traj_mod
    t = traj_mod.traj_from_raw("246", *real_raw["246"])
    res = segment.split_trajectory(t)
    assert res.n_points_kept >= 0.8 * len(t)
