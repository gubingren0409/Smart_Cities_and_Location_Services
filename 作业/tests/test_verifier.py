"""核验器：regret 口径、knee point、约束与准入闸门。"""
from __future__ import annotations

import pytest

from traj_agent.verifier import objective as obj_mod
from traj_agent.verifier import search as search_mod
from traj_agent.verifier import verify as verify_mod


# ---- regret 的三种口径 ----------------------------------------------------
def test_regret_normalized():
    r, basis = verify_mod.compute_regret(0.7, 0.9, 0.5)
    assert r == pytest.approx(0.5)
    assert basis == "normalized"


def test_regret_zero_when_matching_best():
    r, basis = verify_mod.compute_regret(0.9, 0.9, 0.5)
    assert r == 0.0


def test_regret_capped_at_one():
    r, _ = verify_mod.compute_regret(0.1, 0.9, 0.5)
    assert r == pytest.approx(1.0)


def test_regret_small_headroom_falls_back_to_absolute():
    """回归：headroom 仅 0.021 时曾算出归一化 regret=1.0。

    车辆 246 实测：best=0.6366、baseline=0.6330、提议=0.6008。
    分母过小会把微小差距放大，故应退回绝对口径。
    """
    r, basis = verify_mod.compute_regret(0.6008, 0.6366, 0.6330)
    assert basis == "absolute(small_headroom)"
    assert r == pytest.approx(0.0358, abs=1e-4)


def test_regret_not_applicable_returns_zero():
    """目标函数不适用时 regret 必须为 0，由调用方按其它判据决策。"""
    r, basis = verify_mod.compute_regret(0.6, 0.7, 0.5, applicable=False)
    assert r == 0.0
    assert basis == "not_applicable"


def test_regret_unnormalized():
    r, basis = verify_mod.compute_regret(0.6, 0.9, 0.5, normalize=False)
    assert r == pytest.approx(0.3)
    assert basis == "absolute(disabled)"


# ---- 目标函数 ------------------------------------------------------------
def test_objective_compression_and_fidelity_bounds():
    r = obj_mod.compute_objective(n_after=25, n_before=100, max_deviation_m=5.0,
                                  tolerance_m=5.0, length_after_m=990.0,
                                  length_before_m=1000.0, runtime_ms=1.0)
    assert r.compression == pytest.approx(0.75)
    assert 0.0 <= r.fidelity <= 1.0
    assert r.feasible


def test_objective_flags_length_loss_as_infeasible():
    r = obj_mod.compute_objective(n_after=5, n_before=100, max_deviation_m=1.0,
                                  tolerance_m=5.0, length_after_m=500.0,
                                  length_before_m=1000.0, runtime_ms=1.0)
    assert not r.feasible
    assert any("长度" in v for v in r.violations)


def test_objective_flags_deviation_beyond_tolerance():
    r = obj_mod.compute_objective(n_after=10, n_before=100, max_deviation_m=50.0,
                                  tolerance_m=5.0, length_after_m=1000.0,
                                  length_before_m=1000.0, runtime_ms=1.0)
    assert not r.feasible
    assert any("偏差" in v for v in r.violations)


def test_short_trajectory_is_not_applicable():
    """回归：静止轨迹（12m）上目标函数必然不可行，应标为不适用。

    实测车辆 0 清洗后总长 12.29m，任何容差都让长度比崩到 40-86%。
    这不是「方案不好」，而是指标在该尺度上没有意义。
    """
    r = obj_mod.compute_objective(n_after=4, n_before=21, max_deviation_m=0.36,
                                  tolerance_m=0.5, length_after_m=10.5,
                                  length_before_m=12.29, runtime_ms=0.1)
    assert r.applicable is False
    assert "物理含义" in r.inapplicable_reason
    assert r.score == 0.0


def test_applicable_flag_survives_to_dict():
    r = obj_mod.compute_objective(n_after=4, n_before=21, max_deviation_m=0.36,
                                  tolerance_m=0.5, length_after_m=10.5,
                                  length_before_m=12.29, runtime_ms=0.1)
    d = r.to_dict()
    assert d["applicable"] is False
    assert d["inapplicable_reason"]


# ---- knee point ----------------------------------------------------------
def test_knee_point_on_synthetic_curve():
    curve = [(0.1, 0.99), (0.3, 0.97), (0.5, 0.94), (0.7, 0.86), (0.9, 0.5)]
    kp = obj_mod.find_knee_point(curve)
    assert kp.index == 3
    assert kp.compression == pytest.approx(0.7)


def test_knee_point_edge_cases():
    assert obj_mod.find_knee_point([]).index == -1
    single = obj_mod.find_knee_point([(0.5, 0.5)])
    assert single.index == 0


def test_knee_point_handles_unsorted_input():
    unsorted = [(0.7, 0.86), (0.1, 0.99), (0.9, 0.5), (0.3, 0.97)]
    kp = obj_mod.find_knee_point(unsorted)
    assert 0 <= kp.index < 4


def test_pareto_front_excludes_dominated():
    results = [
        _mk_result(0.8, 0.9),    # 好
        _mk_result(0.5, 0.5),    # 被支配
        _mk_result(0.9, 0.8),    # 好（压缩更高但保真略低）
    ]
    front = obj_mod.pareto_front(results)
    assert 1 not in front
    assert set(front) == {0, 2}


def _mk_result(comp: float, fid: float):
    r = obj_mod.ObjectiveResult()
    r.compression = comp
    r.fidelity = fid
    return r


# ---- 搜索 ----------------------------------------------------------------
def test_grid_search_covers_all_combinations():
    tr = search_mod.grid_search({"a": 1.0}, {"x": [1, 2, 3], "y": [10, 20]},
                                evaluator=lambda p: p["x"] + p["y"])
    assert tr.n_evaluations == 6
    assert tr.best_score == 23.0


def test_coordinate_descent_improves_over_start():
    def ev(p):
        return -((p["dp_tolerance"] - 8.0) ** 2)
    tr = search_mod.coordinate_descent(
        {"dp_tolerance": 30.0}, ["dp_tolerance"], ev, n_steps=5, rounds=3)
    assert tr.n_evaluations > 1
    assert tr.best_params["dp_tolerance"] == pytest.approx(8.0, abs=2.0)
    assert tr.best_score > ev({"dp_tolerance": 30.0})


def test_search_records_baseline_fields():
    """baseline_score 必须记录，它是归一化 regret 的分母。"""
    tr = search_mod.coordinate_descent({"dp_tolerance": 5.0}, ["dp_tolerance"],
                                      lambda p: -abs(p["dp_tolerance"] - 5))
    tr.baseline_score = 0.42
    assert tr.to_dict()["baseline_score"] == 0.42


def test_linspace_integer_dedupes():
    vals = search_mod.linspace(15, 16, 5, integer=True)
    assert vals == [15.0, 16.0]


def test_search_stays_within_param_bounds():
    tr = search_mod.grid_search({"dp_tolerance": 5.0},
                                {"dp_tolerance": [0.1, 5.0, 1000.0]},
                                evaluator=lambda p: p["dp_tolerance"])
    lo = obj_mod and None
    from traj_agent.core import params as pm
    spec = pm.get_spec("dp_tolerance")
    for p, _ in tr.evaluations:
        assert spec.in_range(p["dp_tolerance"])


def test_search_clamps_and_records_violations():
    tr = search_mod.grid_search({"dp_tolerance": 5.0},
                                {"dp_tolerance": [1000.0]},
                                evaluator=lambda p: p["dp_tolerance"])
    assert "dp_tolerance" in tr.clamped_params


# ---- 方向一致性 ----------------------------------------------------------
def test_direction_check_agreement():
    checks = verify_mod.check_direction(
        {"quality": "down", "compression": "up"},
        {"quality": 0.8, "compression": 0.7},
        {"quality": 0.9, "compression": 0.5})
    assert all(c.agree for c in checks)
    assert verify_mod.direction_accuracy(checks) == 1.0


def test_direction_check_disagreement():
    checks = verify_mod.check_direction(
        {"quality": "up"}, {"quality": 0.5}, {"quality": 0.9})
    assert checks[0].agree is False


def test_direction_accuracy_none_when_no_valid_checks():
    assert verify_mod.direction_accuracy([]) is None
    checks = verify_mod.check_direction({"x": "maybe"}, {"x": 1}, {"x": 2})
    assert verify_mod.direction_accuracy(checks) is None


def test_direction_ignores_keys_not_in_both_dicts():
    checks = verify_mod.check_direction({"a": "up", "b": "up"},
                                        {"a": 2.0}, {"a": 1.0})
    assert len(checks) == 1


# ---- 准入闸门 ------------------------------------------------------------
def test_admission_gate_rejects_high_regret():
    prop = _obj(score=0.3, feasible=True)
    base = _obj(score=0.5, feasible=True)
    tr = search_mod.SearchTrace(method="t")
    tr.baseline_score = 0.5
    tr.record({"dp_tolerance": 1.0}, 0.9)
    res = verify_mod.verify_proposal({"dp_tolerance": 1.0}, prop, base, tr,
                                     regret_threshold=0.05)
    assert not res.admitted
    assert "regret" in res.admit_reason


def test_admission_gate_accepts_good_proposal():
    prop = _obj(score=0.89, feasible=True)
    base = _obj(score=0.5, feasible=True)
    tr = search_mod.SearchTrace(method="t")
    tr.baseline_score = 0.5
    tr.record({"dp_tolerance": 8.0}, 0.9)
    res = verify_mod.verify_proposal({"dp_tolerance": 8.0}, prop, base, tr,
                                     regret_threshold=0.05)
    assert res.admitted
    assert res.regret < 0.05


def test_admission_gate_accepts_inapplicable_by_constraint_only():
    """目标函数不适用时（静止轨迹）按约束合规性准入，不要求 regret。"""
    prop = _obj(score=0.0, feasible=False, applicable=False)
    base = _obj(score=0.0, feasible=False, applicable=False)
    tr = search_mod.SearchTrace(method="t")
    tr.baseline_score = 0.0
    tr.record({"dp_tolerance": 0.5}, 0.0)
    res = verify_mod.verify_proposal({"dp_tolerance": 0.5}, prop, base, tr)
    assert res.admitted
    assert res.regret_basis == "not_applicable"
    assert "不适用" in res.admit_reason


def test_admission_gate_rejects_out_of_range_params():
    prop = _obj(score=0.89, feasible=True)
    base = _obj(score=0.5, feasible=True)
    tr = search_mod.SearchTrace(method="t")
    tr.baseline_score = 0.5
    tr.record({"dp_tolerance": 8.0}, 0.9)
    res = verify_mod.verify_proposal({"dp_tolerance": 999.0}, prop, base, tr)
    assert not res.admitted
    assert "dp_tolerance" in res.clamped_params
    assert res.constraint_ok is False


def test_regret_uses_trace_baseline_not_proposal_baseline():
    """回归：若 baseline 传成「搜索起点」（等于提议），headroom 会被压到 0。

    核验必须优先用 search_trace.baseline_score 这个固定参照系。
    """
    prop = _obj(score=0.60, feasible=True)
    passed_baseline = _obj(score=0.60, feasible=True)   # 人为压低 headroom
    tr = search_mod.SearchTrace(method="t")
    tr.baseline_score = 0.50                            # 真实参照系
    tr.record({"dp_tolerance": 8.0}, 0.6366)
    res = verify_mod.verify_proposal({"dp_tolerance": 8.0}, prop, passed_baseline,
                                     tr, regret_threshold=0.05)
    assert res.regret_basis == "normalized"
    assert 0.0 < res.regret < 1.0


# ---- 消融聚合 ------------------------------------------------------------
def test_ablation_summary_groups_by_mode():
    cases = [
        {"mode": "llm-only", "regret": 0.3, "admitted": False, "n_evaluations": 0},
        {"mode": "llm-only", "regret": 0.5, "admitted": True, "n_evaluations": 0},
        {"mode": "llm+search", "regret": 0.05, "admitted": True, "n_evaluations": 40},
    ]
    rows = verify_mod.ablation_summary(cases)
    by = {r["mode"]: r for r in rows}
    assert by["llm-only"]["n"] == 2
    assert by["llm-only"]["mean_regret"] == pytest.approx(0.4)
    assert by["llm-only"]["admit_rate"] == pytest.approx(0.5)
    assert by["llm+search"]["mean_evals"] == pytest.approx(40.0)


def test_evals_to_reach():
    seq = [({"a": 1}, 0.1), ({"a": 2}, 0.5), ({"a": 3}, 0.96)]
    assert verify_mod.evals_to_reach(seq, 1.0, 0.95) == 3
    assert verify_mod.evals_to_reach([], 1.0) is None


def _obj(score: float, feasible: bool = True, applicable: bool = True):
    r = obj_mod.ObjectiveResult()
    r.score = score
    r.feasible = feasible
    r.applicable = applicable
    r.compression = 0.5
    r.fidelity = 0.5
    return r
