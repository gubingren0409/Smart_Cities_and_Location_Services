"""任务三修复清单的针对性回归测试。"""
from __future__ import annotations

import pytest

from tests.conftest import pt
from traj_agent.agent import provider as prov
from traj_agent.agent.loop import TrajCleaningAgent
from traj_agent.core import params as params_mod
from traj_agent.experiments.analyze_real_ablation import (
    aggregate_case_costs, cluster_stats, latest_case_rows,
)
from traj_agent.tools.registry import ToolRegistry, attach_dataset
from traj_agent.verifier import objective as obj_mod


def _agent_handle(timestamps, coords):
    registry = ToolRegistry()
    attach_dataset(registry.ctx, {"synthetic": [timestamps, coords]})
    loaded = registry.call(
        "load_trajectory", vehicle_id="synthetic", segment_index=0)
    agent = TrajCleaningAgent(
        registry=registry, use_llm=False, use_search=False, memory=None)
    return agent, loaded["handle"]


def test_integer_rounding_is_not_a_constraint_violation():
    audit = params_mod.normalize_params(
        {"dt_threshold": 35.46}, only=["dt_threshold"])
    assert audit.normalized_params["dt_threshold"] == 35.0
    assert audit.execution_params["dt_threshold"] == 35.0
    assert audit.rounded_params == ["dt_threshold"]
    assert audit.out_of_bounds_params == []
    assert audit.constraint_ok


def test_true_out_of_bounds_is_recorded_separately():
    audit = params_mod.normalize_params(
        {"dt_threshold": 300.0}, only=["dt_threshold"])
    assert audit.normalized_params["dt_threshold"] == 300.0
    assert audit.execution_params["dt_threshold"] == 180.0
    assert audit.out_of_bounds_params == ["dt_threshold"]
    assert not audit.constraint_ok


def test_raw_normalized_and_execution_params_are_preserved(real_raw):
    class FractionalProvider:
        name = "fractional"

        def chat(self, messages, tools=None):
            return prov.LLMResponse(
                content=(
                    '{"params":{"dp_tolerance":5.25,'
                    '"dist_threshold":401.5,"max_speed_mps":38.25,'
                    '"dt_threshold":35.46},'
                    '"expected_effect":{},"rationale":"test"}'),
                finish_reason="stop")

    registry = ToolRegistry()
    attach_dataset(registry.ctx, real_raw)
    result = TrajCleaningAgent(
        registry=registry, llm=FractionalProvider(), memory=None,
        use_search=False).run("153")
    payload = result.to_dict(with_trace=False)
    assert payload["raw_proposal_params"]["dt_threshold"] == 35.46
    assert "dt_threshold" not in payload["execution_params"]
    assert payload["normalized_proposal_params"] == payload["execution_params"]
    assert set(payload["execution_params"]) == set(
        params_mod.ACTIVE_EXECUTION_PARAMS)


def test_active_parameter_space_is_aligned_and_inactive_are_excluded():
    assert params_mod.ACTIVE_EXECUTION_PARAMS == (
        "dp_tolerance", "dist_threshold", "max_speed_mps")
    assert "dt_threshold" not in params_mod.active_default_params()
    assert {row["name"] for row in params_mod.param_bounds_table(
        params_mod.ACTIVE_EXECUTION_PARAMS)} == set(
            params_mod.ACTIVE_EXECUTION_PARAMS)


@pytest.mark.parametrize("name,params_a,params_b,timestamps,coords", [
    (
        "dp_tolerance",
        {"dp_tolerance": 0.5, "dist_threshold": 3000, "max_speed_mps": 55},
        {"dp_tolerance": 30, "dist_threshold": 3000, "max_speed_mps": 55},
        [i * 10 for i in range(10)],
        [pt(i * 100, (i % 2) * 20) for i in range(10)],
    ),
    (
        "dist_threshold",
        {"dp_tolerance": 5, "dist_threshold": 50, "max_speed_mps": 55},
        {"dp_tolerance": 5, "dist_threshold": 3000, "max_speed_mps": 55},
        [i * 10 for i in range(7)],
        [pt(0, 0), pt(100, 0), pt(200, 0), pt(1000, 0),
         pt(300, 0), pt(400, 0), pt(500, 0)],
    ),
    (
        "max_speed_mps",
        {"dp_tolerance": 5, "dist_threshold": 3000, "max_speed_mps": 15},
        {"dp_tolerance": 5, "dist_threshold": 3000, "max_speed_mps": 55},
        [i * 10 for i in range(11)],
        [pt(i * 100 if i != 5 else 800, 0) for i in range(11)],
    ),
])
def test_each_active_parameter_can_change_execution(
        name, params_a, params_b, timestamps, coords):
    agent, handle = _agent_handle(timestamps, coords)
    measured_a, _ = agent._execute(handle, params_a)
    measured_b, _ = agent._execute(handle, params_b)
    assert measured_a != measured_b, f"{name} 未改变执行结果"


def test_runtime_does_not_change_primary_quality_score():
    common = dict(
        n_after=40, n_before=100, max_deviation_m=5,
        tolerance_m=5, length_after_m=995, length_before_m=1000)
    fast = obj_mod.compute_objective(**common, runtime_ms=0.1)
    slow = obj_mod.compute_objective(**common, runtime_ms=40.0)
    assert fast.score == pytest.approx(slow.score)
    assert fast.runtime_term != slow.runtime_term


def test_unavailable_road_component_is_removed_from_denominator():
    common = dict(
        n_after=50, n_before=100, max_deviation_m=0,
        tolerance_m=5, length_after_m=1000, length_before_m=1000,
        runtime_ms=1)
    unavailable = obj_mod.compute_objective(**common, road_match_rate=None)
    available_zero = obj_mod.compute_objective(**common, road_match_rate=0.0)
    assert unavailable.score == pytest.approx(0.75)
    assert available_zero.score < unavailable.score


def test_vehicle_cluster_statistics_separate_rows_and_vehicles():
    rows = [
        {"vehicle_id": "a", "repetition": 1, "value": 0.0},
        {"vehicle_id": "a", "repetition": 2, "value": 0.0},
        {"vehicle_id": "a", "repetition": 3, "value": 0.0},
        {"vehicle_id": "b", "repetition": 1, "value": 1.0},
        {"vehicle_id": "b", "repetition": 2, "value": 1.0},
        {"vehicle_id": "b", "repetition": 3, "value": 1.0},
    ]
    result = cluster_stats(rows, lambda row: row["value"], seed=7)
    assert result["observation_n"] == 6
    assert result["unique_vehicle_n"] == 2
    assert result["repetitions"] == 3
    assert result["bootstrap_unit"] == "vehicle_id"


def test_retry_rows_keep_latest_case_for_statistics():
    base = {
        "phase": "holdout", "repetition": 1,
        "mode": "llm-only", "vehicle_id": "10",
    }
    rows = [{**base, "error": "network"}, {**base, "error": ""}]
    assert latest_case_rows(rows) == [{**base, "error": ""}]


def test_retry_rows_preserve_total_cost():
    base = {
        "phase": "holdout", "repetition": 1,
        "mode": "llm-only", "vehicle_id": "10",
    }
    rows = [
        {**base, "llm_calls": 3, "prompt_tokens": 0, "elapsed_ms": 100},
        {**base, "llm_calls": 1, "prompt_tokens": 500, "elapsed_ms": 20},
    ]
    result = aggregate_case_costs(rows)[0]
    assert result["llm_calls"] == 4
    assert result["prompt_tokens"] == 500
    assert result["elapsed_ms"] == 120


def test_deterministic_output_mode_uses_search_best(real_raw):
    registry = ToolRegistry()
    attach_dataset(registry.ctx, real_raw)
    result = TrajCleaningAgent(
        registry=registry,
        llm=prov.MockProvider(),
        memory=None,
        use_llm=False,
        use_search=True,
        proposal_policy="deterministic-search",
        mode="det-search-output",
    ).run("153")
    assert result.llm_turns == 0
    assert result.proposal_params == pytest.approx(
        result.search["deterministic_search_best_params"], abs=1e-4)
    assert result.search["best_observed_source"] == "deterministic-search"
