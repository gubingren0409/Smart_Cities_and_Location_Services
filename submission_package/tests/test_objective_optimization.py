"""固定 Objective、Search Teacher 与 warm-start 的回归测试。"""
from __future__ import annotations

import json
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess

import pytest

from traj_agent.core import diagnosis, params as params_mod, traj as traj_mod
from traj_agent.experiments import analyze_objective_optimization as analyzer
from traj_agent.experiments import objective_optimization as opt
from traj_agent.experiments import objective_optimization_refined as refined
from traj_agent.experiments import objective_optimization_refined_runner as refined_runner
from traj_agent.memory.store import MemoryStore
from traj_agent.verifier import search as search_mod


def test_teacher_set_and_holdout_are_disjoint(real_raw):
    holdout = ("2", "4", "10")
    manifest = opt.teacher_manifest(
        real_raw, holdout=holdout, excluded_demo=("0", "1"), size=24, seed=7)
    assert manifest["teacher_holdout_overlap"] == []
    assert not set(manifest["vehicle_ids"]) & set(holdout)
    assert len(set(manifest["vehicle_ids"])) == 24


def test_teacher_runner_uses_no_llm_and_best_comes_from_trace(real_raw):
    row = opt.run_teacher_case(real_raw, "153", max_evals=10)
    assert row["llm_calls"] == 0
    assert row["n_evaluations"] == 10
    best = max(row["search_trace"], key=lambda item: item["score"])
    assert row["search_best_score"] == pytest.approx(best["score"])
    assert row["search_best_params"] == best["params"]


def test_memory_source_is_explicit(real_raw):
    card_moving = diagnosis.diagnose(
        traj_mod.traj_from_raw("153", *real_raw["153"]))
    store = MemoryStore(":memory:")
    for index in range(3):
        store.write_case(
            card=card_moving,
            params={
                "dp_tolerance": 4.0 + index,
                "dist_threshold": 300.0 + index,
                "max_speed_mps": 35.0 + index,
            },
            metrics={}, objective={"score": 0.8}, verification={},
            admitted=True, score=0.8, regret=0.0,
            source=opt.TEACHER_SOURCE,
        )
    store.rebuild_procedural(
        param_names=params_mod.ACTIVE_EXECUTION_PARAMS,
        min_samples=3,
        source=opt.TEACHER_SOURCE,
    )
    cases = store.retrieve_similar(
        card_moving, exclude_self=False, sources=(opt.TEACHER_SOURCE,))
    regions = store.query_regions(source=opt.TEACHER_SOURCE)
    assert cases and all(case.source == opt.TEACHER_SOURCE for case in cases)
    assert regions and all(region.source == opt.TEACHER_SOURCE for region in regions)
    store.close()


def test_region_proposal_is_clamped_to_active_bounds():
    normalized = opt.normalize_region_proposal({
        "regions": {
            "dp_tolerance": [-10, 100],
            "dist_threshold": [900, 200],
            "max_speed_mps": {"low": 20, "high": 40},
        }
    })
    bounds = normalized["normalized_regions"]
    assert bounds["dp_tolerance"] == [0.5, 30.0]
    assert bounds["dist_threshold"] == [200.0, 900.0]
    assert bounds["max_speed_mps"] == [20.0, 40.0]
    assert "dp_tolerance" in normalized["clamped_regions"]


def test_warmstart_and_pure_search_use_identical_budget():
    evaluator = lambda p: -sum((float(v) - 5.0) ** 2 for v in p.values())
    defaults = {name: 5.0 for name in params_mod.ACTIVE_EXECUTION_PARAMS}
    pure = search_mod.budgeted_region_search(
        defaults, evaluator, 10, bounds=search_mod.full_parameter_bounds())
    warm = search_mod.budgeted_region_search(
        defaults, evaluator, 10,
        bounds={name: (1.0, 10.0) for name in defaults})
    assert pure.n_evaluations == warm.n_evaluations == 10


def test_gain_recovery_headroom_and_near_search_definition():
    small = opt.optimization_metrics(
        method_score=0.505, default_score=0.50, search_score=0.51,
        epsilon=0.01, headroom_threshold=0.02)
    assert small["gain_recovery"] is None
    assert small["near_search"] is True

    regular = opt.optimization_metrics(
        method_score=0.58, default_score=0.50, search_score=0.60,
        epsilon=0.01, headroom_threshold=0.02)
    assert regular["gain_recovery"] == pytest.approx(0.8)
    assert regular["gap_to_search"] == pytest.approx(0.02)
    assert regular["near_search"] is False


def test_memory_ablation_reads_only_requested_layer(real_raw):
    card_moving = diagnosis.diagnose(
        traj_mod.traj_from_raw("153", *real_raw["153"]))
    store = MemoryStore(":memory:")
    for index in range(3):
        store.write_case(
            card=card_moving,
            params={
                "dp_tolerance": 3.0 + index,
                "dist_threshold": 300.0 + index,
                "max_speed_mps": 35.0 + index,
            }, metrics={}, objective={"score": 0.8}, verification={},
            admitted=True, score=0.8, regret=0.0,
            source=opt.TEACHER_SOURCE,
        )
    store.rebuild_procedural(
        param_names=params_mod.ACTIVE_EXECUTION_PARAMS,
        min_samples=3, source=opt.TEACHER_SOURCE)

    query = replace(card_moving, seg_id="query#0", vehicle_id="query")
    episodic = opt.memory_prior(query, store, "episodic-only")
    procedural = opt.memory_prior(query, store, "procedural-only")
    assert episodic["neighbors"] and episodic["suggested_regions"] == []
    assert procedural["neighbors"] == [] and procedural["suggested_regions"]
    store.close()


def test_warmstart_does_not_change_objective_evaluation(real_raw):
    evaluator = opt.TrajectoryEvaluator(real_raw, "153")
    params = params_mod.active_default_params()
    before = evaluator.evaluate(params, cache=False)
    outcome = opt.run_budgeted_method(
        evaluator,
        bounds=opt.active_bounds(),
        start_params=params,
        budget=3,
        reference_score=before + 0.1,
        reference_params=params,
    )
    assert outcome["trace"][0]["score"] == pytest.approx(before)


def test_retry_rows_use_latest_result_but_preserve_all_cost(tmp_path):
    path = tmp_path / "region_proposals.jsonl"
    base = {
        "repetition": 1, "teacher_size": 12,
        "memory_mode": "episodic-only", "vehicle_id": "2",
    }
    first = {**base, "error": "timeout", "llm_calls": 2,
             "prompt_tokens": 10, "completion_tokens": 1,
             "retry_count": 1, "provider_errors": ["timeout"],
             "elapsed_ms": 100.0, "json_success": False}
    second = {**base, "error": "", "llm_calls": 1,
              "prompt_tokens": 20, "completion_tokens": 2,
              "retry_count": 0, "provider_errors": [],
              "elapsed_ms": 50.0, "json_success": True}
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8")
    rows = analyzer._aggregate_proposal_costs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["error"] == ""
    assert rows[0]["llm_calls"] == 3
    assert rows[0]["prompt_tokens"] == 30
    assert rows[0]["provider_error_count"] == 1


def test_strong_reference_uses_no_llm_and_is_not_weaker(real_raw):
    defaults = params_mod.active_default_params()
    old = {
        "applicable": True,
        "search_best_score": -999.0,
        "search_best_params": defaults,
        "n_evaluations": 1,
    }
    row = refined.run_strong_reference_case(
        real_raw, "153", old,
        global_budget=3, multi_starts=1, cd_max_evals=3)
    assert row["llm_calls"] == 0
    assert row["strong_reference_score"] >= row["old_reference_score"]


def test_reference_containment_is_closed_and_multidimensional():
    bounds = {
        "dp_tolerance": (1.0, 5.0),
        "dist_threshold": (100.0, 500.0),
        "max_speed_mps": (20.0, 40.0),
    }
    assert refined.region_contains({
        "dp_tolerance": 1.0,
        "dist_threshold": 500.0,
        "max_speed_mps": 30.0,
    }, bounds)
    assert not refined.region_contains({
        "dp_tolerance": 1.0,
        "dist_threshold": 501.0,
        "max_speed_mps": 30.0,
    }, bounds)


def test_normalized_region_volume_exact_full_and_point():
    full = opt.active_bounds()
    assert refined.normalized_region_volume(full) == pytest.approx(1.0)
    half = dict(full)
    low, high = full["dp_tolerance"]
    half["dp_tolerance"] = (low, low + (high - low) / 2.0)
    assert refined.normalized_region_volume(half) == pytest.approx(0.5)
    point = {name: (bounds[0], bounds[0]) for name, bounds in full.items()}
    assert refined.normalized_region_volume(point) == pytest.approx(0.0)


def test_deterministic_memory_regions_have_zero_llm_cost():
    prior = {
        "neighbors": [
            {"params": {"dp_tolerance": value,
                        "dist_threshold": 300 + value,
                        "max_speed_mps": 30 + value}}
            for value in (2.0, 3.0, 4.0, 5.0, 6.0)
        ],
        "suggested_regions": [
            {"param": "dp_tolerance", "low": 2.0, "high": 6.0},
        ],
    }
    episodic = refined.episodic_region_bounds(prior)
    procedural = refined.procedural_region_bounds(prior)
    assert refined.normalized_region_volume(episodic) < 1.0
    assert refined.normalized_region_volume(procedural) < 1.0
    assert refined.online_cost(search_elapsed_ms=12.0)["llm_calls"] == 0


def test_online_cost_excludes_offline_teacher_and_sums_llm_search():
    cost = refined.online_cost(
        search_elapsed_ms=25.0,
        proposal={
            "llm_elapsed_ms": 75.0,
            "elapsed_ms": 90.0,
            "llm_calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 2,
        },
    )
    assert cost["total_online_ms"] == pytest.approx(100.0)
    assert cost["end_to_end_online_ms"] == pytest.approx(115.0)
    assert "teacher" not in cost


def test_paired_rows_align_by_vehicle_repetition_and_budget():
    rows = [
        {"vehicle_id": "a", "repetition": 1, "budget": 3},
        {"vehicle_id": "b", "repetition": 2, "budget": 10},
    ]
    assert refined.validate_paired_alignment(rows, list(reversed(rows)))
    assert not refined.validate_paired_alignment(rows, rows[:1])


def test_objective_hash_matches_frozen_experiment_config():
    repo_dir = Path(__file__).resolve().parents[2]
    config = json.loads((
        repo_dir / "作业" / "experiments" /
        refined_runner.OLD_EXPERIMENT / "config.json"
    ).read_text(encoding="utf-8"))
    objective = repo_dir / "作业" / "traj_agent" / "verifier" / "objective.py"
    actual = "sha256:" + hashlib.sha256(objective.read_bytes()).hexdigest()
    assert actual == config["objective_version"]


def test_v2_experiment_is_untouched_since_frozen_baseline():
    repo_dir = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "git", "diff", "--quiet", refined_runner.BASELINE_COMMIT, "--",
            "作业/experiments/llm_assisted_ecnu_20260921_v2",
        ],
        cwd=repo_dir,
        check=False,
    )
    assert result.returncode == 0
