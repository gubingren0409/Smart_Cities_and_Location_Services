from __future__ import annotations

from pathlib import Path

import pytest

from traj_agent.agent import provider as provider_mod
from traj_agent.experiments import objective_optimization as opt
from traj_agent.experiments import objective_optimization_runner as opt_runner
from traj_agent.experiments.task3_full_run import (
    _terminal_case, append_terminal, coverage_report, load_terminal_rows,
    redact_secret,
)
from traj_agent.memory.store import MemoryStore


def _terminal(vehicle_id: str, status: str = "success"):
    return {"vehicle_id": vehicle_id, "terminal_status": status, "route": "test"}


def test_checkpoint_append_and_resume_skips_completed(tmp_path: Path):
    path = tmp_path / "raw_results.jsonl"
    append_terminal(path, _terminal("1"))
    append_terminal(path, _terminal("2", "fallback_success"))
    rows, by_id = load_terminal_rows(path)
    assert len(rows) == 2
    assert set(by_id) == {"1", "2"}
    pending = [value for value in ("1", "2", "3") if value not in by_id]
    assert pending == ["3"]


def test_duplicate_terminal_detection(tmp_path: Path):
    path = tmp_path / "raw_results.jsonl"
    append_terminal(path, _terminal("1"))
    append_terminal(path, _terminal("1"))
    with pytest.raises(ValueError, match="重复"):
        load_terminal_rows(path)


def test_coverage_exactly_11386_has_no_pending_or_duplicates():
    source = [str(value) for value in range(11386)]
    rows = [_terminal(value) for value in source]
    report = coverage_report(source, rows)
    assert report["n_unique_source_vehicle_ids"] == 11386
    assert report["n_unique_terminal_vehicle_ids"] == 11386
    assert report["coverage_rate"] == 1.0
    assert report["pending"] == 0
    assert report["duplicate_terminal_ids"] == []


def test_secret_redaction():
    secret = "sk-1234567890abcdef"
    text = redact_secret(f"authorization failed for {secret}", [secret])
    assert secret not in text
    assert "<redacted>" in text


def test_not_applicable_and_capacity_fallback_terminal_states():
    raw = {
        "stationary": ([0, 10, 20], [(121.47, 31.23), (121.47001, 31.23), (121.47, 31.23)]),
        "moving": ([0, 10, 20, 30], [(121.47, 31.23), (121.473, 31.23),
                                              (121.476, 31.23), (121.479, 31.23)]),
    }
    store = MemoryStore(":memory:")
    try:
        stationary, used = _terminal_case(
            raw, "stationary", store, provider_mod.MockProvider(),
            similarity_threshold=.75, llm_allowed=False, search_budget=1)
        assert stationary["terminal_status"] == "not_applicable"
        assert not used
        moving, used = _terminal_case(
            raw, "moving", store, provider_mod.MockProvider(),
            similarity_threshold=.75, llm_allowed=False, search_budget=1)
        assert moving["terminal_status"] == "fallback_success"
        assert moving["route"] == "deterministic-capacity-fallback"
        assert not used
    finally:
        store.close()


def test_retry_counts_all_attempts_and_usage(monkeypatch):
    raw = {
        "moving": ([0, 10, 20, 30], [(121.47, 31.23), (121.473, 31.23),
                                              (121.476, 31.23), (121.479, 31.23)]),
    }
    evaluator = opt.TrajectoryEvaluator(raw, "moving")

    class Flaky:
        def __init__(self):
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            if self.calls < 3:
                raise provider_mod.ProviderError("503 temporary unavailable")
            return provider_mod.LLMResponse(
                content='{"regions":{"dp_tolerance":[1,5],"dist_threshold":[200,400],"max_speed_mps":[20,40]}}',
                prompt_tokens=100, completion_tokens=20, elapsed_ms=50,
            )

    monkeypatch.setattr(opt_runner.time, "sleep", lambda _: None)
    row = opt_runner.call_region_proposal(
        Flaky(), evaluator, {}, repetition=1, teacher_size=200,
        memory_mode="episodic+procedural", max_retries=2, backoff_s=0,
    )
    assert row["llm_calls"] == 3
    assert row["retry_count"] == 2
    assert len(row["provider_errors"]) == 2
    assert row["prompt_tokens"] == 100
    assert row["completion_tokens"] == 20
    assert row["json_success"]
