"""真实实验 runner 的断点、样本隔离和零调用基线测试。"""
from __future__ import annotations

from traj_agent.agent import provider as prov
from traj_agent.experiments.real_ablation import (
    STAGE1_DEMO,
    STAGE1_HOLDOUT,
    append_jsonl,
    build_demo_candidates,
    case_key,
    read_jsonl,
    run_case,
)
from traj_agent.memory.store import MemoryStore


class ExplodingProvider:
    name = "must-not-be-called"

    def chat(self, messages, tools=None):
        raise AssertionError("search-only 不应调用 LLM")


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "raw_results.jsonl"
    row = {
        "phase": "holdout",
        "repetition": 1,
        "mode": "search-only",
        "vehicle_id": "2",
    }
    append_jsonl(path, row)
    assert read_jsonl(path) == [row]
    assert case_key(row) == ("holdout", 1, "search-only", "2")


def test_candidate_manifest_is_nested_and_disjoint(real_raw):
    manifest = build_demo_candidates(
        real_raw, seed=20260921, sizes=(24, 48)
    )
    ids24 = manifest["candidates"]["24"]["vehicle_ids"]
    ids48 = manifest["candidates"]["48"]["vehicle_ids"]
    assert ids24[:len(STAGE1_DEMO)] == STAGE1_DEMO
    assert set(ids24).issubset(ids48)
    assert not (set(ids48) & set(STAGE1_HOLDOUT))
    assert len(ids24) == len(set(ids24)) == 24
    assert len(ids48) == len(set(ids48)) == 48


def test_runner_search_only_has_zero_llm_calls(real_raw):
    row, _ = run_case(
        real_raw,
        "153",
        "search-only",
        repetition=1,
        phase="holdout",
        base_provider=ExplodingProvider(),
        memory=None,
        max_retries=0,
        backoff_s=0,
    )
    assert row["agent_ok"]
    assert row["llm_calls"] == 0
    assert row["llm_turns"] == 0
    assert row["llm_tool_calls"] == 0


def test_runner_holdout_does_not_write_memory(real_raw):
    memory = MemoryStore(":memory:")
    before = memory.count()
    row, _ = run_case(
        real_raw,
        "153",
        "llm+memory+search",
        repetition=1,
        phase="holdout",
        base_provider=prov.MockProvider(),
        memory=memory,
        max_retries=0,
        backoff_s=0,
    )
    assert row["agent_ok"]
    assert memory.count() == before
    memory.close()
