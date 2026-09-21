"""智能体层：循环、预算、provider 回退、结构化输出解析、消融模式。"""
from __future__ import annotations

import json

import pytest

from traj_agent.agent import prompts
from traj_agent.agent import provider as prov
from traj_agent.agent.loop import TrajCleaningAgent
from traj_agent.memory.store import MemoryStore
from traj_agent.tools.registry import ToolRegistry, attach_dataset


# ---- provider -------------------------------------------------------------
def test_build_provider_defaults_to_mock_without_key(monkeypatch):
    monkeypatch.delenv(prov.DEFAULT_API_KEY_ENV, raising=False)
    monkeypatch.delenv("DSH_TRAJ_LLM_PROVIDER", raising=False)
    p = prov.build_provider()
    assert isinstance(p, prov.MockProvider)
    assert p.available


def test_build_provider_honours_explicit_name(monkeypatch):
    monkeypatch.setenv("DSH_TRAJ_LLM_PROVIDER", "mock")
    assert isinstance(prov.build_provider(), prov.MockProvider)


def test_provider_status_does_not_leak_key(monkeypatch):
    monkeypatch.setenv(prov.DEFAULT_API_KEY_ENV, "sk-secret-value")
    status = prov.provider_status()
    assert "sk-secret-value" not in json.dumps(status)
    assert status["has_api_key"] is True


def test_openai_provider_reports_unavailable_without_key(monkeypatch):
    monkeypatch.delenv(prov.DEFAULT_API_KEY_ENV, raising=False)
    p = prov.OpenAICompatProvider(model="x")
    assert p.available is False


def test_openai_provider_parses_tool_calls_from_fake_client():
    """用假 client 验证工具调用解析，不打真实网络。"""
    class _Fn:
        name = "profile_trajectory"
        arguments = json.dumps({"handle": "h1"})

    class _TC:
        id = "call_1"
        function = _Fn()

    class _Msg:
        content = "thinking"
        tool_calls = [_TC()]

    class _Choice:
        message = _Msg()
        finish_reason = "tool_calls"

    class _Usage:
        prompt_tokens = 11
        completion_tokens = 22

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        def create(self, **kw):
            _Completions.last = kw
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    p = prov.OpenAICompatProvider(model="m", api_key="k", client=_Client())
    r = p.chat([{"role": "user", "content": "hi"}],
               tools=[{"type": "function", "function": {"name": "x"}}])
    assert r.has_tool_calls
    assert r.tool_calls[0].name == "profile_trajectory"
    assert r.tool_calls[0].arguments == {"handle": "h1"}
    assert r.prompt_tokens == 11 and r.completion_tokens == 22
    assert _Completions.last["tool_choice"] == "auto"


def test_openai_provider_wraps_client_errors():
    class _Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("network down")

    p = prov.OpenAICompatProvider(model="m", api_key="k", client=_Boom())
    with pytest.raises(prov.ProviderError):
        p.chat([{"role": "user", "content": "hi"}])


# ---- Mock 状态机 ----------------------------------------------------------
def test_mock_advances_one_step_per_chat():
    """回归：Mock 曾因解析对话历史格式而在第 2 轮跳过全部工具。

    改为按对话轮次推进后，每轮应恰好产生一次工具调用直到提议阶段。
    """
    m = prov.MockProvider()
    m.seed("h@v0", {"regime": "moving", "is_stationary": False,
                    "timeline": {"quality": "ok"},
                    "duplicates": {"consecutive_dup_ratio": 0.05}})
    msgs = [{"role": "system", "content": "sys"}]
    seen = []
    for _ in range(6):
        r = m.chat(msgs)
        if not r.has_tool_calls:
            break
        seen.append(r.tool_calls[0].name)
        m.observe(seen[-1], {"ok": True, "handle": "h@v1"})
        msgs.append({"role": "tool", "content": "{}"})
    assert seen == ["clean_trajectory", "split_trajectory",
                    "run_search", "find_knee"], f"工具序列不符: {seen}"


def test_mock_skips_split_when_timeline_unusable():
    m = prov.MockProvider()
    m.seed("h@v0", {"regime": "mixed", "is_stationary": False,
                    "timeline": {"quality": "unusable"},
                    "duplicates": {"consecutive_dup_ratio": 0.5}})
    msgs = [{"role": "system", "content": "s"}]
    seen = []
    for _ in range(6):
        r = m.chat(msgs)
        if not r.has_tool_calls:
            break
        seen.append(r.tool_calls[0].name)
        m.observe(seen[-1], {"ok": True, "handle": "h@v1"})
        msgs.append({"role": "tool", "content": "{}"})
    assert "split_trajectory" not in seen, "时间轴不可用时不该切分"
    assert "simplify_trajectory" in seen


def test_mock_proposal_is_structurally_valid():
    m = prov.MockProvider()
    m.seed("h@v0", {"regime": "moving", "is_stationary": False,
                    "timeline": {"quality": "ok"},
                    "duplicates": {"consecutive_dup_ratio": 0.0}})
    for _ in range(6):
        r = m.chat([{"role": "system", "content": "s"}])
        if not r.has_tool_calls:
            break
    p = prompts.proposal_from_text(r.content)
    assert p is not None
    assert "params" in p and "expected_effect" in p and "rationale" in p
    assert all(isinstance(v, (int, float)) for v in p["params"].values())


# ---- 输出解析 -------------------------------------------------------------
def test_proposal_from_code_block():
    text = '解释\n```json\n{"params": {"a": 1}, "expected_effect": {}, "rationale": "r"}\n```\n完'
    p = prompts.proposal_from_text(text)
    assert p["params"] == {"a": 1}


def test_proposal_from_bare_json_with_prose():
    text = '我的建议是 {"params": {"a": 2}, "expected_effect": {}, "rationale": "r"} 以上。'
    assert prompts.proposal_from_text(text)["params"] == {"a": 2}


def test_proposal_from_final_wrapper():
    text = '{"thought": "x", "final": {"params": {"a": 3}, "expected_effect": {}, "rationale": "r"}}'
    assert prompts.proposal_from_text(text)["params"] == {"a": 3}


def test_proposal_returns_none_without_params():
    assert prompts.proposal_from_text("我觉得取 8 米比较合适。") is None
    assert prompts.proposal_from_text("") is None


def test_proposal_handles_nested_braces_in_strings():
    text = '{"params": {"a": 1}, "rationale": "含 {花括号} 的说明", "expected_effect": {}}'
    p = prompts.proposal_from_text(text)
    assert p is not None and p["params"] == {"a": 1}


def test_parse_text_action_variants():
    a = prompts.parse_text_action(
        '{"thought": "t", "action": "evaluate", "action_input": {"handle": "h"}}')
    assert a["action"] == "evaluate"
    b = prompts.parse_text_action(
        '{"thought": "t", "final": {"params": {}, "expected_effect": {}, "rationale": ""}}')
    assert "final" in b
    assert prompts.parse_text_action("no json here") is None


def test_system_message_contains_key_sections(real_raw):
    from traj_agent.core import diagnosis, params as pm, traj as tm
    from traj_agent.tools.playbook import playbook_digest
    card = diagnosis.diagnose(tm.traj_from_raw("246", *real_raw["246"]))
    msg = prompts.build_system_message(card.to_dict(), pm.param_bounds_table(),
                                       playbook_digest(), {"available": False})
    for token in ("诊断卡", "物理先验", "人类知识笔记", "历史经验", "expected_effect"):
        assert token in msg


def test_system_message_never_contains_coordinates(real_raw):
    from traj_agent.core import diagnosis, params as pm, traj as tm
    from traj_agent.tools.playbook import playbook_digest
    card = diagnosis.diagnose(tm.traj_from_raw("246", *real_raw["246"]))
    msg = prompts.build_system_message(card.to_dict(), pm.param_bounds_table(),
                                       playbook_digest(), {"available": False})
    assert "121.4" not in msg
    assert "coords" not in msg


# ---- 循环 -----------------------------------------------------------------
@pytest.fixture
def agent(real_raw):
    mem = MemoryStore(":memory:")
    reg = ToolRegistry()
    attach_dataset(reg.ctx, real_raw)
    ag = TrajCleaningAgent(registry=reg, llm=prov.MockProvider(), memory=mem)
    yield ag
    mem.close()


def test_agent_runs_full_loop(agent):
    r = agent.run("246")
    assert r.ok, r.error
    assert r.proposal_source == "llm"
    assert r.proposal_params
    assert r.measured
    assert r.verification
    assert r.search["n_evaluations"] > 0
    assert r.tool_calls >= 3


def test_agent_trace_records_every_phase(agent):
    r = agent.run("246")
    kinds = {s.kind for s in r.trace}
    assert {"tool", "llm", "verify", "memory"} <= kinds
    names = {s.name for s in r.trace}
    assert "load_trajectory" in names
    assert "execute_proposal" in names
    assert "coordinate_descent" in names


def test_agent_writes_memory_even_when_rejected(agent):
    """失败经验也要入库（支撑后续分析），只是不被检索推荐。"""
    r = agent.run("246")
    assert r.memory_written
    assert r.episodic_id is not None
    assert agent.memory.count() >= 1


def test_agent_admits_stationary_by_constraint(agent):
    r = agent.run("0")
    assert r.ok
    assert r.verification["applicable"] is False
    assert r.verification["admitted"] is True


def test_agent_result_is_json_serializable(agent):
    r = agent.run("246")
    json.dumps(r.to_dict(), ensure_ascii=False, default=str)


def test_agent_unknown_vehicle_fails_gracefully(agent):
    r = agent.run("no_such_vehicle")
    assert r.ok is False
    assert r.error


def test_agent_respects_tool_budget(real_raw):
    mem = MemoryStore(":memory:")
    reg = ToolRegistry()
    attach_dataset(reg.ctx, real_raw)
    ag = TrajCleaningAgent(registry=reg, llm=prov.MockProvider(), memory=mem,
                           max_tool_calls=2)
    r = ag.run("246")
    # 预算限制的是 LLM 发起的调用；load_trajectory 由 agent 自己发起，不占预算
    assert r.llm_tool_calls <= 2
    mem.close()


def test_agent_handles_provider_failure(real_raw):
    class _Boom(prov.MockProvider):
        def chat(self, messages, tools=None):
            raise prov.ProviderError("boom")

    mem = MemoryStore(":memory:")
    reg = ToolRegistry()
    attach_dataset(reg.ctx, real_raw)
    ag = TrajCleaningAgent(registry=reg, llm=_Boom(), memory=mem)
    r = ag.run("246")
    # provider 挂掉时应回退到默认参数，而不是崩溃
    assert r.ok
    assert r.proposal_source == "fallback"
    mem.close()


def test_agent_without_memory_still_runs(real_raw):
    reg = ToolRegistry()
    attach_dataset(reg.ctx, real_raw)
    ag = TrajCleaningAgent(registry=reg, llm=prov.MockProvider(), memory=None)
    r = ag.run("246")
    assert r.ok
    assert r.memory_written is False


def test_agent_without_search_uses_baseline(real_raw):
    reg = ToolRegistry()
    attach_dataset(reg.ctx, real_raw)
    ag = TrajCleaningAgent(registry=reg, llm=prov.MockProvider(), memory=None,
                           use_search=False)
    r = ag.run("246")
    assert r.ok
    assert r.search["method"] == "disabled"


def test_agent_measures_against_cleaned_reference(agent):
    """回归：口径必须隔离出纯 DP 误差，不能被清洗形变污染。"""
    r = agent.run("246")
    # 相对清洗后轨迹，Hausdorff 应与 DP 偏差界同量级（< 容差的 1.5 倍）
    assert r.measured["hausdorff_m"] <= r.proposal_params["dp_tolerance"] * 1.5


def test_tool_call_accounting_distinguishes_budget_from_total(agent):
    """回归：预算口径与实际调用数曾经漂移。

    load_trajectory 由 agent 自己发起（非 LLM），记入总调用数但不占预算。
    两个数必须分别暴露，否则 max_tool_calls=2 时实际执行 3 次。
    """
    r = agent.run("246")
    assert r.llm_tool_calls <= agent.max_tool_calls
    assert r.tool_calls >= r.llm_tool_calls
    # agent 自己发起的 load_trajectory 计入总数但不计预算
    assert r.tool_calls == r.llm_tool_calls + 1


def test_agent_run_many_creates_run_record(agent):
    results = agent.run_many(["246", "256"])
    assert len(results) == 2
    assert all(r.ok for r in results)
    with agent.memory._lock:
        row = agent.memory._conn.execute("SELECT * FROM runs").fetchone()
    assert row is not None
    assert row["n_cases"] == 2


def test_memory_accumulates_and_is_retrievable(real_raw):
    """记忆闭环：跑完几条后，新轨迹应能检索到相似案例。"""
    mem = MemoryStore(":memory:")
    reg = ToolRegistry()
    attach_dataset(reg.ctx, real_raw)
    ag = TrajCleaningAgent(registry=reg, llm=prov.MockProvider(), memory=mem,
                           regret_threshold=1.0)   # 放宽阈值以便积累样本
    for vid in ("246", "256", "306", "209"):
        ag.run(vid)
    assert mem.count(admitted_only=True) >= 1

    from traj_agent.core import diagnosis, traj as tm
    from traj_agent.memory import retrieve as retrieve_mod
    card = diagnosis.diagnose(tm.traj_from_raw("246", *real_raw["246"]))
    prior = retrieve_mod.prior_for(card, mem, k=3)
    assert prior.available
    assert prior.n_neighbors >= 1
    mem.close()


def test_ablation_modes_are_structurally_distinct(real_raw):
    """消融实验：四种模式必须结构性不同，而不是换了个标签。

    特别注意 search-only 必须**真的不含 LLM 调用**，
    否则它与含 LLM 的模式不可比。
    """
    from traj_agent.verifier.verify import ablation_summary

    specs = [
        # mode,                use_llm, use_memory, use_search, threshold
        ("llm-only",           True,  False, False, 1.0),
        ("search-only",        False, False, True,  0.05),
        ("llm+search",         True,  False, True,  0.05),
        ("llm+memory+search",  True,  True,  True,  1.0),
    ]
    cases = []
    for mode, use_llm, use_mem, use_search, thr in specs:
        mem = MemoryStore(":memory:") if use_mem else None
        reg = ToolRegistry()
        attach_dataset(reg.ctx, real_raw)
        ag = TrajCleaningAgent(registry=reg, llm=prov.MockProvider(), memory=mem,
                               use_llm=use_llm, use_memory=use_mem,
                               use_search=use_search, regret_threshold=thr,
                               mode=mode)
        for vid in ("246", "306"):
            r = ag.run(vid)
            assert r.ok, f"{mode} 在 {vid} 上失败: {r.error}"
            d = r.to_dict(with_trace=False)
            v = d["verification"]
            cases.append({"mode": mode, "regret": v["regret"],
                          "proposal_score": v["proposal_score"],
                          "n_evaluations": d["llm_tool_calls"],
                          "direction_accuracy": v["direction_accuracy"],
                          "admitted": v["admitted"], "_llm_turns": d["llm_turns"]})

    # search-only 必须真的不调用 LLM
    for c in cases:
        if c["mode"] == "search-only":
            assert c["_llm_turns"] == 0, "search-only 不应产生 LLM 轮次"
        else:
            assert c["_llm_turns"] > 0, f"{c['mode']} 应当调用 LLM"

    rows = ablation_summary(cases)
    assert {r["mode"] for r in rows} == {s[0] for s in specs}
    for r in rows:
        assert r["n"] == 2
        assert r["mean_score"] is not None


def test_search_disabled_marks_caveat(real_raw):
    """关闭搜索时必须在结果里标注 regret 口径不可比。"""
    reg = ToolRegistry()
    attach_dataset(reg.ctx, real_raw)
    ag = TrajCleaningAgent(registry=reg, llm=prov.MockProvider(), memory=None,
                           use_search=False)
    r = ag.run("246")
    assert r.search["method"] == "disabled"
    assert "caveat" in r.search
    assert "不可" in r.search["caveat"]


def test_prior_only_proposal_does_not_use_llm(real_raw):
    reg = ToolRegistry()
    attach_dataset(reg.ctx, real_raw)
    ag = TrajCleaningAgent(registry=reg, llm=prov.MockProvider(), memory=None,
                           use_llm=False, mode="search-only")
    r = ag.run("246")
    assert r.ok
    assert r.proposal_source == "prior-only"
    assert r.llm_turns == 0
