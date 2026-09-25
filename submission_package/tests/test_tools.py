"""工具层：schema 完整性、坐标不外泄、副作用标注、参数越界处理。"""
from __future__ import annotations

import json

import pytest

from traj_agent.core import traj as tm
from traj_agent.tools import playbook
from traj_agent.tools.registry import ToolContext, ToolRegistry, attach_dataset
from traj_agent.tools.store import TrajStore, state_hash


@pytest.fixture
def registry(real_raw):
    reg = ToolRegistry()
    attach_dataset(reg.ctx, real_raw)
    return reg


# ---- schema ---------------------------------------------------------------
def test_every_tool_has_openai_schema(registry):
    tools = registry.openai_tools()
    assert len(tools) == len(registry.names())
    for t in tools:
        assert t["type"] == "function"
        fn = t["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"


def test_tool_schemas_are_json_serializable(registry):
    json.dumps(registry.openai_tools())


def test_expected_tools_present(registry):
    names = set(registry.names())
    required = {"load_trajectory", "profile_trajectory", "detect_anomalies",
                "split_trajectory", "clean_trajectory", "simplify_trajectory",
                "evaluate", "compare_handles", "run_search", "find_knee",
                "query_memory", "query_playbook", "render",
                "apply_road_constraint", "suggest_param_range"}
    assert required <= names


def test_no_memory_write_tool_exists(registry):
    """核心设计约束：LLM 没有任何写记忆的工具。"""
    for name in registry.names():
        assert "write" not in name.lower(), f"{name} 看起来是写操作工具"
        assert "save" not in name.lower()
        assert "delete" not in name.lower()


def test_unknown_tool_returns_error_not_raises(registry):
    out = registry.call("no_such_tool")
    assert out["ok"] is False
    assert "未知工具" in out["error"]


# ---- 坐标不外泄 -----------------------------------------------------------
def test_responses_never_contain_raw_coordinates(registry):
    """诊断卡与所有工具响应都不得包含坐标序列。"""
    h = registry.call("load_trajectory", vehicle_id="246")["handle"]
    for name, args in [
        ("profile_trajectory", {"handle": h}),
        ("detect_anomalies", {"handle": h}),
        ("suggest_param_range", {"handle": h}),
        ("evaluate", {"handle": h}),
    ]:
        out = registry.call(name, **args)
        text = json.dumps(out, ensure_ascii=False, default=str)
        # 上海经度区间的数值（121.xx）不该出现在标量统计里
        assert "121.4" not in text.replace("121.4700", ""), \
            f"{name} 的响应疑似泄漏了坐标: {text[:200]}"


def test_diagnosis_card_has_no_coordinates(registry):
    out = registry.call("load_trajectory", vehicle_id="246")
    text = json.dumps(out["diagnosis"], ensure_ascii=False)
    assert "coords" not in text
    assert "121.4" not in text
    assert "timestamps" not in text


# ---- 工具链 ---------------------------------------------------------------
def test_full_chain_produces_handles(registry):
    h = registry.call("load_trajectory", vehicle_id="246")["handle"]
    assert registry.ctx.store.exists(h)

    hc = registry.call("clean_trajectory", handle=h)["handle"]
    assert registry.ctx.store.exists(hc)
    assert registry.ctx.store.record(hc).parent == h

    sp = registry.call("split_trajectory", handle=hc, dt_threshold=30,
                       dist_threshold=400)
    for sh in sp["segment_handles"]:
        assert registry.ctx.store.exists(sh)

    hs = registry.call("simplify_trajectory", handle=hc, tolerance_m=5.0)["handle"]
    assert registry.ctx.store.exists(hs)


def test_lineage_recorded(registry):
    h = registry.call("load_trajectory", vehicle_id="246")["handle"]
    hc = registry.call("clean_trajectory", handle=h)["handle"]
    hs = registry.call("simplify_trajectory", handle=hc, tolerance_m=5.0)["handle"]
    ops = [l["operation"] for l in registry.ctx.store.lineage(hs)]
    assert ops[0] == "load"
    assert "clean" in ops
    assert "simplify" in ops


def test_unknown_handle_returns_error(registry):
    out = registry.call("profile_trajectory", handle="nope@v0")
    assert out["ok"] is False
    assert "未知句柄" in out["error"]


def test_simplify_clamps_out_of_range_tolerance(registry):
    h = registry.call("load_trajectory", vehicle_id="246")["handle"]
    out = registry.call("simplify_trajectory", handle=h, tolerance_m=999.0)
    assert out["ok"]
    assert "dp_tolerance" in out["clamped_params"]
    assert out["result"]["tolerance_m"] <= 30.0


def test_call_json_parses_arguments(registry):
    h = registry.call("load_trajectory", vehicle_id="246")["handle"]
    payload = registry.call_json("profile_trajectory",
                                 json.dumps({"handle": h}))
    assert json.loads(payload)["ok"] is True


def test_call_json_handles_bad_json(registry):
    payload = registry.call_json("profile_trajectory", "{not json")
    assert json.loads(payload)["ok"] is False


def test_call_log_records_elapsed(registry):
    registry.call("load_trajectory", vehicle_id="246")
    assert registry.ctx.call_log
    assert registry.ctx.call_log[-1]["elapsed_ms"] >= 0.0


# ---- find_knee ------------------------------------------------------------
def test_find_knee_returns_param_value(registry):
    """回归：find_knee 曾因字段名不匹配返回 value=0.0。"""
    h = registry.call("load_trajectory", vehicle_id="246")["handle"]
    hc = registry.call("clean_trajectory", handle=h)["handle"]
    curve = registry.call("run_search", handle=hc, param="dp_tolerance",
                          n=6, reference_handle=hc)["curve"]
    out = registry.call("find_knee", curve_json=json.dumps(curve))
    assert out["ok"]
    knee = out["knee"]
    assert "value" in knee
    assert knee["value"] > 0.0, "knee 的参数值不能为 0"
    assert knee["n_points"] > 0
    assert knee["max_deviation_m"] >= 0.0


def test_find_knee_rejects_bad_json(registry):
    out = registry.call("find_knee", curve_json="[[[")
    assert out["ok"] is False


def test_find_knee_rejects_empty_array(registry):
    out = registry.call("find_knee", curve_json="[]")
    assert out["ok"] is False


# ---- 路网 -----------------------------------------------------------------
def test_road_constraint_produces_handle(registry):
    h = registry.call("load_trajectory", vehicle_id="246")["handle"]
    out = registry.call("apply_road_constraint", handle=h)
    assert out["ok"]
    assert out["road_stats"]["match_rate"] > 0
    assert registry.ctx.store.exists(out["handle"])


def test_road_constraint_rejects_degenerate_road():
    """回归：退化路网（总长 < 50m）必须被拒绝，不能报出误导性的 match_rate=1.0。

    构造一条只有几米的轨迹：简化后折线长度趋零，
    任何点投影到它上面距离都是 0，match_rate 会是 1.0——
    这个数字看起来完美但毫无信息量。
    """
    reg = ToolRegistry()
    from traj_agent.core.traj import Traj
    tiny = Traj("tiny", [0, 10, 20],
                [(121.4700, 31.2300), (121.47001, 31.2300), (121.47002, 31.2300)])
    reg.ctx.store.put(tiny)
    out = reg.call("apply_road_constraint", handle="tiny#0@v0")
    assert out["ok"] is False
    assert "过短" in out["error"]


def test_roads_from_trajectory_refuses_short_polyline():
    from traj_agent.core.traj import Traj
    from traj_agent.road import matcher as road_mod
    short = Traj("s", [0, 10], [(121.4700, 31.2300), (121.47005, 31.2300)])
    assert road_mod.roads_from_trajectory(short) == []


def test_road_match_rate_on_real_trajectory_is_meaningful(real_raw):
    from traj_agent.core.traj import Traj
    from traj_agent.road import matcher as road_mod
    ts, coords = real_raw["246"]
    t = Traj("246", list(ts), [tuple(p) for p in coords])
    roads = road_mod.roads_from_trajectory(t)
    assert len(roads) == 1
    m = road_mod.PolylineRoadMatcher(roads, match_tolerance_m=30.0)
    stats = m.match(t).stats()
    # 用轨迹自身几何构造的路网，匹配率应当很高但不该是恒等的 1.0
    assert stats["match_rate"] > 0.5
    assert stats["distance_p95_m"] < 30.0


# ---- compare_handles 口径 -------------------------------------------------
def test_compare_handles_includes_metric_caveat(registry):
    h = registry.call("load_trajectory", vehicle_id="246")["handle"]
    hc = registry.call("clean_trajectory", handle=h)["handle"]
    hs = registry.call("simplify_trajectory", handle=hc, tolerance_m=5.0)["handle"]
    out = registry.call("compare_handles", handle_a=hs, handle_b=h)
    assert "caveat" in out
    assert "清洗" in out["caveat"]


def test_hausdorff_matches_dp_bound_via_tools(registry):
    h = registry.call("load_trajectory", vehicle_id="246")["handle"]
    hc = registry.call("clean_trajectory", handle=h)["handle"]
    out = registry.call("simplify_trajectory", handle=hc, tolerance_m=5.0)
    hs = out["handle"]
    cmp = registry.call("compare_handles", handle_a=hs, handle_b=hc)
    # result 里的 max_deviation_m 经过了 to_dict 的 3 位舍入，
    # 故此处用 5e-3 容差比对，而不是 1e-6。
    assert cmp["hausdorff_m"] == pytest.approx(out["result"]["max_deviation_m"],
                                              abs=5e-3)


# ---- playbook -------------------------------------------------------------
def test_playbook_loads_stub_notes():
    notes = playbook.load_notes()
    assert len(notes) >= 3
    assert any(n.param == "dp_tolerance" for n in notes)


def test_playbook_frontmatter_dates_are_json_safe():
    """回归：yaml 把 updated 解析成 date，直接 json.dumps 会失败。"""
    digest = playbook.playbook_digest()
    json.dumps(digest)
    for n in digest["notes"]:
        assert isinstance(n.get("updated"), (str, type(None)))


def test_playbook_is_read_only():
    status = playbook.vault_status()
    assert status["read_only"] is True


def test_playbook_missing_vault_returns_empty(tmp_path):
    notes = playbook.load_notes(str(tmp_path / "does_not_exist"))
    assert notes == []
    assert playbook.playbook_digest(str(tmp_path / "nope"))["available"] is False


def test_playbook_parses_wikilinks():
    notes = playbook.load_notes()
    all_links = [l for n in notes for l in n.wikilinks]
    assert all_links, "stub 笔记里应当有 WikiLink，否则图谱视图无意义"


def test_playbook_digest_filters_by_param():
    only = playbook.playbook_digest(params=["dp_tolerance"])
    assert only["n_notes"] == 1


# ---- store ----------------------------------------------------------------
def test_state_hash_differs_for_different_content():
    from traj_agent.core.traj import Traj
    a = Traj("a", [0, 10, 20], [(121.47, 31.23), (121.48, 31.23), (121.49, 31.23)])
    b = Traj("a", [0, 10, 20], [(121.47, 31.23), (121.48, 31.23), (121.50, 31.23)])
    assert state_hash(a) != state_hash(b)
    assert state_hash(a) == state_hash(a)


def test_store_never_overwrites_handle():
    store = TrajStore()
    from traj_agent.core.traj import Traj
    t = Traj("v", [0, 10], [(121.47, 31.23), (121.48, 31.23)])
    h1 = store.put(t)
    h2 = store.put(t, parent=h1)
    assert h1 != h2
    assert store.n_handles() == 2
