"""记忆层：特征、检索、蒸馏、以及「只有核验器能写」的约束。"""
from __future__ import annotations

import json

import pytest

from traj_agent.core import diagnosis
from traj_agent.core.traj import Traj
from traj_agent.memory import features as feat_mod
from traj_agent.memory import retrieve as retrieve_mod
from traj_agent.memory.store import MemoryStore


@pytest.fixture
def card_moving(real_raw):
    from traj_agent.core import traj as tm
    return diagnosis.diagnose(tm.traj_from_raw("246", *real_raw["246"]))


@pytest.fixture
def card_stationary(real_raw):
    from traj_agent.core import traj as tm
    return diagnosis.diagnose(tm.traj_from_raw("0", *real_raw["0"]))


@pytest.fixture
def store():
    st = MemoryStore(":memory:")
    yield st
    st.close()


def _seed(store, card, n=4, admitted=True, mode="test", seg_id=None):
    """写入 n 条模拟经验。

    seg_id 默认在 card.seg_id 后加后缀：检索默认开启自检索排除
    （exclude_self=True，防止留出集评测时把答案喂回来），
    若种子与查询用同一个 seg_id，检索会正确地返回空 —— 那会掩盖真正的检索逻辑。
    需要测试「排除自身」这一行为时请显式传 seg_id=card.seg_id。
    """
    import copy
    for i in range(n):
        c = copy.copy(card)
        c.seg_id = seg_id if seg_id is not None else f"{card.seg_id}-seed{i}"
        store.write_case(
            card=c,
            params={"dp_tolerance": 3.0 + i, "dt_threshold": 30.0},
            metrics={"compression_ratio": 0.5 + 0.05 * i, "max_deviation_m": 2.0 + i},
            objective={"score": 0.5 + 0.1 * i},
            verification={"admitted": admitted},
            admitted=admitted, admit_reason="seed",
            score=0.5 + 0.1 * i, regret=0.02 * i, mode=mode)
    return n


# ---- 特征 ----------------------------------------------------------------
def test_feature_vector_length_and_range(card_moving):
    vec = feat_mod.featurize(card_moving)
    assert len(vec) == len(feat_mod.FEATURE_NAMES)
    assert all(0.0 <= v <= 1.0 for v in vec), f"特征越界: {vec}"


def test_features_differ_between_regimes(card_moving, card_stationary):
    a = feat_mod.featurize(card_moving)
    b = feat_mod.featurize(card_stationary)
    assert a != b
    d = feat_mod.euclidean(a, b)
    assert d > 0.1, f"两个 regime 的特征距离过小: {d}"


def test_features_are_deterministic(card_moving):
    assert feat_mod.featurize(card_moving) == feat_mod.featurize(card_moving)


def test_feature_dict_has_names(card_moving):
    d = feat_mod.vector_to_dict(feat_mod.featurize(card_moving))
    assert set(d) == set(feat_mod.FEATURE_NAMES)


def test_similarity_identical_is_one(card_moving):
    v = feat_mod.featurize(card_moving)
    assert feat_mod.similarity(v, v) == 1.0
    assert feat_mod.cosine_similarity(v, v) == pytest.approx(1.0)


def test_similarity_handles_zero_vector():
    z = [0.0] * len(feat_mod.FEATURE_NAMES)
    assert feat_mod.cosine_similarity(z, z) == 0.0
    assert feat_mod.similarity(z, z) == 1.0


def test_dt_cv_proxy_monotone(card_moving):
    """p95/median 越大，离散度代理应越大。"""
    import copy
    a = copy.copy(card_moving)
    b = copy.copy(card_moving)
    a.dt_median_s, a.dt_p95_s = 10.0, 10.0
    b.dt_median_s, b.dt_p95_s = 10.0, 40.0
    assert feat_mod.dt_coefficient_of_variation(a) == 0.0
    assert feat_mod.dt_coefficient_of_variation(b) == pytest.approx(3.0)


# ---- 写入与统计 -----------------------------------------------------------
def test_write_and_count(store, card_moving):
    _seed(store, card_moving, n=4)
    assert store.count() == 4
    assert store.count(admitted_only=True) == 4
    stats = store.stats()
    assert stats["n_episodic"] == 4
    assert stats["admit_rate"] == 1.0


def test_rejected_cases_are_recorded_but_not_retrievable(store, card_moving):
    """失败经验也要记录（支撑「这个方向不行」），但不应被检索推荐。"""
    _seed(store, card_moving, n=3, admitted=False)
    assert store.count() == 3
    assert store.count(admitted_only=True) == 0
    assert store.retrieve(feat_mod.featurize(card_moving), k=5) == []


def test_recent_lists_newest_first(store, card_moving):
    _seed(store, card_moving, n=3)
    rows = store.recent(limit=2)
    assert len(rows) == 2
    assert rows[0]["id"] > rows[1]["id"]


# ---- 检索 ----------------------------------------------------------------
def test_retrieve_returns_most_similar_first(store, real_raw):
    from traj_agent.core import traj as tm
    for vid in ("246", "256", "306", "209", "0"):
        card = diagnosis.diagnose(tm.traj_from_raw(vid, *real_raw[vid]))
        _seed(store, card, n=2)

    query = diagnosis.diagnose(tm.traj_from_raw("246", *real_raw["246"]))
    cases = store.retrieve_similar(query, k=3)
    assert cases
    assert cases[0].similarity >= cases[-1].similarity
    assert cases[0].vehicle_id == "246"


def test_retrieve_excludes_self_by_default(store, card_moving):
    """回归：检索默认排除同 seg_id 的记录。

    否则留出集评测时 agent 只要重复跑同一条轨迹，
    就能把「自己的上一次结果」检索回来当先验 —— 信息泄漏。
    """
    _seed(store, card_moving, n=2, seg_id=card_moving.seg_id)
    assert store.retrieve_similar(card_moving, k=3) == []
    assert len(store.retrieve_similar(card_moving, k=3, exclude_self=False)) == 2


def test_retrieve_is_regime_gated(store, real_raw):
    """静止轨迹的检索不该返回行驶轨迹的案例。"""
    from traj_agent.core import traj as tm
    moving = diagnosis.diagnose(tm.traj_from_raw("246", *real_raw["246"]))
    for vid in ("246", "256", "306"):
        _seed(store, diagnosis.diagnose(tm.traj_from_raw(vid, *real_raw[vid])), n=2)

    stationary = diagnosis.diagnose(tm.traj_from_raw("0", *real_raw["0"]))
    cases = store.retrieve(feat_mod.featurize(stationary), k=3,
                           regime="stationary")
    assert cases == [], "stationary 门控下不应命中 moving 的案例"


def test_retrieve_skips_mismatched_feature_length(store, card_moving):
    """特征版本不一致的记录应被跳过，而不是错误匹配。"""
    import sqlite3
    _seed(store, card_moving, n=2)
    with store._lock:
        store._conn.execute(
            "UPDATE episodic SET features=?", (json.dumps([0.1, 0.2]),))
        store._conn.commit()
    assert store.retrieve(feat_mod.featurize(card_moving), k=5) == []


# ---- 蒸馏 ----------------------------------------------------------------
def test_rebuild_procedural_only_uses_admitted(store, card_moving):
    _seed(store, card_moving, n=3, admitted=True)
    _seed(store, card_moving, n=3, admitted=False)
    n = store.rebuild_procedural(min_samples=2)
    assert n >= 1
    regions = store.query_regions(param_name="dp_tolerance")
    assert regions
    assert all(r.n_samples == 3 for r in regions), "只应统计 admitted 记录"


def test_rebuild_procedural_needs_min_samples(store, card_moving):
    _seed(store, card_moving, n=1, admitted=True)
    assert store.rebuild_procedural(min_samples=5) == 0


def test_procedural_uses_iqr_not_minmax(store, card_moving):
    """区间应取 IQR 而非 min/max，避免极端值把区间撑得过宽。"""
    for v in (1.0, 5.0, 5.0, 5.0, 100.0):
        store.write_case(card=card_moving,
                         params={"dp_tolerance": v},
                         metrics={}, objective={}, verification={},
                         admitted=True, score=0.5, regret=0.0)
    store.rebuild_procedural(min_samples=3)
    r = store.query_regions(param_name="dp_tolerance")[0]
    assert r.high < 100.0, f"IQR 上界不该被 100 拉满: {r.high}"
    assert r.low > 0.0


def test_procedural_upsert_is_idempotent(store, card_moving):
    _seed(store, card_moving, n=3)
    store.rebuild_procedural(min_samples=2)
    first = store.query_regions(param_name="dp_tolerance")
    store.rebuild_procedural(min_samples=2)
    second = store.query_regions(param_name="dp_tolerance")
    assert len(first) == len(second)


# ---- 先验组装 -------------------------------------------------------------
def test_prior_cold_start(store, card_moving):
    prior = retrieve_mod.prior_for(card_moving, store)
    assert prior.available is False
    assert "冷启动" in prior.caution


def test_prior_with_no_store(card_moving):
    prior = retrieve_mod.prior_for(card_moving, None)
    assert prior.available is False
    assert prior.caution


def test_prior_available_after_seeding(store, card_moving):
    _seed(store, card_moving, n=3)   # seg_id 加后缀，避免被自检索排除
    store.rebuild_procedural(min_samples=2)
    prior = retrieve_mod.prior_for(card_moving, store, k=3)
    assert prior.available is True
    assert prior.n_neighbors == 3
    assert prior.suggested_regions
    assert prior.feature_vector


def test_prior_caution_on_low_similarity(store, card_moving, real_raw):
    """相似度过低时必须给出警示，避免先验被过度信任。"""
    from traj_agent.core import traj as tm
    _seed(store, diagnosis.diagnose(tm.traj_from_raw("0", *real_raw["0"])), n=2)
    # 人为把查询卡改成 moving，regime 门控会落空 -> 走 timeline 回退
    prior = retrieve_mod.prior_for(card_moving, store, k=3)
    assert prior.caution or not prior.neighbors


def test_explain_neighbors_is_serializable(store, card_moving):
    _seed(store, card_moving, n=2)
    out = retrieve_mod.explain_neighbors(card_moving, store, k=2)
    json.dumps(out, ensure_ascii=False, default=str)
    assert out["feature_names"] == list(feat_mod.FEATURE_NAMES)
    assert len(out["neighbors"]) == 2


def test_region_hint_text_empty():
    prior = retrieve_mod.MemoryPrior()
    assert "无" in retrieve_mod.region_hint_text(prior)


def test_region_hint_text_renders(store, card_moving):
    _seed(store, card_moving, n=3)
    store.rebuild_procedural(min_samples=2)
    prior = retrieve_mod.prior_for(card_moving, store)
    text = retrieve_mod.region_hint_text(prior)
    assert "dp_tolerance" in text


# ---- 运行记录 -------------------------------------------------------------
def test_run_bookkeeping(store, card_moving):
    store.start_run("run-1", mode="llm+memory+search")
    _seed(store, card_moving, n=2)
    store.finish_run("run-1", 2)
    with store._lock:
        row = store._conn.execute(
            "SELECT * FROM runs WHERE run_id=?", ("run-1",)).fetchone()
    assert row["n_cases"] == 2
    assert row["mode"] == "llm+memory+search"


def test_persistence_roundtrip(tmp_path, card_moving):
    """落到磁盘后重开应能读回，验证 schema 与序列化。"""
    path = str(tmp_path / "mem.sqlite")
    st = MemoryStore(path)
    _seed(st, card_moving, n=3)
    st.close()

    st2 = MemoryStore(path)
    assert st2.count() == 3
    cases = st2.retrieve(feat_mod.featurize(card_moving), k=3)
    assert len(cases) == 3
    assert cases[0].params["dt_threshold"] == 30.0
    st2.close()
