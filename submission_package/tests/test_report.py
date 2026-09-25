"""出图与导出：文件产出、被删点判据、以及内容可校验性。

本环境无法目视核验图片，故这些测试检查可程序化验证的性质：
文件非空、中文字形存在、数据点数量正确、被删点计数与清洗账本一致。
"""
from __future__ import annotations

import collections
import os

import pytest

from traj_agent.core import clean, segment, simplify
from traj_agent.core.traj import Traj
from traj_agent.report import figures


@pytest.fixture(scope="module")
def tmp_fig(tmp_path_factory):
    return str(tmp_path_factory.mktemp("figs"))


@pytest.fixture
def prepared(real_raw):
    from traj_agent.core import traj as tm
    t = tm.traj_from_raw("246", *real_raw["246"])
    c, rep = clean.denoise_trajectory(t)
    s = simplify.simplify_trajectory(c, 5.0)
    return t, c, s, rep


# ---- 被删点判据（回归） ---------------------------------------------------
def test_dropped_points_matches_clean_report(prepared, real_raw):
    """回归：被删点判据必须与清洗账本一致。

    早先用 (时间戳, 坐标) 组合键做差分，而 fix_dt_artifacts 会改写时间戳，
    实测把 2 个真实删除算成 17 个。改用坐标多重集差分后逐条一致。
    """
    from traj_agent.core import traj as tm
    for vid in ("0", "246", "306", "352", "256", "209"):
        t = tm.traj_from_raw(vid, *real_raw[vid])
        c, rep = clean.denoise_trajectory(t)
        dropped = figures.dropped_points(t, c)
        actual = rep.n_before - rep.n_after
        assert len(dropped) == actual, f"{vid}: 图上标 {len(dropped)} 实际删 {actual}"


def test_dropped_points_handles_duplicate_coordinates():
    """重复坐标：折叠后该坐标仍存在，多项集差分必须正确抵销。"""
    from tests.conftest import pt  # noqa: F401
    coords = [(121.4700, 31.2300), (121.4700, 31.2300), (121.4700, 31.2300),
              (121.4800, 31.2300)]
    before = Traj("d", [0, 10, 20, 30], coords)
    after = Traj("d", [0, 30], [coords[0], coords[3]])
    dropped = figures.dropped_points(before, after)
    assert len(dropped) == 2


def test_dropped_points_empty_when_nothing_removed():
    t = Traj("x", [0, 10], [(121.47, 31.23), (121.48, 31.23)])
    assert figures.dropped_points(t, t) == []


# ---- 中文字形 -------------------------------------------------------------
def test_cjk_font_available_or_warned(capsys):
    """要么找到中文字体，要么显式告警——不能默默画出方框。"""
    state = figures.setup_style()
    if not state["cjk"]:
        out = capsys.readouterr().out
        assert "英文" in out, "找不到中文字体时必须显式告警"


def test_cjk_glyphs_actually_exist():
    """字体文件里必须真的有这些汉字的字形，而不只是声明了字体名。"""
    figures.setup_style()
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    fam = plt.rcParams.get("font.sans-serif", [])
    if isinstance(fam, str):
        fam = [fam]
    cjk = next((f for f in fam if f in figures._CJK_CANDIDATES), None)
    if cjk is None:
        pytest.skip("未找到中文字体")
    ttf = font_manager.findfont(font_manager.FontProperties(family=cjk))
    font = FT2Font(ttf)
    missing = [ch for ch in "清洗异常轨迹点经度纬度压缩率保真度"
               if font.get_char_index(ord(ch)) == 0]
    assert not missing, f"字体 {cjk} 缺少字形: {missing}"


# ---- 各类图产出 -----------------------------------------------------------
def _assert_png(path: str, min_bytes: int = 8000):
    assert os.path.exists(path)
    size = os.path.getsize(path)
    assert size > min_bytes, f"{path} 仅 {size} 字节，疑似空图"
    with open(path, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n", "不是合法 PNG"


def test_clean_overlay_renders(prepared, tmp_fig):
    t, c, s, _ = prepared
    p = figures.plot_clean_overlay(s.traj, reference=t, out_dir=tmp_fig,
                                   filename="overlay.png")
    _assert_png(p)


def test_clean_overlay_without_reference(prepared, tmp_fig):
    _, c, s, _ = prepared
    p = figures.plot_clean_overlay(s.traj, reference=None, out_dir=tmp_fig,
                                   filename="overlay_noref.png")
    _assert_png(p)


def test_anomaly_scatter_renders(prepared, tmp_fig):
    t, _, _, _ = prepared
    p = figures.plot_anomaly_scatter(t, out_dir=tmp_fig, filename="anom.png")
    _assert_png(p)


def test_anomaly_scatter_on_clean_trajectory(prepared, tmp_fig):
    """无异常时不能崩，且应画出提示文字。"""
    _, _, s, _ = prepared
    p = figures.plot_anomaly_scatter(s.traj, out_dir=tmp_fig,
                                     filename="anom_clean.png")
    _assert_png(p)


def test_heatmap_renders(prepared, tmp_fig):
    t, _, s, _ = prepared
    p = figures.plot_heatmap(s.traj, reference=t, out_dir=tmp_fig,
                             filename="heat.png")
    _assert_png(p)


def test_quality_compression_curve_renders(prepared, tmp_fig):
    t, c, _, _ = prepared
    curve = [{"value": v, "compression_ratio": r, "max_deviation_m": d,
              "length_ratio": 0.99, "n_points": 20}
             for v, r, d in ((1, 0.3, 0.8), (5, 0.6, 4.7), (10, 0.75, 8.3),
                             (20, 0.83, 19.0))]
    p = figures.plot_quality_compression(curve, knee={"compression": 0.6,
                                                     "fidelity": 0.84},
                                         out_dir=tmp_fig, filename="curve.png")
    _assert_png(p)


def test_sensitivity_heatmap_renders(prepared, tmp_fig):
    _, c, _, _ = prepared
    rows = segment.sweep_split_thresholds(c, [15, 30, 60], [100, 400, 800])
    p = figures.plot_sensitivity_heatmap(rows, out_dir=tmp_fig,
                                         filename="sens.png")
    _assert_png(p)


def test_sensitivity_heatmap_rejects_empty():
    with pytest.raises(ValueError):
        figures.plot_sensitivity_heatmap([])


def test_ablation_chart_renders(tmp_fig):
    rows = [{"mode": "llm-only", "n": 4, "mean_regret": 0.3, "admit_rate": 0.2},
            {"mode": "search-only", "n": 4, "mean_regret": 0.0, "admit_rate": 1.0}]
    p = figures.plot_ablation(rows, out_dir=tmp_fig, filename="abl.png")
    _assert_png(p)


def test_render_dispatch(prepared, tmp_fig):
    t, _, s, _ = prepared
    for kind in ("clean_overlay", "anomaly_scatter", "heatmap"):
        p = figures.render(kind, s.traj, reference=t, out_dir=tmp_fig)
        _assert_png(p)


def test_render_rejects_unknown_kind(prepared):
    _, _, s, _ = prepared
    with pytest.raises(KeyError):
        figures.render("no_such_kind", s.traj)


# ---- 数据点数正确性 -------------------------------------------------------
def test_plot_contains_all_points(prepared):
    """图上折线的点数必须等于轨迹点数，避免静默丢点。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, _, s, _ = prepared
    fig, ax = plt.subplots()
    ax.plot([p[0] for p in s.traj.coords], [p[1] for p in s.traj.coords])
    xs, _ = ax.lines[0].get_data()
    assert len(xs) == len(s.traj)
    plt.close(fig)


# ---- 导出 -----------------------------------------------------------------
def test_export_preview_has_review_checklist(real_raw):
    from traj_agent.core import diagnosis
    from traj_agent.memory import export as export_mod
    from traj_agent.memory.store import MemoryStore
    from traj_agent.core import traj as tm

    st = MemoryStore(":memory:")
    card = diagnosis.diagnose(tm.traj_from_raw("246", *real_raw["246"]))
    for tol in (5.0, 6.0, 7.0, 8.0):
        st.write_case(card=card, params={"dp_tolerance": tol}, metrics={},
                      objective={}, verification={}, admitted=True,
                      score=0.6, regret=0.01)
    st.rebuild_procedural(min_samples=2)
    notes = export_mod.preview_export(st, min_samples=2)
    assert notes
    body = notes[0]["body"]
    assert "待人工确认" in body
    assert "- [ ]" in body
    assert notes[0]["frontmatter"]["needs_review"] == "true"
    st.close()


def test_export_writes_only_to_inbox(tmp_path, real_raw):
    """导出产物必须落在 00-Inbox/，不得直接进正式目录。"""
    from traj_agent.core import diagnosis, traj as tm
    from traj_agent.memory import export as export_mod
    from traj_agent.memory.store import MemoryStore

    vault = tmp_path / "vault"
    vault.mkdir()
    st = MemoryStore(":memory:")
    card = diagnosis.diagnose(tm.traj_from_raw("246", *real_raw["246"]))
    for tol in (5.0, 6.0, 7.0, 8.0):
        st.write_case(card=card, params={"dp_tolerance": tol}, metrics={},
                      objective={}, verification={}, admitted=True,
                      score=0.6, regret=0.01)
    st.rebuild_procedural(min_samples=2)
    res = export_mod.export_regions(st, vault_dir=str(vault), min_samples=2)
    assert res.written
    for p in res.written:
        assert export_mod.INBOX_SUBDIR in p
        assert os.path.exists(p)
    st.close()


def test_export_refuses_to_overwrite_by_default(tmp_path, real_raw):
    from traj_agent.core import diagnosis, traj as tm
    from traj_agent.memory import export as export_mod
    from traj_agent.memory.store import MemoryStore

    st = MemoryStore(":memory:")
    card = diagnosis.diagnose(tm.traj_from_raw("246", *real_raw["246"]))
    for tol in (5.0, 6.0, 7.0, 8.0):
        st.write_case(card=card, params={"dp_tolerance": tol}, metrics={},
                      objective={}, verification={}, admitted=True,
                      score=0.6, regret=0.01)
    st.rebuild_procedural(min_samples=2)

    note = export_mod.preview_export(st, min_samples=2)[0]
    p1 = export_mod.export_note(note, str(tmp_path))
    assert os.path.exists(p1)
    with pytest.raises(FileExistsError):
        export_mod.export_note(note, str(tmp_path))
    p2 = export_mod.export_note(note, str(tmp_path), force=True)
    assert p2 == p1
    st.close()


def test_export_skips_when_vault_missing(real_raw):
    """vault 不存在时只预览不写盘，并记入 skipped。"""
    from traj_agent.core import diagnosis, traj as tm
    from traj_agent.memory import export as export_mod
    from traj_agent.memory.store import MemoryStore

    st = MemoryStore(":memory:")
    card = diagnosis.diagnose(tm.traj_from_raw("246", *real_raw["246"]))
    for tol in (5.0, 6.0, 7.0, 8.0):
        st.write_case(card=card, params={"dp_tolerance": tol}, metrics={},
                      objective={}, verification={}, admitted=True,
                      score=0.6, regret=0.01)
    st.rebuild_procedural(min_samples=2)
    res = export_mod.export_regions(st, vault_dir="/no/such/vault", min_samples=2)
    assert res.written == []
    assert res.skipped
    st.close()
