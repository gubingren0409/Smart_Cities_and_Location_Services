"""可视化：清洗前后叠加图、异常点分布图、热力图、曲线与消融表。

对应必做任务①的第三种输出，以及任务②③的结果呈现。

中文字体：脚本会尝试已知的中文字体名；找不到时**显式告警并退回英文标签**，
而不是画出一堆方框——方框图看起来像 bug，会浪费课堂时间。
"""
from __future__ import annotations

import collections
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")        # 无显示环境（notebook/CI）下必须
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

from ..core import anomalies, geo, metrics   # noqa: E402
from ..core.traj import Traj                 # noqa: E402

# 原因 → 颜色（异常分布图用，保持跨图一致）
REASON_COLORS: Dict[str, str] = {
    anomalies.REASON_ILLEGAL_COORD: "#d62728",
    anomalies.REASON_CONSECUTIVE_DUP: "#8c564b",
    anomalies.REASON_SPEED_SPIKE: "#ff7f0e",
    anomalies.REASON_SPEED_OVER_LIMIT: "#bcbd22",
    anomalies.REASON_ACCEL_OUTLIER: "#e377c2",
    anomalies.REASON_TURN_ANGLE: "#9467bd",
    anomalies.REASON_DRIFT: "#17becf",
    anomalies.REASON_TIME_GAP: "#7f7f7f",
    anomalies.REASON_SPACE_JUMP: "#1f77b4",
    anomalies.REASON_NONPOSITIVE_DT: "#000000",
    anomalies.REASON_STATIONARY_DRIFT: "#2ca02c",
    anomalies.REASON_DT_ARTIFACT: "#aec7e8",
}

_CJK_CANDIDATES = [
    "PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS",
    "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "Microsoft YaHei",
]


def setup_style(prefer_cjk: bool = True) -> Dict[str, Any]:
    """配置绘图样式与中文字体。返回生效状态，便于在报告里说明。"""
    state: Dict[str, Any] = {"cjk": False, "font": None}
    if prefer_cjk:
        try:
            from matplotlib import font_manager
            available = {f.name for f in font_manager.fontManager.ttflist}
            for name in _CJK_CANDIDATES:
                if name in available:
                    plt.rcParams["font.sans-serif"] = [name] + \
                        list(plt.rcParams.get("font.sans-serif", []))
                    plt.rcParams["axes.unicode_minus"] = False
                    state["cjk"] = True
                    state["font"] = name
                    break
        except Exception:
            pass
    if not state["cjk"]:
        # 显式告警：让调用方知道图上会是英文，而不是默默画出方框
        print("[traj_agent.report] 未找到中文字体，图标签将使用英文。"
              "如需中文，请安装字体或用 lab() 覆盖标签。")
    return state


_STYLE_DONE = False


def _ensure_style() -> None:
    global _STYLE_DONE
    if not _STYLE_DONE:
        setup_style()
        _STYLE_DONE = True


def lab(zh: str, en: str) -> str:
    """按中文字体是否可用选择标签文本。"""
    return zh if plt.rcParams.get("font.sans-serif") and _has_cjk() else en


def _has_cjk() -> bool:
    fonts = plt.rcParams.get("font.sans-serif", [])
    if isinstance(fonts, str):
        fonts = [fonts]
    return any(f in _CJK_CANDIDATES for f in fonts)


def _out_path(out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)


def dropped_points(reference: Traj, cleaned: Traj) -> List[geo.LonLat]:
    """求出被删除的坐标点。

    用**坐标多重集差分**，不用 set 也不用坐标+时间戳的组合键：
      - set 判据：折叠重复段后该坐标仍可能出现，于是把已删的点算成未删（漏标）；
        反之若某坐标被删光但结果里有等值坐标，则会把合法保留点算成被删（多标）。
      - 坐标+时间戳键：fix_dt_artifacts 会**改写时间戳**，键随之变化，
        实测把 2 个真实删除算成 17 个。时间戳不是稳定的身份。
    多重集差分对重复坐标的处理是正确的，且只依赖坐标。

    注意：若两个点在空间上恰好完全重合而只删掉其中一个，图上只画一个叉，
    这是坐标层面的正确行为（它们不可区分）。
    """
    before = collections.Counter((float(p[0]), float(p[1])) for p in reference.coords)
    after = collections.Counter((float(p[0]), float(p[1])) for p in cleaned.coords)
    remaining = before - after
    out: List[geo.LonLat] = []
    for p in reference.coords:
        key = (float(p[0]), float(p[1]))
        if remaining.get(key, 0) > 0:
            remaining[key] -= 1
            out.append(p)
    return out


def _lonlat_arrays(traj: Traj) -> Tuple[np.ndarray, np.ndarray]:
    if len(traj) == 0:
        return np.zeros(0), np.zeros(0)
    lons = np.array([p[0] for p in traj.coords], dtype=float)
    lats = np.array([p[1] for p in traj.coords], dtype=float)
    return lons, lats


# ---------------------------------------------------------------------------
# 1. 清洗前后叠加图
# ---------------------------------------------------------------------------
def plot_clean_overlay(cleaned: Traj, reference: Optional[Traj] = None,
                       out_dir: str = "figures",
                       filename: str = "clean_overlay.png",
                       title: str = "") -> str:
    """清洗前后轨迹叠加图。

    原始轨迹 = 灰色细线，清洗后 = 彩色粗线，
    被删除的点 = 红叉（这是「哪些点被删了」最直观的呈现）。
    """
    _ensure_style()
    fig, ax = plt.subplots(figsize=(9, 8))

    if reference is not None and len(reference):
        rl, rt = _lonlat_arrays(reference)
        ax.plot(rl, rt, "-", color="#9e9e9e", lw=1.2, label=lab("原始轨迹", "raw"),
                zorder=1)
        dropped = dropped_points(reference, cleaned)
        if dropped:
            dl = [p[0] for p in dropped]
            dt_ = [p[1] for p in dropped]
            ax.scatter(dl, dt_, marker="x", s=42, color="#d62728",
                       label=lab(f"被删点 ({len(dropped)})", f"dropped ({len(dropped)})"),
                       zorder=3)

    if len(cleaned):
        cl, ct = _lonlat_arrays(cleaned)
        ax.plot(cl, ct, "-o", color="#1f77b4", lw=1.8, ms=3.2,
                label=lab(f"清洗后 ({len(cleaned)} 点)", f"cleaned ({len(cleaned)})"),
                zorder=2)
        ax.scatter(cl[:1], ct[:1], marker="^", s=90, color="#2ca02c",
                   label=lab("起点", "start"), zorder=4)
        ax.scatter(cl[-1:], ct[-1:], marker="s", s=80, color="#d62728",
                   label=lab("终点", "end"), zorder=4)

    ax.set_xlabel(lab("经度", "lon"))
    ax.set_ylabel(lab("纬度", "lat"))
    ax.set_title(title or lab(f"清洗前后轨迹叠加 · {cleaned.seg_id}",
                             f"Clean overlay · {cleaned.seg_id}"))
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    path = _out_path(out_dir, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. 异常点分布图
# ---------------------------------------------------------------------------
def plot_anomaly_scatter(traj: Traj, rules=None, out_dir: str = "figures",
                         filename: str = "anomaly_scatter.png",
                         title: str = "") -> str:
    """异常点分布图：按原因分色，含图例计数。

    图例会给出每个原因的实际点数——这是「异常原因字段」在报告里的落点。
    """
    _ensure_style()
    res = anomalies.detect_anomalies(traj, rules)
    lons, lats = _lonlat_arrays(traj)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(lons, lats, "-", color="#c8c8c8", lw=1.0, zorder=1,
            label=lab("轨迹", "trajectory"))

    counts = res.counts()
    drawn = 0
    for reason, idxs in sorted(res.flags.items(), key=lambda kv: -len(kv[1])):
        if not idxs:
            continue
        xs = [lons[i] for i in idxs if 0 <= i < len(lons)]
        ys = [lats[i] for i in idxs if 0 <= i < len(lats)]
        zh = anomalies.REASON_ZH.get(reason, reason)
        ax.scatter(xs, ys, s=48, alpha=0.85,
                   color=REASON_COLORS.get(reason, "#333333"),
                   marker="o", edgecolors="white", linewidths=0.5,
                   label=f"{zh} ({len(idxs)})", zorder=3)
        drawn += 1

    if drawn == 0:
        ax.text(0.5, 0.5, lab("未检出异常点", "no anomaly detected"),
                transform=ax.transAxes, ha="center", va="center", fontsize=13,
                color="#666666")

    ax.set_xlabel(lab("经度", "lon"))
    ax.set_ylabel(lab("纬度", "lat"))
    ax.set_title(title or lab(f"异常点分布 · {traj.seg_id}",
                             f"Anomaly distribution · {traj.seg_id}"))
    if drawn:
        ax.legend(loc="best", fontsize=8, ncol=1)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    path = _out_path(out_dir, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. 热力图（网格密度）
# ---------------------------------------------------------------------------
def plot_heatmap(cleaned: Traj, reference: Optional[Traj] = None,
                 out_dir: str = "figures",
                 filename: str = "heatmap.png",
                 bins: int = 40, title: str = "") -> str:
    """停留/密度热力图。给出 reference 时左右并排对比清洗前后。"""
    _ensure_style()
    panels = []
    if reference is not None and len(reference):
        panels.append((reference, lab("清洗前", "before")))
    panels.append((cleaned, lab("清洗后", "after")))
    n = len(panels)

    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 7), squeeze=False)
    for k, (traj, name) in enumerate(panels):
        ax = axes[0][k]
        lons, lats = _lonlat_arrays(traj)
        if len(lons) == 0:
            ax.set_axis_off()
            continue
        # 用米作为单位，避免经纬度尺度差异导致热力图被压扁
        xs = (lons - lons.mean()) * 95274.27
        ys = (lats - lats.mean()) * 110873.43
        h = ax.hist2d(xs, ys, bins=bins, cmap="inferno")
        fig.colorbar(h[3], ax=ax, shrink=0.82,
                     label=lab("点数", "point count"))
        ax.set_title(f"{name} · {len(traj)} " + lab("点", "pts"))
        ax.set_xlabel(lab("东向偏移 (m)", "east offset (m)"))
        ax.set_ylabel(lab("北向偏移 (m)", "north offset (m)"))
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(title or lab(f"轨迹热力图 · {cleaned.seg_id}",
                             f"Density heatmap · {cleaned.seg_id}"), y=1.00)
    fig.tight_layout()
    path = _out_path(out_dir, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. 质量—压缩率曲线（任务②核心图，也是任务③的 terrain map）
# ---------------------------------------------------------------------------
def plot_quality_compression(curve: Sequence[Dict[str, Any]],
                             knee: Optional[Dict[str, Any]] = None,
                             out_dir: str = "figures",
                             filename: str = "quality_compression.png",
                             title: str = "") -> str:
    """质量—压缩率曲线，标注 knee point。

    这张图同时服务于两个任务：
      任务② 看阈值扫描的形状（质量换压缩率的边际递减）；
      任务③ 它就是 verifier 计算 regret 的目标函数地形图。
    """
    _ensure_style()
    comp = [float(c.get("compression_ratio", 0.0)) for c in curve]
    dev = [float(c.get("max_deviation_m", 0.0)) for c in curve]
    vals = [float(c.get("value", 0.0)) for c in curve]
    fid = [max(0.0, 1.0 - d / 30.0) for d in dev]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(comp, fid, "-o", color="#1f77b4", lw=1.6, ms=6)
    for c, f, v in zip(comp, fid, vals):
        ax1.annotate(f"{v:g}", (c, f), textcoords="offset points",
                     xytext=(5, 5), fontsize=8, color="#555555")
    if knee and knee.get("compression") is not None:
        kx = float(knee.get("compression", 0.0))
        ky = float(knee.get("fidelity", 0.0))
        ax1.scatter([kx], [ky], s=220, marker="*", color="#d62728",
                    zorder=5, label=lab("knee point", "knee point"))
        ax1.legend(fontsize=9)
    ax1.set_xlabel(lab("压缩率", "compression ratio"))
    ax1.set_ylabel(lab("保真度 (1 - 偏差/30m)", "fidelity (1 - dev/30m)"))
    ax1.set_title(lab("质量—压缩率曲线（标注为参数值）",
                     "Quality-compression curve (labels = param value)"))
    ax1.grid(alpha=0.3)

    ax2.plot(vals, dev, "-s", color="#ff7f0e", lw=1.6, ms=6,
             label=lab("最大偏差", "max deviation"))
    ax2.plot(vals, comp, "-^", color="#2ca02c", lw=1.6, ms=6,
             label=lab("压缩率", "compression"))
    ax2.axhline(8.0, ls="--", color="#d62728", lw=1.2,
                label=lab("GPS CEP ≈ 8m", "GPS CEP ~8m"))
    ax2.set_xlabel(lab("参数值 (DP 容差 m)", "param value (DP tolerance, m)"))
    ax2.set_ylabel(lab("数值", "value"))
    ax2.set_title(lab("偏差与压缩率随容差变化", "Deviation & compression vs tolerance"))
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(title or lab("道格拉斯-普克容差扫描", "DP tolerance sweep"), y=1.02)
    fig.tight_layout()
    path = _out_path(out_dir, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. 二维敏感性热图（任务②）
# ---------------------------------------------------------------------------
def plot_sensitivity_heatmap(rows: Sequence[Dict[str, float]],
                             out_dir: str = "figures",
                             filename: str = "sensitivity.png",
                             title: str = "") -> str:
    """参数敏感性热图：时间阈值 × 速度/距离阈值的二维响应面。

    输入是 sweep_split_thresholds 之类的输出行，每行含两个自变量与一个因变量。
    """
    _ensure_style()
    if not rows:
        raise ValueError("rows 为空")
    xs = sorted({float(r["dt_threshold"]) for r in rows})
    ys = sorted({float(r.get("dist_threshold", r.get("max_speed_mps", 0.0)))
                 for r in rows})
    key = "n_points_kept" if "n_points_kept" in rows[0] else "n_segments"
    grid = np.full((len(ys), len(xs)), np.nan)
    for r in rows:
        xi = xs.index(float(r["dt_threshold"]))
        yv = float(r.get("dist_threshold", r.get("max_speed_mps", 0.0)))
        yi = ys.index(yv)
        grid[yi, xi] = float(r.get(key, 0.0))

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45, fontsize=8)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([f"{y:g}" for y in ys], fontsize=8)
    ax.set_xlabel(lab("时间间隔阈值 (s)", "dt threshold (s)"))
    ax.set_ylabel(lab("空间跳跃阈值 (m)", "dist threshold (m)"))
    ylabel = lab("保留点数", "points kept") if key == "n_points_kept" else \
        lab("分段数", "segments")
    fig.colorbar(im, ax=ax, shrink=0.85, label=ylabel)
    ax.set_title(title or lab("参数敏感性：阈值对结果的影响",
                             "Sensitivity: thresholds vs outcome"))
    fig.tight_layout()
    path = _out_path(out_dir, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. 消融表（任务③结论页）
# ---------------------------------------------------------------------------
def plot_ablation(rows: Sequence[Dict[str, Any]],
                  out_dir: str = "figures",
                  filename: str = "ablation.png",
                  title: str = "") -> str:
    """消融实验柱状图：各方案的 regret 与准入率对比。"""
    _ensure_style()
    modes = [str(r.get("mode", "?")) for r in rows]
    regret = [float(r.get("mean_regret") or 0.0) for r in rows]
    admit = [float(r.get("admit_rate") or 0.0) for r in rows]
    n = [int(r.get("n") or 0) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(modes))
    ax1.bar(x, regret, color="#d62728", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{m}\n(n={c})" for m, c in zip(modes, n)],
                        rotation=20, fontsize=8)
    ax1.set_ylabel(lab("平均 regret", "mean regret"))
    ax1.set_title(lab("各方案相对确定性搜索最优的差距（越小越好）",
                      "Regret vs deterministic optimum (lower is better)"))
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(x, admit, color="#2ca02c", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{m}\n(n={c})" for m, c in zip(modes, n)],
                        rotation=20, fontsize=8)
    ax2.set_ylabel(lab("记忆准入率", "admission rate"))
    ax2.set_ylim(0, 1.05)
    ax2.set_title(lab("通过核验的比例", "Fraction passing verification"))
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(title or lab("消融实验：各方案的定量对比",
                             "Ablation: quantitative comparison"), y=1.02)
    fig.tight_layout()
    path = _out_path(out_dir, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
RENDERERS = {
    "clean_overlay": "plot_clean_overlay",
    "anomaly_scatter": "plot_anomaly_scatter",
    "heatmap": "plot_heatmap",
}


def render(kind: str, traj: Traj, reference: Optional[Traj] = None,
           out_dir: str = "figures", road_matcher: Any = None,
           filename: Optional[str] = None) -> str:
    """供工具层调用的统一出图入口。返回文件路径。"""
    if kind not in RENDERERS:
        raise KeyError(f"未知图类型 {kind!r}；可用 {sorted(RENDERERS)}")
    fn = globals()[RENDERERS[kind]]
    fname = filename or f"{kind}_{traj.seg_id.replace('#', '_').replace('@', '_')}.png"
    if kind == "clean_overlay":
        return fn(traj, reference=reference, out_dir=out_dir, filename=fname)
    if kind == "anomaly_scatter":
        return fn(traj, out_dir=out_dir, filename=fname)
    return fn(traj, reference=reference, out_dir=out_dir, filename=fname)
