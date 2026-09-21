"""生成讲义配图。全部基于真实数据，不用示意图冒充真实结果。"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体：本机可用 Songti SC / Arial Unicode MS
CJK = "Songti SC"
if CJK not in {f.name for f in font_manager.fontManager.ttflist}:
    CJK = "Arial Unicode MS"
plt.rcParams["font.sans-serif"] = [CJK, "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from traj_agent.core import clean, diagnosis, segment, simplify, traj as tm
from traj_agent.report import figures

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)
raw = tm.load_raw("traj_dict.json")
ids = sorted(raw.keys(), key=lambda k: (len(k), k))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name:34s} {os.path.getsize(p)/1024:7.1f} KB")
    return p


# ---- 图 1：两个 regime 的分布（真实统计） ----
cards = [diagnosis.diagnose(tm.traj_from_raw(v, *raw[v]), with_anomalies=False)
         for v in ids[:200]]
cnt = {}
for c in cards:
    cnt[c.regime] = cnt.get(c.regime, 0) + 1
fig, ax = plt.subplots(figsize=(7.6, 4.4))
keys = ["stationary", "mixed", "moving"]
zh = {"stationary": "静止轨迹", "mixed": "混合", "moving": "行驶轨迹"}
vals = [cnt.get(k, 0) for k in keys]
bars = ax.bar([zh[k] for k in keys], vals,
              color=["#9e9e9e", "#f0ad4e", "#2b7bba"], width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}\n{v/len(cards)*100:.0f}%",
            ha="center", va="bottom", fontsize=14, fontweight="bold")
ax.set_ylabel("车辆数", fontsize=14)
ax.set_title(f"200 条抽样中的轨迹类型分布", fontsize=16, fontweight="bold", pad=14)
ax.tick_params(labelsize=14)
ax.set_ylim(0, max(vals) * 1.28)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
save(fig, "chart_regime.png")

# ---- 图 2：DP 容差扫描的真实曲线 ----
t_raw = tm.traj_from_raw("246", *raw["246"])
t_clean, _ = clean.denoise_trajectory(t_raw)
tols = [1, 2, 3, 5, 8, 12, 16, 22, 30]
comps, devs = [], []
for tol in tols:
    r = simplify.simplify_trajectory(t_clean, tol)
    comps.append(r.compression_ratio * 100)
    devs.append(r.max_deviation_m)
fig, ax1 = plt.subplots(figsize=(7.8, 4.4))
ax1.plot(tols, comps, "-o", color="#2b7bba", lw=2.6, ms=8, label="压缩率")
ax1.set_xlabel("DP 容差（米）", fontsize=14)
ax1.set_ylabel("压缩率（%）", fontsize=14, color="#2b7bba")
ax1.tick_params(labelsize=13)
ax1.set_ylim(0, 100)
ax2 = ax1.twinx()
ax2.plot(tols, devs, "-s", color="#d9534f", lw=2.6, ms=8, label="最大偏差")
ax2.axhline(8, ls="--", color="#5cb85c", lw=2.2)
ax2.text(30, 8.6, "GPS 精度 8m", ha="right", fontsize=12, color="#3d8b40")
ax2.set_ylabel("最大偏差（米）", fontsize=14, color="#d9534f")
ax2.tick_params(labelsize=13)
ax1.set_title("车辆 246：容差越大，压缩越多，偏差也越大", fontsize=15,
              fontweight="bold", pad=14)
for s in ("top",):
    ax1.spines[s].set_visible(False)
    ax2.spines[s].set_visible(False)
save(fig, "chart_dp_curve.png")

# ---- 图 3：重复点占比（真实分布，说明去噪价值） ----
dups = [c.consecutive_dup_ratio * 100 for c in cards]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.hist(dups, bins=24, color="#2b7bba", edgecolor="white", linewidth=0.8)
med = sorted(dups)[len(dups) // 2]
ax.axvline(med, color="#d9534f", lw=2.6, ls="--")
ax.text(med + 2, ax.get_ylim()[1] * 0.82, f"中位数 {med:.0f}%",
        color="#d9534f", fontsize=15, fontweight="bold")
ax.set_xlabel("连续重复点占比（%）", fontsize=14)
ax.set_ylabel("车辆数", fontsize=14)
ax.set_title("超过一半的相邻点对是重复点", fontsize=16, fontweight="bold", pad=14)
ax.tick_params(labelsize=13)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
save(fig, "chart_dup.png")

# ---- 图 4：四类必做图（清洗叠加 / 异常分布 / 热力图），用真实结果 ----
figures.plot_clean_overlay(
    simplify.simplify_trajectory(t_clean, 5.0).traj, reference=t_raw,
    out_dir=OUT, filename="chart_overlay.png", title="清洗前后轨迹叠加（车辆 246）")
figures.plot_anomaly_scatter(t_raw, out_dir=OUT, filename="chart_anomaly.png",
                             title="异常点分布（车辆 246）")
figures.plot_heatmap(simplify.simplify_trajectory(t_clean, 5.0).traj,
                     reference=t_raw, out_dir=OUT, filename="chart_heatmap.png",
                     title="轨迹热力图（车辆 246）")

# ---- 图 5：敏感性热图 ----
rows = segment.sweep_split_thresholds(t_clean, [15, 30, 60, 120],
                                      [100, 200, 400, 800, 1600])
figures.plot_sensitivity_heatmap(rows, out_dir=OUT, filename="chart_sens.png",
                                 title="阈值敏感性：时间阈值与距离阈值")
print("完成，字体:", CJK)
