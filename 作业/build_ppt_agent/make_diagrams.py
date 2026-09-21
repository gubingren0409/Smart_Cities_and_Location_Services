"""为「LLM 辅助轨迹清洗评估智能体」框架生成配图。"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import font_manager

CJK = "Songti SC"
if CJK not in {f.name for f in font_manager.fontManager.ttflist}:
    CJK = "Arial Unicode MS"
plt.rcParams["font.sans-serif"] = [CJK, "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)

INK, MUTED, ACCENT, WARN, OK = "#1F2A37", "#5B6B7C", "#2B7BBA", "#C0392B", "#2E8B57"


def box(ax, x, y, w, h, text, fc, ec=None, tc="white", fs=11, bold=True, radius=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={radius}",
                                fc=fc, ec=ec or fc, lw=1.2, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold" if bold else "normal", zorder=3,
            linespacing=1.5)


def arrow(ax, p1, p2, color=MUTED, style="-|>", lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color, lw=lw,
                                 mutation_scale=14, zorder=4, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def blank(w=11, h=6.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name:30s} {os.path.getsize(p)/1024:7.1f} KB")


# ============ 图 1：分层架构 ============
fig, ax = blank(10.5, 6.6)
layers = [
    ("agent/", "ReAct 主循环 · provider（OpenAI兼容 / Mock）· 预算护栏 · trace", "#2B4A63"),
    ("verifier/", "目标函数 · knee point · 有界搜索 · regret 核验  ← 唯一有记忆写权限", "#1F6F5C"),
    ("tools/", "15 个 JSON-Schema 工具 · 轨迹句柄存储 · Obsidian 只读挂载  ← LLM 唯一能碰的层", "#8A6D1F"),
    ("memory/", "SQLite 情景记忆 + 程序记忆 · 12 维特征 kNN · 单向导出", "#6B4A7A"),
    ("core/", "geo · segment · anomalies · clean · simplify · metrics · diagnosis  ← 零 LLM 依赖", "#2B7BBA"),
    ("road/", "路网匹配协议（接口就位，可选依赖）", "#7A8A96"),
]
y = 0.86
for name, desc, color in layers:
    box(ax, 0.30, y, 0.62, 0.115, "", color, radius=0.01)
    ax.text(0.325, y + 0.0575, name, ha="left", va="center", color="white",
            fontsize=12.5, fontweight="bold", zorder=3, family="monospace")
    ax.text(0.455, y + 0.0575, desc, ha="left", va="center", color="white",
            fontsize=9.3, zorder=3)
    y -= 0.135
# 依赖方向
arrow(ax, (0.26, 0.92), (0.26, 0.20), color=ACCENT, lw=2.2)
ax.text(0.215, 0.56, "单向依赖", rotation=90, ha="center", va="center",
        color=ACCENT, fontsize=10.5, fontweight="bold")
ax.text(0.5, 0.955, "分层架构：core 不知道上面任何一层的存在", ha="center",
        fontsize=14, fontweight="bold", color=INK)
ax.text(0.5, 0.075, "由 tests/test_architecture.py 用 AST 强制检查依赖方向",
        ha="center", fontsize=9.5, color=MUTED, style="italic")
save(fig, "arch_layers.png")

# ============ 图 2：闭环流程 ============
fig, ax = blank(11.4, 6.4)
steps = [
    (0.04, 0.72, "1 载入轨迹", "句柄 + 诊断卡\n（不含坐标）", "#2B7BBA"),
    (0.28, 0.72, "2 检索记忆", "L2 参数区间\nL3 人类知识", "#6B4A7A"),
    (0.52, 0.72, "3 LLM 提议", "params\nexpected_effect\nrationale", "#8A6D1F"),
    (0.76, 0.72, "4 核验约束", "纯代码\n无 LLM", "#1F6F5C"),
    (0.04, 0.20, "5 执行提议", "实测指标", "#2B7BBA"),
    (0.28, 0.20, "6 确定性搜索", "ground truth 最优", "#1F6F5C"),
    (0.52, 0.20, "7 regret", "提议 vs 最优", "#C0392B"),
    (0.76, 0.20, "8 记忆准入", "达标才写入 SQLite", "#6B4A7A"),
]
for x, yy, title, sub, c in steps:
    box(ax, x, yy + 0.10, 0.20, 0.115, title, c, fs=11.5, radius=0.012)
    box(ax, x, yy - 0.045, 0.20, 0.125, sub, "#F4F6F8", ec="#D8DEE4",
        tc=INK, fs=8.8, bold=False, radius=0.012)
for i in range(3):
    arrow(ax, (0.24 + i * 0.24, 0.7775), (0.28 + i * 0.24, 0.7775), color=MUTED)
arrow(ax, (0.86, 0.70), (0.14, 0.40), color=MUTED, rad=0.25, lw=1.6)
for i in range(3):
    arrow(ax, (0.24 + i * 0.24, 0.2775), (0.28 + i * 0.24, 0.2775), color=MUTED)
# 高亮 7-8 之间的价值
ax.add_patch(Rectangle((0.50, 0.135), 0.48, 0.30, fc="#FDECEA", ec=WARN,
                       lw=1.8, zorder=1, linestyle="--"))
ax.text(0.74, 0.075, "核心：把「LLM 说得好不好」变成可自动判分的数字",
        ha="center", fontsize=11, color=WARN, fontweight="bold", zorder=5)
ax.text(0.5, 0.955, "一条轨迹的完整生命周期", ha="center",
        fontsize=14.5, fontweight="bold", color=INK)
save(fig, "arch_loop.png")

# ============ 图 3：regret 三种口径 ============
fig, ax = plt.subplots(figsize=(10.4, 4.9))
cases = ["headroom 充足\n→ 归一化", "headroom 过小\n→ 绝对口径", "轨迹过短\n→ 不适用"]
vals = [0.75, 0.036, 0.0]
labels = ["归一化 regret\n0.75", "绝对 regret\n0.0358", "按 regime\n正确性准入"]
colors = [WARN, "#E8A33D", OK]
bars = ax.bar(cases, [1, 1, 1], color=colors, width=0.5, alpha=0.16)
for i, (b, v, lb) in enumerate(zip(bars, vals, labels)):
    ax.text(b.get_x() + b.get_width() / 2, 0.55,
            lb.split("\n")[0], ha="center", fontsize=13.5, fontweight="bold",
            color=colors[i])
    ax.text(b.get_x() + b.get_width() / 2, 0.33,
            lb.split("\n")[1], ha="center", fontsize=12, color=colors[i])
    ax.text(b.get_x() + b.get_width() / 2, 0.72,
            ["分母 = best − baseline", "分母过小会放大微小差距", "压缩率无物理含义"][i],
            ha="center", fontsize=9.5, color=MUTED)
ax.set_yticks([]); ax.set_ylim(0, 1)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("regret 的三种口径，必须显式标注", fontsize=15.5,
             fontweight="bold", color=INK, pad=16)
ax.tick_params(axis="x", labelsize=12)
save(fig, "regret_bases.png")

# ============ 图 4：四层记忆 ============
fig, ax = blank(11, 5.6)
tiers = [
    ("L0 工作记忆", "当前轨迹诊断卡 + 已试参数", "上下文", "agent", "#B0BEC5", INK),
    ("L1 情景记忆", "全量 (诊断→参数→指标) 记录", "SQLite", "核验器", "#2B7BBA", "white"),
    ("L2 程序记忆", "诊断签名 → 参数区间（12维特征 kNN）", "SQLite", "核验器", "#1F6F5C", "white"),
    ("L3 人类知识", "结论 + 证据 + 反例（Obsidian）", "Markdown", "人", "#8A6D1F", "white"),
]
y = 0.76
for name, desc, store, writer, color, tc in tiers:
    box(ax, 0.06, y, 0.875, 0.135, "", color, radius=0.012)
    ax.text(0.09, y + 0.0675, name, ha="left", va="center", color=tc,
            fontsize=12.5, fontweight="bold", zorder=3)
    ax.text(0.245, y + 0.0675, desc, ha="left", va="center", color=tc,
            fontsize=10, zorder=3)
    ax.text(0.745, y + 0.0675, store, ha="center", va="center", color=tc,
            fontsize=9.5, zorder=3)
    ax.text(0.885, y + 0.0675, writer, ha="center", va="center", color=tc,
            fontsize=10, fontweight="bold", zorder=3)
    y -= 0.16
ax.text(0.5, 0.955, "四层记忆：机器校准与人类知识分离", ha="center",
        fontsize=14, fontweight="bold", color=INK)
ax.text(0.06, 0.055, "硬规则：agent 对 Obsidian vault 只有读权限，永远不能写。"
        "自动导出落到 00-Inbox/，人工 review 后才升格。",
        ha="left", fontsize=10, color=WARN, fontweight="bold")
save(fig, "memory_tiers.png")

# ============ 图 5：三段式调参 ============
fig, ax = blank(11, 4.4)
seg = [
    (0.04, "① 物理先验定区间", "9 个参数各有合法区间\n由数据分布或物理量反推", "确定性代码", "#2B7BBA"),
    (0.36, "② LLM 提起点与方向", "params + expected_effect\n强制结构化输出", "LLM", "#8A6D1F"),
    (0.68, "③ 有界搜索精调", "坐标下降 + knee point\n给出 ground truth", "确定性代码", "#1F6F5C"),
]
for x, title, sub, who, c in seg:
    box(ax, x, 0.48, 0.28, 0.15, title, c, fs=12, radius=0.014)
    box(ax, x, 0.24, 0.28, 0.22, sub, "#F4F6F8", ec="#D8DEE4", tc=INK,
        fs=9.3, bold=False, radius=0.014)
    ax.text(x + 0.14, 0.155, who, ha="center", fontsize=10,
            color=c, fontweight="bold")
arrow(ax, (0.32, 0.555), (0.36, 0.555), color=MUTED)
arrow(ax, (0.64, 0.555), (0.68, 0.555), color=MUTED)
ax.text(0.5, 0.88, "三段式调参：LLM 不是优化器", ha="center",
        fontsize=15, fontweight="bold", color=INK)
ax.text(0.5, 0.80, "它只负责给起点和方向，好不好由确定性搜索当标尺", ha="center",
        fontsize=10.5, color=MUTED)
ax.text(0.5, 0.05, "expected_effect 让方向准确率可以零成本计算（只需比对符号）",
        ha="center", fontsize=10, color=ACCENT, style="italic")
save(fig, "param_three_stage.png")

# ============ 图 6：真实消融柱状图 ============
fig, ax = plt.subplots(figsize=(10.4, 4.8))
modes = ["llm-only", "search-only", "llm+search", "llm+memory\n+search"]
gap = [0.0026, -0.0004, 0.0025, 0.0025]
beat = [0.50, 0.25, 0.4167, 0.4167]
x = range(len(modes))
ax.bar([i - 0.19 for i in x], gap, width=0.36, color="#2B7BBA",
       label="相对基线提升（绝对口径）")
ax.bar([i + 0.19 for i in x], beat, width=0.36, color="#E8A33D",
       label="超基线率")
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(list(x)); ax.set_xticklabels(modes, fontsize=11)
ax.set_ylabel("数值", fontsize=12)
ax.legend(fontsize=10, loc="upper left")
ax.set_title("留出集消融：三种含 LLM 模式的得分完全相同", fontsize=15,
             fontweight="bold", color=INK, pad=14)
ax.text(0.5, -0.30, "记忆层在本配置下没有带来可测增益：moving 类轨迹在记忆里没有已验证案例，"
        "检索到的参数区间数为 0",
        transform=ax.transAxes, ha="center", fontsize=10.5, color=WARN)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
save(fig, "ablation_real.png")
print("完成")
