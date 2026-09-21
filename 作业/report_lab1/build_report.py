from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report_lab1"
FIG_DIR = OUT_DIR / "figures"
DOCX_PATH = ROOT / "轨迹数据预处理评估报告.docx"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    }
)

BLUE = "#1F4E78"
MID_BLUE = "#5B9BD5"
LIGHT_BLUE = "#DDEBF7"
ORANGE = "#ED7D31"
GOLD = "#D6A21D"
RED = "#C0392B"
GRAY = "#7F8C8D"


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_figures() -> dict[str, Path]:
    figures: dict[str, Path] = {}

    # 1. Processing-stage point counts and the source of split-stage losses.
    stages = ["原始", "分段后", "去噪后", "简化后"]
    counts = np.array([1_173_410, 631_602, 628_412, 202_336])
    retained = counts / counts[0] * 100
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    bars = ax.bar(stages, counts / 1e6, color=[GRAY, BLUE, MID_BLUE, GOLD], width=0.62)
    ax.set_ylabel("轨迹点数（百万）")
    ax.set_title("各处理阶段轨迹点数量")
    ax.grid(axis="y", alpha=0.22)
    for bar, n, r in zip(bars, counts, retained):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{n:,}\n原始数据的 {r:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.32)
    figures["stages"] = save_fig(fig, "01_stage_counts.png")

    labels = ["最终保留", "点数少于 5", "长度小于 65 m"]
    values = np.array([631_602, 54_059, 487_749])
    fig, ax = plt.subplots(figsize=(9.2, 2.5))
    left = 0
    colors = [BLUE, ORANGE, "#A5A5A5"]
    for label, value, color in zip(labels, values, colors):
        ax.barh([0], [value], left=left, color=color, label=label, height=0.42)
        ax.text(
            left + value / 2,
            0,
            f"{label}\n{value:,} 点  {value / values.sum() * 100:.2f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if color == BLUE else "black",
        )
        left += value
    ax.set_xlim(0, values.sum())
    ax.set_yticks([])
    ax.set_xlabel("原始轨迹点数")
    ax.set_title("分段阶段的数据保留与过滤来源")
    ax.grid(False)
    figures["loss"] = save_fig(fig, "02_split_loss.png")

    # 2. Split-threshold sensitivity heatmap.
    split_matrix = np.array(
        [
            [11.25, 26.69, 29.37, 29.42],
            [34.58, 51.35, 56.02, 56.52],
            [34.82, 51.71, 56.48, 57.08],
            [35.00, 51.98, 56.74, 57.35],
        ]
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    im = ax.imshow(split_matrix, cmap="Blues", vmin=10, vmax=60, aspect="auto")
    ax.set_xticks(range(4), ["95", "200", "400", "800"])
    ax.set_yticks(range(4), ["15", "30", "60", "120"])
    ax.set_xlabel("空间阈值（m）")
    ax.set_ylabel("时间阈值（s）")
    ax.set_title("分段参数敏感性：轨迹点保留率")
    for i in range(split_matrix.shape[0]):
        for j in range(split_matrix.shape[1]):
            color = "white" if split_matrix[i, j] > 42 else "black"
            ax.text(j, i, f"{split_matrix[i, j]:.2f}%", ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, label="保留率（%）", shrink=0.86)
    figures["split_heatmap"] = save_fig(fig, "03_split_sensitivity.png")

    # 3. Denoising sensitivity.
    direction = np.array([20, 35, 50, 70])
    removed = np.array([177, 130, 114, 82])
    high_speed = np.array([2.121, 2.121, 2.120, 2.121])
    fig, ax1 = plt.subplots(figsize=(8.2, 4.4))
    ax1.plot(direction, removed, marker="o", lw=2.2, color=BLUE, label="删除点数")
    ax1.set_xlabel("方向阈值（°）")
    ax1.set_ylabel("删除点数", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.grid(alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(direction, high_speed, marker="s", lw=2, color=ORANGE, label="超 25 m/s 比例")
    ax2.set_ylabel("超 25 m/s 比例（%）", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.set_ylim(2.115, 2.126)
    ax1.set_title("去噪参数敏感性：删除量变化明显，独立速度指标基本不变")
    figures["denoise"] = save_fig(fig, "04_denoise_sensitivity.png")

    # 4. DP tradeoff.
    tolerance = np.array([2, 5, 10, 20])
    compression = np.array([55.143, 67.036, 75.887, 82.280])
    p95_error = np.array([1.448, 3.599, 7.385, 13.900])
    mean_error = np.array([0.291, 0.853, 1.982, 4.014])
    fig, ax1 = plt.subplots(figsize=(8.4, 4.7))
    ax1.plot(tolerance, compression, "o-", color=BLUE, lw=2.3, label="压缩率")
    ax1.set_xlabel("DP 容差（m）")
    ax1.set_ylabel("压缩率（%）", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_ylim(50, 86)
    ax1.grid(alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(tolerance, p95_error, "s-", color=RED, lw=2.1, label="P95 几何误差")
    ax2.plot(tolerance, mean_error, "^-", color=ORANGE, lw=1.8, label="平均几何误差")
    ax2.set_ylabel("原始点到简化折线的距离（m）")
    ax2.set_ylim(0, 15.5)
    ax1.axvline(5, color=GOLD, ls="--", lw=1.5)
    ax1.text(5.3, 52.2, "候选基准 5 m", color="#7F6000", fontsize=9)
    lines = ax1.get_lines()[:1] + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    ax1.set_title("DP 容差的压缩率与几何误差权衡")
    figures["dp"] = save_fig(fig, "05_dp_tradeoff.png")

    # 5. Dynamic threshold comparison.
    names = ["固定 400 m", "固定 400 m + 动态约束"]
    retention = [56.018, 55.243]
    high_speed_ratio = [2.122, 1.099]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    axes[0].bar(names, retention, color=[BLUE, MID_BLUE], width=0.58)
    axes[0].set_ylabel("轨迹点保留率（%）")
    axes[0].set_ylim(54.5, 56.5)
    axes[0].set_title("保留率")
    for i, value in enumerate(retention):
        axes[0].text(i, value + 0.08, f"{value:.2f}%", ha="center")
    axes[1].bar(names, high_speed_ratio, color=[ORANGE, GOLD], width=0.58)
    axes[1].set_ylabel("超 25 m/s 比例（%）")
    axes[1].set_ylim(0, 2.5)
    axes[1].set_title("独立速度异常指标")
    for i, value in enumerate(high_speed_ratio):
        axes[1].text(i, value + 0.06, f"{value:.2f}%", ha="center")
    for ax in axes:
        ax.tick_params(axis="x", rotation=8)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("动态空间约束的收益与代价", fontweight="bold")
    fig.tight_layout()
    figures["dynamic"] = save_fig(fig, "06_dynamic_threshold.png")

    figures["examples"] = ROOT / "figures" / "hw1_real_examples.png"
    return figures


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color: str = "D9D9D9", size: str = "6") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        node = borders.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), size)
        node.set(qn("w:color"), color)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_run_font(run, cn="宋体", latin="Times New Roman", size=10.5, bold=None, color=None):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), cn)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_paragraph_spacing(paragraph, before=0, after=6, line=1.35):
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    fmt.line_spacing = line


def add_text(doc: Document, text: str, bold_lead: str | None = None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(0.74)
    set_paragraph_spacing(p, after=5, line=1.30)
    if bold_lead and text.startswith(bold_lead):
        r1 = p.add_run(bold_lead)
        set_run_font(r1, bold=True)
        r2 = p.add_run(text[len(bold_lead) :])
        set_run_font(r2)
    else:
        r = p.add_run(text)
        set_run_font(r)
    return p


def add_bullet(doc: Document, text: str, level=0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.paragraph_format.left_indent = Cm(0.75 + 0.65 * level)
    p.paragraph_format.first_line_indent = Cm(-0.35)
    set_paragraph_spacing(p, after=3, line=1.25)
    set_run_font(p.add_run(text))
    return p


def add_caption(doc: Document, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    set_paragraph_spacing(p, before=3, after=8, line=1.1)
    set_run_font(p.add_run(text), cn="宋体", size=9, color="595959")


def add_figure(doc: Document, path: Path, caption: str, width_inches=6.25):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width_inches))
    add_caption(doc, caption)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths=None, font_size=9):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for j, header in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        set_cell_shading(cell, "1F4E78")
        set_cell_border(cell)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_paragraph_spacing(p, after=0, line=1.05)
        set_run_font(p.add_run(header), cn="微软雅黑", size=font_size, bold=True, color="FFFFFF")
        if widths:
            cell.width = Cm(widths[j])
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        for j, value in enumerate(row):
            cell = cells[j]
            cell.text = ""
            set_cell_border(cell)
            set_cell_margins(cell)
            if i % 2:
                set_cell_shading(cell, "F3F6F9")
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            set_paragraph_spacing(p, after=0, line=1.05)
            set_run_font(p.add_run(str(value)), size=font_size)
            if widths:
                cell.width = Cm(widths[j])
    after = doc.add_paragraph()
    set_paragraph_spacing(after, after=2, line=1.0)
    return table


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr_text, fld_char2])
    set_run_font(run, size=9, color="777777")


def configure_styles(doc: Document):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)

    title = styles["Title"]
    title.font.name = "Microsoft YaHei"
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    title.font.size = Pt(24)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)

    for style_name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)):
        style = styles[style_name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.keep_with_next = True


def build_docx(figures: dict[str, Path]):
    doc = Document()
    configure_styles(doc)
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)
    section.header_distance = Cm(1.0)
    section.footer_distance = Cm(1.0)

    # Cover.
    for _ in range(4):
        doc.add_paragraph()
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # LibreOffice may inherit a blue bottom rule from the built-in Title style.
    # An explicit nil border keeps the required Title style without a decorative line.
    title_ppr = title._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "nil")
    borders.append(bottom)
    title_ppr.append(borders)
    title.add_run("轨迹数据预处理评估报告")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_spacing(subtitle, before=14, after=8, line=1.2)
    set_run_font(subtitle.add_run("《智慧城市与位置服务》实验一"), cn="微软雅黑", size=15, bold=True)
    scope = doc.add_paragraph()
    scope.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_spacing(scope, before=30, after=6, line=1.3)
    set_run_font(scope.add_run("数据范围：11,386 条轨迹，1,173,410 个轨迹点"), size=11)
    date_p = doc.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(date_p.add_run("报告日期：2026 年 9 月 21 日"), size=11)
    doc.add_page_break()

    # Static contents.
    h = doc.add_heading("报告目录", level=1)
    for item in [
        "1 任务要求与数据说明",
        "2 处理方法与实现口径",
        "3 评价指标设计",
        "4 实验结果与分析",
        "5 AI 建议的技术批判",
        "6 可复现性与程序检查",
        "7 结论与局限性",
        "参考资料",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        set_paragraph_spacing(p, after=5, line=1.2)
        set_run_font(p.add_run(item), cn="微软雅黑", size=11)

    doc.add_heading("摘要", level=1)
    add_text(
        doc,
        "本报告评估轨迹分段、方向去噪和 Douglas–Peucker 简化三个步骤。全量数据从 1,173,410 个点处理为 202,336 个点。分段后保留 631,602 个点，损失主要来自长度不足 65 m 的子段过滤；去噪删除 3,190 个点；5 m 容差的 DP 简化在去噪结果上压缩 67.80%，原始点到简化折线的平均、P95 和最大距离分别为 0.89 m、3.65 m 和 5.00 m。规则违规率用于验证分段代码是否满足既定条件，速度分位数、超物理速度比例和几何误差用于评价独立效果。结果表明，当前参数能形成可复现的候选基准，但 400 m 分段距离、35° 方向阈值和 5 m DP 容差仍需结合交通方式、采样周期、道路真值与人工漂移标签调整。",
    )

    doc.add_heading("1 任务要求与数据说明", level=1)
    add_text(
        doc,
        "实验课 PPT 要求完成轨迹数据分段、去噪和简化，并自行设计评价指标，说明指标逻辑与结果。PPT 还要求检查 AI 建议中的单位、坐标和插值假设，至少构造一组反例。老师标注的思考与讨论以及拓展内容按项目约定一并完成。",
    )
    add_table(
        doc,
        ["项目", "本次数据或设置"],
        [
            ["原始规模", "11,386 条轨迹，1,173,410 个点"],
            ["坐标", "经纬度输入；分段距离用球面距离，DP 几何运算用 WGS84 / UTM 51N"],
            ["时间", "Unix 时间戳；子轨迹内要求严格递增"],
            ["主要流程", "分段 → 去噪 → 简化"],
            ["抽样实验", "固定随机种子 20260917；低、高首尾净位移各抽 250 条"],
        ],
        widths=[4.0, 12.2],
        font_size=9.5,
    )
    add_text(
        doc,
        "数据经度范围为 120.386365°E 至 121.915853°E，纬度范围为 30.578117°N 至 32.239372°N。DP 使用 EPSG:32651。对数据范围内全部点计算投影尺度因子后，线性尺度约为 0.99973 至 1.00036，因此 5 m 投影容差可近似理解为 5 m 地面距离。分段阶段继续采用 Haversine 球面距离，避免把平面投影距离与既有切分规则混用。",
    )

    doc.add_heading("2 处理方法与实现口径", level=1)
    doc.add_heading("2.1 参数设置", level=2)
    add_table(
        doc,
        ["环节", "参数", "默认值", "物理含义"],
        [
            ["分段", "时间间隔阈值", "30 s", "相邻点间隔超过该值时切分"],
            ["分段", "空间距离阈值", "400 m", "相邻点球面距离超过该值时切分"],
            ["过滤", "最少点数", "5", "少于 5 点的候选子段过滤"],
            ["过滤", "最短长度", "65 m", "路线长度不足 65 m 的候选子段过滤"],
            ["去噪", "方向阈值", "35°", "候选点前后方向均出现较大差异"],
            ["去噪", "两侧最小位移", "5 m", "抑制静止抖动引起的误删"],
            ["简化", "DP 容差", "5 m", "原始点到候选简化线段的距离阈值"],
        ],
        widths=[2.3, 3.5, 2.2, 8.2],
        font_size=8.8,
    )
    add_text(
        doc,
        "PPT 给出的 95 m 来自约 12 s 采样间隔与 25 km/h 骑行速度上界，是骑行场景示例。当前 Notebook 的 400 m 属于车辆数据候选基准，报告不将它表述为普遍最优值，而是用参数敏感性与独立速度指标检查其影响。",
    )

    doc.add_heading("2.2 轨迹分段", level=2)
    add_text(
        doc,
        "相邻点出现非正时间间隔、时间间隔超过 30 s 或球面距离超过 400 m 时切开。切开事件本身不删除点。算法在切开后分别检查子段点数与路线长度，只有不满足最小条件的子段才被过滤。切分原因采用互斥统计，时间与空间同时异常单列，避免重复归因。",
    )

    doc.add_heading("2.3 轨迹去噪", level=2)
    add_text(
        doc,
        "去噪以方向突变为主要线索，同时增加两侧位移均超过 5 m、跨点直连距离小于至少一条折线路径的条件。这样可减少静止 GPS 抖动和正常转弯被误删的风险。删除后重新计算速度与方向，保证派生数组和轨迹点数一致。",
    )

    doc.add_heading("2.4 轨迹简化", level=2)
    add_text(
        doc,
        "简化使用索引式 Douglas–Peucker 实现。算法反复寻找当前首尾线段之外距离最大的点，若最大距离超过容差则保留该点并继续细分，否则只保留该段首尾点。索引式实现避免重复坐标通过坐标值反查时间戳时产生歧义，并保证首尾点不变。",
    )

    doc.add_heading("3 评价指标设计", level=1)
    add_text(doc, "评价分为规则一致性、独立数据质量和几何保持三类。三类指标回答不同问题，不能互相替代。")
    add_table(
        doc,
        ["指标", "计算方式", "用途与解释边界"],
        [
            ["规则违规比例", "检查非正时间、Δt > 30 s 或距离 > 400 m 的相邻点对", "验证分段代码是否执行正确；0% 不等于真实数据质量已改善"],
            ["速度 P95 / P99", "对正时间间隔计算球面距离除以 Δt", "观察速度分布尾部；受保留样本组成影响"],
            ["超 25 m/s 比例", "速度超过 25 m/s 的相邻点对占比", "独立的宽松城市机动车筛查线，不是人工真值"],
            ["压缩率", "1 − 简化后点数 / 简化前点数", "评价存储与计算量下降"],
            ["路线长度变化", "简化后总长度 / 简化前总长度 − 1", "检查总长度是否明显改变，但不能代表局部形状"],
            ["点到简化折线距离", "原始点到对应简化折线的最短距离", "直接评价局部几何误差，报告平均值、P95 与最大值"],
            ["合成真值案例", "正常与异常轨迹的预期保留/删除结果", "检查规则行为；不能替代真实轨迹人工标注"],
        ],
        widths=[3.2, 5.4, 7.8],
        font_size=8.5,
    )

    doc.add_heading("4 实验结果与分析", level=1)
    doc.add_heading("4.1 全量处理结果", level=2)
    add_figure(doc, figures["stages"], "图 1  全量数据各处理阶段的轨迹点数量", width_inches=6.25)
    add_text(
        doc,
        "分段保留 631,602 点，占原始点数的 53.83%；去噪后剩余 628,412 点，仅比完成分段的数据减少 3,190 点；5 m DP 简化后剩余 202,336 点，占原始数据的 17.24%。本次从头运行中，分段、去噪和简化分别耗时 5.83 s、6.46 s 和 8.15 s。耗时受硬件和运行环境影响，仅作为工程信息。",
    )
    add_figure(doc, figures["loss"], "图 2  分段后 46.17% 点未进入保留结果的具体来源", width_inches=6.25)
    add_table(
        doc,
        ["结果类别", "子段数", "点数", "占原始点数"],
        [
            ["保留子段", "19,143", "631,602", "53.83%"],
            ["点数少于 5", "41,053", "54,059", "4.61%"],
            ["长度小于 65 m", "7,223", "487,749", "41.57%"],
            ["合计", "67,419", "1,173,410", "100.00%"],
        ],
        widths=[4.4, 3.2, 4.0, 4.0],
        font_size=9,
    )
    add_text(
        doc,
        "切断边界共有 56,033 个，其中非正时间间隔 42,185 个，只有时间超限 6,696 个，只有空间超限 5,576 个，时间与空间同时超限 1,576 个。切断次数描述发生分段的原因，不能直接换算成丢失点数。实际损失发生在短子段过滤阶段，其中长度不足 65 m 贡献了 487,749 个点，是主要来源。这个规则也可能过滤有意义的静止活动，因此保留率低并不能单独说明处理质量。",
    )

    doc.add_heading("4.2 规则一致性与独立速度指标", level=2)
    add_table(
        doc,
        ["阶段", "轨迹数", "点数", "规则违规", "速度 P95", "速度 P99", "超 25 m/s"],
        [
            ["原始", "11,386", "1,173,410", "4.822%", "17.564 m/s", "29.418 m/s", "1.528%"],
            ["分段", "19,143", "631,602", "0.000%", "19.812 m/s", "37.030 m/s", "2.206%"],
            ["去噪", "19,143", "628,412", "0.082%", "19.822 m/s", "37.011 m/s", "2.207%"],
            ["简化", "19,143", "202,336", "31.327%", "21.849 m/s", "39.507 m/s", "2.911%"],
        ],
        widths=[2.2, 2.2, 3.0, 2.4, 2.4, 2.4, 2.4],
        font_size=7.8,
    )
    add_text(
        doc,
        "分段后的规则违规比例为 0%，说明所有保留子轨迹均满足既定切分条件。速度 P95 与超 25 m/s 比例没有同步下降。原始与分段后的超 25 m/s 相邻点绝对数从 17,108 降为 13,511，但正时间相邻点分母从 1,119,839 降为 612,459，比例反而由 1.53% 升为 2.21%。这反映了过滤后样本组成发生变化，不能据此宣称速度质量改善。去噪和 DP 会连接原本不相邻的点，因此后续规则违规率可以回升；特别是 DP 后相邻点跨越更长时间与距离，不宜直接和原始相邻点比较。",
    )

    doc.add_heading("4.3 分段参数敏感性", level=2)
    add_text(
        doc,
        "参数实验使用同一批 500 条轨迹，共 51,478 个点。按照首尾净位移是否达到 100 m 分层，每层固定随机抽取 250 条。分层用于比较参数趋势，不代表全体数据的自然比例；回环轨迹也可能因净位移较小而被划入低位移层。",
    )
    add_figure(doc, figures["split_heatmap"], "图 3  时间阈值与空间阈值对点数保留率的影响", width_inches=5.75)
    add_text(
        doc,
        "空间阈值固定为 400 m 时，时间阈值从 15 s 增至 30 s，保留率由 29.37% 增至 56.02%；继续放宽到 120 s，保留率仅增至 56.74%。因此只能说 30 s 之后保留率增幅明显变小。时间阈值固定为 30 s 时，空间阈值从 95 m 放宽到 400 m，保留率由 34.58% 增至 56.02%。95 m 对当前数据明显更严格，但 400 m 的物理合理性仍需结合交通方式和道路真值判断。",
    )

    doc.add_heading("4.4 去噪效果", level=2)
    add_figure(doc, figures["denoise"], "图 4  方向阈值对删除点数与独立速度指标的影响", width_inches=5.9)
    add_text(
        doc,
        "在抽样数据上，方向阈值从 20° 增至 70°，删除点数从 177 降到 82，删除比例从 0.614% 降到 0.284%。超 25 m/s 比例始终约为 2.12%，说明这组去噪规则主要针对局部形状突变，没有在速度尾部指标上表现出明显改善。删除比例低只能说明规则较保守，不能当作误删率低的证据。",
    )
    add_table(
        doc,
        ["合成案例", "期望", "实际", "结果"],
        [
            ["正常直线", "保留", "保留", "符合"],
            ["正常 90° 转弯", "保留", "保留", "符合"],
            ["U-turn", "保留", "保留", "符合"],
            ["静止 GPS 抖动", "保留", "保留", "符合"],
            ["单点漂移", "删除", "删除", "符合"],
        ],
        widths=[5.6, 3.4, 3.4, 3.4],
        font_size=9,
    )
    add_text(
        doc,
        "五组合成案例均得到预期结果，说明新增距离条件能保留正常转弯、回转与小幅静止抖动，同时识别单点漂移。合成案例的真值由构造过程给出，但真实 GPS 数据尚未进行人工标注，因此不能据此计算真实准确率或误删率。",
    )

    doc.add_heading("4.5 DP 简化效果", level=2)
    add_figure(doc, figures["dp"], "图 5  DP 容差的压缩率与局部几何误差", width_inches=5.9)
    add_table(
        doc,
        ["容差", "压缩率", "长度变化", "平均误差", "P95 误差", "最大误差"],
        [
            ["2 m", "55.143%", "−0.102%", "0.291 m", "1.448 m", "2.000 m"],
            ["5 m", "67.036%", "−0.283%", "0.853 m", "3.599 m", "4.994 m"],
            ["10 m", "75.887%", "−0.615%", "1.982 m", "7.385 m", "9.993 m"],
            ["20 m", "82.280%", "−1.084%", "4.014 m", "13.900 m", "19.998 m"],
        ],
        widths=[2.4, 2.8, 2.8, 2.8, 2.8, 2.8],
        font_size=8.5,
    )
    add_text(
        doc,
        "5 m 容差在抽样数据上压缩 67.04%，P95 几何误差为 3.60 m；全量数据上的压缩率为 67.80%，平均、P95 与最大误差为 0.89 m、3.65 m 和 5.00 m，全部轨迹首尾点保持不变。总路线长度仅变化 −0.28%，但总长度接近并不保证局部位置接近，因此参数判断以压缩率和点到折线误差共同为依据。5 m 可作为当前候选基准，不能称为绝对最优。",
    )

    doc.add_heading("4.6 动态空间阈值拓展", level=2)
    add_text(
        doc,
        "拓展实验在固定 400 m 上限之外增加“25 m/s × 实际时间差 × 1.5 安全系数”的动态约束，用于切断短时间内不合理的大位移。该方案与教师基准并行比较，没有直接替换原方案。",
    )
    add_figure(doc, figures["dynamic"], "图 6  动态空间约束的点数代价与速度异常变化", width_inches=5.95)
    add_text(
        doc,
        "动态约束额外触发 337 个切分边界，保留点数从 28,837 降为 28,438，保留率由 56.02% 降为 55.24%；速度 P99 由 35.96 m/s 降为 25.38 m/s，超 25 m/s 比例由 2.12% 降为 1.10%。该结果显示速度异常筛查的潜在收益，同时付出 399 个点的损失。由于时间戳也可能出错，且没有道路真值，当前仍保留固定 400 m 方案作为基准。",
    )

    doc.add_heading("4.7 真实轨迹形状核验", level=2)
    add_figure(doc, figures["examples"], "图 7  中断、漂移与弯道轨迹的处理前后叠加", width_inches=6.55)
    add_text(
        doc,
        "轨迹 2200 检出 5 个切断位置，分段后保留 2 段；直接连接断点会形成图中的灰色虚假路径。轨迹 7552_1 删除 8 个候选漂移点，红叉对应明显离开主线的绕行，但该样本尚无人工真值。弯道轨迹 7355_0 从 101 点简化为 61 点，关键转向位置仍有保留点。三条轨迹用于解释指标，不代表总体准确率。图中坐标采用 UTM 51N 米制投影。",
    )

    doc.add_heading("5 AI 建议的技术批判", level=1)
    add_text(
        doc,
        "按照 PPT 要求，将“缺失轨迹点一律线性插值”视为待验证建议。该建议隐含缺失期间沿直线匀速运动。在构造的 90° 转弯反例中，起点、拐点和终点沿折线路径运动，直接连接起终点得到的中点偏离已知路线 47.65 m。距离使用与 DP 一致的 UTM 51N 米制口径重新计算。",
    )
    add_bullet(doc, "拒绝无条件线性插值。长缺口优先分段，短缺口也要核对运动状态和道路形状。")
    add_bullet(doc, "拒绝把经纬度差直接解释为米。分段使用球面距离，几何运算使用局部米制投影。")
    add_bullet(doc, "拒绝不写单位的速度阈值。25 km/h 等于约 6.94 m/s，25 m/s 等于 90 km/h，二者适用场景不同。")
    add_text(
        doc,
        "这些结论来自物理假设检查和可复现实验。AI 建议只用于提出待检验方案，最终参数与处理方式由数据、单位、坐标口径和反例共同约束。AI 使用过程与交互证据另见独立的 AI 使用记录，不在本评估报告中重复。",
    )

    doc.add_heading("6 可复现性与程序检查", level=1)
    add_text(doc, "Notebook 从头运行时执行以下不变量检查，全部通过：")
    add_bullet(doc, "时间戳数组与坐标数组长度始终一致。")
    add_bullet(doc, "分段后的每条子轨迹时间戳严格递增，点数不少于 5，路线长度不少于 65 m。")
    add_bullet(doc, "去噪和简化后重新计算的速度、方向数组与轨迹点数一致。")
    add_bullet(doc, "简化后点数不超过简化前，且首尾时间戳与坐标保持一致。")
    add_text(
        doc,
        "三组参数实验使用同一批分层样本与固定随机种子 20260917，保证不同参数结果可比。完整实现、输出和图像保存在作业 Notebook 中。报告中的全量统计、抽样参数表和轨迹编号均来自该 Notebook 的实际执行结果。",
    )

    doc.add_heading("7 结论与局限性", level=1)
    add_text(
        doc,
        "本次实现完成了分段、去噪与简化，并为每一步设置了与其目标相匹配的证据。分段代码满足既定规则，但独立速度指标没有证明质量整体改善；分段后的点数损失主要来自 65 m 最小长度过滤。方向去噪规则在合成案例中表现符合预期，在真实数据上删除比例约 0.51%，但缺少人工标签，不能计算误删率。5 m DP 容差在当前数据上获得约 67.8% 压缩率和 3.65 m 的 P95 几何误差，形成了可解释的候选基准。",
    )
    add_text(
        doc,
        "当前最大限制是缺少道路真值、交通方式标签和真实漂移点人工标注。低于 65 m 的子段过滤了 487,749 个点，其中可能包含静止、短程或采样稀疏但有意义的活动。固定 400 m 距离阈值对不同时间间隔采用同一上限，动态速度约束虽然降低了速度异常比例，也依赖时间戳可靠性。后续若获得标注，可按交通方式和运动状态分层，报告去噪的精确率与召回率，并用道路匹配误差检验简化轨迹。",
    )
    add_text(
        doc,
        "因此，30 s、400 m、35° 和 5 m 应被理解为当前数据与任务下的候选参数。更换采样周期、交通方式或下游任务时，需要重新执行相同的敏感性实验与形状核验。",
    )

    doc.add_heading("参考资料", level=1)
    refs = [
        "[1] 《智慧城市与位置服务》实验课 1：轨迹数据质量提升与轨迹数据预处理，课程 PPT，2026-09-17。",
        "[2] Douglas, D. H., & Peucker, T. K. Algorithms for the Reduction of the Number of Points Required to Represent a Digitized Line or Its Caricature. The Canadian Cartographer, 10(2), 112–122, 1973. DOI: 10.3138/FM57-6770-U75U-7727.",
        "[3] PROJ Contributors. Universal Transverse Mercator (UTM) documentation. https://proj.org/en/stable/operations/projections/utm.html",
        "[4] 作业 1 轨迹数据预处理.ipynb，本项目实现与实验输出。",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.7)
        p.paragraph_format.first_line_indent = Cm(-0.7)
        set_paragraph_spacing(p, after=2, line=1.08)
        set_run_font(p.add_run(ref), size=8.5)

    # Header and footer after body is built.
    for sec in doc.sections:
        header = sec.header.paragraphs[0]
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        set_run_font(header.add_run("轨迹数据预处理评估报告"), cn="微软雅黑", size=8.5, color="777777")
        add_page_number(sec.footer.paragraphs[0])

    props = doc.core_properties
    props.title = "轨迹数据预处理评估报告"
    props.subject = "《智慧城市与位置服务》实验一"
    props.keywords = "轨迹分段, 轨迹去噪, Douglas-Peucker, 数据质量评估"
    props.comments = "由实验 Notebook 的实际运行结果整理"

    doc.save(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    figs = make_figures()
    build_docx(figs)
