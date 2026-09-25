"""Render the final Lab 1 Markdown report to a submission-ready PDF.

The Markdown file remains the report source of truth.  Pandoc creates a DOCX,
python-docx applies the course report layout, and Microsoft Word exports PDF.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "作业" / "实验一_轨迹数据预处理_最终实验报告.md"
BUILD_DIR = Path(tempfile.gettempdir()) / "smart_cities_lab1_report_build"
DOCX_PATH = BUILD_DIR / "实验一_轨迹数据预处理_最终实验报告.docx"
PDF_PATH = ROOT / "output" / "pdf" / "实验一_轨迹数据预处理_最终实验报告.pdf"

BLUE = "1F4E78"
LIGHT_BLUE = "DCE6F1"
GRID = "AAB7C4"
BODY = "1F2933"
MUTED = "657381"


def set_east_asia_font(run, font_name: str = "Microsoft YaHei") -> None:
    run.font.name = font_name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), font_name)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top: int = 75, start: int = 90, bottom: int = 75, end: int = 90) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instruction, separate, text, end))
    set_east_asia_font(run)
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string(MUTED)


def has_drawing(paragraph) -> bool:
    return bool(paragraph._p.xpath(".//w:drawing | .//w:pict"))


def add_docx_page_field_start(section, start: int = 0) -> None:
    sect_pr = section._sectPr
    page_num = sect_pr.find(qn("w:pgNumType"))
    if page_num is None:
        page_num = OxmlElement("w:pgNumType")
        sect_pr.append(page_num)
    page_num.set(qn("w:start"), str(start))


def convert_markdown_to_docx() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pandoc",
            str(SOURCE),
            "--from=gfm-implicit_figures",
            "--to=docx",
            f"--resource-path={SOURCE.parent}",
            f"--output={DOCX_PATH}",
        ],
        cwd=SOURCE.parent,
        check=True,
    )


def style_docx() -> None:
    doc = Document(DOCX_PATH)

    section = doc.sections[0]
    section.start_type = WD_SECTION_START.NEW_PAGE
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.55)
    section.bottom_margin = Cm(1.45)
    section.left_margin = Cm(1.65)
    section.right_margin = Cm(1.65)
    section.header_distance = Cm(0.65)
    section.footer_distance = Cm(0.55)
    section.different_first_page_header_footer = True
    add_docx_page_field_start(section, 0)

    normal = doc.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(9.4)
    normal.font.color.rgb = RGBColor.from_string(BODY)
    normal.paragraph_format.line_spacing = 1.22
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.widow_control = True

    heading_specs = {
        "Heading 1": (15.5, 12, 5),
        "Heading 2": (13.0, 10, 4),
        "Heading 3": (10.7, 7, 2.5),
    }
    for name, (size, before, after) in heading_specs.items():
        style = doc.styles[name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(BLUE)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True

    for style_name in ("Source Code", "Verbatim Char"):
        if style_name in doc.styles:
            style = doc.styles[style_name]
            style.font.name = "Consolas"
            style.font.size = Pt(8.3)

    # Body text and inline formatting.
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        paragraph.paragraph_format.widow_control = True
        for run in paragraph.runs:
            if run.style and run.style.name == "Verbatim Char":
                run.font.name = "Consolas"
                run.font.size = Pt(8.3)
            else:
                set_east_asia_font(run)

        if text.startswith("图 "):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(5)
            paragraph.paragraph_format.keep_together = True
            for run in paragraph.runs:
                set_east_asia_font(run)
                run.font.size = Pt(8.1)
                run.font.color.rgb = RGBColor.from_string(MUTED)
        elif has_drawing(paragraph):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.space_after = Pt(1)
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.keep_together = True

    # Cover page: title, report label and metadata table.
    if len(doc.paragraphs) >= 2:
        title = doc.paragraphs[0]
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.paragraph_format.space_before = Pt(110)
        title.paragraph_format.space_after = Pt(16)
        title.paragraph_format.keep_with_next = True
        for run in title.runs:
            set_east_asia_font(run)
            run.font.size = Pt(22)
            run.font.bold = True
            run.font.color.rgb = RGBColor.from_string(BLUE)

        label = doc.paragraphs[1]
        label.alignment = WD_ALIGN_PARAGRAPH.CENTER
        label.paragraph_format.space_after = Pt(24)
        for run in label.runs:
            set_east_asia_font(run)
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor.from_string(BLUE)

    # Start the body on a new page and keep the advanced exploration distinct.
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text == "摘要":
            paragraph.paragraph_format.page_break_before = True
        elif text.startswith("进阶探索："):
            paragraph.paragraph_format.page_break_before = True

    # Keep image dimensions within the printable area while preserving aspect.
    max_width = Cm(15.8)
    max_height = Cm(9.2)
    for shape in doc.inline_shapes:
        scale = min(1.0, max_width / shape.width, max_height / shape.height)
        shape.width = int(shape.width * scale)
        shape.height = int(shape.height * scale)

    # Tables use compact typography, repeated headers and unsplit rows.
    for table_index, table in enumerate(doc.tables):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True
        for row_index, row in enumerate(table.rows):
            prevent_row_split(row)
            for cell in row.cells:
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                set_cell_margins(cell)
                if row_index == 0:
                    set_cell_shading(cell, LIGHT_BLUE)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.line_spacing = 1.05
                    if row_index < len(table.rows) - 1:
                        paragraph.paragraph_format.keep_with_next = True
                    for run in paragraph.runs:
                        set_east_asia_font(run)
                        run.font.size = Pt(8.0 if table_index else 9.2)
                        if row_index == 0:
                            run.font.bold = True
                            run.font.color.rgb = RGBColor.from_string(BLUE)
        # Metadata table on the cover has a little more vertical space.
        if table_index == 0:
            for row in table.rows:
                for cell in row.cells:
                    set_cell_margins(cell, top=115, bottom=115)

    # Header and footer; the first page stays clean.
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.paragraph_format.space_after = Pt(0)
    header_run = header.add_run("《智慧城市与位置服务》实验一")
    set_east_asia_font(header_run)
    header_run.font.size = Pt(8)
    header_run.font.color.rgb = RGBColor.from_string(MUTED)

    footer = section.footer.paragraphs[0]
    footer.paragraph_format.space_before = Pt(0)
    footer.paragraph_format.tab_stops.add_tab_stop(Cm(17.1), WD_TAB_ALIGNMENT.RIGHT)
    footer_run = footer.add_run("智慧城市与位置服务 · 实验一\t")
    set_east_asia_font(footer_run)
    footer_run.font.size = Pt(8)
    footer_run.font.color.rgb = RGBColor.from_string(MUTED)
    add_page_number(footer)

    props = doc.core_properties
    props.title = "实验一：轨迹数据预处理"
    props.subject = "《智慧城市与位置服务》课程实验报告"
    props.author = "谷秉仁"
    props.keywords = "轨迹分段, 轨迹去噪, Douglas-Peucker, OSM, LLM"
    props.comments = "由最终 Markdown 报告构建"

    doc.save(DOCX_PATH)


def export_pdf_with_word() -> None:
    import win32com.client  # type: ignore[import-untyped]

    if PDF_PATH.exists():
        PDF_PATH.unlink()
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(str(DOCX_PATH.resolve()), ReadOnly=False)
        document.Repaginate()
        document.Fields.Update()
        document.ExportAsFixedFormat(
            OutputFileName=str(PDF_PATH.resolve()),
            ExportFormat=17,
            OpenAfterExport=False,
            OptimizeFor=0,
            Range=0,
            Item=0,
            IncludeDocProps=True,
            KeepIRM=True,
            CreateBookmarks=1,
            DocStructureTags=True,
            BitmapMissingFonts=True,
            UseISO19005_1=False,
        )
    finally:
        if document is not None:
            document.Close(False)
        word.Quit()


def normalize_pdf_metadata() -> None:
    """Replace Word's locale-dependent metadata with UTF-8-safe values."""
    from pypdf import PdfReader, PdfWriter

    temporary = PDF_PATH.with_name(f"{PDF_PATH.stem}.metadata.pdf")
    reader = PdfReader(str(PDF_PATH))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    writer.add_metadata(
        {
            "/Title": "实验一：轨迹数据预处理",
            "/Subject": "《智慧城市与位置服务》课程实验报告",
            "/Author": "谷秉仁",
            "/Keywords": "轨迹分段, 轨迹去噪, Douglas-Peucker, OSM, LLM",
        }
    )
    with temporary.open("wb") as stream:
        writer.write(stream)
    temporary.replace(PDF_PATH)


def main() -> None:
    convert_markdown_to_docx()
    style_docx()
    export_pdf_with_word()
    normalize_pdf_metadata()
    print(PDF_PATH)
    print(f"bytes={PDF_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
