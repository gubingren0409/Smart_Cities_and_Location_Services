"""程序化校验 pptx：解析内部 XML，检查几何越界、字号、图片与文本完整性。

没有 LibreOffice 时无法渲染成图肉眼检查，故改用几何断言：
pptx 的每个 shape 都有明确的 EMU 坐标，越界和过小是可判定的。
"""
import sys, zipfile, re
from xml.etree import ElementTree as ET

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
EMU_PER_PX = 9525          # 96 DPI
SLIDE_W, SLIDE_H = 960, 540  # 16:9


def px(v):
    return int(v) / EMU_PER_PX


def main(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    slide_names = sorted(
        [n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)],
        key=lambda n: int(re.search(r"(\d+)", n.split("/")[-1]).group(1)))
    print(f"文件: {path.split('/')[-1]}")
    print(f"幻灯片数: {len(slide_names)}")
    media = [n for n in names if n.startswith("ppt/media/")]
    print(f"内嵌媒体: {len(media)} 个")
    print()

    problems, warnings_ = [], []
    total_text = 0

    for idx, sn in enumerate(slide_names, 1):
        root = ET.fromstring(z.read(sn))
        texts, min_font, n_pic, n_shape = [], None, 0, 0

        for sp in root.iter():
            tag = sp.tag.split("}")[-1]
            if tag in ("sp", "pic", "graphicFrame"):
                n_shape += 1
                if tag == "pic":
                    n_pic += 1
                # 几何
                xfrm = sp.find(".//a:xfrm", NS)
                if xfrm is not None:
                    off = xfrm.find("a:off", NS); ext = xfrm.find("a:ext", NS)
                    if off is not None and ext is not None:
                        x, y = px(off.get("x")), px(off.get("y"))
                        w, h = px(ext.get("cx")), px(ext.get("cy"))
                        if x < -2 or y < -2:
                            problems.append(f"P{idx} 图形起点越界 ({x:.0f},{y:.0f})")
                        if x + w > SLIDE_W + 6:
                            problems.append(
                                f"P{idx} 图形右越界 x+w={x+w:.0f} > {SLIDE_W}")
                        if y + h > SLIDE_H + 6:
                            problems.append(
                                f"P{idx} 图形下越界 y+h={y+h:.0f} > {SLIDE_H}")
                # 文本与字号
                for t in sp.findall(".//a:t", NS):
                    if t.text and t.text.strip():
                        texts.append(t.text)
                for rpr in sp.findall(".//a:rPr", NS):
                    sz = rpr.get("sz")
                    if sz:
                        pt = int(sz) / 100.0
                        min_font = pt if min_font is None else min(min_font, pt)

        total_text += sum(len(t) for t in texts)
        joined = " ".join(texts)
        if not joined.strip():
            problems.append(f"P{idx} 无任何文本")
        if min_font is not None and min_font < 9.5:
            warnings_.append(f"P{idx} 最小字号 {min_font}pt < 9.5pt")

        head = joined[:56].replace("\n", " ")
        print(f"  P{idx:2d}  图形{n_shape:2d} 图{n_pic} "
              f"最小字号{min_font if min_font else '-':>5}   {head}")

    print()
    print(f"文本总字符数: {total_text}")
    if problems:
        print("\n[错误]")
        for p in problems:
            print("  ", p)
    else:
        print("\n几何检查: 通过（无越界、无空页）")
    if warnings_:
        print("\n[提醒]")
        for w in warnings_:
            print("  ", w)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
