import pptxgen from "pptxgenjs";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

// 必须用 fileURLToPath：中文目录名在 URL.pathname 里是百分号编码的。
const HERE = path.dirname(fileURLToPath(import.meta.url));
const ASSETS = path.join(HERE, "assets");
const OUT = path.join(HERE, "out");
fs.mkdirSync(OUT, { recursive: true });

const ZH = "Microsoft YaHei";
const EN = "Arial";
const INK = "1F2A37", MUTED = "5B6B7C", ACCENT = "2B7BBA";
const WARN = "C0392B", OK = "2E8B57", LIGHT = "EAF2F8";
const PAPER = "FFFFFF", SOFT = "F4F6F8";

const pptx = new pptxgen();
pptx.layout = "LAYOUT_16x9";
pptx.author = "轨迹处理实验课";
pptx.title = "轨迹数据清洗：从原始数据到可用轨迹";

const SLIDE_W = 960, SLIDE_H = 540;

// ---------- 备注：每页讲解要点与数据来源 ----------
// 备注按**页面标题**绑定，不按页码。
// 原因：页码会随插入新页而整体漂移，实测因此产生过一次整体错位
// （P6 是「重复点」却写着「任务一」的备注，P16 无备注）。
// 标题是稳定的语义键，插页不会破坏映射。
const NOTES = {
 "轨迹数据预处理": "开场。这门课的数据是 11386 辆车、117 万个点，采样间隔以 10 秒为主，单条记录约 20 分钟。数据来源 traj_dict.json。",
 "这门课要解决什么问题": "先讲清楚为什么需要预处理。学生容易直接跳到写代码，结果在地图匹配那一步才发现数据根本不能用。强调预处理是所有后续分析的前提。",
 "先看三个数据事实，它们决定了整条流水线": "本课最重要的铺垫，建议讲慢。三个数字都来自前 200 条抽样的真实统计。可以提问：如果对静止轨迹用行驶轨迹的速度阈值会怎样？答案是速度计算全部失效，因为位移几乎为零。",
 "数据格式：车辆 ID 到时间戳与坐标": "讲数据格式。两个列表必须等长且按时间升序，一旦错位后面所有速度计算都会出错。可以现场打开 traj_dict.json 让学生看一眼真实结构。",
 "两种轨迹类型必须分开处理": "核心概念页。静止与行驶两类必须分开处理。判断依据用位移加运动点占比双判据，不要只看总长度。实测反例：有一条轨迹停车 19 分钟后开走，只看长度会误判为静止。",
 "重复点的分布与处置顺序": "这一页解释为什么必须先折叠重复点再压缩。让学生注意只折叠连续重复点，非连续回访属于真实几何。可以出一道思考题：闭合环路该怎么处理？",
 "任务一：轨迹分段": "任务一。两条切分规则互相独立，一次切分可能同时命中，两个原因都要记录。特别提醒：讲义里的 30 秒阈值在这份数据上偏大，会把正常的 20 秒采样误切，建议让学生自己用分位数验证。",
 "任务二：异常点规则与原因字段": "任务二。四类异常对应四种处置，重点是保留异常原因字段。学生最常见的错误是只输出删了多少点，说不出删的是哪一类，导致后面无法调参和复核。",
 "最容易做错的地方：在噪声上算航向": "本课最重要的易错点，建议留足时间。现场做两个实验：一是不加噪声地板检测静止轨迹的转向角，会看到几十个假掉头；二是不加尺度归一化检测行驶轨迹的漂移，误判率高达一半。结论是任何涉及方向或偏离的判据都必须做尺度处理。",
 "任务三：去噪与 Douglas-Peucker 压缩": "任务三。强调去噪顺序：先折叠重复点再压缩，因为重复点会污染 DP 的最远点选择。DP 容差不要超过 GPS 精度 8 米。右图是车辆 246 的实测曲线，绿虚线标出 8 米精度线。",
 "任务四：四个对比指标": "任务四。四个指标都要给出具体数字，不能只贴图。重点讲 Hausdorff 的口径：必须算点到折线，不能算顶点到顶点。实测对比是 548 米对 4.77 米，这个坑很隐蔽，因为代码不会报错。",
 "任务五：三张必做图": "任务五。三张图都是车辆 246 的真实结果。判断作业时看三点：被删点有没有标出来、异常图例有没有给出每类计数、热力图有没有说清单位。",
 "选做一：阈值敏感性实验": "选做一。核心是让学生理解参数不是拍出来的。建议扫一遍 DP 容差，画出质量压缩率曲线，用拐点作为推荐值。这张曲线图后面讲 LLM 评测时还会用到，它就是目标函数的地形图。",
 "选做二：加入路网约束": "选做二。路网约束的价值在于让阈值有物理来源。关键要求是把路网做成可选依赖，没有路网数据时整条流水线必须照常运行，只是匹配率返回空值。",
 "作业要求一览": "作业要求汇总，右侧进阶项可作加分。评分标准是代码能跑通、异常原因说得清、指标对比有数字支撑。",
 "提交前自检清单": "提交前自检清单，可以直接发给学生当检查表。建议逐条对照，尤其是前三条，它们是本次作业主要的失分点。",
};

let pageNo = 0;
let current = null;
let currentTitle = "";

function beginSlide(bg = PAPER) {
  current = pptx.addSlide();
  current.background = { color: bg };
  pageNo += 1;
  return current;
}

function endSlide() {
  const note = NOTES[currentTitle];
  if (note) {
    current.addNotes(note);
  } else {
    // 缺备注时显式报错，而不是静默留空 —— 静默会让错位再次发生而不被发现
    throw new Error(`第 ${pageNo} 页「${currentTitle}」没有对应备注`);
  }
  current = null;
  currentTitle = "";
}

function head(s, title, kicker) {
  currentTitle = title;
  if (kicker) {
    s.addText(kicker, { x: 0.55, y: 0.26, w: 8.9, h: 0.24,
      fontFace: ZH, fontSize: 11, color: ACCENT, bold: true, charSpacing: 1 });
  }
  s.addText(title, { x: 0.55, y: kicker ? 0.48 : 0.36, w: 8.9, h: 0.55,
    fontFace: ZH, fontSize: 25, color: INK, bold: true });
  s.addShape(pptx.ShapeType.rect, { x: 0.55, y: kicker ? 1.06 : 0.96,
    w: 0.9, h: 0.035, fill: { color: ACCENT } });
}

function bullets(s, items, o = {}) {
  const fs0 = o.fontSize ?? 15;
  const runs = items.map((it) => ({
    text: it.t,
    options: {
      bullet: { characterCode: "25CF" },
      fontSize: it.big ? fs0 + 2 : fs0,
      color: it.color ?? INK,
      bold: !!it.bold,
      breakLine: true,
      paraSpaceAfter: it.gap ?? 8,
      indentLevel: it.level ?? 0,
    },
  }));
  s.addText(runs, { x: o.x ?? 0.6, y: o.y ?? 1.3, w: o.w ?? 5.0, h: o.h ?? 3.6,
    fontFace: ZH, valign: "top", lineSpacingMultiple: 1.22 });
}

function caption(s, text, y = 5.0) {
  s.addText(text, { x: 0.6, y, w: 8.85, h: 0.3,
    fontFace: ZH, fontSize: 10.5, color: MUTED, italic: true });
}

function callout(s, text, x, y, w, h, color = ACCENT) {
  s.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: SOFT }, line: { color: SOFT } });
  s.addShape(pptx.ShapeType.rect, { x, y, w: 0.055, h, fill: { color } });
  s.addText(text, { x: x + 0.24, y: y + 0.1, w: w - 0.42, h: h - 0.2,
    fontFace: ZH, fontSize: 13.5, color: INK, valign: "middle", lineSpacingMultiple: 1.2 });
}

function pageNum(s, n) {
  s.addText(String(n), { x: 9.05, y: 5.14, w: 0.5, h: 0.3,
    fontFace: EN, fontSize: 11, color: "AAB4BE", align: "right" });
}

function img(s, file, x, y, w, h) {
  s.addImage({ path: path.join(ASSETS, file), x, y, w, h,
    sizing: { type: "contain", w, h } });
}

// ================= 1. 封面 =================
{
  const s = beginSlide("0E2436");
  currentTitle = "轨迹数据预处理";
  s.addText("轨迹数据预处理", { x: 0.8, y: 1.55, w: 8.4, h: 0.5,
    fontFace: ZH, fontSize: 17, color: "7FB4D8", bold: true, charSpacing: 3 });
  s.addText("从原始数据到可用轨迹", { x: 0.8, y: 2.05, w: 8.4, h: 1.0,
    fontFace: ZH, fontSize: 40, color: "FFFFFF", bold: true });
  s.addShape(pptx.ShapeType.rect, { x: 0.8, y: 3.2, w: 1.5, h: 0.05,
    fill: { color: "4FA3D1" } });
  s.addText("11386 辆车 · 117 万点 · 20 秒采样轨迹的清洗、压缩与评价",
    { x: 0.8, y: 3.45, w: 8.4, h: 0.4, fontFace: ZH, fontSize: 15, color: "B8CEDA" });
  s.addText("实验课 1 · 作业讲解", { x: 0.8, y: 4.55, w: 8.4, h: 0.35,
    fontFace: ZH, fontSize: 13, color: "7A93A3" });
  endSlide();
}

// ================= 2. 问题 =================
{
  const s = beginSlide();
  head(s, "这门课要解决什么问题", "起点");
  bullets(s, [
    { t: "你手里有一条轨迹：一串带时间戳的经纬度点。", big: true },
    { t: "它不能直接用来做地图匹配、路网推理或速度分析。" },
    { t: "因为真实数据里有重复点、跳变点、时间戳故障。" },
    { t: "所以需要一条流水线：分段 → 去噪 → 压缩 → 评价。", bold: true, color: ACCENT, gap: 2 },
  ], { y: 1.35, w: 5.15, h: 3.5, fontSize: 15.5 });

  s.addShape(pptx.ShapeType.rect, { x: 6.0, y: 1.42, w: 3.42, h: 3.3,
    fill: { color: SOFT }, line: { color: SOFT } });
  s.addText([
    { text: "本课数据\n", options: { fontSize: 14, bold: true, color: INK, breakLine: true, paraSpaceAfter: 10 } },
    { text: "11386 辆车\n", options: { fontSize: 13, color: MUTED, breakLine: true, paraSpaceAfter: 6 } },
    { text: "1,173,410 个点\n", options: { fontSize: 13, color: MUTED, breakLine: true, paraSpaceAfter: 6 } },
    { text: "每车中位 100 点\n", options: { fontSize: 13, color: MUTED, breakLine: true, paraSpaceAfter: 6 } },
    { text: "时间间隔主频 10 秒\n", options: { fontSize: 13, color: MUTED, breakLine: true, paraSpaceAfter: 6 } },
    { text: "单条约 20 分钟\n", options: { fontSize: 13, color: MUTED, breakLine: true, paraSpaceAfter: 6 } },
    { text: "\n坐标格式 (经度, 纬度)", options: { fontSize: 11.5, color: "8A98A5", italic: true } },
  ], { x: 6.3, y: 1.75, w: 2.9, h: 2.7, fontFace: ZH, valign: "top", lineSpacingMultiple: 1.15 });
  pageNum(s, pageNo);
  endSlide();
}

// ================= 3. 三个数据事实 =================
{
  const s = beginSlide();
  head(s, "先看三个数据事实，它们决定了整条流水线", "关键前提");
  const items = [
    { big: "63%", t: "的车辆全程静止", d: "20 分钟内位移中位数只有 42 米", c: ACCENT },
    { big: "52.8%", t: "的相邻点对是重复点", d: "静止轨迹里重复点可占 90%", c: WARN },
    { big: "1037 秒", t: "是最大的时间间隔", d: "而主频是 10 秒，存在长间隔断点", c: OK },
  ];
  items.forEach((it, i) => {
    const y = 1.3 + i * 1.02;
    s.addText(it.big, { x: 0.6, y, w: 2.15, h: 0.55,
      fontFace: EN, fontSize: 27, color: it.c, bold: true });
    s.addText(it.t, { x: 2.75, y: y + 0.02, w: 2.6, h: 0.33,
      fontFace: ZH, fontSize: 14, color: INK, bold: true });
    s.addText(it.d, { x: 2.75, y: y + 0.35, w: 2.6, h: 0.5,
      fontFace: ZH, fontSize: 11.5, color: MUTED, lineSpacingMultiple: 1.15 });
  });
  // 右侧用真实分布图支撑这三个数字
  img(s, "chart_regime.png", 5.6, 1.28, 3.85, 2.75);
  callout(s, "静止轨迹与行驶轨迹必须用两套参数。用同一套阈值，不是效果差一点，而是对其中一半数据完全失效。",
    0.6, 4.28, 8.85, 0.78, ACCENT);
  caption(s, "左图为 200 条抽样的真实类型分布，右图的三项统计均来自同一份抽样", 4.72);
  pageNum(s, pageNo);
  endSlide();
}

// ================= 4. 数据格式 =================
{
  const s = beginSlide();
  head(s, "数据格式：车辆 ID 到时间戳与坐标", "数据概览");
  s.addShape(pptx.ShapeType.rect, { x: 0.6, y: 1.35, w: 5.0, h: 1.5, fill: { color: "1B2B3A" } });
  s.addText([
    { text: '{ "0": [', options: { color: "9CDCFE", breakLine: true } },
    { text: '    [1523293209, 1523293229, ...],', options: { color: "CE9178", breakLine: true } },
    { text: '    [[121.472353, 31.31464], ...]', options: { color: "CE9178", breakLine: true } },
    { text: '  ] }', options: { color: "9CDCFE" } },
  ], { x: 0.78, y: 1.5, w: 4.7, h: 1.2, fontFace: "Menlo", fontSize: 10.5,
    valign: "top", lineSpacingMultiple: 1.25 });
  s.addText("车辆 ID → [时间戳列表, 坐标列表]", { x: 0.6, y: 2.92, w: 5.0, h: 0.3,
    fontFace: ZH, fontSize: 11.5, color: MUTED });
  bullets(s, [
    { t: "两个列表等长、按时间升序", gap: 6 },
    { t: "坐标是 (经度, 纬度)，单位是度" },
    { t: "时间戳是 Unix 秒" },
  ], { x: 6.0, y: 1.35, w: 3.45, h: 1.5, fontSize: 13 });
  callout(s, "顺序很重要。一旦两个列表错位，后面所有速度计算都会出错，而且代码不会报错。",
    0.6, 3.5, 8.85, 0.75, WARN);
  caption(s, "可以现场打开 traj_dict.json 让学生看一眼真实结构");
  pageNum(s, pageNo);
  endSlide();
}

// ================= 5. 两种轨迹类型 =================
{
  const s = beginSlide();
  head(s, "两种轨迹类型必须分开处理", "核心概念");
  s.addText("同样是 20 分钟的记录，它们的参数完全不能共用",
    { x: 0.6, y: 1.06, w: 8.85, h: 0.3, fontFace: ZH, fontSize: 12.5, color: MUTED });

  const cols = [
    { x: 0.6, title: "静止轨迹", sub: "约 63% 的车辆", c: "6C7A89", rows: [
      ["位移中位数", "42 米"], ["连续重复点", "可占 90%"],
      ["中位速度", "0.1 m/s"], ["切分结果", "按长度阈值全部滤除"] ] },
    { x: 5.05, title: "行驶轨迹", sub: "约 37% 的车辆", c: ACCENT, rows: [
      ["位移中位数", "数公里至 20 公里"], ["连续重复点", "通常低于 10%"],
      ["中位速度", "10 至 20 m/s"], ["切分结果", "保留大部分点"] ] },
  ];
  cols.forEach((col) => {
    s.addShape(pptx.ShapeType.rect, { x: col.x, y: 1.45, w: 4.35, h: 0.52, fill: { color: col.c } });
    s.addText(col.title, { x: col.x + 0.18, y: 1.45, w: 2.6, h: 0.52,
      fontFace: ZH, fontSize: 16, bold: true, color: "FFFFFF", valign: "middle" });
    s.addText(col.sub, { x: col.x + 2.5, y: 1.45, w: 1.75, h: 0.52,
      fontFace: ZH, fontSize: 11.5, color: "E8F0F6", valign: "middle", align: "right" });
    col.rows.forEach((r, i) => {
      const y = 2.01 + i * 0.44;
      s.addShape(pptx.ShapeType.rect, { x: col.x, y, w: 4.35, h: 0.44,
        fill: { color: i % 2 === 0 ? SOFT : PAPER } });
      s.addText(r[0], { x: col.x + 0.18, y, w: 1.9, h: 0.44,
        fontFace: ZH, fontSize: 12.5, color: MUTED, valign: "middle" });
      s.addText(r[1], { x: col.x + 2.05, y, w: 2.2, h: 0.44,
        fontFace: ZH, fontSize: 12.5, color: INK, bold: true, valign: "middle" });
    });
  });
  callout(s, "判断依据不能只看总长度。先看位移，再看运动点占比，两者结合才稳。只看长度会把「停车 19 分钟后开走」误判为静止。",
    0.6, 3.95, 8.85, 0.75, WARN);
  caption(s, "占比来自 200 条抽样的实测统计", 4.82);
  pageNum(s, pageNo);
  endSlide();
}

// ================= 5b. 重复点：为什么先折叠再压缩 =================
{
  const s = beginSlide();
  head(s, "重复点的分布与处置顺序", "核心概念");
  img(s, "chart_dup.png", 0.55, 1.3, 4.7, 2.85);
  bullets(s, [
    { t: "超过一半的相邻点对是重复点。", big: true, color: WARN, gap: 9 },
    { t: "这些点不是噪声，而是车辆静止时的重复上报。" },
    { t: "但它们会让 DP 的最远点选择退化到重复段内部。", gap: 9 },
    { t: "所以顺序必须是：先折叠重复点，再压缩。", bold: true, color: OK, gap: 6 },
    { t: "反过来做，压缩率会明显下降。", level: 1, color: MUTED },
  ], { x: 5.5, y: 1.32, w: 3.95, h: 3.0, fontSize: 13.5 });
  callout(s, "只折叠「连续」重复点。非连续的回访点属于真实几何，删掉会改变轨迹形状。闭合环路和折返都属于这一类。",
    0.6, 4.3, 8.85, 0.78, WARN);
  pageNum(s, pageNo);
  endSlide();
}

// ================= 6. 任务一：分段 =================
{
  const s = beginSlide();
  head(s, "任务一：轨迹分段", "必做");
  bullets(s, [
    { t: "为什么要分段？一条记录里可能包含停车、行驶、失联三段。", gap: 7 },
    { t: "两条独立的切分规则：", bold: true, color: ACCENT, gap: 5 },
    { t: "时间间隔超过阈值 → 断开", level: 1, gap: 5 },
    { t: "空间跳跃超过阈值 → 断开", level: 1, gap: 7 },
    { t: "一次切分可以同时命中两条规则，两个原因都要记录。", gap: 7 },
    { t: "分段之后再做长度与点数过滤，滤掉静止碎片。", color: MUTED },
  ], { y: 1.32, w: 5.15, h: 3.6, fontSize: 14.5 });

  s.addShape(pptx.ShapeType.rect, { x: 6.0, y: 1.32, w: 3.42, h: 1.68, fill: { color: LIGHT } });
  s.addText([
    { text: "阈值怎么定？\n", options: { fontSize: 13, bold: true, color: INK, breakLine: true, paraSpaceAfter: 7 } },
    { text: "不是拍脑袋。用 Δt 分布的分位数。\n", options: { fontSize: 12, color: INK, breakLine: true, paraSpaceAfter: 5 } },
    { text: "本课数据主频 10 秒，p95 是 20 秒。\n", options: { fontSize: 12, color: INK, breakLine: true, paraSpaceAfter: 5 } },
    { text: "讲义的 30 秒 = 3×中位数，会误切正常的 20 秒采样。", options: { fontSize: 12, color: WARN, bold: true } },
  ], { x: 6.24, y: 1.5, w: 3.0, h: 1.4, fontFace: ZH, valign: "top", lineSpacingMultiple: 1.18 });

  s.addText("空间阈值有物理推导", { x: 6.0, y: 3.15, w: 3.42, h: 0.3,
    fontFace: ZH, fontSize: 13, bold: true, color: INK });
  s.addText("阈值 ≈ Δt × 路段限速 × 安全系数", { x: 6.0, y: 3.47, w: 3.42, h: 0.35,
    fontFace: ZH, fontSize: 12.5, color: ACCENT, bold: true });
  s.addText("例：20 秒 × 20 m/s × 1.5 = 600 米", { x: 6.0, y: 3.8, w: 3.42, h: 0.3,
    fontFace: ZH, fontSize: 11.5, color: MUTED });
  pageNum(s, pageNo);
  endSlide();
}

// ================= 7. 任务二：异常规则 =================
{
  const s = beginSlide();
  head(s, "任务二：异常点规则与原因字段", "必做");
  const rows = [
    ["速度突变", "单点速度超过阈值，且两侧正常", "剔除或中值平滑"],
    ["非法坐标", "NaN、越界、或 (0,0) 空值占位", "直接删除"],
    ["连续重复点", "与紧邻前一点坐标完全相同", "折叠，保留每段最后一个"],
    ["明显漂移", "偏离相邻点连线且偏离量异常", "删除或修坐标"],
  ];
  s.addTable([
    [
      { text: "异常类型", options: { bold: true, color: "FFFFFF", fill: { color: "2B4A63" } } },
      { text: "判据", options: { bold: true, color: "FFFFFF", fill: { color: "2B4A63" } } },
      { text: "处置", options: { bold: true, color: "FFFFFF", fill: { color: "2B4A63" } } },
    ],
    ...rows.map((r) => r.map((c, i) => ({
      text: c, options: { fontSize: 12, color: INK, fill: { color: i === 0 ? SOFT : "FFFFFF" } },
    }))),
  ], { x: 0.6, y: 1.3, w: 8.85, colW: [1.8, 4.35, 2.7],
    border: { type: "solid", color: "D8DEE4", pt: 0.5 },
    fontFace: ZH, rowH: 0.44, valign: "middle" });
  callout(s, "必须保留「异常原因」字段。不能只输出「删了 78 个点」，而要能说出删的是哪一类。这是后面调参和复核的唯一依据。",
    0.6, 3.6, 8.85, 0.8, WARN);
  caption(s, "还有一类容易漏掉：时间戳故障。下一页展开。");
  pageNum(s, pageNo);
  endSlide();
}

// ================= 8. 易错点：噪声地板 =================
{
  const s = beginSlide();
  head(s, "最容易做错的地方：在噪声上算航向", "易错点");
  bullets(s, [
    { t: "错误做法：在相邻两点之间直接算航向角。", color: WARN, bold: true, gap: 8 },
    { t: "如果这两点只差几厘米，得到的航向完全是随机数。", gap: 8 },
    { t: "实测后果：一条全程静止的轨迹报出 37 个掉头点。", gap: 8 },
    { t: "正确做法：加一个 10 米的位移下限。", bold: true, color: OK, gap: 5 },
    { t: "低于 10 米不判航向、不判漂移。", level: 1, gap: 8 },
    { t: "漂移判据也要做尺度归一化，用「垂距 ÷ 弦长」这个无量纲量。", gap: 6 },
    { t: "否则单步 235 米的正常行驶会被判成漂移，实测误判 50% 的点。", color: MUTED },
  ], { y: 1.32, w: 5.3, h: 3.6, fontSize: 14 });

  s.addShape(pptx.ShapeType.rect, { x: 6.15, y: 1.35, w: 3.28, h: 3.3, fill: { color: SOFT } });
  s.addText("修正前后对比", { x: 6.35, y: 1.5, w: 2.9, h: 0.3,
    fontFace: ZH, fontSize: 13, bold: true, color: INK });
  s.addText([
    { text: "轨迹 306 的漂移误判\n", options: { fontSize: 12, bold: true, color: INK, breakLine: true, paraSpaceAfter: 4 } },
    { text: "修正前  56 / 111 点  (50%)\n", options: { fontSize: 12, color: WARN, breakLine: true, paraSpaceAfter: 4 } },
    { text: "修正后   3 / 111 点  (2.7%)\n", options: { fontSize: 12, color: OK, breakLine: true, paraSpaceAfter: 12 } },
    { text: "静止轨迹的假掉头\n", options: { fontSize: 12, bold: true, color: INK, breakLine: true, paraSpaceAfter: 4 } },
    { text: "修正前  37 个\n", options: { fontSize: 12, color: WARN, breakLine: true, paraSpaceAfter: 4 } },
    { text: "修正后    0 个", options: { fontSize: 12, color: OK } },
  ], { x: 6.35, y: 1.85, w: 2.9, h: 2.6, fontFace: ZH, valign: "top", lineSpacingMultiple: 1.1 });
  pageNum(s, pageNo);
  endSlide();
}

// ================= 9. 任务三：去噪与 DP =================
{
  const s = beginSlide();
  head(s, "任务三：去噪与 Douglas-Peucker 压缩", "必做");
  bullets(s, [
    { t: "去噪的顺序有讲究。", bold: true, color: ACCENT, gap: 7 },
    { t: "先折叠重复点，再压缩。", level: 1, gap: 7 },
    { t: "原因：重复点会污染 DP 的「最远点」选择，压缩效率明显下降。", level: 1, color: MUTED, gap: 9 },
    { t: "DP 的原理：保留首尾，找离连线最远的点。", gap: 6 },
    { t: "若最远距离超过容差，就在该点切开，递归处理两段。", gap: 9 },
    { t: "容差不要超过 GPS 精度（约 8 米）。", bold: true, color: WARN, gap: 5 },
    { t: "超过它，删掉的就是真实几何，不是噪声。", level: 1, color: MUTED },
  ], { y: 1.32, w: 5.2, h: 3.6, fontSize: 14 });
  img(s, "chart_dp_curve.png", 5.95, 1.4, 3.5, 3.1);
  caption(s, "车辆 246 实测：容差 8 米时压缩约 75%，偏差仍在 GPS 精度内", 4.62);
  pageNum(s, pageNo);
  endSlide();
}

// ================= 10. 任务四：指标 =================
{
  const s = beginSlide();
  head(s, "任务四：四个对比指标", "必做");
  const items = [
    ["点数", "压缩率 = 1 − 压缩后点数 / 原点数", "直接反映压缩收益"],
    ["轨迹长度", "逐段地面距离累加", "压缩后长度不该明显变短"],
    ["Hausdorff 距离", "一个点到另一条折线的最大距离", "回答离原线有多远"],
    ["运行时间", "单条轨迹的处理耗时", "评估算法是否可用"],
  ];
  items.forEach((it, i) => {
    const y = 1.3 + i * 0.86;
    s.addShape(pptx.ShapeType.rect, { x: 0.6, y, w: 0.05, h: 0.7, fill: { color: ACCENT } });
    s.addText(it[0], { x: 0.82, y, w: 2.0, h: 0.7,
      fontFace: ZH, fontSize: 15, bold: true, color: INK, valign: "middle" });
    s.addText(it[1], { x: 2.9, y, w: 3.5, h: 0.7,
      fontFace: ZH, fontSize: 12.5, color: INK, valign: "middle" });
    s.addText(it[2], { x: 6.5, y, w: 2.95, h: 0.7,
      fontFace: ZH, fontSize: 12, color: MUTED, valign: "middle" });
  });
  callout(s, "Hausdorff 必须算「点到折线」，不能算「顶点到顶点」。压缩后顶点变稀疏，顶点集算法会给出完全错误的结果：实测 548 米，真实值 4.77 米。",
    0.6, 4.72, 8.85, 0.78, WARN);
  pageNum(s, pageNo);
  endSlide();
}

// ================= 11. 任务五：三张图 =================
{
  const s = beginSlide();
  head(s, "任务五：三张必做图", "必做");
  const imgs = [
    ["chart_overlay.png", "清洗前后叠加图", "灰色是原始，蓝线是清洗后，红叉是被删点"],
    ["chart_anomaly.png", "异常点分布图", "按原因分色，图例给出每类点数"],
    ["chart_heatmap.png", "轨迹热力图", "看停留密度，左右对比清洗前后"],
  ];
  imgs.forEach((it, i) => {
    const x = 0.55 + i * 3.08;
    s.addShape(pptx.ShapeType.rect, { x, y: 1.28, w: 2.92, h: 2.3,
      fill: { color: SOFT }, line: { color: "E1E7EC" } });
    s.addImage({ path: path.join(ASSETS, it[0]), x: x + 0.1, y: 1.38,
      w: 2.72, h: 2.1, sizing: { type: "contain", w: 2.72, h: 2.1 } });
    s.addText(it[1], { x, y: 3.66, w: 2.92, h: 0.32,
      fontFace: ZH, fontSize: 13.5, bold: true, color: INK, align: "center" });
    s.addText(it[2], { x, y: 4.0, w: 2.92, h: 0.6,
      fontFace: ZH, fontSize: 11, color: MUTED, align: "center", lineSpacingMultiple: 1.12 });
  });
  caption(s, "三张图都来自车辆 246 的真实清洗结果，不是示意图");
  pageNum(s, pageNo);
  endSlide();
}

// ================= 12. 选做一：敏感性 =================
{
  const s = beginSlide();
  head(s, "选做一：阈值敏感性实验", "进阶");
  bullets(s, [
    { t: "改变一个阈值，看结果怎么变。", gap: 7 },
    { t: "时间阈值、速度阈值、DP 容差各扫一遍。", gap: 7 },
    { t: "画「质量—压缩率曲线」。", bold: true, color: ACCENT, gap: 5 },
    { t: "曲线拐点就是最佳折衷。", level: 1, gap: 9 },
    { t: "为什么重要：这张曲线图就是后面 LLM 评测的目标函数地形图。", color: WARN, bold: true },
  ], { y: 1.32, w: 4.7, h: 3.6, fontSize: 14 });
  img(s, "chart_sens.png", 5.45, 1.32, 4.0, 3.4);
  caption(s, "横轴时间阈值、纵轴距离阈值，颜色是保留点数", 4.82);
  pageNum(s, pageNo);
  endSlide();
}

// ================= 13. 选做二：路网约束 =================
{
  const s = beginSlide();
  head(s, "选做二：加入路网约束", "进阶");
  s.addTable([
    [
      { text: "对比项", options: { bold: true, color: "FFFFFF", fill: { color: "2B4A63" } } },
      { text: "仅几何规则", options: { bold: true, color: "FFFFFF", fill: { color: "2B4A63" } } },
      { text: "加路网约束", options: { bold: true, color: "FFFFFF", fill: { color: "2B4A63" } } },
    ],
    ...[
      ["速度阈值来源", "全局常数 38 m/s", "按所在道路等级取限速"],
      ["跳跃阈值来源", "固定 400 米", "由路段限速推导"],
      ["漂移判定", "只看几何偏离", "还要求偏离道路中心线"],
      ["可解释性", "弱", "能说出这条轨迹在快速路上"],
    ].map((r) => r.map((c, i) => ({
      text: c, options: { fontSize: 12, color: INK, fill: { color: i === 0 ? SOFT : "FFFFFF" } },
    }))),
  ], { x: 0.6, y: 1.3, w: 8.85, colW: [2.5, 3.1, 3.25],
    border: { type: "solid", color: "D8DEE4", pt: 0.5 },
    fontFace: ZH, rowH: 0.45, valign: "middle" });
  callout(s, "关键：把路网做成可选依赖。没有路网数据时整条流水线必须照常运行，只是路网匹配率返回空值。",
    0.6, 3.95, 8.85, 0.8, OK);
  caption(s, "先做纯几何方案，再把路网作为可插拔组件接进来，两者对比才有意义");
  pageNum(s, pageNo);
  endSlide();
}

// ================= 14. 作业要求 =================
{
  const s = beginSlide();
  head(s, "作业要求一览", "提交清单");
  s.addText("必做", { x: 0.6, y: 1.28, w: 4.2, h: 0.3,
    fontFace: ZH, fontSize: 13, bold: true, color: ACCENT });
  bullets(s, [
    "切分：按车辆 ID、时间间隔、空间跳跃",
    "异常：速度突变、非法坐标、重复点、漂移",
    "清洗：去噪并保留异常原因字段",
    "压缩：Douglas-Peucker 简化",
  ].map((t) => ({ t, gap: 10 })), { x: 0.6, y: 1.62, w: 4.2, h: 2.0, fontSize: 13.5 });
  s.addText("选做与进阶", { x: 5.2, y: 1.28, w: 4.3, h: 0.3,
    fontFace: ZH, fontSize: 13, bold: true, color: OK });
  bullets(s, [
    "指标：点数、长度、Hausdorff、运行时间",
    "出图：叠加图、异常分布图、热力图",
    "选做：路网约束对比",
    "选做：阈值敏感性与质量—压缩率曲线",
  ].map((t) => ({ t, gap: 10 })), { x: 5.2, y: 1.62, w: 4.25, h: 2.0, fontSize: 13.5 });
  callout(s, "评分看三件事：代码能跑通、异常原因说得清、指标对比有数字支撑。只贴图不给数字，不算完成。",
    0.6, 4.0, 8.85, 0.75, WARN);
  pageNum(s, pageNo);
  endSlide();
}

// ================= 15. 自检清单 =================
{
  const s = beginSlide();
  head(s, "提交前自检清单", "收尾");
  bullets(s, [
    "分段函数对时间间隔和空间跳跃都做了判断，两个原因都记录",
    "异常原因字段贯穿全流程，能回答「删的是哪一类点」",
    "静止轨迹与行驶轨迹分别验证过，没有用同一套阈值硬套",
    "Hausdorff 用的是点到折线口径，不是顶点到顶点",
    "DP 的实际偏差不超过容差（这是数学保证，可以断言）",
    "三张图都有，且图例与坐标轴标清单位",
    "指标对比表里点数和长度对得上，没有悄悄丢点",
  ].map((t) => ({ t, gap: 7 })), { y: 1.3, w: 8.85, h: 3.6, fontSize: 13.5 });
  s.addText("有疑问随时在群里问，或者直接看 DECISIONS.md 里的踩坑记录。",
    { x: 0.6, y: 4.85, w: 8.85, h: 0.35,
      fontFace: ZH, fontSize: 12, color: ACCENT, italic: true });
  pageNum(s, pageNo);
  endSlide();
}

const outPath = path.join(OUT, "轨迹数据清洗_学生讲义.pptx");
await pptx.writeFile({ fileName: outPath });
console.log("written:", outPath);
console.log("slides:", pageNo);
