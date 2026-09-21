import pptxgen from "pptxgenjs";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

// 全部用 PPT 原生形状/表格/文本构建，不使用任何位图。
// 理由：位图在 PowerPoint 里无法编辑，而这个框架的架构会持续演进，
// 讲授者需要能自己改动流程图的文字与结构。
const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, "out2");
fs.mkdirSync(OUT, { recursive: true });

const ZH = "Microsoft YaHei", EN = "Arial", MONO = "Consolas";
const INK = "1F2A37", MUTED = "5B6B7C", ACCENT = "2B7BBA";
const WARN = "C0392B", OK = "2E8B57", GOLD = "8A6D1F", PURPLE = "6B4A7A";
const DARK = "2B4A63";
const PAPER = "FFFFFF", SOFT = "F4F6F8", LIGHT = "EAF2F8", BORDER = "D8DEE4";

const pptx = new pptxgen();
pptx.layout = "LAYOUT_16x9";
pptx.author = "轨迹处理实验课";
pptx.title = "基于 LLM 的轨迹数据清洗评估";

// 备注按页面标题绑定（页码会随插页漂移）
const NOTES = {
 "基于 LLM 的轨迹数据清洗评估":
   "开场。这套方法要把轨迹清洗的参数决策交给 LLM，但它的建议必须能被自动判分。今天讲的是完整做法：怎么组织输入、让 LLM 调哪些工具、参数怎么定、怎么核验它的建议、以及怎么判断这套流程真的有效。",
 "要解决什么问题": "这是动机页。核心是第二点：没法判分。做工程的人最容易忽略这点，结果做出来的东西只能演示不能评价。建议点明：本课所有设计都是为了让第三点（成本）和第二点（判分）同时成立。",
 "整体做法：四步":
   "这是方法论总览，后面每一页都在展开其中一步。建议强调第一步是替代 LLM 的语感，第三步是替代人的判断。LLM 只出现在第二步。",
 "第一步：把数据变成 LLM 能读的输入":
   "这一步的关键是纪律。117 万个点不可能进上下文，所以只给标量和句柄。诊断卡里最有用的是 regime 和时间轴质量，因为它们直接决定该用哪套参数。",
 "为什么要先分类型": "固定随机 200 条样本中，静止、混合和行驶分别占 47%、24% 和 29%。这只是该样本的分布，但足以说明参数实验需要分层，不能让某一类轨迹主导。",
 "第二步：给 LLM 的工具箱":
   "讲工具分类。重点是刻意不提供写记忆的工具，这条规则让记忆不会被幻觉污染。另外要说明工具返回的都是标量统计，不是坐标。",
 "第三步：参数怎么定": "回答参数怎么调这个核心问题。LLM 独立给出候选参数和方向，确定性搜索再独立建立内部参考。LLM 不负责搜索最优，这也是建议能够被自动判分的基础。",
 "第四步：怎么判断 LLM 提得对不对":
   "核心页，建议讲慢。regret 需要统一内部参考，所以从默认参数独立跑一次确定性搜索。三种口径一定要讲清楚，否则会得出错误结论。方向准确率是零成本的额外判据，值得强调。",
 "记忆怎么用：只复用已准入案例":
   "讲清楚为什么要分层。L2 是机器算出来的统计，L3 是人写的解释，两者不冲突。注意最后一行的成本含义：跑少数、复用多数，这是 11386 条能落地的关键。",
 "消融实验：怎么证明这套流程有效":
   "方法论页。两个陷阱都会让结论完全相反，而且都很隐蔽。信息泄漏尤其容易犯，因为它在单条轨迹上看不出任何异常。",
  "ECNU 三轮结果：尚未检出稳定增益":
    "这页使用 ecnu-plus 的三轮真实调用。Memory 配对均值略为正，但置信区间跨过零。moving/degraded 三轮只准入一条，每个参数区域也只有一到两个样本，因此当前证据还不足以判断 Memory 的稳定贡献。",
 "六个被数据推翻的假设":
   "这页对做过实际数据处理的人最有价值。挑两三个讲透即可，比如墨卡托放大 17% 和 Hausdorff 的 548 米对 4.77 米。这些坑都不会让代码报错，只会让结论错。",
 "现在能跑到什么程度": "如实交代边界。左侧可直接用，右侧是还没验证的部分。不要把未验证的说成已验证。",
 "怎么接着做": "三条按收益排序。第一条最紧，如果不解决，后面两条得出的结论都不可信。",
};

let pageNo = 0, current = null, currentTitle = "";
const S = pptx.ShapeType;

function beginSlide(bg = PAPER) {
  current = pptx.addSlide();
  current.background = { color: bg };
  pageNo += 1;
  return current;
}
function endSlide() {
  const n = NOTES[currentTitle];
  if (n) current.addNotes(n);
  else throw new Error(`第 ${pageNo} 页「${currentTitle}」缺备注`);
  current = null; currentTitle = "";
}
function head(s, title, kicker) {
  currentTitle = title;
  if (kicker) s.addText(kicker, { x: 0.55, y: 0.24, w: 8.9, h: 0.24,
    fontFace: ZH, fontSize: 11, color: ACCENT, bold: true, charSpacing: 1 });
  s.addText(title, { x: 0.55, y: kicker ? 0.46 : 0.34, w: 8.9, h: 0.55,
    fontFace: ZH, fontSize: 24, color: INK, bold: true });
  s.addShape(S.rect, { x: 0.55, y: kicker ? 1.03 : 0.93, w: 0.9, h: 0.035,
    fill: { color: ACCENT } });
}
function bullets(s, items, o = {}) {
  const f0 = o.fontSize ?? 14;
  s.addText(items.map((it) => ({
    text: it.t,
    options: { bullet: { characterCode: "25CF" }, fontSize: it.big ? f0 + 1.5 : f0,
      color: it.color ?? INK, bold: !!it.bold, breakLine: true,
      paraSpaceAfter: it.gap ?? 8, indentLevel: it.level ?? 0 },
  })), { x: o.x ?? 0.6, y: o.y ?? 1.28, w: o.w ?? 5.0, h: o.h ?? 3.6,
    fontFace: ZH, valign: "top", lineSpacingMultiple: 1.18 });
}
function caption(s, t, y = 5.02) {
  s.addText(t, { x: 0.6, y, w: 8.85, h: 0.3, fontFace: ZH, fontSize: 10.5,
    color: MUTED, italic: true });
}
function callout(s, t, x, y, w, h, color = ACCENT) {
  s.addShape(S.rect, { x, y, w, h, fill: { color: SOFT }, line: { color: SOFT } });
  s.addShape(S.rect, { x, y, w: 0.055, h, fill: { color } });
  s.addText(t, { x: x + 0.24, y: y + 0.06, w: w - 0.42, h: h - 0.12,
    fontFace: ZH, fontSize: 12, color: INK, valign: "middle", lineSpacingMultiple: 1.16 });
}
function pageNum(s) {
  s.addText(String(pageNo), { x: 9.05, y: 5.14, w: 0.5, h: 0.3, fontFace: EN,
    fontSize: 11, color: "AAB4BE", align: "right" });
}
// 原生方块
function node(s, x, y, w, h, text, fill, o = {}) {
  s.addShape(o.round ? S.roundRect : S.rect, {
    x, y, w, h, fill: { color: fill },
    line: { color: o.border ?? fill, width: o.bw ?? 1 },
    rectRadius: o.round ? 0.06 : undefined,
  });
  s.addText(text, { x: x + 0.05, y: y + 0.03, w: w - 0.1, h: h - 0.06,
    fontFace: ZH, fontSize: o.fs ?? 10.5, color: o.tc ?? "FFFFFF",
    bold: o.bold !== false, align: "center", valign: "middle",
    lineSpacingMultiple: o.ls ?? 1.1 });
}
// 原生箭头（用线条 + 箭头端点）
function arrow(s, x, y, w, h, color = MUTED, dir = "right") {
  const map = {
    right: { x, y, w, h: 0, line: { beginArrowType: "none", endArrowType: "triangle" } },
    down:  { x, y, w: 0, h, line: { beginArrowType: "none", endArrowType: "triangle" } },
    up:    { x, y, w: 0, h, line: { beginArrowType: "triangle", endArrowType: "none" } },
  };
  const m = map[dir];
  s.addShape(S.line, { x: m.x, y: m.y, w: m.w, h: m.h,
    line: { color, width: 1.6, ...m.line } });
}
function table(s, rows, o) {
  s.addTable(rows, { x: o.x, y: o.y, w: o.w, colW: o.colW,
    border: { type: "solid", color: BORDER, pt: 0.5 },
    fontFace: ZH, rowH: o.rowH ?? 0.45, valign: "middle",
    fill: { color: PAPER } });
}
function hdrRow(cells, color = DARK) {
  return cells.map((t) => ({ text: t,
    options: { bold: true, color: "FFFFFF", fill: { color } } }));
}

// ================= 1 封面 =================
{
  const s = beginSlide("10202F"); currentTitle = "基于 LLM 的轨迹数据清洗评估";
  s.addText("任务三 · 方法与实现", { x: 0.8, y: 1.32, w: 8.4, h: 0.4,
    fontFace: ZH, fontSize: 15, color: "7FB4D8", bold: true, charSpacing: 3 });
  s.addText("基于 LLM 的轨迹数据清洗评估", { x: 0.8, y: 1.8, w: 8.6, h: 1.1,
    fontFace: ZH, fontSize: 34, color: "FFFFFF", bold: true });
  s.addShape(S.rect, { x: 0.8, y: 3.0, w: 1.5, h: 0.05, fill: { color: "4FA3D1" } });
  s.addText("让 LLM 选评估工具、提参数建议，再用定量指标核验建议", {
    x: 0.8, y: 3.24, w: 8.6, h: 0.4, fontFace: ZH, fontSize: 15, color: "B8CEDA" });
  s.addText("四步做法 · 15 个工具 · 四层记忆 · 留出集消融", {
    x: 0.8, y: 3.72, w: 8.6, h: 0.35, fontFace: ZH, fontSize: 12.5, color: "7A93A3" });
  endSlide();
}

// ================= 2 要解决什么问题 =================
{
  const s = beginSlide();
  head(s, "要解决什么问题", "动机");
  const rows = [
    ["① 参数靠猜", "清洗参数（比如 DP 容差、速度阈值）通常是手工试出来的，换个数据集就失效。", ACCENT],
    ["② 没法判分", "让 LLM 给个参数很容易，但「它提得对不对」没有量化判据，只能靠人看。", WARN],
    ["③ 成本不可行", "11386 条轨迹，每条都调 LLM 不现实，需要让结果可以复用。", GOLD],
  ];
  rows.forEach((r, i) => {
    const y = 1.26 + i * 1.2;
    s.addShape(S.rect, { x: 0.6, y, w: 0.055, h: 1.02, fill: { color: r[2] } });
    s.addText(r[0], { x: 0.85, y, w: 2.2, h: 1.02, fontFace: ZH, fontSize: 16.5,
      bold: true, color: r[2], valign: "middle" });
    s.addText(r[1], { x: 3.1, y, w: 6.35, h: 1.02, fontFace: ZH, fontSize: 13,
      color: INK, valign: "middle", lineSpacingMultiple: 1.2 });
  });
  callout(s, "本课要讲的就是怎么同时解决②和③：把 LLM 的建议变成可自动判分的数字，并让结果可以复用。",
    0.6, 4.74, 8.85, 0.68, OK);
  pageNum(s);
  endSlide();
}

// ================= 3 整体做法：四步（原生流程图） =================
{
  const s = beginSlide();
  head(s, "整体做法：四步", "方法总览");
  const steps = [
    ["第一步", "把数据变成", "LLM 能读的输入", "诊断卡 + 句柄", ACCENT],
    ["第二步", "让 LLM 决定", "用什么工具、提什么参数", "函数调用", GOLD],
    ["第三步", "参数不靠猜", "物理先验定区间", "确定性代码给边界", OK],
    ["第四步", "核验它的建议", "跟确定性搜索最优比", "regret 判分", WARN],
  ];
  const w = 2.12, gap = 0.19, x0 = 0.6;
  steps.forEach((st, i) => {
    const x = x0 + i * (w + gap);
    node(s, x, 1.24, w, 0.52, st[0], st[4], { fs: 12, round: true });
    node(s, x, 1.8, w, 1.5, "", SOFT, { border: BORDER, round: true });
    s.addText([
      { text: st[1] + "\n", options: { fontSize: 11.5, color: INK, breakLine: true, paraSpaceAfter: 3 } },
      { text: st[2] + "\n", options: { fontSize: 11.5, color: INK, bold: true, breakLine: true, paraSpaceAfter: 8 } },
      { text: st[3], options: { fontSize: 9.5, color: MUTED } },
    ], { x: x + 0.08, y: 1.86, w: w - 0.16, h: 1.38, fontFace: ZH,
      valign: "top", align: "center", lineSpacingMultiple: 1.1 });
    if (i < 3) arrow(s, x + w + 0.02, 1.5, gap - 0.04, 0);
  });
  callout(s, "LLM 只出现在第二步。第一步用确定性统计替代它的语感，第三、四步用确定性搜索替代人的判断。",
    0.6, 3.52, 8.85, 0.7, ACCENT);
  bullets(s, [
    { t: "这样做的结果：LLM 的建议好坏变成一个数字，而不是一段需要人读的解释。", gap: 6 },
    { t: "同时因为第三、四步是确定性的，它们不消耗 API 调用，成本是可控的。", color: MUTED },
  ], { y: 4.32, w: 8.85, h: 0.85, fontSize: 12 });
  pageNum(s);
  endSlide();
}

// ================= 4 第一步：输入设计 =================
{
  const s = beginSlide();
  head(s, "第一步：把数据变成 LLM 能读的输入", "做法");
  bullets(s, [
    { t: "117 万个点不可能进上下文。", big: true, color: WARN, gap: 8 },
    { t: "所以只给两样东西：", gap: 5 },
    { t: "句柄字符串，例如 246#0@v3", level: 1, gap: 4 },
    { t: "一张诊断卡，只含标量统计", level: 1, gap: 9 },
    { t: "诊断卡约 400 到 600 字节。11386 条全量也只有几 MB。", color: MUTED, gap: 9 },
    { t: "每次清洗或压缩产生新句柄并记录血缘，过程可完整重放。", color: ACCENT },
  ], { y: 1.28, w: 5.05, h: 3.6, fontSize: 13 });

  s.addShape(S.rect, { x: 5.85, y: 1.26, w: 3.6, h: 3.72, fill: { color: "1B2B3A" } });
  s.addText([
    { text: "// 诊断卡（无任何坐标）\n", options: { color: "6A9955", breakLine: true } },
    { text: "seg_id:     246#0@v3\n", options: { color: "9CDCFE", breakLine: true } },
    { text: "n_points:   106\n", options: { color: "9CDCFE", breakLine: true } },
    { text: "regime:     moving\n", options: { color: "CE9178", breakLine: true } },
    { text: "timeline:   ok\n", options: { color: "CE9178", breakLine: true } },
    { text: "dt_median:  10.0\n", options: { color: "9CDCFE", breakLine: true } },
    { text: "dt_p95:     20.0\n", options: { color: "9CDCFE", breakLine: true } },
    { text: "dup_ratio:  0.019\n", options: { color: "9CDCFE", breakLine: true } },
    { text: "speed_p99:  100.77\n", options: { color: "9CDCFE", breakLine: true } },
    { text: "anomalies:  {\n", options: { color: "9CDCFE", breakLine: true } },
    { text: "  dt_artifact: 4,\n", options: { color: "9CDCFE", breakLine: true } },
    { text: "  space_jump:  9 }\n", options: { color: "9CDCFE", breakLine: true } },
    { text: "// 没有 coords 字段", options: { color: "6A9955" } },
  ], { x: 6.0, y: 1.4, w: 3.3, h: 3.45, fontFace: MONO, fontSize: 9.5,
    valign: "top", lineSpacingMultiple: 1.15 });
  caption(s, "诊断卡里最有用的是 regime 和时间轴质量，它们直接决定该用哪套参数");
  pageNum(s);
  endSlide();
}

// ================= 5 为什么要先分类型（原生柱状图） =================
{
  const s = beginSlide();
  head(s, "为什么要先分类型", "关键前提");
  // 原生条形图
  const data = [["静止轨迹", 47, "6C7A89"], ["混合", 24, "E8A33D"], ["行驶轨迹", 29, "2B7BBA"]];
  const x0 = 0.75, y0 = 1.32, bw = 0.62, bh = 2.55, gapx = 1.0;
  s.addShape(S.line, { x: x0 - 0.12, y: y0 + bh, w: 3.6, h: 0,
    line: { color: BORDER, width: 1.2 } });
  data.forEach((d, i) => {
    const h = bh * d[1] / 70;
    const x = x0 + i * (bw + gapx);
    node(s, x, y0 + bh - h, bw, h, "", d[2]);
    s.addText(d[1] + "%", { x, y: y0 + bh - h - 0.34, w: bw, h: 0.3,
      fontFace: EN, fontSize: 14, bold: true, color: d[2], align: "center" });
    s.addText(d[0], { x: x - 0.18, y: y0 + bh + 0.06, w: bw + 0.36, h: 0.3,
      fontFace: ZH, fontSize: 11.5, color: INK, align: "center" });
  });
  s.addText("固定随机 200 条样本（seed=20260920）", { x: 0.6, y: 4.32, w: 3.9, h: 0.3,
    fontFace: ZH, fontSize: 11, color: MUTED, align: "center" });

  bullets(s, [
    { t: "三类轨迹的运动尺度和时间轴问题不同", big: true, color: WARN, gap: 8 },
    { t: "固定样本计数：静止 94、混合 48、行驶 58。", gap: 7 },
    { t: "该分布只描述这 200 条样本，不能写成全量 11386 辆的比例。", color: MUTED, gap: 9 },
    { t: "参数实验按 regime 与 timeline_quality 分层，避免某一类样本主导结论。", bold: true, color: ACCENT, gap: 9 },
    { t: "诊断卡先给轨迹类型，模型据此选择参数", color: MUTED },
  ], { x: 5.35, y: 1.3, w: 4.1, h: 3.6, fontSize: 12 });
  caption(s, "占比来自固定随机样本，不代表全量数据分布");
  pageNum(s);
  endSlide();
}

// ================= 6 第二步：工具箱 =================
{
  const s = beginSlide();
  head(s, "第二步：给 LLM 的工具箱", "做法");
  table(s, [
    hdrRow(["类别", "工具", "作用"]),
    ...[
      ["查数据", "profile · detect_anomalies · evaluate · compare_handles", "返回标量统计，不含坐标"],
      ["改数据", "split · clean · simplify · apply_road_constraint", "产生新句柄"],
      ["做实验", "run_search · find_knee · suggest_param_range", "扫描参数、取拐点"],
      ["查经验", "query_memory · query_playbook", "相似案例 + 人类笔记"],
      ["出图", "render", "叠加图 / 异常分布 / 热力图"],
      ["刻意不给", "write_memory · write_playbook", "LLM 无权写记忆"],
    ].map((r) => r.map((c, i) => ({
      text: c,
      options: { fontSize: 11, color: r[0] === "刻意不给" && i < 2 ? WARN : INK,
        fill: { color: i === 0 ? SOFT : PAPER } } }))),
  ], { x: 0.6, y: 1.28, w: 8.85, colW: [1.45, 4.4, 3.0], rowH: 0.47 });

  callout(s, "为什么故意不给写记忆的工具：确定性代码可把所有案例写入 L1 留档；只有 verifier 准入的案例会被检索并蒸馏到 L2。",
    0.6, 4.3, 8.85, 0.78, WARN);
  caption(s, "所有工具都是确定性函数，没有随机性、不联网、不调用 LLM", 5.14);
  pageNum(s);
  endSlide();
}

// ================= 7 第三步：参数怎么定 =================
{
  const s = beginSlide();
  head(s, "第三步：参数怎么定", "做法");
  // 原生三段
  const segs = [
    ["① 物理先验定区间", "9 个参数各有合法区间\n由数据分布或物理量反推", "确定性代码", ACCENT],
    ["② LLM 独立提候选与方向", "不作为搜索起点\n参数值 + 预期效果\n强制结构化输出", "LLM", GOLD],
    ["③ 独立确定性搜索", "坐标下降 + 拐点\n产出内部参考", "确定性代码", OK],
  ];
  const w = 2.72, gap = 0.34, x0 = 0.6;
  segs.forEach((sg, i) => {
    const x = x0 + i * (w + gap);
    node(s, x, 1.26, w, 0.5, sg[0], sg[3], { fs: 12, round: true });
    node(s, x, 1.8, w, 0.95, sg[1], SOFT, { border: BORDER, tc: INK, fs: 10.5, bold: false, round: true });
    s.addText(sg[2], { x, y: 2.78, w, h: 0.28, fontFace: ZH, fontSize: 10.5,
      bold: true, color: sg[3], align: "center" });
    if (i < 2) arrow(s, x + w + 0.04, 1.51, gap - 0.08, 0);
  });

  bullets(s, [
    { t: "关键：LLM 给独立候选与方向，不作为确定性搜索的起点。", bold: true, color: ACCENT, gap: 7 },
    { t: "dp_tolerance 区间 [0.5, 30] 米，上限约等于 GPS 定位精度。", gap: 5 },
    { t: "dt_threshold 区间 [15, 180] 秒，下界保住正常的 20 秒采样。", gap: 5 },
    { t: "dist_threshold 由 Δt × 限速 × 安全系数推导，不由位移分布推导。", gap: 5 },
  ], { y: 3.2, w: 8.85, h: 1.5, fontSize: 11.5 });
  callout(s, "「预期效果」这一栏是白送的评分抓手。LLM 必须预测每个参数的升降方向，核验时只需比对符号，就能零成本算出方向准确率。",
    0.6, 4.5, 8.85, 0.62, OK);
  pageNum(s);
  endSlide();
}

// ================= 8 第四步：怎么判分 =================
{
  const s = beginSlide();
  head(s, "第四步：怎么判断 LLM 提得对不对", "核心");
  bullets(s, [
    { t: "regret = 搜索最优分 − LLM 提议分", big: true, bold: true, color: ACCENT, gap: 9 },
    { t: "它需要统一内部参考，因此从默认参数独立运行确定性搜索作为标尺。", gap: 9 },
    { t: "实测一条：提议 0.6008、基线 0.6330、最优 0.6366，得 regret 0.0358。", gap: 9 },
    { t: "另外两个零成本判据：", bold: true, gap: 5 },
    { t: "方向准确率：预测的升降符号与实测是否一致", level: 1, gap: 4 },
    { t: "约束满足率：提议是否落在物理先验区间内", level: 1 },
  ], { y: 1.28, w: 5.1, h: 3.7, fontSize: 12.5 });

  table(s, [
    hdrRow(["口径", "触发条件", "怎么读"]),
    ...[
      ["归一化", "最优与基线差距充足", "丢掉了多少比例的可得收益"],
      ["绝对", "最优与基线差距过小", "分母趋零会放大微小差距"],
      ["不适用", "轨迹过短（< 200 米）", "压缩率在该尺度无物理含义"],
    ].map((r) => r.map((c, i) => ({
      text: c, options: { fontSize: 10.5, color: INK,
        fill: { color: i === 0 ? SOFT : PAPER } } }))),
  ], { x: 5.85, y: 1.3, w: 3.6, colW: [0.85, 1.5, 1.25], rowH: 0.62 });
  callout(s, "三种口径必须显式标注。混用会让结论完全反过来：headroom 只有 0.021 时，一个与基线持平的提议会被算成 regret 1.0。",
    0.6, 4.42, 8.85, 0.72, WARN);
  caption(s, "这就是「用定量指标核验建议」的具体落地", 5.16);
  pageNum(s);
  endSlide();
}

// ================= 9 记忆怎么用 =================
{
  const s = beginSlide();
  head(s, "记忆怎么用：只复用已准入案例", "做法");
  table(s, [
    hdrRow(["层", "存什么", "谁写", "怎么用"]),
    ...[
      ["工作记忆", "当前轨迹的诊断卡与已试参数", "agent", "单条轨迹内"],
      ["情景记忆", "全量（诊断 → 参数 → 实测指标）", "核验器", "积累原始经验"],
      ["程序记忆", "诊断签名 → 参数区间", "核验器", "12 维特征 kNN 检索"],
      ["人类知识", "结论 + 证据 + 反例", "人", "只读，给 LLM 因果解释"],
    ].map((r) => r.map((c, i) => ({
      text: c, options: { fontSize: 10.5, color: INK,
        fill: { color: i === 0 ? SOFT : PAPER } } }))),
  ], { x: 0.6, y: 1.28, w: 8.85, colW: [1.35, 3.6, 1.05, 2.85], rowH: 0.55 });

  bullets(s, [
    { t: "L1 保存全部案例；默认检索与 L2 蒸馏只用 verifier 已准入案例。", color: MUTED, gap: 5 },
    { t: "L2 给参数区间，L3 给人写的物理解释；agent 对 L3 只读。", gap: 5 },
    { t: "已准入经验可供其余轨迹查区间，从而减少 LLM 调用。", color: ACCENT, bold: true },
  ], { y: 3.76, w: 8.85, h: 1.25, fontSize: 9.8 });
  pageNum(s);
  endSlide();
}

// ================= 10 消融实验怎么设计 =================
{
  const s = beginSlide();
  head(s, "消融实验：怎么证明这套流程有效", "验证方法");
  const items = [
    { t: "四种模式必须结构性不同", c: ACCENT, d: [
      "llm-only：只有 LLM，没有实测依据",
      "search-only：只有确定性搜索，不含任何 LLM 调用",
      "llm+search：加上实测依据",
      "llm+memory+search：再加上历史经验",
    ]},
    { t: "两个必须避开的陷阱", c: WARN, d: [
      "信息泄漏：在评测轨迹上边跑边攒记忆，agent 会把这条轨迹自己的上次结果检索回来。等于考试时把答案摆桌上。",
      "标签造假：search-only 若仍调用 LLM，与含 LLM 的模式就不可比。所以它必须真的不调用，并有测试断言轮次为 0。",
    ]},
  ];
  let y = 1.26;
  items.forEach((it) => {
    const h = it.t.startsWith("四种") ? 1.55 : 1.85;
    s.addShape(S.rect, { x: 0.6, y, w: 0.055, h, fill: { color: it.c } });
    s.addText(it.t, { x: 0.85, y, w: 8.5, h: 0.32, fontFace: ZH, fontSize: 14.5,
      bold: true, color: it.c });
    s.addText(it.d.map((d) => ({ text: d, options: {
      bullet: { characterCode: "25CF" }, breakLine: true, paraSpaceAfter: 5 } })),
      { x: 0.85, y: y + 0.34, w: 8.5, h: h - 0.4, fontFace: ZH, fontSize: 11.5,
        color: INK, valign: "top", lineSpacingMultiple: 1.16 });
    y += h + 0.22;
  });
  callout(s, "做法：先在示范集上积累记忆，再在完全不重叠的留出集上评测。评测阶段只读不写。",
    0.6, 4.56, 8.85, 0.58, OK);
  pageNum(s);
  endSlide();
}

// ================= 11 ECNU 真实结果（原生柱状图） =================
{
  const s = beginSlide();
  head(s, "ECNU 三轮结果：尚未检出稳定增益", "真实模型");
  // 原生柱状图：三轮 holdout 的超基线率
  const modes = ["LLM", "Search", "LLM+S", "LLM+M+S"];
  const beat = [58.3, 37.5, 54.2, 58.3];
  const x0 = 0.75, y0 = 1.4, bw = 0.5, bh = 2.15, gapx = 0.62;
  s.addShape(S.line, { x: x0 - 0.1, y: y0 + bh, w: 4.15, h: 0,
    line: { color: BORDER, width: 1.2 } });
  beat.forEach((v, i) => {
    const h = bh * v / 60;
    const x = x0 + i * (bw + gapx);
    node(s, x, y0 + bh - h, bw, h, "", i === 1 ? "E8A33D" : ACCENT);
    s.addText(v + "%", { x: x - 0.1, y: y0 + bh - h - 0.3, w: bw + 0.2, h: 0.28,
      fontFace: EN, fontSize: 12, bold: true, color: i === 1 ? GOLD : ACCENT,
      align: "center" });
    s.addText(modes[i], { x: x - 0.12, y: y0 + bh + 0.05, w: bw + 0.24, h: 0.6,
      fontFace: ZH, fontSize: 10, color: INK, align: "center", valign: "top" });
  });
  s.addText("超基线率（提议显著优于基线的比例）", { x: 0.6, y: 4.28, w: 4.3, h: 0.28,
    fontFace: ZH, fontSize: 10.5, color: MUTED, align: "center" });

  bullets(s, [
    { t: "每模式 24 个适用样本，重复 3 轮。", big: true, bold: true, color: ACCENT, gap: 8 },
    { t: "Memory − 无 Memory：", bold: true, gap: 4 },
    { t: "平均 +0.001236，95% CI [-0.000851, 0.003498]。", level: 1, color: MUTED, gap: 4 },
    { t: "胜 / 平 / 负为 16 / 0 / 8，但区间覆盖 0。", level: 1, color: MUTED, gap: 7 },
    { t: "L2 每个参数区域只有 1–2 个样本。", color: WARN, gap: 5 },
    { t: "结论：当前尚未检出稳定 Memory 增益。", color: OK, bold: true },
  ], { x: 5.35, y: 1.32, w: 4.1, h: 3.5, fontSize: 11.5 });
  pageNum(s);
  endSlide();
}

// ================= 12 六个坑 =================
{
  const s = beginSlide();
  head(s, "六个被数据推翻的假设", "实践提醒");
  table(s, [
    hdrRow(["问题", "实测症状", "修法"], WARN),
    ...[
      ["距离用墨卡托", "31°N 系统性放大 17%，容差预算凭空偏 17%", "换局部等距投影"],
      ["Hausdorff 口径错", "顶点集算法给 548 米，真实值 4.77 米", "改点到折线"],
      ["DP 偏差估计错", "容差 2 米时报出 8349 米", "从递归结构取精确界"],
      ["在噪声上算航向", "单条静止轨迹假掉头 37 个", "加 10 米噪声地板"],
      ["漂移用绝对偏移", "误判 50% 的正常行驶点", "改无量纲曲率"],
      ["毛刺用位移法识别", "一处毛刺污染两个位置，真毛刺抓不到", "改相对插值位置偏差"],
    ].map((r) => r.map((c, i) => ({
      text: c, options: { fontSize: 10.5, color: INK,
        fill: { color: i === 0 ? SOFT : PAPER } } }))),
  ], { x: 0.6, y: 1.28, w: 8.85, colW: [2.3, 4.5, 2.05], rowH: 0.47 });
  callout(s, "这些坑的共同点：代码都不会报错，只会让结论错。所以每条都写了回归测试。",
    0.6, 4.62, 8.85, 0.6, WARN);
  pageNum(s);
  endSlide();
}

// ================= 13 现在能跑到什么程度 =================
{
  const s = beginSlide();
  head(s, "现在能跑到什么程度", "现状");
  s.addText("可以直接用", { x: 0.6, y: 1.24, w: 4.25, h: 0.32,
    fontFace: ZH, fontSize: 13.5, bold: true, color: OK });
  bullets(s, [
    "离线模式可完整跑通全流程",
    "ECNU 四模式真实实验已重复 3 轮",
    "180 个 case 无失败；search-only 零次 LLM",
    "七类真实结果图与 case 级证据已保存",
  ].map((t) => ({ t, gap: 9 })), { x: 0.6, y: 1.6, w: 4.25, h: 2.1, fontSize: 12.5 });
  s.addText("还没验证", { x: 5.2, y: 1.24, w: 4.3, h: 0.32,
    fontFace: ZH, fontSize: 13.5, bold: true, color: WARN });
  bullets(s, [
    "稳定的 LLM 或 Memory 增益",
    "独立 OSM 或人工标注质量",
    "100/200 条 Memory 扩展（成本较高）",
    "全量 11386 条真实模型验证",
  ].map((t) => ({ t, gap: 9 })), { x: 5.2, y: 1.6, w: 4.25, h: 2.1, fontSize: 12.5 });
  callout(s, "真实运行：167 次 LLM 请求，529980 + 37761 tokens，case 时间合计 1416.83 秒，失败与重试均为 0。",
    0.6, 3.94, 8.85, 0.78, WARN);
  pageNum(s);
  endSlide();
}

// ================= 14 怎么接着做 =================
{
  const s = beginSlide();
  head(s, "怎么接着做", "下一步");
  const steps = [
    ["1", "先比较 24 与 48 条分层 demo", "优先增加 moving 证据，再看配对区间是否收窄。预计分别需要 88 和 176 次调用。", WARN],
    ["2", "加入小规模独立质量核验", "对 moving 子集做人工标注或 OSM 中心线距离，只作为 secondary evaluation。", ACCENT],
    ["3", "根据前两步决定是否扩到 100/200", "三轮 demo 预计需要 367/733 次调用，先看收益再承担成本。", OK],
  ];
  steps.forEach((st, i) => {
    const y = 1.3 + i * 1.14;
    s.addShape(S.ellipse, { x: 0.6, y: y + 0.12, w: 0.5, h: 0.5, fill: { color: st[3] } });
    s.addText(st[0], { x: 0.6, y: y + 0.12, w: 0.5, h: 0.5, fontFace: EN, fontSize: 15,
      bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(st[1], { x: 1.3, y, w: 8.1, h: 0.42, fontFace: ZH, fontSize: 15,
      bold: true, color: st[3] });
    s.addText(st[2], { x: 1.3, y: y + 0.42, w: 8.1, h: 0.6, fontFace: ZH,
      fontSize: 12.5, color: MUTED, lineSpacingMultiple: 1.16 });
  });
  s.addText("100/200 条候选清单已生成，本轮没有自动执行高成本批量调用。",
    { x: 0.6, y: 4.72, w: 8.85, h: 0.35, fontFace: ZH, fontSize: 12,
      color: ACCENT, italic: true });
  pageNum(s);
  endSlide();
}

const outPath = path.join(OUT, "基于LLM的轨迹数据清洗评估.pptx");
await pptx.writeFile({ fileName: outPath });
console.log("written:", outPath);
console.log("slides:", pageNo);
