import pptxgen from "pptxgenjs";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ASSETS = path.join(HERE, "assets");
const OUT = path.join(HERE, "out");
fs.mkdirSync(OUT, { recursive: true });

const ZH = "Microsoft YaHei", EN = "Arial", MONO = "Consolas";
const INK = "1F2A37", MUTED = "5B6B7C", ACCENT = "2B7BBA";
const WARN = "C0392B", OK = "2E8B57", GOLD = "8A6D1F", PURPLE = "6B4A7A";
const PAPER = "FFFFFF", SOFT = "F4F6F8", LIGHT = "EAF2F8";

const pptx = new pptxgen();
pptx.layout = "LAYOUT_16x9";
pptx.author = "轨迹处理实验课";
pptx.title = "LLM 辅助的轨迹清洗评估智能体";

// 备注按页面标题绑定：页码会随插页漂移，标题是稳定语义键
const NOTES = {
 "LLM 辅助的轨迹清洗评估智能体":
   "开场。这个框架要解决的问题是：让 LLM 参与轨迹清洗的参数决策，但它的建议必须能被自动判分，而不是靠人去读一段解释。整个设计围绕这一条展开。",
 "为什么不能让 LLM 直接清洗":
   "这三个问题必须讲透，否则后面所有设计看起来都像过度工程。猜不准、没法判分、成本不可行。特别强调第二点：没法判分是致命的，因为你无法评价 11386 条轨迹上的建议质量。",
 "五条不可让步的设计原则":
   "这是整个框架的骨架，后面每个模块都是为了兑现其中某一条。建议逐条讲清动机，尤其第二条（LLM 不能写记忆）和第四条（提议与核验分离）。",
 "分层架构": "强调依赖方向是严格的：core 不知道上面任何一层的存在。好处是学生能单独复用算法、敏感性实验不用启动智能体、智能体的 bug 不会污染算法正确性。这个约束由测试强制。",
 "一条轨迹的完整生命周期":
   "核心页。走一遍八步。重点停在第 3 步和第 7 到 8 步之间：那里把 LLM 建议质量变成了一个数字。其余步骤都是为了让这个数字可信而存在的支撑。",
 "核心机制一：regret 让建议可判分":
   "regret 需要 ground truth，而 ground truth 来自确定性搜索。这里有个联动值得指出：任务二的敏感性实验产物就是任务三目标函数的地形图。三种口径必须显式标注，否则会得出错误结论。",
 "核心机制二：句柄与诊断卡":
   "上下文纪律。117 万个点不可能进上下文，所以只传句柄和标量诊断卡。血缘链让 trace 可以完整重放，也让结果可追溯。",
 "核心机制三：四层记忆":
   "重点是 L2 和 L3 的分工。L2 是机器算出来的统计区间，L3 是人写的因果解释。两者不重复：L2 给数字锚点，L3 给物理直觉。硬规则是 agent 对 vault 只读，这样它的 bug 不可能损坏知识库。",
 "核心机制四：三段式调参":
   "回答「参数怎么调」这个核心问题。LLM 不是优化器，它只给起点和方向。expected_effect 是白送的评分抓手，核验器只需比对符号。",
 "工具层：LLM 的全部能力":
   "逐类讲工具。特别强调刻意缺席的 write_memory：这不是遗漏，是设计。有一条测试专门断言工具名里不含 write、save、delete。",
 "消融实验的两个方法学陷阱":
   "这一页是方法论价值所在。信息泄漏和模式标签造假都是很隐蔽的错误，做错了两者都会得出完全错误的结论。我们是用代码强制避免的，不是靠自觉。",
 "真实结果：记忆层没有带来可测增益":
   "必须如实报告。三种含 LLM 模式的得分完全相同。根因已经定位到具体层面：moving 类轨迹在记忆里没有已验证案例。这不是设计失败，是数据饥饿。",
 "六个被数据推翻的假设":
   "这页对做工程的人最有价值。每一条都是真实踩过的坑，而且都配了回归测试。可以挑两三个讲透，比如墨卡托 17% 和 Hausdorff 548 米对 4.77 米。",
 "当前状态与已知短板": "诚实交代边界。已完成的部分和已知短板要分开讲，不要把短板藏起来。",
 "下一步": "三条改进路径按收益排序。第一条最紧，第二条才能让消融表真正有信息量。",
};

let pageNo = 0, current = null, currentTitle = "";

function beginSlide(bg = PAPER) {
  current = pptx.addSlide();
  current.background = { color: bg };
  pageNo += 1;
  return current;
}
function endSlide() {
  const note = NOTES[currentTitle];
  if (note) current.addNotes(note);
  else throw new Error(`第 ${pageNo} 页「${currentTitle}」缺备注`);
  current = null; currentTitle = "";
}
function head(s, title, kicker) {
  currentTitle = title;
  if (kicker) s.addText(kicker, { x: 0.55, y: 0.24, w: 8.9, h: 0.24,
    fontFace: ZH, fontSize: 11, color: ACCENT, bold: true, charSpacing: 1 });
  s.addText(title, { x: 0.55, y: kicker ? 0.46 : 0.34, w: 8.9, h: 0.55,
    fontFace: ZH, fontSize: 24, color: INK, bold: true });
  s.addShape(pptx.ShapeType.rect, { x: 0.55, y: kicker ? 1.03 : 0.93,
    w: 0.9, h: 0.035, fill: { color: ACCENT } });
}
function bullets(s, items, o = {}) {
  const f0 = o.fontSize ?? 14.5;
  s.addText(items.map((it) => ({
    text: it.t,
    options: { bullet: { characterCode: "25CF" },
      fontSize: it.big ? f0 + 2 : f0, color: it.color ?? INK, bold: !!it.bold,
      breakLine: true, paraSpaceAfter: it.gap ?? 8, indentLevel: it.level ?? 0 },
  })), { x: o.x ?? 0.6, y: o.y ?? 1.28, w: o.w ?? 5.0, h: o.h ?? 3.6,
    fontFace: ZH, valign: "top", lineSpacingMultiple: 1.2 });
}
function caption(s, t, y = 5.0) {
  s.addText(t, { x: 0.6, y, w: 8.85, h: 0.3, fontFace: ZH, fontSize: 10.5,
    color: MUTED, italic: true });
}
function callout(s, t, x, y, w, h, color = ACCENT) {
  s.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: SOFT }, line: { color: SOFT } });
  s.addShape(pptx.ShapeType.rect, { x, y, w: 0.055, h, fill: { color } });
  s.addText(t, { x: x + 0.24, y: y + 0.08, w: w - 0.42, h: h - 0.16,
    fontFace: ZH, fontSize: 12.5, color: INK, valign: "middle", lineSpacingMultiple: 1.18 });
}
function pageNum(s, n) {
  s.addText(String(n), { x: 9.05, y: 5.14, w: 0.5, h: 0.3, fontFace: EN,
    fontSize: 11, color: "AAB4BE", align: "right" });
}
function img(s, f, x, y, w, h) {
  s.addImage({ path: path.join(ASSETS, f), x, y, w, h, sizing: { type: "contain", w, h } });
}
function code(s, lines, x, y, w, h) {
  s.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: "1B2B3A" } });
  s.addText(lines.map((l) => ({ text: l, options: { breakLine: true } })),
    { x: x + 0.16, y: y + 0.1, w: w - 0.3, h: h - 0.2, fontFace: MONO,
      fontSize: 9.5, color: "9CDCFE", valign: "top", lineSpacingMultiple: 1.3 });
}

// ===== 1 封面 =====
{
  const s = beginSlide("10202F"); currentTitle = "LLM 辅助的轨迹清洗评估智能体";
  s.addText("任务三 · 架构与设计", { x: 0.8, y: 1.32, w: 8.4, h: 0.4,
    fontFace: ZH, fontSize: 15, color: "7FB4D8", bold: true, charSpacing: 3 });
  s.addText("LLM 辅助的轨迹清洗评估智能体", { x: 0.8, y: 1.8, w: 8.6, h: 1.1,
    fontFace: ZH, fontSize: 34, color: "FFFFFF", bold: true });
  s.addShape(pptx.ShapeType.rect, { x: 0.8, y: 3.0, w: 1.5, h: 0.05, fill: { color: "4FA3D1" } });
  s.addText("让 LLM 选工具、提参数，由确定性代码判定它的建议好不好", {
    x: 0.8, y: 3.24, w: 8.6, h: 0.4, fontFace: ZH, fontSize: 15, color: "B8CEDA" });
  s.addText("293 项测试 · 15 个工具 · 四层记忆 · 留出集消融", {
    x: 0.8, y: 3.72, w: 8.6, h: 0.35, fontFace: ZH, fontSize: 12.5, color: "7A93A3" });
  endSlide();
}

// ===== 2 为什么不能让 LLM 直接清洗 =====
{
  const s = beginSlide();
  head(s, "为什么不能让 LLM 直接清洗", "问题定义");
  const rows = [
    ["猜不准", "LLM 没读过你的数据，dp_tolerance 说 5 还是 15 全靠语感", ACCENT],
    ["没法判分", "「这条建议看着挺合理」不是判据，无法评价 11386 条轨迹上的建议质量", WARN],
    ["成本不可行", "11386 辆车 × 每车多次 API 调用，烧不起", GOLD],
  ];
  rows.forEach((r, i) => {
    const y = 1.28 + i * 1.18;
    s.addShape(pptx.ShapeType.rect, { x: 0.6, y, w: 0.055, h: 1.0, fill: { color: r[2] } });
    s.addText(r[0], { x: 0.85, y, w: 2.1, h: 1.0, fontFace: ZH, fontSize: 17,
      bold: true, color: r[2], valign: "middle" });
    s.addText(r[1], { x: 3.0, y, w: 6.45, h: 1.0, fontFace: ZH, fontSize: 13.5,
      color: INK, valign: "middle", lineSpacingMultiple: 1.2 });
  });
  callout(s, "针对性解法：物理先验定边界、确定性搜索做标尺、分层记忆摊成本。",
    0.6, 4.72, 8.85, 0.68, OK);
  pageNum(s, pageNo);
  endSlide();
}

// ===== 3 五条原则 =====
{
  const s = beginSlide();
  head(s, "五条不可让步的设计原则", "设计约束");
  bullets(s, [
    { t: "LLM 不碰数值，也不当优化器。", big: true, color: ACCENT, gap: 4 },
    { t: "所有几何与统计计算在 core/ 里，是纯函数、可单测、不知道 LLM 存在。", level: 1, color: MUTED, gap: 11 },
    { t: "LLM 不能写记忆。", big: true, color: WARN, gap: 4 },
    { t: "工具清单里故意没有 write_memory，记忆写入只发生在核验确认有效之后。", level: 1, color: MUTED, gap: 11 },
    { t: "坐标永不进上下文。", big: true, color: ACCENT, gap: 4 },
    { t: "只传句柄和标量诊断卡，117 万点留在进程内存里。", level: 1, color: MUTED, gap: 11 },
    { t: "提议与核验分离。", big: true, color: ACCENT, gap: 4 },
    { t: "两个独立步骤，专门制造 generator-verifier gap。", level: 1, color: MUTED, gap: 11 },
    { t: "core/ 不得依赖上层。", big: true, color: ACCENT, gap: 4 },
    { t: "由 tests/test_architecture.py 用 AST 强制检查。", level: 1, color: MUTED },
  ], { y: 1.26, w: 8.85, h: 3.7, fontSize: 13.5 });
  pageNum(s, pageNo);
  endSlide();
}

// ===== 4 分层架构 =====
{
  const s = beginSlide();
  head(s, "分层架构", "结构");
  img(s, "arch_layers.png", 1.05, 1.2, 7.9, 3.9);
  caption(s, "共 10584 行，293 项测试全绿", 5.06);
  pageNum(s, pageNo);
  endSlide();
}

// ===== 5 闭环流程 =====
{
  const s = beginSlide();
  head(s, "一条轨迹的完整生命周期", "执行流程");
  img(s, "arch_loop.png", 1.15, 1.18, 7.7, 3.85);
  caption(s, "第 3 步与第 7 到 8 步之间是全部价值所在", 5.02);
  pageNum(s, pageNo);
  endSlide();
}

// ===== 6 regret =====
{
  const s = beginSlide();
  head(s, "核心机制一：regret 让建议可判分", "机制");
  bullets(s, [
    { t: "regret = score（搜索最优）− score（LLM 提议）", bold: true, color: ACCENT, gap: 9 },
    { t: "它需要 ground truth，而 ground truth 来自确定性搜索。" },
    { t: "联动：任务二的敏感性实验产物，就是任务三目标函数的地形图。", color: GOLD, bold: true, gap: 9 },
    { t: "实测一条：提议 0.6008、基线 0.6330、最优 0.6366，得 regret 0.0358。" },
  ], { y: 1.26, w: 5.35, h: 2.0, fontSize: 13.5 });
  img(s, "regret_bases.png", 0.6, 2.72, 8.85, 2.35);
  pageNum(s, pageNo);
  endSlide();
}

// ===== 7 句柄与诊断卡 =====
{
  const s = beginSlide();
  head(s, "核心机制二：句柄与诊断卡", "机制");
  bullets(s, [
    { t: "117 万点不可能进上下文。", big: true, color: WARN, gap: 9 },
    { t: "LLM 只拿到句柄字符串和一张标量诊断卡。" },
    { t: "诊断卡约 400 到 600 字节，11386 条全量也只有几 MB。", gap: 9 },
    { t: "每次变换产生新句柄并记录血缘，trace 可完整重放。", color: ACCENT },
  ], { y: 1.3, w: 5.1, h: 3.5, fontSize: 13.5 });
  code(s, [
    "TrajHandle(",
    "  vehicle_id='246',",
    "  version=3,          # 每次变换递增",
    "  n_points=106,",
    "  state_hash='a3f...' # 内容指纹",
    ")",
    "",
    "# 诊断卡字段（无任何坐标）",
    "regime: moving",
    "timeline.quality: ok",
    "dt_bimodal: false",
    "dup_ratio: 0.019",
    "speed_p99: 100.77",
    "anomaly_counts: {...}",
  ], 5.85, 1.3, 3.6, 3.5);
  pageNum(s, pageNo);
  endSlide();
}

// ===== 8 四层记忆 =====
{
  const s = beginSlide();
  head(s, "核心机制三：四层记忆", "机制");
  img(s, "memory_tiers.png", 0.9, 1.2, 8.2, 3.7);
  caption(s, "L2 给数字锚点，L3 给物理直觉，两者不重复", 5.02);
  pageNum(s, pageNo);
  endSlide();
}

// ===== 9 三段式调参 =====
{
  const s = beginSlide();
  head(s, "核心机制四：三段式调参", "机制");
  img(s, "param_three_stage.png", 0.95, 1.18, 8.1, 2.9);
  bullets(s, [
    { t: "dp_tolerance 区间 [0.5, 30] 米，上限约等于 GPS 精度 CEP。", gap: 6 },
    { t: "dt_threshold 区间 [15, 180] 秒，下界保住正常的 20 秒采样。", gap: 6 },
    { t: "dist_threshold 由 dt × 限速 × 安全系数推导，不由位移分布推导。", gap: 6 },
  ], { y: 4.08, w: 8.85, h: 1.0, fontSize: 12 });
  pageNum(s, pageNo);
  endSlide();
}

// ===== 10 工具层 =====
{
  const s = beginSlide();
  head(s, "工具层：LLM 的全部能力", "接口");
  s.addTable([
    [{ text: "类别", options: { bold: true, color: "FFFFFF", fill: { color: "2B4A63" } } },
     { text: "工具", options: { bold: true, color: "FFFFFF", fill: { color: "2B4A63" } } },
     { text: "说明", options: { bold: true, color: "FFFFFF", fill: { color: "2B4A63" } } }],
    ...[
      ["只读（9 个）", "load · profile · detect_anomalies · evaluate · compare_handles · find_knee · suggest_param_range · query_memory · query_playbook", "不改变状态"],
      ["产生句柄（6 个）", "split · clean · simplify · apply_road_constraint · run_search · render", "写入句柄存储"],
      ["刻意缺席", "write_memory · write_playbook · 任何 delete_*", "LLM 无权写记忆"],
    ].map((r) => r.map((c, i) => ({
      text: c, options: { fontSize: i === 2 ? 10.5 : 11, color: i === 2 && r[0] === "刻意缺席" ? WARN : INK,
        fill: { color: i === 0 ? SOFT : "FFFFFF" } } }))),
  ], { x: 0.6, y: 1.3, w: 8.85, colW: [1.85, 5.05, 1.95],
    border: { type: "solid", color: "D8DEE4", pt: 0.5 },
    fontFace: ZH, rowH: 0.95, valign: "middle" });
  callout(s, "有一条测试专门断言工具名里不含 write、save、delete。这是设计，不是遗漏。",
    0.6, 4.42, 8.85, 0.7, WARN);
  pageNum(s, pageNo);
  endSlide();
}

// ===== 11 消融的两个陷阱 =====
{
  const s = beginSlide();
  head(s, "消融实验的两个方法学陷阱", "方法论");
  const items = [
    { t: "陷阱一：信息泄漏", c: WARN, d: [
      "在评测轨迹上边跑边攒记忆，agent 会把「这条轨迹自己的上次结果」检索回来当先验。",
      "等于考试时把答案摆在桌上。",
      "三处强制：两阶段评测、read_only 不写回、exclude_self 排除同 seg_id。demo 与 holdout 重叠直接抛错。",
    ]},
    { t: "陷阱二：模式标签造假", c: GOLD, d: [
      "search-only 若仍调用 LLM，与含 LLM 的模式就不可比。",
      "用 use_llm=False 真正跳过，并有测试断言其 LLM 轮次为 0。",
      "regret 口径随模式变化，关闭搜索时是绝对口径，不能与归一化口径直接比较。",
    ]},
  ];
  let y = 1.26;
  items.forEach((it) => {
    s.addShape(pptx.ShapeType.rect, { x: 0.6, y, w: 0.055, h: 1.42, fill: { color: it.c } });
    s.addText(it.t, { x: 0.82, y, w: 8.5, h: 0.32, fontFace: ZH, fontSize: 15,
      bold: true, color: it.c });
    s.addText(it.d.map((d) => ({ text: d, options: { bullet: { characterCode: "25CF" },
      breakLine: true, paraSpaceAfter: 4 } })),
      { x: 0.82, y: y + 0.34, w: 8.5, h: 1.05, fontFace: ZH, fontSize: 11.5,
        color: INK, valign: "top", lineSpacingMultiple: 1.16 });
    y += 1.62;
  });
  caption(s, "两者都会让消融表得出完全错误的结论，而且都是隐蔽的", 4.7);
  pageNum(s, pageNo);
  endSlide();
}

// ===== 12 真实结果 =====
{
  const s = beginSlide();
  head(s, "真实结果：记忆层没有带来可测增益", "实测");
  img(s, "ablation_real.png", 0.5, 1.16, 5.5, 3.7);
  bullets(s, [
    { t: "三种含 LLM 模式的提议分逐位相同。", bold: true, color: WARN, gap: 9 },
    { t: "根因已定位：", bold: true, color: INK, gap: 4 },
    { t: "phase 1 准入的 5 条里，4 条静止、1 条 mixed。", level: 1, color: MUTED, gap: 5 },
    { t: "moving 类轨迹在记忆里没有已验证案例。", level: 1, color: MUTED, gap: 5 },
    { t: "4 条 moving holdout 检索到的区间数是 0。", level: 1, color: MUTED, gap: 9 },
    { t: "瓶颈是 demo 集只有 12 条，不是设计失败。", color: OK, bold: true },
  ], { x: 6.2, y: 1.3, w: 3.25, h: 3.5, fontSize: 12 });
  caption(s, "如实报告：当前配置下不能声称记忆有效", 4.95);
  pageNum(s, pageNo);
  endSlide();
}

// ===== 13 六个被推翻的假设 =====
{
  const s = beginSlide();
  head(s, "六个被数据推翻的假设", "工程发现");
  const rows = [
    ["墨卡托放大距离", "31°N 系统性偏大 17%，容差预算凭空偏 17%", "换局部等距投影"],
    ["Hausdorff 口径错", "顶点集算法给 548 米，真实值 4.77 米", "改点到折线"],
    ["DP 偏差估计错", "tol=2 米时报出 8349 米", "从递归结构取精确界"],
    ["在噪声上算航向", "单条静止轨迹假掉头 37 个", "加 10 米噪声地板"],
    ["漂移用绝对偏移", "误判 50% 的正常行驶点", "改无量纲曲率"],
    ["毛刺用位移法识别", "一处毛刺污染两位，真毛刺抓不到", "改相对插值位置偏差"],
  ];
  s.addTable([
    [{ text: "问题", options: { bold: true, color: "FFFFFF", fill: { color: "C0392B" } } },
     { text: "实测症状", options: { bold: true, color: "FFFFFF", fill: { color: "C0392B" } } },
     { text: "修法", options: { bold: true, color: "FFFFFF", fill: { color: "2E8B57" } } }],
    ...rows.map((r) => r.map((c, i) => ({
      text: c, options: { fontSize: 11, color: INK,
        fill: { color: i === 0 ? SOFT : "FFFFFF" } } }))),
  ], { x: 0.6, y: 1.28, w: 8.85, colW: [2.35, 4.45, 2.05],
    border: { type: "solid", color: "D8DEE4", pt: 0.5 },
    fontFace: ZH, rowH: 0.53, valign: "middle" });
  caption(s, "每条都配了回归测试，详见 DECISIONS.md", 5.02);
  pageNum(s, pageNo);
  endSlide();
}

// ===== 14 状态与短板 =====
{
  const s = beginSlide();
  head(s, "当前状态与已知短板", "边界");
  s.addText("已完成", { x: 0.6, y: 1.24, w: 4.2, h: 0.3,
    fontFace: ZH, fontSize: 13.5, bold: true, color: OK });
  bullets(s, [
    "10584 行代码，293 项测试全绿",
    "15 个工具、四层记忆、六类报告图",
    "离线 Mock provider 可完整跑通",
    "接真实模型只需设一个环境变量",
  ].map((t) => ({ t, gap: 9 })), { x: 0.6, y: 1.58, w: 4.25, h: 2.2, fontSize: 12.5 });

  s.addText("已知短板", { x: 5.2, y: 1.24, w: 4.3, h: 0.3,
    fontFace: ZH, fontSize: 13.5, bold: true, color: WARN });
  bullets(s, [
    "记忆层无可测增益（数据饥饿）",
    "尚未用真实 LLM 跑过消融",
    "路网仅有协议与离线实现",
    "未全量跑 11386 条",
  ].map((t) => ({ t, gap: 9 })), { x: 5.2, y: 1.58, w: 4.25, h: 2.2, fontSize: 12.5 });
  callout(s, "Mock provider 每次给出同一个 knee point，天然抹平模式差异。不接真实模型，消融表就没有信息量。",
    0.6, 3.94, 8.85, 0.78, WARN);
  pageNum(s, pageNo);
  endSlide();
}

// ===== 15 下一步 =====
{
  const s = beginSlide();
  head(s, "下一步", "改进方向");
  const steps = [
    ["1", "扩大 demo 集到 200 条以上", "让 moving 类轨迹积累到足够样本。这是当前最紧的瓶颈。", WARN],
    ["2", "接真实 LLM 重跑消融", "真实模型的提议有方差，才能显出搜索兜底与记忆先验的价值。", ACCENT],
    ["3", "放松准入阈值做敏感性扫描", "观察准入率与记忆增益的权衡曲线，这本身就是个好实验。", OK],
  ];
  steps.forEach((st, i) => {
    const y = 1.3 + i * 1.16;
    s.addShape(pptx.ShapeType.ellipse, { x: 0.6, y: y + 0.14, w: 0.5, h: 0.5,
      fill: { color: st[3] } });
    s.addText(st[0], { x: 0.6, y: y + 0.14, w: 0.5, h: 0.5, fontFace: EN,
      fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(st[1], { x: 1.3, y, w: 8.1, h: 0.42, fontFace: ZH, fontSize: 15,
      bold: true, color: st[3] });
    s.addText(st[2], { x: 1.3, y: y + 0.42, w: 8.1, h: 0.6, fontFace: ZH,
      fontSize: 12.5, color: MUTED, lineSpacingMultiple: 1.18 });
  });
  s.addText("按预期收益排序。第一条不解决，后面两条的结论都不可信。", {
    x: 0.6, y: 4.78, w: 8.85, h: 0.35, fontFace: ZH, fontSize: 12,
    color: ACCENT, italic: true });
  pageNum(s, pageNo);
  endSlide();
}

const outPath = path.join(OUT, "LLM辅助轨迹清洗评估智能体_架构与设计.pptx");
await pptx.writeFile({ fileName: outPath });
console.log("written:", outPath);
console.log("slides:", pageNo);
