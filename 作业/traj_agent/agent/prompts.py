"""提示词：把诊断卡、物理先验、人类知识、记忆先验组织成 LLM 的输入。

设计约束
-------
1. 提示词里**不出现任何坐标**。坐标从不离开进程内存。
2. 物理先验与人类知识（Obsidian 笔记）一起给出，前者给数字锚点，
   后者给因果解释。LLM 的建议要同时被这两者约束。
3. 强制结构化输出。自由文本没法自动判分，
   而「可判分」正是本工作流相对「让 LLM 谈谈看法」的全部价值。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

SYSTEM_PROMPT = """你是轨迹清洗的参数决策助手。你决定用什么工具、取什么参数。
数值计算由确定性代码执行，不归你管。

## 硬规则
1. **先给答案，再按需查证**。直接给出你的参数提议，然后最多用 3 次工具调用核对。
2. 不要反复调用同一个工具。查过的数据不要再查。
3. 参数必须落在下方给出的物理先验区间内。
4. 距离单位是米，时间单位是秒。
5. 你**没有**写记忆的权限。

## 按情形调整策略
- 时间轴不可用（时间跨度过小）：速度类规则全部失效，改用纯几何规则。
- 静止轨迹：重复点占比高属正常，不要做漂移与转向角判定。
- 相邻点时间间隔呈双峰：不要套用「中位数×3」这类全局规则。
- 掉头、持续超速、绕行是真实行为，只标记不删除。

## 输出格式
结束工具调用后，**只输出一个 JSON 对象**，不要任何解释文字：

```json
{
  "params": {"dp_tolerance": 8.0, "dt_threshold": 30.0, "max_speed_mps": 38.0},
  "expected_effect": {"quality": "down", "compression": "up"},
  "rationale": "一句话说明依据，引用实测数字"
}
```

`expected_effect` 的取值只能是 "up" / "down" / "same"。
它会与实测符号逐项比对，用于计算方向准确率。
可用键：quality（保真度）、compression（压缩率）、以及你提议的每个参数名。
"""


def diagnosis_block(card_dict: Dict[str, Any]) -> str:
    """渲染诊断卡。"""
    return ("## 本条轨迹的诊断卡\n"
            "（只含标量统计，不含坐标。坐标不会进入你的上下文。）\n"
            "```json\n" + json.dumps(card_dict, ensure_ascii=False, indent=1, sort_keys=True)
            + "\n```")


def param_bounds_block(specs: Sequence[Dict[str, Any]]) -> str:
    """渲染参数物理先验区间。"""
    lines = ["## 参数的物理先验区间（提议必须落在区间内）", ""]
    lines.append("| 参数 | 下界 | 上界 | 默认 | 单位 | 依据 |")
    lines.append("|---|---|---|---|---|---|")
    for s in specs:
        lines.append(f"| {s['name']} | {s['low']} | {s['high']} | {s['default']} | "
                     f"{s['unit']} | {s['rationale']} |")
    return "\n".join(lines)


def playbook_block(digest: Dict[str, Any]) -> str:
    """渲染人类知识笔记（Obsidian，只读）。"""
    if not digest.get("available"):
        return "## 人类知识笔记\n（未挂载 Obsidian vault，本层知识为空）"
    lines = ["## 人类知识笔记（由人维护，只读）",
             "这些是已验证的因果解释与反例，应作为你判断的依据之一。", ""]
    for n in digest.get("notes", [])[:6]:
        lines.append(f"### {n.get('title') or n.get('path')}  "
                     f"[param={n.get('param')}, scope={n.get('scope')}]")
        lines.append((n.get("body") or "").strip()[:900])
        if n.get("wikilinks"):
            lines.append(f"关联案例：{', '.join(n['wikilinks'][:5])}")
        lines.append("")
    return "\n".join(lines)


def memory_block(prior: Dict[str, Any]) -> str:
    """渲染记忆先验（相似案例 + 推荐参数区间）。"""
    if not prior.get("available"):
        return ("## 历史经验（记忆库）\n"
                f"{prior.get('caution') or '（无可用历史经验，冷启动）'}\n"
                "此时只能依据物理先验与本次实测做决策。")
    lines = ["## 历史经验（记忆库，仅含已通过核验的案例）", ""]
    lines.append(f"检索到你这条轨迹的 {prior.get('n_neighbors', 0)} 个相似历史案例"
                 f"（regime={prior.get('regime')}）：")
    lines.append("")
    lines.append("| 案例 | 相似度 | 参数 | 得分 | 归一化 regret |")
    lines.append("|---|---|---|---|---|")
    for nb in prior.get("neighbors", [])[:6]:
        p = ", ".join(f"{k}={v}" for k, v in list(nb.get("params", {}).items())[:4])
        lines.append(f"| {nb.get('seg_id')} | {nb.get('similarity')} | {p} | "
                     f"{nb.get('score')} | {nb.get('regret')} |")
    if prior.get("suggested_regions"):
        lines.append("")
        lines.append("**已验证的参数推荐区间**：")
        for r in prior["suggested_regions"][:6]:
            lines.append(f"- {r['param']}: {r['low']}–{r['high']} "
                         f"（中位 {r['median']}，{r['n_samples']} 条案例，"
                         f"平均 regret {r['mean_regret']}）")
    if prior.get("caution"):
        lines.append("")
        lines.append(f"**注意**：{prior['caution']}")
    return "\n".join(lines)


def tool_catalog_block(catalog: Sequence[Dict[str, Any]]) -> str:
    """渲染工具目录（用于文本 ReAct 模式）。"""
    lines = ["## 可用工具", ""]
    for t in catalog:
        flag = "（会产生新句柄）" if t.get("side_effect") else "（只读）"
        lines.append(f"- **{t['name']}** {flag}: {t['description']}")
        props = (t.get("parameters") or {}).get("properties", {})
        if props:
            ps = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
            lines.append(f"  参数: {ps}")
    return "\n".join(lines)


def build_system_message(card_dict: Dict[str, Any],
                         specs: Sequence[Dict[str, Any]],
                         playbook: Dict[str, Any],
                         memory: Dict[str, Any],
                         catalog: Optional[Sequence[Dict[str, Any]]] = None,
                         text_protocol: bool = False) -> str:
    """组装完整的系统消息。"""
    parts = [SYSTEM_PROMPT, diagnosis_block(card_dict),
             param_bounds_block(specs), playbook_block(playbook),
             memory_block(memory)]
    if text_protocol and catalog is not None:
        parts.append(tool_catalog_block(catalog))
        parts.append(
            "## 文本协议（本模型不支持原生函数调用）\n"
            "每一步只输出一个 JSON 对象，形如：\n"
            '{"thought": "…", "action": "工具名", "action_input": {…}}\n'
            "需要结束并给出最终提议时，输出：\n"
            '{"thought": "…", "final": {"params": {…}, "expected_effect": {…}, '
            '"rationale": "…"}}')
    return "\n\n".join(parts)


def proposal_from_text(text: str) -> Optional[Dict[str, Any]]:
    """从自由文本里抠出最终提议的 JSON。

    容错策略：先找 ```json 代码块，再退回扫描第一个平衡的 JSON 对象。
    LLM 经常在 JSON 前后加解释性文字，直接 json.loads 会失败。
    """
    if not text:
        return None
    candidates: List[str] = []
    # 1) 代码块
    idx = 0
    while True:
        start = text.find("```", idx)
        if start < 0:
            break
        end = text.find("```", start + 3)
        if end < 0:
            break
        block = text[start + 3:end]
        if block.lstrip().lower().startswith("json"):
            block = block.lstrip()[4:]
        candidates.append(block.strip())
        idx = end + 3
    # 2) 全文与所有平衡花括号片段
    candidates.append(text.strip())
    candidates.extend(_balanced_objects(text))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        payload = obj.get("final") if isinstance(obj.get("final"), dict) else obj
        if isinstance(payload, dict) and "params" in payload:
            return payload
    return None


def _balanced_objects(text: str) -> List[str]:
    """扫描所有花括号平衡的子串（跳过字符串内的括号）。"""
    out: List[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
    return out


def parse_text_action(text: str) -> Optional[Dict[str, Any]]:
    """解析文本 ReAct 协议下的一步动作。"""
    for cand in [text.strip()] + _balanced_objects(text):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "final" in obj:
            return {"final": obj["final"], "thought": obj.get("thought", "")}
        if "action" in obj:
            return {"action": str(obj["action"]),
                    "action_input": obj.get("action_input") or {},
                    "thought": obj.get("thought", "")}
    return None
