"""单向导出：SQLite（L1/L2）→ Obsidian markdown（L3 候选）。

**这是唯一会写文件到 vault 的代码，且它不由 agent 调用。**
运行方式：人工或定时任务执行，产物落在 `00-Inbox/`，
经人工 review 后才升格为正式笔记。

为什么不让 agent 直接写 vault：
    vault 是人类知识的唯一真源。若 agent 能写，
    它的任何 bug 都会污染知识库，而且污染会被后续检索放大。
    分离之后，机器积累在 SQLite（可重建），人类知识在 vault（受保护）。
"""
from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .store import MemoryStore, ParamRegion

INBOX_SUBDIR = "00-Inbox"
DEFAULT_MAX_CASES = 20
MAX_SCORE_SLUG = 60


@dataclass
class ExportResult:
    written: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    target_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"written": len(self.written), "skipped": len(self.skipped),
                "target_dir": self.target_dir, "files": self.written}


def _slug(text: str) -> str:
    """把任意文本变成安全的文件名片段。"""
    s = re.sub(r"[^\w\u4e00-\u9fff\-]+", "-", str(text)).strip("-")
    return (s[:MAX_SCORE_SLUG] or "case").lower()


def _frontmatter(data: Dict[str, Any]) -> str:
    lines = ["---"]
    for k, v in data.items():
        if isinstance(v, (list, tuple)):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def export_param_region_note(region: ParamRegion, store: Optional[MemoryStore] = None,
                             vault_dir: Optional[str] = None,
                             cases: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """把一个 L2 参数区间导出为候选笔记内容（不写盘）。"""
    evidence_links = []
    for c in (cases or [])[:8]:
        evidence_links.append(f"- [[case-{_slug(c.get('seg_id', 'unknown'))}]] "
                              f"得分 {c.get('score')}，regret {c.get('regret')}")
    body = [
        f"# {region.param_name} 的推荐区间（{region.regime} / {region.timeline_quality}）",
        "",
        "## 结论",
        f"在 {region.regime} 且时间轴质量为 {region.timeline_quality} 的轨迹上，",
        f"`{region.param_name}` 的已验证区间是 **{region.low:.4g} – {region.high:.4g}**",
        f"（中位 {region.median:.4g}，基于 {region.n_samples} 条通过核验的案例，",
        f"平均 regret {region.mean_regret}）。",
        "",
        "> 区间取 IQR 而非 min/max，避免个别极端值把推荐范围撑得过宽。",
        "",
        "## 证据",
    ]
    body.extend(evidence_links or ["- （本次导出未附带案例列表）"])
    body += [
        "",
        "## 待人工确认",
        "- [ ] 上述区间是否有物理上的合理解释？",
        "- [ ] 是否需要补充反例（该区间之外的失败案例）？",
        "- [ ] 确认后请把本文件从 `00-Inbox/` 移到正式目录，并补上 `## 反直觉的地方`。",
        "",
        "*本文件由 memory/export.py 自动生成，属于候选知识，未经人工确认。*",
    ]
    fm = {
        "type": "traj-playbook",
        "param": region.param_name,
        "scope": f"{region.regime} / {region.timeline_quality}",
        "updated": datetime.date.today().isoformat(),
        "source": "auto-export",
        "needs_review": "true",
    }
    return {
        "frontmatter": fm,
        "body": "\n".join(body),
        "filename": f"{_slug('region-' + region.regime + '-' + region.param_name)}.md",
    }


def export_note(note: Dict[str, Any], vault_dir: str,
                force: bool = False) -> str:
    """把候选笔记写入 vault 的 00-Inbox/。返回路径。"""
    target_dir = os.path.join(vault_dir, INBOX_SUBDIR)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, note["filename"])
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"{path} 已存在；加 force=True 覆盖，"
            "或先人工处理已有候选笔记（避免覆盖你正在编辑的内容）")
    content = _frontmatter(note["frontmatter"]) + "\n\n" + note["body"] + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def export_regions(store: MemoryStore,
                   vault_dir: Optional[str] = None,
                   min_samples: int = 3,
                   max_notes: int = DEFAULT_MAX_CASES,
                   force: bool = False) -> ExportResult:
    """批量导出 L2 参数区间为候选笔记。

    vault_dir 未提供或不存在时**只返回内容、不写盘**，
    并把这些项记入 skipped —— 这样在没有 vault 的环境里也能预览导出内容。
    """
    regions = [r for r in store.query_regions() if r.n_samples >= min_samples]
    res = ExportResult(target_dir=os.path.join(vault_dir or "", INBOX_SUBDIR))
    recent = store.recent(limit=100, admitted_only=True)
    for region in regions[:max_notes]:
        note = export_param_region_note(region, store, vault_dir, cases=recent)
        if not vault_dir or not os.path.isdir(vault_dir):
            res.skipped.append(f"{note['filename']}（vault 不存在，未写盘）")
            continue
        try:
            path = export_note(note, vault_dir, force=force)
            res.written.append(path)
        except FileExistsError as exc:
            res.skipped.append(str(exc))
    return res


def preview_export(store: MemoryStore, min_samples: int = 3) -> List[Dict[str, Any]]:
    """预览将要导出的笔记内容（不写盘），便于课堂展示与人工审核。"""
    out = []
    for r in store.query_regions():
        if r.n_samples < min_samples:
            continue
        out.append(export_param_region_note(r, store))
    return out
