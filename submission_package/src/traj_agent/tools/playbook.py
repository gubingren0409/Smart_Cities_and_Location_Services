"""Obsidian vault 只读挂载：人类知识层（L3）。

**硬规则：agent 对 vault 只有读权限，永远不能写。**
这条规则的价值在于：智能体的任何 bug 都不可能损坏你的第二大脑，
而它仍然能积累经验——积累在 SQLite（L1/L2），
只有经人工确认的知识才由 `memory/export.py` 单向导出为 markdown。

Obsidian 的 vault 就是一堆 .md 文件，Python 用标准库即可读写，
不需要 Obsidian 在运行，也不需要任何插件或 API。

笔记格式（frontmatter + 正文）：

    ---
    type: traj-playbook
    param: dp_tolerance
    scope: 上海城区 / 静止漂移段
    updated: 2026-01-15
    ---

    # DP 容差取值

    ## 结论
    GPS CEP 约 8 m 时，dp_tolerance 取 6-10 m 收益最高。

    ## 证据
    - 见 [[case-0007-dp-sweep]]：knee point 在 8 m

    ## 反直觉的地方
    重复点比例高时，先折叠重复点再压缩更好。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import yaml
except ImportError:      # pragma: no cover
    yaml = None

DEFAULT_VAULT_ENV = "DSH_TRAJ_VAULT_DIR"
PLAYBOOK_TYPE = "traj-playbook"
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class PlaybookNote:
    """一条人类知识笔记。"""

    path: str
    title: str = ""
    frontmatter: Dict[str, object] = field(default_factory=dict)
    body: str = ""
    wikilinks: List[str] = field(default_factory=list)

    @property
    def param(self) -> Optional[str]:
        v = self.frontmatter.get("param")
        return str(v) if v else None

    @property
    def scope(self) -> Optional[str]:
        v = self.frontmatter.get("scope")
        return str(v) if v else None

    @property
    def note_type(self) -> Optional[str]:
        v = self.frontmatter.get("type")
        return str(v) if v else None

    def to_dict(self, max_body: int = 1200) -> Dict[str, object]:
        body = self.body.strip()
        return {
            "path": self.path,
            "title": self.title,
            "param": self.param,
            "scope": self.scope,
            "updated": self.frontmatter.get("updated"),
            "wikilinks": list(self.wikilinks),
            "body": body[:max_body],
            "truncated": len(body) > max_body,
        }


def _jsonify(value):
    """把 frontmatter 里的值转成可 JSON 序列化的形式。

    yaml.safe_load 会把 `updated: 2026-01-15` 解析成 datetime.date，
    而日期不能直接 json.dumps——工具返回一律要能过 JSON，
    故统一转 ISO 字符串。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    return str(value)


def _parse_frontmatter(text: str) -> tuple:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    body = text[m.end():]
    if yaml is not None:
        try:
            data = yaml.safe_load(raw) or {}
            if isinstance(data, dict):
                return {str(k): _jsonify(v) for k, v in data.items()}, body
        except Exception:
            pass
    # 无 yaml 或解析失败时的极简解析（key: value 逐行）
    data: Dict[str, object] = {}
    for line in raw.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            data[k.strip()] = v.strip()
    return data, body


def resolve_vault_dir(vault_dir: Optional[str] = None) -> Optional[Path]:
    """解析 vault 目录。优先显式参数，其次环境变量，最后默认写作区。"""
    cand = vault_dir or os.environ.get(DEFAULT_VAULT_ENV)
    if cand:
        p = Path(cand).expanduser()
        return p if p.exists() else None
    # 默认写作区：工作目录下的 vault_stub
    p = Path.cwd() / "vault_stub"
    return p if p.exists() else None


def load_notes(vault_dir: Optional[str] = None,
               note_type: Optional[str] = PLAYBOOK_TYPE,
               param: Optional[str] = None) -> List[PlaybookNote]:
    """遍历 vault 读取笔记。vault 不存在时返回空列表（不抛异常）。

    刻意不抛异常：无 vault 时整条流水线仍应可运行，
    只是 L3 知识层为空。
    """
    root = resolve_vault_dir(vault_dir)
    if root is None:
        return []
    notes: List[PlaybookNote] = []
    for path in sorted(root.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        fm, body = _parse_frontmatter(text)
        if note_type is not None and str(fm.get("type", "")) != note_type:
            continue
        if param is not None and str(fm.get("param", "")) != param:
            continue
        title = ""
        for line in body.splitlines():
            if line.strip().startswith("#"):
                title = line.lstrip("#").strip()
                break
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        notes.append(PlaybookNote(
            path=rel,
            title=title,
            frontmatter=fm,
            body=body,
            wikilinks=WIKILINK_RE.findall(body),
        ))
    return notes


def notes_for_params(params: Sequence[str],
                     vault_dir: Optional[str] = None) -> Dict[str, List[PlaybookNote]]:
    """按参数名取相关笔记，用于注入 LLM 上下文。"""
    out: Dict[str, List[PlaybookNote]] = {}
    for p in params:
        out[p] = load_notes(vault_dir, param=p)
    return out


def playbook_digest(vault_dir: Optional[str] = None,
                    params: Optional[Sequence[str]] = None,
                    max_chars_per_note: int = 700) -> Dict[str, object]:
    """给 LLM 的精简知识摘要：结论与反例，不含实现细节。"""
    notes = load_notes(vault_dir)
    if params:
        wanted = set(params)
        notes = [n for n in notes if n.param in wanted]
    return {
        "available": bool(notes),
        "n_notes": len(notes),
        "notes": [n.to_dict(max_body=max_chars_per_note) for n in notes],
    }


def vault_status(vault_dir: Optional[str] = None) -> Dict[str, object]:
    """vault 挂载状态，供诊断与报错信息使用。"""
    explicit = vault_dir
    env = os.environ.get(DEFAULT_VAULT_ENV)
    root = resolve_vault_dir(vault_dir)
    return {
        "vault_dir": str(root) if root else None,
        "source": ("argument" if explicit else
                   ("env:" + DEFAULT_VAULT_ENV if env else "default(vault_stub)")),
        "mounted": root is not None,
        "read_only": True,
        "n_playbook_notes": len(load_notes(root)) if root else 0,
    }
