"""会话存储：轨迹句柄 → 实际数据。

这是「坐标永不进 LLM 上下文」的执行机制。
LLM 只拿到形如 `246#0@v3` 的句柄字符串；坐标留在进程内存里。

每个 handle 记录血缘（父句柄与操作），使 trace 可以完整重放，
也让「这个结果是怎么来的」在报告里可追溯。
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core.traj import Traj


@dataclass
class HandleRecord:
    """一个句柄的血缘与状态。"""

    handle: str
    traj: Traj
    version: int
    parent: Optional[str] = None
    operation: str = "load"
    params: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.traj)

    def line(self) -> Dict[str, object]:
        return {
            "handle": self.handle,
            "version": self.version,
            "parent": self.parent,
            "operation": self.operation,
            "params": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                       for k, v in sorted(self.params.items())},
            "n_points": self.n_points,
        }


class TrajStore:
    """线程安全的句柄存储。"""

    def __init__(self) -> None:
        self._records: Dict[str, HandleRecord] = {}
        self._lineage: Dict[str, List[str]] = {}     # 谱系根 -> 句柄链
        self._lock = threading.RLock()
        self._counter = 0

    # ---- 写入 -----------------------------------------------------------
    def new_handle(self, vehicle_id: str, segment_index: int = 0,
                   version: int = 0) -> str:
        base = f"{vehicle_id}#{segment_index}"
        return f"{base}@v{version}"

    def put(self, traj: Traj, parent: Optional[str] = None,
            operation: str = "load",
            params: Optional[Dict[str, float]] = None,
            version: Optional[int] = None,
            meta: Optional[Dict[str, object]] = None) -> str:
        with self._lock:
            if version is None:
                if parent and parent in self._records:
                    version = self._records[parent].version + 1
                else:
                    version = 0
            handle = self.new_handle(traj.vehicle_id, traj.segment_index, version)
            # 防止同一句柄被覆盖：同名则递增到唯一
            while handle in self._records:
                version += 1
                handle = self.new_handle(traj.vehicle_id, traj.segment_index, version)
            rec = HandleRecord(handle=handle, traj=traj, version=version,
                               parent=parent, operation=operation,
                               params=dict(params or {}), meta=dict(meta or {}))
            self._records[handle] = rec
            root = self._root_of(parent) if parent else handle
            self._lineage.setdefault(root, []).append(handle)
            if parent is None:
                self._lineage.setdefault(handle, [handle])
            self._counter += 1
            return handle

    def _root_of(self, handle: Optional[str]) -> str:
        seen = set()
        cur = handle
        while cur and cur in self._records and self._records[cur].parent and cur not in seen:
            seen.add(cur)
            cur = self._records[cur].parent
        return cur or (handle or "")

    # ---- 读取 -----------------------------------------------------------
    def get(self, handle: str) -> Traj:
        rec = self.record(handle)
        return rec.traj

    def record(self, handle: str) -> HandleRecord:
        with self._lock:
            if handle not in self._records:
                raise KeyError(
                    f"未知句柄 {handle!r}；可用句柄共 {len(self._records)} 个。"
                    f"请先用 load_trajectory 或其它工具生成句柄。")
            return self._records[handle]

    def try_get(self, handle: str) -> Optional[Traj]:
        with self._lock:
            rec = self._records.get(handle)
            return rec.traj if rec else None

    def exists(self, handle: str) -> bool:
        with self._lock:
            return handle in self._records

    def lineage(self, handle: str) -> List[Dict[str, object]]:
        """从根到该句柄的血缘链。"""
        with self._lock:
            root = self._root_of(handle)
            chain = self._lineage.get(root, [])
            if handle not in chain:
                # 兜底：沿 parent 回溯
                out: List[Dict[str, object]] = []
                cur: Optional[str] = handle
                while cur and cur in self._records:
                    out.append(self._records[cur].line())
                    cur = self._records[cur].parent
                return list(reversed(out))
            return [self._records[h].line() for h in chain if h in self._records]

    def handles(self) -> List[str]:
        with self._lock:
            return sorted(self._records.keys())

    def n_handles(self) -> int:
        with self._lock:
            return len(self._records)

    def stats(self) -> Dict[str, object]:
        with self._lock:
            ops: Dict[str, int] = {}
            for rec in self._records.values():
                ops[rec.operation] = ops.get(rec.operation, 0) + 1
            return {
                "n_handles": len(self._records),
                "operations": dict(sorted(ops.items())),
            }


def state_hash(traj: Traj) -> str:
    """轨迹内容指纹，用于缓存与防串号。

    只取长度与首末点等少量特征——全量哈希在 117 万点上代价过高，
    而本用途只需要区分不同的处理结果。
    """
    if len(traj) == 0:
        return "empty"
    parts = [traj.vehicle_id, str(traj.segment_index), str(len(traj)),
             str(traj.timestamps[0]), str(traj.timestamps[-1]),
             f"{traj.coords[0][0]:.6f}", f"{traj.coords[0][1]:.6f}",
             f"{traj.coords[-1][0]:.6f}", f"{traj.coords[-1][1]:.6f}"]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
