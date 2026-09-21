"""记忆存储：SQLite 承载机器校准（L1 情景记忆 + L2 程序记忆）。

**写入权限的唯一持有者是核验器。** LLM 没有任何工具可以写这张表。
理由：未经验证的建议一旦进入记忆，下一条轨迹就会被它带偏，
而错误会被后续检索不断放大。只有实测确认有效的经验才值得复用。

L1 情景记忆：每条 (诊断 → 参数 → 实测指标 → 核验结论) 一笔，
    是全量原始经验，不删不改。
L2 程序记忆：从 L1 蒸馏出的「诊断签名 → 参数区间」，
    是检索时实际使用的那一层。

为什么 L2 仍留在 SQLite 而不进 Obsidian：
    它是机器统计量（区间、样本数、置信度），量会随轨迹数增长，
    且需要按数值特征检索。人写的因果解释在 Obsidian（L3），
    由 memory/export.py 单向导出，agent 只读。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.diagnosis import DiagnosisCard
from . import features as feat_mod

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    seg_id TEXT NOT NULL,
    vehicle_id TEXT,
    regime TEXT,
    timeline_quality TEXT,
    feature_version TEXT NOT NULL,
    features TEXT NOT NULL,          -- JSON 数组
    diagnosis_json TEXT NOT NULL,    -- 完整诊断卡
    params_json TEXT NOT NULL,       -- LLM 提议的参数
    metrics_json TEXT NOT NULL,      -- 实测指标
    objective_json TEXT NOT NULL,    -- 目标函数结果
    verification_json TEXT NOT NULL, -- 核验结论（含 regret）
    admitted INTEGER NOT NULL,       -- 是否通过准入闸门
    admit_reason TEXT,
    score REAL,
    regret REAL,
    mode TEXT,                       -- 消融实验的分组标签
    run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodic_seg ON episodic(seg_id);
CREATE INDEX IF NOT EXISTS idx_episodic_regime ON episodic(regime, admitted);
CREATE INDEX IF NOT EXISTS idx_episodic_run ON episodic(run_id);

CREATE TABLE IF NOT EXISTS procedural (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    updated_at REAL NOT NULL,
    regime TEXT NOT NULL,
    timeline_quality TEXT NOT NULL,
    signature TEXT NOT NULL,          -- 人群签名（regime+timeline+分桶）
    param_name TEXT NOT NULL,
    low REAL, high REAL, median REAL,
    n_samples INTEGER NOT NULL,
    mean_regret REAL,
    mean_score REAL,
    evidence_ids TEXT,                -- 支撑该区间的 episodic id 列表
    UNIQUE(regime, timeline_quality, signature, param_name)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    mode TEXT,
    n_cases INTEGER,
    notes TEXT
);
"""


@dataclass
class MemoryCase:
    """一条检索到的历史案例。"""

    episodic_id: int
    seg_id: str
    vehicle_id: str
    regime: str
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    regret: Optional[float] = None
    similarity: float = 0.0
    diagnosis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, max_features: int = 0) -> Dict[str, Any]:
        return {
            "episodic_id": self.episodic_id,
            "seg_id": self.seg_id,
            "vehicle_id": self.vehicle_id,
            "regime": self.regime,
            "similarity": round(self.similarity, 4),
            "params": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                       for k, v in sorted(self.params.items())},
            "score": round(self.score, 4),
            "regret": None if self.regret is None else round(self.regret, 4),
            "metrics": {k: v for k, v in sorted(self.metrics.items())
                        if k in ("compression_ratio", "max_deviation_m",
                                 "hausdorff_m", "n_points", "road_match_rate")},
        }


@dataclass
class ParamRegion:
    """L2 蒸馏出的参数推荐区间。"""

    regime: str
    timeline_quality: str
    param_name: str
    low: float
    high: float
    median: float
    n_samples: int
    mean_regret: Optional[float] = None
    mean_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "timeline_quality": self.timeline_quality,
            "param": self.param_name,
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "median": round(self.median, 4),
            "n_samples": self.n_samples,
            "mean_regret": None if self.mean_regret is None else round(self.mean_regret, 4),
            "mean_score": None if self.mean_score is None else round(self.mean_score, 4),
        }


class MemoryStore:
    """SQLite 记忆存储。线程安全。"""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._lock = threading.RLock()
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- 写入（仅核验器调用） -------------------------------------------
    def write_case(self,
                   card: DiagnosisCard,
                   params: Dict[str, Any],
                   metrics: Dict[str, Any],
                   objective: Dict[str, Any],
                   verification: Dict[str, Any],
                   admitted: bool,
                   admit_reason: str = "",
                   mode: str = "",
                   run_id: str = "",
                   score: Optional[float] = None,
                   regret: Optional[float] = None) -> int:
        """记录一条经验。返回 episodic id。

        即使 admitted=False 也**照常写入**——失败经验同样有价值：
        它支撑「这个方向不行」的结论，而且 L2 蒸馏需要看全量分布。
        准入闸门控制的是「是否进入 L2 与是否被检索推荐」，
        不是「是否记录」。
        """
        vec = feat_mod.featurize(card)
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO episodic
                   (created_at, seg_id, vehicle_id, regime, timeline_quality,
                    feature_version, features, diagnosis_json, params_json,
                    metrics_json, objective_json, verification_json,
                    admitted, admit_reason, score, regret, mode, run_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (time.time(), card.seg_id, card.vehicle_id, card.regime,
                 card.timeline_quality, feat_mod.FEATURE_VERSION,
                 json.dumps(vec), json.dumps(card.to_dict(), ensure_ascii=False),
                 json.dumps(params, ensure_ascii=False, default=str),
                 json.dumps(metrics, ensure_ascii=False, default=str),
                 json.dumps(objective, ensure_ascii=False, default=str),
                 json.dumps(verification, ensure_ascii=False, default=str),
                 1 if admitted else 0, admit_reason,
                 None if score is None else float(score),
                 None if regret is None else float(regret),
                 mode, run_id))
            self._conn.commit()
            return int(cur.lastrowid)

    def start_run(self, run_id: str, mode: str = "", notes: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, created_at, mode, notes) VALUES (?,?,?,?)",
                (run_id, time.time(), mode, notes))
            self._conn.commit()

    def finish_run(self, run_id: str, n_cases: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE runs SET n_cases=? WHERE run_id=?",
                               (int(n_cases), run_id))
            self._conn.commit()

    # ---- 读取 -----------------------------------------------------------
    def count(self, admitted_only: bool = False) -> int:
        sql = "SELECT COUNT(*) AS c FROM episodic"
        if admitted_only:
            sql += " WHERE admitted=1"
        with self._lock:
            return int(self._conn.execute(sql).fetchone()["c"])

    def retrieve(self, features: Sequence[float], k: int = 5,
                 regime: Optional[str] = None,
                 admitted_only: bool = True,
                 metric: str = "euclidean",
                 timeline_quality: Optional[str] = None,
                 exclude_seg_ids: Optional[Sequence[str]] = None,
                 exclude_ids: Optional[Sequence[int]] = None) -> List[MemoryCase]:
        """按特征相似度检索历史案例。

        regime 是**门控**而非一个距离维度：静止轨迹与行驶轨迹的
        特征分布完全不同，混在一起检索会让结果失去参考价值。
        """
        where: List[str] = []
        args: List[Any] = []
        if admitted_only:
            where.append("admitted=1")
        if regime:
            where.append("regime=?")
            args.append(regime)
        if timeline_quality:
            where.append("timeline_quality=?")
            args.append(timeline_quality)
        excluded = set(exclude_seg_ids or ())
        excluded_ids = set(int(i) for i in (exclude_ids or ()))
        sql = "SELECT * FROM episodic"
        if where:
            sql += " WHERE " + " AND ".join(where)

        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()

        scored: List[Tuple[float, sqlite3.Row]] = []
        for row in rows:
            # 排除自身与指定 id：评测留出集时必须排除「正在评估的那条轨迹」，
            # 否则记忆会把答案直接喂回来，消融实验因信息泄漏而失去意义。
            if row["seg_id"] in excluded or int(row["id"]) in excluded_ids:
                continue
            try:
                vec = json.loads(row["features"])
            except (json.JSONDecodeError, TypeError):
                continue
            if len(vec) != len(features):
                continue      # 特征版本不一致，跳过而不是错误匹配
            s = feat_mod.similarity(features, vec, metric=metric)
            scored.append((s, row))
        scored.sort(key=lambda t: -t[0])

        out: List[MemoryCase] = []
        for s, row in scored[:max(0, int(k))]:
            out.append(MemoryCase(
                episodic_id=int(row["id"]),
                seg_id=row["seg_id"],
                vehicle_id=row["vehicle_id"] or "",
                regime=row["regime"] or "",
                params=_load_json(row["params_json"]),
                metrics=_load_json(row["metrics_json"]),
                score=float(row["score"] or 0.0),
                regret=(None if row["regret"] is None else float(row["regret"])),
                similarity=s,
                diagnosis=_load_json(row["diagnosis_json"]),
            ))
        return out

    def retrieve_similar(self, card: DiagnosisCard, k: int = 5,
                         metric: str = "euclidean",
                         exclude_self: bool = True) -> List[MemoryCase]:
        """按诊断卡检索相似案例。regime 作为硬门控。

        exclude_self 默认开启：评测场景下必须排除同一条轨迹（同一 seg_id）的
        历史记录，否则 agent 只要重复跑同一条轨迹就能「检索到」自己的答案，
        消融实验会得出「记忆毫无增益」或「记忆效果惊人」这类假结论。
        """
        excl = [card.seg_id] if exclude_self else None
        cases = self.retrieve(feat_mod.featurize(card), k=k,
                              regime=card.regime, metric=metric,
                              exclude_seg_ids=excl)
        if not cases:
            # regime 门控下无结果时放宽到同 timeline 质量，便于冷启动
            cases = self.retrieve(feat_mod.featurize(card), k=k, regime=None,
                                  timeline_quality=card.timeline_quality,
                                  metric=metric, exclude_seg_ids=excl)
        return cases

    def all_features(self) -> List[Tuple[int, str, List[float]]]:
        """返回 (id, regime, 特征向量)，用于离线分析。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, regime, features FROM episodic WHERE admitted=1").fetchall()
        out = []
        for r in rows:
            try:
                out.append((int(r["id"]), r["regime"] or "", json.loads(r["features"])))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    # ---- L2 蒸馏 ---------------------------------------------------------
    def rebuild_procedural(self, param_names: Optional[Sequence[str]] = None,
                          min_samples: int = 3,
                          quantile_low: float = 0.25,
                          quantile_high: float = 0.75) -> int:
        """从 L1 蒸馏 L2：按 (regime, timeline_quality, param) 给出推荐区间。

        只用 **admitted=1** 的记录——这正是准入闸门的价值所在：
        未经验证的建议不会影响后续推荐。

        区间取 admitted 样本的 IQR，而不是 min/max：
        避免个别极端值把推荐区间撑得过宽而失去指导意义。
        """
        names = list(param_names) if param_names else list(
            _collect_param_names(self))
        n_written = 0
        with self._lock:
            rows = self._conn.execute(
                """SELECT regime, timeline_quality, params_json, score, regret
                   FROM episodic WHERE admitted=1""").fetchall()

        buckets: Dict[Tuple[str, str, str], List[Tuple[float, float, float]]] = {}
        for row in rows:
            params = _load_json(row["params_json"])
            regime = row["regime"] or "unknown"
            tl = row["timeline_quality"] or "unknown"
            score = float(row["score"] or 0.0)
            regret = float(row["regret"] or 0.0)
            for name in names:
                if name not in params:
                    continue
                try:
                    v = float(params[name])
                except (TypeError, ValueError):
                    continue
                buckets.setdefault((regime, tl, name), []).append((v, score, regret))

        with self._lock:
            for (regime, tl, name), items in buckets.items():
                if len(items) < min_samples:
                    continue
                vals = sorted(v for v, _, _ in items)
                low = _quantile(vals, quantile_low)
                high = _quantile(vals, quantile_high)
                med = _quantile(vals, 0.5)
                mean_score = sum(s for _, s, _ in items) / len(items)
                mean_regret = sum(r for _, _, r in items) / len(items)
                signature = f"{regime}|{tl}"
                self._conn.execute(
                    """INSERT INTO procedural
                       (updated_at, regime, timeline_quality, signature, param_name,
                        low, high, median, n_samples, mean_regret, mean_score, evidence_ids)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(regime, timeline_quality, signature, param_name)
                       DO UPDATE SET updated_at=excluded.updated_at,
                                     low=excluded.low, high=excluded.high,
                                     median=excluded.median,
                                     n_samples=excluded.n_samples,
                                     mean_regret=excluded.mean_regret,
                                     mean_score=excluded.mean_score,
                                     evidence_ids=excluded.evidence_ids""",
                    (time.time(), regime, tl, signature, name, low, high, med,
                     len(items), mean_regret, mean_score, json.dumps([])))
                n_written += 1
            self._conn.commit()
        return n_written

    def query_regions(self, regime: Optional[str] = None,
                      timeline_quality: Optional[str] = None,
                      param_name: Optional[str] = None) -> List[ParamRegion]:
        where: List[str] = []
        args: List[Any] = []
        for col, val in (("regime", regime), ("timeline_quality", timeline_quality),
                         ("param_name", param_name)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        sql = "SELECT * FROM procedural"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY param_name"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [ParamRegion(
            regime=r["regime"], timeline_quality=r["timeline_quality"],
            param_name=r["param_name"], low=float(r["low"]), high=float(r["high"]),
            median=float(r["median"]), n_samples=int(r["n_samples"]),
            mean_regret=r["mean_regret"], mean_score=r["mean_score"],
        ) for r in rows]

    # ---- 统计 -----------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) c FROM episodic").fetchone()["c"]
            adm = self._conn.execute(
                "SELECT COUNT(*) c FROM episodic WHERE admitted=1").fetchone()["c"]
            proc = self._conn.execute("SELECT COUNT(*) c FROM procedural").fetchone()["c"]
            by_regime = self._conn.execute(
                """SELECT regime, COUNT(*) c, AVG(score) s FROM episodic
                   WHERE admitted=1 GROUP BY regime""").fetchall()
        return {
            "n_episodic": int(total),
            "n_admitted": int(adm),
            "admit_rate": round(adm / total, 4) if total else 0.0,
            "n_procedural": int(proc),
            "by_regime": [{"regime": r["regime"], "n": int(r["c"]),
                           "mean_score": None if r["s"] is None else round(r["s"], 4)}
                          for r in by_regime],
            "path": self.path,
        }

    def recent(self, limit: int = 10, admitted_only: bool = False) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM episodic"
        if admitted_only:
            sql += " WHERE admitted=1"
        sql += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (int(limit),)).fetchall()
        return [{
            "id": r["id"], "seg_id": r["seg_id"], "regime": r["regime"],
            "admitted": bool(r["admitted"]), "score": r["score"], "regret": r["regret"],
            "mode": r["mode"], "admit_reason": r["admit_reason"],
        } for r in rows]


def _load_json(text: Any) -> Dict[str, Any]:
    if text is None:
        return {}
    if isinstance(text, dict):
        return text
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _collect_param_names(store: MemoryStore) -> List[str]:
    with store._lock:
        rows = store._conn.execute("SELECT params_json FROM episodic").fetchall()
    names: set = set()
    for r in rows:
        names.update(_load_json(r["params_json"]).keys())
    return sorted(names)


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo]) * (1 - frac) + float(sorted_vals[hi]) * frac
