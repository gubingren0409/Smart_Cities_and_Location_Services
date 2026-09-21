"""诊断卡 → 归一化特征向量。

这是 L2 程序记忆的检索基础。**刻意不用 embedding**：
  1. 可解释——学生能把检索到的邻居和特征值直接打印出来看；
  2. 离线、零成本、零延迟；
  3. 特征都有物理含义，检索结果可以归因。

维度选择的原则：每个维度都要能区分「该用哪套参数」。
选了 10 维而不是把诊断卡所有字段都塞进去，
因为无关维度会稀释相似度。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.diagnosis import DiagnosisCard

# 特征名（顺序即向量顺序，改动需同步 FEATURE_VERSION）
FEATURE_NAMES: Tuple[str, ...] = (
    "log_n_points",
    "duration_ratio",
    "log_dt_median",
    "dt_cv",
    "dup_ratio",
    "log_length",
    "log_displacement",
    "sinuosity",
    "log_speed_median",
    "log_speed_max",
    "anomaly_density",
    "dt_zero_ratio",
)

FEATURE_VERSION = "v1"

# 每个特征的参考尺度（用于归一化），取本项目实测的量级
_SCALES: Dict[str, float] = {
    "log_n_points": math.log1p(200.0),
    "duration_ratio": 1.0,
    "log_dt_median": math.log1p(60.0),
    "dt_cv": 2.0,
    "dup_ratio": 1.0,
    "log_length": math.log1p(25000.0),
    "log_displacement": math.log1p(20000.0),
    "sinuosity": 3.0,
    "log_speed_median": math.log1p(25.0),
    "log_speed_max": math.log1p(60.0),
    "anomaly_density": 1.0,
    "dt_zero_ratio": 1.0,
}

_REGIME_CODE = {"stationary": 0.0, "mixed": 0.5, "moving": 1.0, "unknown": 0.25}


def _safe_log1p(x: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return math.log1p(max(0.0, x))


def dt_coefficient_of_variation(card: DiagnosisCard) -> float:
    """时间间隔的变异系数代理。

    诊断卡里只有 median 与 p95，故用 (p95/median - 1) 作为离散度代理。
    真实 CV 需要全量 Δt，而诊断卡刻意不含原始序列。
    """
    med = float(card.dt_median_s or 0.0)
    p95 = float(card.dt_p95_s or 0.0)
    if med <= 0:
        return 0.0
    return max(0.0, p95 / med - 1.0)


def featurize(card: DiagnosisCard) -> List[float]:
    """把诊断卡转成归一化特征向量，每维落在 [0,1]。"""
    n = max(1, int(card.n_points or 0))
    raw: Dict[str, float] = {
        "log_n_points": _safe_log1p(n),
        "duration_ratio": min(1.0, float(card.duration_s or 0.0) / 1200.0),
        "log_dt_median": _safe_log1p(card.dt_median_s or 0.0),
        "dt_cv": dt_coefficient_of_variation(card),
        "dup_ratio": float(card.consecutive_dup_ratio or 0.0),
        "log_length": _safe_log1p(card.length_m or 0.0),
        "log_displacement": _safe_log1p(card.displacement_m or 0.0),
        "sinuosity": max(0.0, float(card.sinuosity or 1.0) - 1.0),
        "log_speed_median": _safe_log1p(card.speed_median_mps or 0.0),
        "log_speed_max": _safe_log1p(card.speed_max_mps or 0.0),
        "anomaly_density": min(1.0, float(card.n_anomalous_points or 0) / n),
        "dt_zero_ratio": float(card.dt_zero_ratio or 0.0),
    }
    out: List[float] = []
    for name in FEATURE_NAMES:
        v = raw.get(name, 0.0)
        scale = _SCALES.get(name, 1.0) or 1.0
        out.append(max(0.0, min(1.0, v / scale)))
    return out


def featurize_with_regime(card: DiagnosisCard) -> List[float]:
    """在特征末尾追加 regime 编码。

    检索时另作处理：regime 是**硬分类**，不该被欧氏距离平滑掉
    （静止轨迹与行驶轨迹的特征完全不同，混合会污染结果）。
    故单独返回，由 retrieve 做门控。
    """
    return featurize(card) + [_REGIME_CODE.get(card.regime, 0.25)]


def regime_of(card: DiagnosisCard) -> str:
    return card.regime or "unknown"


def vector_to_dict(vec: Sequence[float]) -> Dict[str, float]:
    return {name: round(float(v), 4) for name, v in zip(FEATURE_NAMES, vec)}


def feature_table(cards: Sequence[DiagnosisCard]) -> List[Dict[str, object]]:
    """批量导出特征表，便于在 notebook 里直接看（可解释性的落点）。"""
    rows: List[Dict[str, object]] = []
    for c in cards:
        row: Dict[str, object] = {"seg_id": c.seg_id, "regime": c.regime}
        row.update(vector_to_dict(featurize(c)))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 相似度
# ---------------------------------------------------------------------------
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) ** 2 for x in a))
    nb = math.sqrt(sum(float(y) ** 2 for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def similarity(a: Sequence[float], b: Sequence[float], metric: str = "euclidean") -> float:
    """返回「越大越相似」的分数，便于统一排序。

    euclidean：score = 1 / (1 + d)。特征已归一化到 [0,1]，
    故 d 的上界约 sqrt(len)，score 落在 (0,1]。
    """
    if metric == "cosine":
        return cosine_similarity(a, b)
    d = euclidean(a, b)
    return 1.0 / (1.0 + d)
