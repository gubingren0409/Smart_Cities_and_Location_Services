"""面向 LLM 的记忆检索：把 L1/L2 组织成可直接使用的先验。

对外只暴露两个东西：
    prior_for(card)  —— 相似案例 + 参数推荐区间，注入 LLM 上下文
    explain_neighbors(card) —— 检索结果的可解释视图（课堂展示用）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..core.diagnosis import DiagnosisCard
from . import features as feat_mod
from .store import MemoryCase, MemoryStore, ParamRegion


@dataclass
class MemoryPrior:
    """给 LLM 的先验包。"""

    available: bool = False
    regime: str = ""
    timeline_quality: str = ""
    n_neighbors: int = 0
    neighbors: List[Dict[str, Any]] = field(default_factory=list)
    suggested_regions: List[Dict[str, Any]] = field(default_factory=list)
    feature_vector: Dict[str, float] = field(default_factory=dict)
    caution: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "regime": self.regime,
            "timeline_quality": self.timeline_quality,
            "n_neighbors": self.n_neighbors,
            "neighbors": self.neighbors,
            "suggested_regions": self.suggested_regions,
            "feature_vector": self.feature_vector,
            "caution": self.caution,
        }


def prior_for(card: DiagnosisCard, store: Optional[MemoryStore],
              k: int = 5, max_params: int = 6) -> MemoryPrior:
    """组装检索到的先验。无记忆时返回 available=False 而不是报错。"""
    vec = feat_mod.featurize(card)
    prior = MemoryPrior(
        regime=card.regime,
        timeline_quality=card.timeline_quality,
        feature_vector=feat_mod.vector_to_dict(vec),
    )
    if store is None or store.count(admitted_only=True) == 0:
        prior.caution = (
            "记忆库为空（冷启动）。以下建议只能依据物理先验与本次实测，"
            "不能引用历史案例。"
        )
        return prior

    # exclude_self=True：绝不把「正在评估的这条轨迹」自己的历史记录当成先验，
    # 那属于信息泄漏，会让消融实验失真。
    cases: List[MemoryCase] = store.retrieve_similar(card, k=k, exclude_self=True)
    prior.available = bool(cases)
    prior.n_neighbors = len(cases)
    prior.neighbors = [c.to_dict() for c in cases]

    regions: List[ParamRegion] = store.query_regions(
        regime=card.regime, timeline_quality=card.timeline_quality)
    if not regions:
        regions = store.query_regions(regime=card.regime)
    prior.suggested_regions = [r.to_dict() for r in regions[:max_params]]

    if not cases:
        prior.caution = (
            f"同 regime({card.regime}) 下没有已验证的相似案例，"
            "历史先验的参考价值有限。"
        )
    elif max((c.similarity for c in cases), default=0.0) < 0.9:
        prior.caution = (
            f"最相似案例的相似度仅 {max(c.similarity for c in cases):.3f}，"
            "说明本条轨迹的特征与历史案例差异较大，先验应当弱化使用。"
        )
    return prior


def explain_neighbors(card: DiagnosisCard, store: Optional[MemoryStore],
                      k: int = 5) -> Dict[str, Any]:
    """检索过程的可解释视图：特征向量、每个邻居的特征与相似度。

    这是课堂展示「记忆是怎么工作的」的落点——
    可以把特征向量直接打出来看，而不是面对一个黑盒 embedding。
    """
    vec = feat_mod.featurize(card)
    out: Dict[str, Any] = {
        "query": {
            "seg_id": card.seg_id,
            "regime": card.regime,
            "features": feat_mod.vector_to_dict(vec),
        },
        "feature_names": list(feat_mod.FEATURE_NAMES),
        "neighbors": [],
        "note": (
            "检索用 12 维归一化特征 + regime 硬门控，不用 embedding。"
            "特征都有物理含义，可直接归因。"
        ),
    }
    if store is None:
        out["note"] = "记忆库未接入"
        return out

    # 拉取候选的原始特征以展示
    all_feats = dict((i, (r, f)) for i, r, f in store.all_features())
    for c in store.retrieve_similar(card, k=k):
        row = all_feats.get(c.episodic_id)
        d = c.to_dict()
        d["features"] = (feat_mod.vector_to_dict(row[1]) if row else {})
        d["regime_match"] = (row[0] == card.regime) if row else None
        out["neighbors"].append(d)
    return out


def region_hint_text(prior: MemoryPrior) -> str:
    """把参数区间渲染成给 LLM 的一段文本。"""
    if not prior.suggested_regions:
        return "（无已验证的参数区间）"
    lines = []
    for r in prior.suggested_regions:
        lines.append(
            f"- {r['param']}: 推荐 {r['low']}–{r['high']}（中位 {r['median']}，"
            f"基于 {r['n_samples']} 条已验证案例，平均 regret {r['mean_regret']}）")
    return "\n".join(lines)
