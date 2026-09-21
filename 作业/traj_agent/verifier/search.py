"""有界确定性搜索：为 LLM 的提议提供 ground-truth 参照。

这是任务③「三段式调参」的第三段，也是 regret 的分母。

为什么需要它
-----------
没有 ground truth 就无法判断 LLM 的建议好不好——「看起来合理」不是判据。
本模块在参数的物理先验区间内做有界搜索，给出可复现的最优解，
LLM 的得分即为它相对这个最优解的差距（regret）。

搜索策略
-------
- `grid_search`：完整网格，用于生成质量—压缩率曲线的**地形图**（任务②）。
- `coordinate_descent`：坐标下降，用远少于网格的评估次数逼近最优，
  用于 regret 的分子/分母计算，省算力。
- 两者都在 `params.PARAM_SPECS` 的区间内取值，超界一律夹紧并记录。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..core import params as params_mod


Evaluator = Callable[[Dict[str, float]], float]


@dataclass
class SearchTrace:
    """搜索过程记录：评估过的点与分数。"""

    evaluations: List[Tuple[Dict[str, float], float]] = field(default_factory=list)
    best_params: Dict[str, float] = field(default_factory=dict)
    best_score: float = -float("inf")
    n_evaluations: int = 0
    method: str = ""
    clamped_params: List[str] = field(default_factory=list)
    # baseline_score：固定参照系的分数（物理先验默认参数），
    # 用于归一化 regret 的分母。它**不能**取搜索起点的分数——
    # 那等于拿提议自己当基准，会把 headroom 压到接近 0，
    # 使归一化 regret 虚高（实测：best−baseline 仅 0.021 却算出 regret=1.0）。
    baseline_score: float = 0.0
    baseline_params: Dict[str, float] = field(default_factory=dict)

    def record(self, p: Dict[str, float], score: float) -> None:
        self.evaluations.append((dict(p), float(score)))
        self.n_evaluations += 1
        if score > self.best_score:
            self.best_score = float(score)
            self.best_params = dict(p)

    def to_dict(self, max_evals: int = 0) -> Dict[str, object]:
        d: Dict[str, object] = {
            "method": self.method,
            "n_evaluations": self.n_evaluations,
            "best_score": round(self.best_score, 6),
            "best_params": {k: round(float(v), 4) for k, v in sorted(self.best_params.items())},
            "clamped_params": list(self.clamped_params),
            "baseline_score": round(self.baseline_score, 6),
        }
        if max_evals:
            d["evaluations"] = [
                {**{k: round(float(v), 4) for k, v in sorted(p.items())},
                 "score": round(s, 6)}
                for p, s in self.evaluations[:max_evals]
            ]
        return d


def _clamp(p: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    return params_mod.clamp_params(p)


def linspace(low: float, high: float, n: int, integer: bool = False) -> List[float]:
    """在 [low, high] 上取 n 个值。integer=True 时取整并去重。"""
    n = max(1, int(n))
    if n == 1:
        vals = [(float(low) + float(high)) / 2.0]
    else:
        step = (float(high) - float(low)) / (n - 1)
        vals = [float(low) + step * i for i in range(n)]
    if integer:
        vals = sorted({float(round(v)) for v in vals})
    return vals


def grid_search(base_params: Dict[str, float],
                dimensions: Dict[str, Sequence[float]],
                evaluator: Evaluator,
                max_evals: Optional[int] = None) -> SearchTrace:
    """在给定维度上做完整网格搜索，其余参数固定为 base_params。"""
    trace = SearchTrace(method="grid")
    names = list(dimensions.keys())
    if not names:
        p, clamped = _clamp(base_params)
        trace.clamped_params = clamped
        trace.record(p, evaluator(p))
        return trace

    def rec(i: int, acc: Dict[str, float]) -> None:
        if max_evals is not None and trace.n_evaluations >= max_evals:
            return
        if i == len(names):
            merged, clamped = _clamp({**base_params, **acc})
            for c in clamped:
                if c not in trace.clamped_params:
                    trace.clamped_params.append(c)
            trace.record(merged, evaluator(merged))
            return
        name = names[i]
        for v in dimensions[name]:
            rec(i + 1, {**acc, name: float(v)})
            if max_evals is not None and trace.n_evaluations >= max_evals:
                return

    rec(0, {})
    return trace


def coordinate_descent(base_params: Dict[str, float],
                       search_params: Optional[Sequence[str]] = None,
                       evaluator: Evaluator = None,
                       n_steps: int = 5,
                       rounds: int = 3,
                       max_evals: Optional[int] = None) -> SearchTrace:
    """坐标下降：轮流在每个参数维度上做一维网格细化。

    评估次数约为 `rounds × 维度数 × n_steps`，
    在 4 个参数上约 60 次，远少于完整网格的数千次。
    """
    if evaluator is None:
        raise ValueError("需要 evaluator")
    names = list(search_params) if search_params else list(base_params.keys())
    trace = SearchTrace(method="coordinate_descent")

    cur, clamped = _clamp(base_params)
    trace.clamped_params = list(clamped)
    trace.record(cur, evaluator(cur))

    for r_i in range(max(1, rounds)):
        improved = False
        for name in names:
            spec = params_mod.PARAM_SPECS.get(name)
            if spec is None:
                continue
            center = float(cur.get(name, spec.default))
            span = (spec.high - spec.low)
            # 半径随轮次收缩，但**不能过早**：早先版本用 span/2^rounds，
            # 在 f(x)=-(x-8)^2、起点 30 的测试里只收敛到 10.64，
            # 使 ground truth 偏弱、regret 被系统性低估。
            # 改为线性衰减（rounds=3 时依次 0.50 / 0.33 / 0.17 倍跨度），
            # 先在粗网格上跨过最优，再逐步细化。
            frac = 1.0 / (r_i + 2.0)
            radius = span * frac
            low = max(spec.low, center - radius)
            high = min(spec.high, center + radius)
            if high <= low:
                continue
            best_local = center
            best_score = -float("inf")
            for v in linspace(low, high, n_steps, integer=spec.integer):
                if max_evals is not None and trace.n_evaluations >= max_evals:
                    return trace
                cand, cl = _clamp({**cur, name: v})
                for c in cl:
                    if c not in trace.clamped_params:
                        trace.clamped_params.append(c)
                s_val = evaluator(cand)
                trace.record(cand, s_val)
                if s_val > best_score:
                    best_score, best_local = s_val, float(cand.get(name, v))
            if abs(best_local - center) > 1e-12:
                cur = {**cur, name: best_local}
                improved = True
        if not improved:
            break

    # 收尾：在历史最优点周围做一次细粒度精修。
    # 各维度已各自收敛后容易停在粗网格点上，这一步把 ground truth 逼近真最优，
    # 使 regret 的分母可靠。
    if trace.best_params:
        for name in names:
            spec = params_mod.PARAM_SPECS.get(name)
            if spec is None:
                continue
            center = float(trace.best_params.get(name, spec.default))
            radius = (spec.high - spec.low) * 0.03
            low = max(spec.low, center - radius)
            high = min(spec.high, center + radius)
            if high <= low:
                continue
            for v in linspace(low, high, 3, integer=spec.integer):
                if max_evals is not None and trace.n_evaluations >= max_evals:
                    return trace
                cand, cl = _clamp({**trace.best_params, name: v})
                for c in cl:
                    if c not in trace.clamped_params:
                        trace.clamped_params.append(c)
                trace.record(cand, evaluator(cand))
    return trace


def sweep_quality_compression(base_params: Dict[str, float],
                              evaluator: Evaluator,
                              param: str = "dp_tolerance",
                              n: int = 12) -> List[Dict[str, float]]:
    """单参数扫描，产出「质量—压缩率曲线」的原始点。

    参数取值由物理先验区间决定（params.PARAM_SPECS），
    曲线上每点的 (compression, fidelity) 由调用方从评分返回里取。
    """
    spec = params_mod.PARAM_SPECS[param]
    rows: List[Dict[str, float]] = []
    for v in linspace(spec.low, spec.high, n, integer=spec.integer):
        p, _ = _clamp({**base_params, param: v})
        rows.append({"params": p, "score": float(evaluator(p))})
    return rows
