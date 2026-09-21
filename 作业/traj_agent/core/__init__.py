"""core：纯确定性算法层。

硬约束：本包**不得** import 任何 LLM / 网络 / agent 相关模块。
tests/test_architecture.py 会对这条约束做强制检查。
"""
from __future__ import annotations

from . import (anomalies, clean, diagnosis, geo, metrics, params,  # noqa: F401
               segment, simplify, traj)

__all__ = [
    "anomalies", "clean", "diagnosis", "geo", "metrics", "params",
    "segment", "simplify", "traj",
]
