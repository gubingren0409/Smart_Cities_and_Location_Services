"""任务三修复版 ECNU 工作流组件消融实验入口。

从作业目录执行：
    python experiments/llm_assisted_ecnu_20260921_v2/run_real_ablation.py --run

不传 --run 时只分析已完成结果，不产生 API 费用。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK))

from traj_agent.experiments.analyze_real_ablation import analyze  # noqa: E402
from traj_agent.experiments.real_ablation import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="执行真实 API 实验")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.run:
        run_experiment(
            HERE,
            WORK / "traj_dict.json",
            repetitions=args.repetitions,
            retry_failed=args.retry_failed,
        )
    analyze(HERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
