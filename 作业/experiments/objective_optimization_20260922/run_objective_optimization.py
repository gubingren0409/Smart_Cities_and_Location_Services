from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORK_DIR = Path(__file__).resolve().parents[2]
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from traj_agent.experiments import objective_optimization_runner as runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("teacher", "formal", "analyze", "all"),
        default="analyze")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    experiment_dir = Path(__file__).resolve().parent
    data_path = WORK_DIR / "traj_dict.json"
    if args.stage in {"teacher", "all"}:
        runner.run_teacher_stage(experiment_dir, data_path)
    if args.stage in {"formal", "all"}:
        runner.run_formal_stage(
            experiment_dir, data_path, retry_failed=args.retry_failed)
    if args.stage in {"analyze", "all"}:
        from traj_agent.experiments import analyze_objective_optimization
        analyze_objective_optimization.analyze(experiment_dir)


if __name__ == "__main__":
    main()
