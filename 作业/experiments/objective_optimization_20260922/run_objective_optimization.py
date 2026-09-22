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
    parser.add_argument(
        "--teacher-sizes",
        help="可选的逗号分隔 Teacher 规模；用于互不重叠的结果分片",
    )
    parser.add_argument("--results-file", default="raw_results.jsonl")
    parser.add_argument("--runtime-tag", default="")
    parser.add_argument("--no-shared-writes", action="store_true")
    args = parser.parse_args()

    experiment_dir = Path(__file__).resolve().parent
    data_path = WORK_DIR / "traj_dict.json"
    if args.stage in {"teacher", "all"}:
        runner.run_teacher_stage(experiment_dir, data_path)
    if args.stage in {"formal", "all"}:
        teacher_sizes = (
            [int(value) for value in args.teacher_sizes.split(",") if value.strip()]
            if args.teacher_sizes else None
        )
        runner.run_formal_stage(
            experiment_dir,
            data_path,
            retry_failed=args.retry_failed,
            teacher_sizes=teacher_sizes,
            results_filename=args.results_file,
            runtime_tag=args.runtime_tag,
            write_shared_artifacts=not args.no_shared_writes,
        )
    if args.stage in {"analyze", "all"}:
        from traj_agent.experiments import analyze_objective_optimization
        analyze_objective_optimization.analyze(experiment_dir)


if __name__ == "__main__":
    main()
