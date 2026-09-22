from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORK_DIR = Path(__file__).resolve().parents[2]
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from traj_agent.experiments import objective_optimization_refined_runner as runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("strong", "regions", "deterministic", "analyze", "all"),
        default="analyze",
    )
    args = parser.parse_args()
    experiment_dir = Path(__file__).resolve().parent
    data_path = WORK_DIR / "traj_dict.json"
    if args.stage in {"strong", "all"}:
        runner.run_strong_reference_stage(experiment_dir, data_path)
    if args.stage in {"regions", "all"}:
        runner.prepare_config(experiment_dir)
        runner.write_region_quality(experiment_dir)
    if args.stage in {"deterministic", "all"}:
        runner.run_deterministic_memory_stage(experiment_dir, data_path)
    if args.stage in {"analyze", "all"}:
        from traj_agent.experiments import analyze_objective_optimization_refined
        analyze_objective_optimization_refined.analyze(experiment_dir)


if __name__ == "__main__":
    main()
