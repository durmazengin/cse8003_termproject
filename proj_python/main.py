"""Orchestrator for the VSA term-project tasks.

Each task lives in its own module and exposes a single ``run_*`` entry
point that returns a structured result and prints to the console. This
script wires them together; downstream tasks (2..6) can consume the
``EstimationResult`` produced by Task 1.

Run:
    python main.py [--csv PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from estimation import DEFAULT_CSV, run_estimation


def main() -> None:
    parser = argparse.ArgumentParser(description="VSA term-project orchestrator")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Path to the dataset CSV (default: %(default)s)")
    args = parser.parse_args()

    # Task 1 - Estimation
    # When invoked from main.py we do NOT pass an output_dir, so estimation.py
    # returns its result object instead of writing files. The console report
    # is still printed by run_estimation itself.
    estimation = run_estimation(
        args.csv,
        output_dir=None,
        print_to_console=True,
    )

    # Downstream tasks will consume `estimation`:
    # task2 = run_optimal_action(estimation)
    # task3 = run_epsilon_optimality(estimation)
    # task4 = run_simulation(estimation, args.csv)
    # task5 = run_perturbation(args.csv)
    # task6 = run_model_check(args.csv, estimation)

    print("\n" + "=" * 72)
    print("main.py: tasks finished")
    print("=" * 72)
    print(f"  rows processed     : {estimation.n_rows}")
    print(f"  policy matrix      : {estimation.policy.shape[0]}x{estimation.policy.shape[1]}")
    print(f"  transition tables  : {len(estimation.transition)} (one per action)")
    print(f"  consistency checks : "
          f"{sum(estimation.checks.values())}/{len(estimation.checks)} passed")


if __name__ == "__main__":
    main()
