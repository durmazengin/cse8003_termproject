"""Orchestrator for the VSA term-project tasks.

Each task lives in its own module and exposes a single ``run_*`` entry
point that returns a structured result and prints to the console. This
script wires them together; the dataset fingerprint runs first, then
Tasks~1--3 (and later tasks) consume the loaded CSV or prior results.

Run:
    python main.py [--csv PATH]

Entry point: ``load_all()`` wires fingerprint, Tasks~1--3 (and later tasks).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from util import DEFAULT_CSV
from estimation import run_estimation
from fingerprint import run_fingerprint
from optimal_action_identification import run_optimal_action
from e_optimality_analysis import run_epsilon_optimality


def load_all() -> None:
    parser = argparse.ArgumentParser(description="VSA term-project orchestrator")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Path to the dataset CSV (default: %(default)s)")
    args = parser.parse_args()

    # Dataset fingerprint (before modeling)
    fp = run_fingerprint(
        args.csv,
        output_dir=None,
        print_to_console=True,
    )

    # Task 1 - Estimation
    # When invoked from load_all() we do NOT pass an output_dir, so estimation.py
    # returns its result object instead of writing files. The console report
    # is still printed by run_estimation itself.
    estimation = run_estimation(
        args.csv,
        output_dir=None,
        print_to_console=True,
    )

    # Task 2 - Optimal Action Identification
    # load_all() forwards estimation's return value directly; no disk I/O here.
    # optimal_action_identification.py still prints its report to the console.
    optimal = run_optimal_action(
        estimation=estimation,
        output_dir=None,
        print_to_console=True,
    )

    # Task 3 — ε-optimality
    epsilon_res = run_epsilon_optimality(
        estimation=estimation,
        optimal=optimal,
        output_dir=None,
        print_to_console=True,
    )

    # Downstream tasks will consume `estimation`, `optimal`, and `epsilon_res`:
    # task4 = run_simulation(estimation, args.csv)
    # task5 = run_perturbation(args.csv)
    # task6 = run_model_check(args.csv, estimation)

    print("\n" + "=" * 72)
    print("load_all(): tasks finished")
    print("=" * 72)
    print(f"  rows processed     : {estimation.n_rows}")
    print(f"  policy matrix      : {estimation.policy.shape[0]}x{estimation.policy.shape[1]}")
    print(f"  transition tables  : {len(estimation.transition)} (one per action)")
    print(f"  consistency checks : "
          f"{sum(estimation.checks.values())}/{len(estimation.checks)} passed")
    print(f"  fingerprint checksum : {fp.checksum}")
    print(f"  sequences / mean len : {fp.n_sequences} / {fp.mean_sequence_length:.4f}")
    print(f"  optimal action     : alpha* = {optimal.alpha_star}  "
          f"(c_star = {optimal.c_star:.6f})")
    print(f"  epsilon (Task 3)   : ε = {epsilon_res.epsilon:.6f}  "
          f"(min P(α*|φ) = {epsilon_res.min_p_alpha_star:.6f})")

if __name__ == "__main__":
    load_all()
