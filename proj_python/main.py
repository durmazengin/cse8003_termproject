"""Orchestrator for the VSA term-project tasks.

1. Dataset fingerprint
2. Load CSV into a DataFrame
3. Tasks 1--4 on full data (baseline), always
4. If ``perturbation_rate`` > 0: additional Tasks 1--4 passes on row-subsampled data

Run::

    python main.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from e_optimality_analysis import run_epsilon_optimality
from estimation import run_estimation
from fingerprint import run_fingerprint
from optimal_action_identification import run_optimal_action
from simulation import run_simulation
from util import DEFAULT_CSV, load_vsa_dataset


def _section(title: str) -> str:
    bar = "=" * 72
    return f"\n{bar}\n{title}\n{bar}"


def _drop_rows_randomly(
    df: pd.DataFrame,
    fraction_remove: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, int]:
    """Return a copy of ``df`` with ``round(fraction_remove * n)`` rows removed."""
    n = len(df)
    if n <= 1 or fraction_remove <= 0.0:
        return df.copy(), 0
    n_remove = int(round(fraction_remove * n))
    n_remove = min(max(0, n_remove), n - 1)
    if n_remove <= 0:
        return df.copy(), 0
    drop_positions = rng.choice(n, size=n_remove, replace=False)
    out = df.drop(df.index[drop_positions]).reset_index(drop=True)
    return out, n_remove


@dataclass
class PipelineRunResult:
    """Metrics from one pass of Tasks 1--4 on a given DataFrame."""

    label: str
    n_rows: int
    alpha_star: str
    c_star: float
    c_k_per_state: pd.DataFrame
    epsilon: float
    min_p_alpha_star: float
    l1_action: float
    l1_phi_src: float
    l1_phi_dst: float


def _spread_across_states(c_k_per_state: pd.DataFrame) -> pd.Series:
    """Per action: max_phi c_k(phi) - min_phi c_k(phi) (P-model spread diagnostic)."""
    return c_k_per_state.max(axis=0) - c_k_per_state.min(axis=0)


def _max_abs_table_delta(reference: pd.DataFrame, other: pd.DataFrame) -> float:
    """Largest |c_k(phi, alpha) - reference| over aligned cells."""
    aligned = other.reindex_like(reference).fillna(0.0)
    return float(np.abs(aligned - reference).max().max())


def _format_c_k_per_state_lines(run: PipelineRunResult) -> List[str]:
    """Human-readable block for one run's per-state penalty table."""
    indented = run.c_k_per_state.round(6).to_string()
    return [
        f"{run.label}  (n={run.n_rows} rows):",
        indented,
        "",
    ]


def _run_tasks_1_to_4(
    df: pd.DataFrame,
    csv_path: Path,
    *,
    dataset_label: str,
    print_to_console: bool,
) -> PipelineRunResult:
    estimation = run_estimation(
        df,
        dataset_label=dataset_label,
        output_dir=None,
        print_to_console=print_to_console,
        print_penalty_per_state=False,
    )
    optimal = run_optimal_action(
        estimation=estimation,
        output_dir=None,
        print_to_console=print_to_console,
    )
    epsilon_res = run_epsilon_optimality(
        estimation=estimation,
        optimal=optimal,
        output_dir=None,
        print_to_console=print_to_console,
    )
    sim_res = run_simulation(
        estimation=estimation,
        csv_path=csv_path,
        output_dir=None,
        print_to_console=print_to_console,
    )
    return PipelineRunResult(
        label=dataset_label,
        n_rows=len(df),
        alpha_star=optimal.alpha_star,
        c_star=optimal.c_star,
        c_k_per_state=estimation.penalty_per_state.copy(),
        epsilon=epsilon_res.epsilon,
        min_p_alpha_star=epsilon_res.min_p_alpha_star,
        l1_action=sim_res.l1_action,
        l1_phi_src=sim_res.l1_phi_src,
        l1_phi_dst=sim_res.l1_phi_dst,
    )


def _print_run_results(run: PipelineRunResult) -> None:
    print("\n" + "=" * 72)
    print(f"main: {run.label} — Tasks 1--4 finished")
    print("=" * 72)
    print(f"  rows processed     : {run.n_rows}")
    print(f"  optimal action     : alpha* = {run.alpha_star}  (c* = {run.c_star:.6f})")
    print(f"  epsilon (Task 3)   : ε = {run.epsilon:.6f}  "
          f"(min P(α*|φ) = {run.min_p_alpha_star:.6f})")
    print(
        f"  simulation L1      : actions={run.l1_action:.4f}  "
        f"phi_src={run.l1_phi_src:.4f}  phi_dst={run.l1_phi_dst:.4f}"
    )


def _print_c_k_per_state_runs(runs: List[PipelineRunResult]) -> None:
    """Print per-state penalty tables for one or more pipeline runs."""
    lines = [
        _section("Task 6 — Model assumption check"),
        "c_k(phi_i) = P(beta=1 | alpha_k, phi_i)",
        "",
    ]
    for run in runs:
        lines.extend(_format_c_k_per_state_lines(run))
    print("\n".join(lines))


def _print_experiments_summary(
    *,
    csv_path: Path,
    n_full: int,
    perturbation_rate: float,
    experiment_count: int,
    baseline: PipelineRunResult,
    perturbed: List[PipelineRunResult],
) -> None:
    epsilons = np.array([r.epsilon for r in perturbed], dtype=float)
    baseline_spread = _spread_across_states(baseline.c_k_per_state)
    perturbed_spreads = np.array(
        [_spread_across_states(r.c_k_per_state).values for r in perturbed],
        dtype=float,
    )
    table_deltas = np.array(
        [_max_abs_table_delta(baseline.c_k_per_state, r.c_k_per_state) for r in perturbed],
        dtype=float,
    )
    actions = list(baseline.c_k_per_state.columns)

    lines = [
        _section(
            f"Task 5 — Perturbation "
            f"({perturbation_rate:.0f}% rows removed, {experiment_count} trial(s))"
        ),
        f"Dataset : {csv_path.resolve()}",
        f"Rows    : {n_full} full",
        "",
        "Baseline (full data, Tasks 1--4)",
        f"  alpha* = {baseline.alpha_star}   c* = {baseline.c_star:.6f}",
        f"  epsilon = {baseline.epsilon:.6f}",
        "",
        "Perturbed runs",
    ]
    for r in perturbed:
        lines.append(
            f"  {r.label}:  alpha*={r.alpha_star}  epsilon={r.epsilon:.6f}  "
            f"L1_act={r.l1_action:.4f}"
        )
    if len(perturbed) > 1:
        lines.extend(
            [
                "",
                f"  epsilon mean={epsilons.mean():.6f}  std={epsilons.std(ddof=1):.6f}  "
                f"delta_mean_vs_baseline={epsilons.mean() - baseline.epsilon:+.6f}",
            ]
        )
    elif len(perturbed) == 1:
        lines.append(
            f"\n  delta_epsilon (perturbed - baseline) = "
            f"{perturbed[0].epsilon - baseline.epsilon:+.6f}"
        )

    lines.extend(
        [
            "Spread across states per action  (max_phi c_k - min_phi c_k):",
            "  Baseline:  "
            + "  ".join(f"{a}={baseline_spread[a]:.6f}" for a in actions),
        ]
    )
    if perturbed:
        if len(perturbed) > 1:
            spread_mean = perturbed_spreads.mean(axis=0)
            spread_std = perturbed_spreads.std(axis=0, ddof=1)
            lines.append(
                "  Perturbed: "
                + "  ".join(
                    f"{a}=mean {spread_mean[i]:.6f}  std {spread_std[i]:.6f}"
                    for i, a in enumerate(actions)
                )
            )
            lines.extend(
                [
                    "",
                    "  |c_k_per_state - baseline| max cell delta:",
                    f"    mean={table_deltas.mean():.6f}  "
                    f"std={table_deltas.std(ddof=1):.6f}  "
                    f"max={table_deltas.max():.6f}",
                ]
            )
        else:
            p_spread = _spread_across_states(perturbed[0].c_k_per_state)
            lines.append(
                "  Perturbed: "
                + "  ".join(f"{a}={p_spread[a]:.6f}" for a in actions)
            )
            lines.append(
                f"\n  |c_k_per_state - baseline| max cell delta: "
                f"{table_deltas[0]:.6f}"
            )

    print("\n".join(lines))
    _print_c_k_per_state_runs([baseline, *perturbed])


def run_all(
    csv_path: Union[Path, str],
    *,
    perturbation_rate: float,
    experiment_count: int = 1,
    random_seed: int,
    print_to_console: bool = True,
) -> List[PipelineRunResult]:
    """Fingerprint, baseline Tasks 1--4 on full data, then optional perturbed passes.

    Parameters
    ----------
    csv_path
        Path to the VSA interaction CSV.
    perturbation_rate
        Percent of rows to remove at random before each perturbed pass (0--99).
        ``0``: baseline only. ``20``: baseline plus ``experiment_count`` trials
        with 20% of rows removed.
    experiment_count
        Number of perturbed passes when ``perturbation_rate > 0``.
    random_seed
        Base RNG seed for row removal.
    print_to_console
        Whether to print fingerprint, per-task reports (baseline only when rate is 0),
        and the final summary.

    Returns
    -------
    list of PipelineRunResult
        ``[baseline, ...perturbed_trials]``.
    """
    if perturbation_rate < 0 or perturbation_rate >= 100:
        raise ValueError("perturbation_rate must be in [0, 100).")
    if experiment_count < 1:
        raise ValueError("experiment_count must be >= 1.")
    if perturbation_rate == 0 and experiment_count != 1:
        warnings.warn(
            "experiment_count is ignored when perturbation_rate is 0.",
            stacklevel=2,
        )

    csv_path = Path(csv_path)
    df_full = load_vsa_dataset(csv_path)

    run_fingerprint(
        csv_path,
        output_dir=None,
        print_to_console=print_to_console,
    )

    # first run without perturbation
    baseline = _run_tasks_1_to_4(
        df_full,
        csv_path,
        dataset_label=f"{csv_path.name} (baseline)",
        print_to_console=print_to_console,
    )
    results: List[PipelineRunResult] = [baseline]
    if print_to_console:
        _print_run_results(baseline)

    if perturbation_rate > 0:
        fraction = perturbation_rate / 100.0
        rng_master = np.random.default_rng(random_seed)
        for t in range(experiment_count):
            trial_seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(trial_seed)
            df_p, n_removed = _drop_rows_randomly(df_full, fraction, rng)
            perturbation_run = _run_tasks_1_to_4(
                df_p,
                csv_path,
                dataset_label=f"{csv_path.name} (trial {t}, removed {n_removed})",
                print_to_console=print_to_console,
            )
            if print_to_console:
                _print_run_results(perturbation_run)
            results.append(perturbation_run)

        if print_to_console:
            _print_experiments_summary(
                csv_path=csv_path,
                n_full=len(df_full),
                perturbation_rate=perturbation_rate,
                experiment_count=experiment_count,
                baseline=baseline,
                perturbed=results[1:],
            )
    elif print_to_console:
        _print_c_k_per_state_runs([baseline])

    return results


def load_all() -> List[PipelineRunResult]:
    """Entry point for ``python main.py`` — fixed run configuration."""
    return run_all(
        csv_path = DEFAULT_CSV,
        perturbation_rate=20,
        experiment_count=10,
        random_seed=42,
    )


if __name__ == "__main__":
    load_all()
