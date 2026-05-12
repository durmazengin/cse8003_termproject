"""Task 1 - Estimation for the VSA term project.

Estimates from the observed interaction-sequence dataset:

    1. Policy            P(alpha_k | phi_i)
    2. Transition model  P(phi_j | phi_i, alpha_k)
    3. Penalty probs.    c_k = P(beta = 1 | alpha_k)

and reports the supporting raw count tables

    N(phi_i, alpha_k),  N(phi_i, alpha_k, phi_j),  N(alpha_k, beta)

together with the three consistency checks required by the assignment.

Two run modes
-------------
Direct (writes a report file + CSVs to ./outputs/task1_estimation/, also prints
to the console):

    python estimation.py [--csv PATH] [--out DIR]

Imported by main.py (returns the result object, also prints to the console):

    from estimation import run_estimation
    result = run_estimation(csv_path)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TERM_PROJECT_DIR = HERE.parent
DEFAULT_CSV = TERM_PROJECT_DIR / "Engin-vsa_dataset_full.csv"
DEFAULT_OUT = HERE / "outputs" / "task1_estimation"

COLUMNS = ["sequence_id", "step", "phi_src", "phi_dst", "alpha", "beta"]


@dataclass
class EstimationResult:
    """Structured return value of :func:`run_estimation`."""

    counts_phi_alpha: pd.DataFrame                    # N(phi_i, alpha_k)
    counts_phi_alpha_phi: Dict[str, pd.DataFrame]     # N(phi_i, alpha_k, phi_j) per alpha
    counts_alpha_beta: pd.DataFrame                   # N(alpha_k, beta)
    policy: pd.DataFrame                              # P(alpha_k | phi_i)
    transition: Dict[str, pd.DataFrame]               # P(phi_j | phi_i, alpha_k) per alpha
    penalty: pd.Series                                # c_k = P(beta=1 | alpha_k)
    empirical_penalty: pd.Series                      # df.groupby('alpha')['beta'].mean()
    checks: Dict[str, bool]                           # named pass/fail flags
    states: list
    actions: list
    n_rows: int


def _load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=COLUMNS, encoding="utf-8")
    df["sequence_id"] = df["sequence_id"].astype(int)
    df["step"] = df["step"].astype(int)
    df["beta"] = df["beta"].astype(int)
    return df


def _section(title: str) -> str:
    bar = "=" * 72
    return f"\n{bar}\n{title}\n{bar}"


def _fmt(obj) -> str:
    return obj.to_string()


def run_estimation(
    csv_path: Union[Path, str] = DEFAULT_CSV,
    *,
    output_dir: Optional[Union[Path, str]] = None,
    print_to_console: bool = True,
) -> EstimationResult:
    """Compute counts, conditional probabilities, and consistency checks.

    Parameters
    ----------
    csv_path : path to the VSA dataset CSV (headerless, UTF-8).
    output_dir : if given, also write per-table CSVs and a ``report.txt``
        summary to this directory. When ``None`` (the default, used by
        ``main.py``) nothing is written to disk.
    print_to_console : whether to echo the human-readable report to stdout.
        ``True`` in both run modes per the task spec.

    Returns
    -------
    EstimationResult
        Counts, probabilities, penalty vector, and a dict of check flags.
    """
    csv_path = Path(csv_path)
    df = _load_dataset(csv_path)

    states = sorted(set(df["phi_src"].unique()) | set(df["phi_dst"].unique()))
    actions = sorted(df["alpha"].unique())
    betas = [0, 1]

    # Raw counts ---------------------------------------------------------
    N_phi_alpha = (
        df.groupby(["phi_src", "alpha"]).size()
          .unstack(fill_value=0)
          .reindex(index=states, columns=actions, fill_value=0)
    )
    N_phi_alpha.index.name = "phi_src"
    N_phi_alpha.columns.name = "alpha"

    N_phi_alpha_phi: Dict[str, pd.DataFrame] = {}
    for a in actions:
        tab = (
            df[df["alpha"] == a]
              .groupby(["phi_src", "phi_dst"]).size()
              .unstack(fill_value=0)
              .reindex(index=states, columns=states, fill_value=0)
        )
        tab.index.name = "phi_src"
        tab.columns.name = "phi_dst"
        N_phi_alpha_phi[a] = tab

    N_alpha_beta = (
        df.groupby(["alpha", "beta"]).size()
          .unstack(fill_value=0)
          .reindex(index=actions, columns=betas, fill_value=0)
    )
    N_alpha_beta.index.name = "alpha"
    N_alpha_beta.columns.name = "beta"

    # Conditional probabilities -----------------------------------------
    row_totals_phi = N_phi_alpha.sum(axis=1)
    policy = N_phi_alpha.div(row_totals_phi.replace(0, np.nan), axis=0).fillna(0.0)

    transition: Dict[str, pd.DataFrame] = {}
    for a in actions:
        N = N_phi_alpha_phi[a]
        T = N.div(N.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        transition[a] = T

    total_per_action = N_alpha_beta.sum(axis=1).replace(0, np.nan)
    penalty = (N_alpha_beta[1] / total_per_action).fillna(0.0)
    penalty.name = "c_k"

    empirical_penalty = (
        df.groupby("alpha")["beta"].mean().reindex(actions, fill_value=0.0)
    )
    empirical_penalty.name = "empirical_penalty"

    # Consistency checks -------------------------------------------------
    nonzero_phi = row_totals_phi > 0
    policy_rows_ok = bool(
        np.allclose(policy.loc[nonzero_phi].sum(axis=1).values, 1.0, atol=1e-9)
    )

    transition_rows_ok = True
    for a, T in transition.items():
        N = N_phi_alpha_phi[a]
        nonzero = N.sum(axis=1) > 0
        if not np.allclose(T.loc[nonzero].sum(axis=1).values, 1.0, atol=1e-9):
            transition_rows_ok = False
            break

    penalty_matches_empirical = bool(
        np.allclose(penalty.values, empirical_penalty.values, atol=1e-9)
    )

    checks = {
        "sum_k P(alpha_k | phi_i) == 1": policy_rows_ok,
        "sum_j P(phi_j | phi_i, alpha_k) == 1": transition_rows_ok,
        "c_k matches empirical frequencies": penalty_matches_empirical,
    }

    result = EstimationResult(
        counts_phi_alpha=N_phi_alpha,
        counts_phi_alpha_phi=N_phi_alpha_phi,
        counts_alpha_beta=N_alpha_beta,
        policy=policy,
        transition=transition,
        penalty=penalty,
        empirical_penalty=empirical_penalty,
        checks=checks,
        states=states,
        actions=actions,
        n_rows=len(df),
    )

    # Human-readable report --------------------------------------------
    lines: list[str] = []
    lines.append(_section(f"Task 1 - Estimation   (dataset: {csv_path.name})"))
    lines.append(f"Rows in dataset : {len(df)}")
    lines.append(f"States observed : {states}")
    lines.append(f"Actions observed: {actions}")

    lines.append(_section("N(phi_i, alpha_k)   -- raw counts"))
    lines.append(_fmt(N_phi_alpha))

    for a in actions:
        lines.append(_section(f"N(phi_i, alpha_k={a}, phi_j)   -- raw counts"))
        lines.append(_fmt(N_phi_alpha_phi[a]))

    lines.append(_section("N(alpha_k, beta)   -- raw counts"))
    lines.append(_fmt(N_alpha_beta))

    lines.append(_section("Policy   P(alpha_k | phi_i)"))
    lines.append(_fmt(policy.round(6)))

    for a in actions:
        lines.append(_section(f"Transition   P(phi_j | phi_i, alpha_k={a})"))
        lines.append(_fmt(transition[a].round(6)))

    lines.append(_section("Penalty probabilities   c_k = P(beta=1 | alpha_k)"))
    side_by_side = pd.concat([penalty, empirical_penalty], axis=1).round(6)
    lines.append(_fmt(side_by_side))

    lines.append(_section("Consistency checks"))
    for name, ok in checks.items():
        lines.append(f"  [{'PASS' if ok else 'FAIL'}]  {name}")

    report = "\n".join(lines)

    if print_to_console:
        print(report)

    # File output (direct-run mode) -------------------------------------
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        N_phi_alpha.to_csv(out / "N_phi_alpha.csv", encoding="utf-8")
        for a, T in N_phi_alpha_phi.items():
            T.to_csv(out / f"N_phi_alpha_phi__{a}.csv", encoding="utf-8")
        N_alpha_beta.to_csv(out / "N_alpha_beta.csv", encoding="utf-8")

        policy.to_csv(out / "policy.csv", encoding="utf-8")
        for a, T in transition.items():
            T.to_csv(out / f"transition__{a}.csv", encoding="utf-8")
        side_by_side.to_csv(out / "penalty.csv", encoding="utf-8")

        (out / "report.txt").write_text(report, encoding="utf-8")
        if print_to_console:
            print(f"\nWrote outputs to: {out}")

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 1 - Estimation (VSA term project)")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help="Path to the dataset CSV (default: %(default)s)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output directory for files (default: %(default)s)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_estimation(args.csv, output_dir=args.out, print_to_console=True)
