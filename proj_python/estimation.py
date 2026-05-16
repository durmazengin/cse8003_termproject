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
    result = run_estimation(dataframe)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from util import (
    CLMN_ALPHA,
    CLMN_BETA,
    CLMN_PHI_DST,
    CLMN_PHI_SRC,
    DEFAULT_CSV,
    EstimationResult,
    HERE,
    load_vsa_dataset,
)


DEFAULT_OUT = HERE / "outputs" / "task1_estimation"


def _section(title: str) -> str:

    # ========================================================================  (72 ='s)
    # title
    # ========================================================================  (72 ='s)
    bar = "=" * 72
    return f"\n{bar}\n{title}\n{bar}"


def _fmt(obj) -> str:
    return obj.to_string()


def run_estimation(
    dataframe: pd.DataFrame,
    *,
    dataset_label: Optional[str] = None,
    output_dir: Optional[Union[Path, str]] = None,
    print_to_console: bool = True,
) -> EstimationResult:
    """Compute counts, conditional probabilities, and consistency checks.

    Parameters
    ----------
    dataframe
        Interaction-sequence data (same columns as ``load_vsa_dataset``).
        ``main.run_all`` (or ``__main__`` here) loads the CSV and passes the frame.
    dataset_label
        Short name for the report header (e.g. CSV filename). Defaults to
        ``n=<row count> rows``.
    output_dir : if given, also write per-table CSVs and a ``report.txt``
        summary to this directory. When ``None`` (the default, used by
        ``main.run_all``) nothing is written to disk.
    print_to_console : whether to echo the human-readable report to stdout.
        ``True`` in both run modes per the task spec.

    Returns
    -------
    EstimationResult
        Counts, probabilities, penalty vector, and a dict of check flags.
    """
    df = dataframe.copy()
    if dataset_label is None:
        dataset_label = f"n={len(df)} rows"

    states = sorted(set(df[CLMN_PHI_SRC].unique()) | set(df[CLMN_PHI_DST].unique()))
    actions = sorted(df[CLMN_ALPHA].unique())
    betas = [0, 1]

    # count by grouping how many action alpha taken in state phi
    N_phi_alpha = (
        df.groupby([CLMN_PHI_SRC, CLMN_ALPHA]).size()
          .unstack(fill_value=0)
          .reindex(index=states, columns=actions, fill_value=0)
    )
    N_phi_alpha.index.name = CLMN_PHI_SRC
    N_phi_alpha.columns.name = CLMN_ALPHA

    # count by grouping how many transitions from state phi_src to state phi_dst for action a
    N_phi_alpha_phi: Dict[str, pd.DataFrame] = {}
    for a in actions:
        tab = (
            df[df[CLMN_ALPHA] == a]
              .groupby([CLMN_PHI_SRC, CLMN_PHI_DST]).size()
              .unstack(fill_value=0)
              .reindex(index=states, columns=states, fill_value=0)
        )
        tab.index.name = CLMN_PHI_SRC
        tab.columns.name = CLMN_PHI_DST
        N_phi_alpha_phi[a] = tab

    # count by grouping how many action alpha valuated with beta
    N_alpha_beta = (
        df.groupby([CLMN_ALPHA, CLMN_BETA]).size()
          .unstack(fill_value=0)
          .reindex(index=actions, columns=betas, fill_value=0)
    )
    N_alpha_beta.index.name = CLMN_ALPHA
    N_alpha_beta.columns.name = CLMN_BETA

    # Conditional probabilities -----------------------------------------
    # N_phi_alpha       : how many action alpha taken in state phi
    # N_phi_alpha_phi   : how many transitions from state phi_src to state phi_dst for action a
    # N_alpha_beta      : how many action alpha valuated with beta

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
        df.groupby(CLMN_ALPHA)[CLMN_BETA].mean().reindex(actions, fill_value=0.0)
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
    lines.append(_section(f"Task 1 - Estimation   (dataset: {dataset_label})"))
    lines.append(f"Rows in dataset : {len(df)}")
    lines.append(f"States observed : {states}")
    lines.append(f"Actions observed: {actions}")

    # how many action k taken in state i
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
    df = load_vsa_dataset(args.csv)
    run_estimation(
        df,
        dataset_label=Path(args.csv).name,
        output_dir=args.out,
        print_to_console=True,
    )
