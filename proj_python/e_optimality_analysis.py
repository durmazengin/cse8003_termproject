"""Task 3 — ε-optimality analysis for the VSA term project.

Estimates (assignment Task 3):

    ε = 1 − min_i P(α* | φ_i)

where α* is the optimal action from Task 2 (minimal c_k) and P(α|φ) is the
policy from Task 1. Only states with at least one observed visit (positive
row total in N(φ_i, ·)) are included in the minimum, so never-visited states
do not dominate ε with artificial zeros.

Two run modes
-------------
Direct (reads ``policy.csv`` and ``N_phi_alpha.csv`` from Task 1 output dir,
``alpha_star.csv`` from Task 2 output dir; writes under
``./outputs/task3_epsilon_optimality/``; also prints to the console):

    python e_optimality_analysis.py [--policy-dir DIR] [--optimal-dir DIR] [--out DIR]

Imported by ``load_all()`` (receives ``EstimationResult`` and
``OptimalActionResult`` in memory; no disk read for those; also prints):

    from e_optimality_analysis import run_epsilon_optimality
    result = run_epsilon_optimality(estimation=..., optimal=...)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from util import EstimationResult, HERE, OptimalActionResult

DEFAULT_POLICY_DIR = HERE / "outputs" / "task1_estimation"
DEFAULT_OPTIMAL_DIR = HERE / "outputs" / "task2_optimal_action"
DEFAULT_OUT = HERE / "outputs" / "task3_epsilon_optimality"

POLICY_FILE = "policy.csv"
N_PHI_ALPHA_FILE = "N_phi_alpha.csv"
ALPHA_STAR_FILE = "alpha_star.csv"


@dataclass
class EpsilonOptimalityResult:
    """Structured return value of ``run_epsilon_optimality``."""

    epsilon: float
    min_p_alpha_star: float
    alpha_star: str
    p_alpha_star_per_visited_state: pd.Series
    visited_state_count: int
    argmin_states: List[str]
    interpretation: Dict[str, str]
    source: str


def _section(title: str) -> str:
    bar = "=" * 72
    return f"\n{bar}\n{title}\n{bar}"


def _interpret(epsilon: float, p_series: pd.Series) -> Dict[str, str]:

    # epsilon = 1.0 - min_p

    """Short qualitative answers for the assignment bullet prompts."""
    std = float(p_series.std()) if len(p_series) > 1 else 0.0
    spread = float(p_series.max() - p_series.min()) if len(p_series) > 0 else 0.0

    if epsilon <= 0.05:
        strong = "Yes — ε is small: even the least favorable (visited) state "
        strong += "still places substantial mass on α*."
    elif epsilon <= 0.25:
        strong = "Moderately — ε is not negligible; some visited states "
        strong += "underweight α* compared to the best state."
    else:
        strong = "Weakly — ε is large: at least one visited state rarely "
        strong += "chooses α* under the empirical policy."

    if epsilon >= 0.3 or std >= 0.2:
        noisy = "Yes — high ε and/or high spread across states suggests "
        noisy += "heterogeneous or stochastic-looking action choices."
    elif std >= 0.1:
        noisy = "Somewhat — moderate spread in P(α*|φ) across visited states."
    else:
        noisy = "Not particularly — P(α*|φ) is fairly tight across visited states."

    if spread <= 0.08:
        cons = "Yes — P(α*|φ) is nearly flat across visited states."
    elif spread <= 0.2:
        cons = "Partially — some variation across states but not extreme."
    else:
        cons = "No — behavior differs markedly across visited states "
        cons += "(large spread in P(α*|φ))."

    return {
        "Is the system strongly optimal?": strong,
        "Is it noisy?": noisy,
        "Is behavior consistent across states?": cons,
        "spread_max_minus_min_P": f"{spread:.6f}",
        "std_P_across_visited_states": f"{std:.6f}",
    }


def _compute_core(
    policy: pd.DataFrame,
    counts_phi_alpha: pd.DataFrame,
    alpha_star: str,
) -> Tuple[float, float, pd.Series, List[str]]:
    if alpha_star not in policy.columns:
        raise ValueError(
            f"α* column {alpha_star!r} not in policy columns {list(policy.columns)}. "
            "Regenerate policy.csv / run Task 1 after Task 2."
        )
    row_totals = counts_phi_alpha.sum(axis=1)
    visited = row_totals > 0
    if not bool(visited.any()):
        raise ValueError("No visited states in count matrix; cannot compute ε.")

    p_col = policy.loc[visited, alpha_star].astype(float)
    min_p = float(np.min(p_col.values))
    epsilon = 1.0 - min_p
    tol = 1e-9
    argmin_states = [str(s) for s in p_col.index[np.isclose(p_col.values, min_p, atol=tol)]]
    return epsilon, min_p, p_col, argmin_states


def _load_alpha_star_from_csv(optimal_dir: Path) -> str:
    path = optimal_dir / ALPHA_STAR_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {path} (from Task 2). Run optimal_action_identification.py first."
        )
    df = pd.read_csv(path, index_col=0, encoding="utf-8")
    if "value" not in df.columns:
        raise ValueError(f"{path} must have a 'value' column (found {list(df.columns)}).")
    if "alpha_star" not in df.index:
        raise ValueError(f"{path} must index row 'alpha_star' (found {list(df.index)}).")
    return str(df.loc["alpha_star", "value"])


def _load_policy_and_counts(policy_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p_path = policy_dir / POLICY_FILE
    n_path = policy_dir / N_PHI_ALPHA_FILE
    if not p_path.exists():
        raise FileNotFoundError(
            f"Expected {p_path} (from Task 1). Run estimation.py first."
        )
    if not n_path.exists():
        raise FileNotFoundError(
            f"Expected {n_path} (from Task 1). Run estimation.py first."
        )
    policy = pd.read_csv(p_path, index_col=0, encoding="utf-8")
    counts = pd.read_csv(n_path, index_col=0, encoding="utf-8")
    return policy, counts


def run_epsilon_optimality(
    estimation: Optional[EstimationResult] = None,
    optimal: Optional[OptimalActionResult] = None,
    *,
    policy_dir: Optional[Union[Path, str]] = None,
    optimal_dir: Optional[Union[Path, str]] = None,
    output_dir: Optional[Union[Path, str]] = None,
    print_to_console: bool = True,
) -> EpsilonOptimalityResult:
    """Compute ε = 1 − min_i P(α*|φ_i) over visited states.

    When ``estimation`` and ``optimal`` are provided (``load_all()`` path),
    policy and counts come from memory. Otherwise files are read from
    ``policy_dir`` (default: Task 1 output) and ``optimal_dir`` (default: Task 2
    output).
    """
    use_memory = estimation is not None and optimal is not None
    use_files = estimation is None and optimal is None
    if not use_memory and not use_files:
        raise ValueError(
            "Pass both `estimation` and `optimal`, or pass neither and use "
            "`policy_dir` / `optimal_dir` to load from disk."
        )

    if estimation is not None and optimal is not None:
        policy = estimation.policy
        counts_phi_alpha = estimation.counts_phi_alpha
        alpha_star = optimal.alpha_star
        source = "estimation-result + optimal-result"
    else:
        pdir = Path(policy_dir) if policy_dir is not None else DEFAULT_POLICY_DIR
        odir = Path(optimal_dir) if optimal_dir is not None else DEFAULT_OPTIMAL_DIR
        policy, counts_phi_alpha = _load_policy_and_counts(pdir)
        alpha_star = _load_alpha_star_from_csv(odir)
        source = f"files: {pdir / POLICY_FILE}, {odir / ALPHA_STAR_FILE}"

    epsilon, min_p, p_series, argmin_states = _compute_core(
        policy, counts_phi_alpha, alpha_star
    )
    visited_n = int((counts_phi_alpha.sum(axis=1) > 0).sum())
    interp = _interpret(epsilon, p_series)

    result = EpsilonOptimalityResult(
        epsilon=epsilon,
        min_p_alpha_star=min_p,
        alpha_star=alpha_star,
        p_alpha_star_per_visited_state=p_series.copy(),
        visited_state_count=visited_n,
        argmin_states=argmin_states,
        interpretation=interp,
        source=source,
    )

    lines: List[str] = []
    lines.append(_section(f"Task 3 — ε-optimality   (source: {source})"))
    lines.append(f"α*  = {alpha_star}")
    lines.append(f"min_i P(α*|φ_i) over visited states = {min_p:.6f}")
    lines.append(f"ε   = 1 − min_i P(α*|φ_i) = {epsilon:.6f}")
    lines.append(f"Visited states (count): {visited_n}")
    lines.append(f"Argmin state(s) (ties): {argmin_states}")
    lines.append("")
    lines.append("P(α*|φ_i) on visited states:")
    lines.append(p_series.sort_index().round(6).to_string())
    lines.append("")
    lines.append(_section("Interpretation (assignment prompts)"))
    for k, v in interp.items():
        if k.startswith("spread_") or k.startswith("std_"):
            continue
        lines.append(f"• {k}")
        lines.append(f"  {v}")
    lines.append("")
    lines.append(
        f"Diagnostics: spread(max−min P)={interp['spread_max_minus_min_P']}, "
        f"std(P)={interp['std_P_across_visited_states']}"
    )

    report = "\n".join(lines)
    if print_to_console:
        print(report)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(
            [
                {
                    "epsilon": epsilon,
                    "min_p_alpha_star": min_p,
                    "alpha_star": alpha_star,
                    "visited_state_count": visited_n,
                    "argmin_states": ";".join(argmin_states),
                }
            ]
        )
        summary.to_csv(out / "epsilon_summary.csv", index=False, encoding="utf-8")
        p_series.to_frame(name="P_alpha_star_given_phi").to_csv(
            out / "p_alpha_star_per_state.csv", encoding="utf-8"
        )
        pd.DataFrame(
            {"question": list(interp.keys()), "answer": list(interp.values())}
        ).to_csv(out / "interpretation.csv", index=False, encoding="utf-8")
        (out / "report.txt").write_text(report, encoding="utf-8")
        if print_to_console:
            print(f"\nWrote outputs to: {out}")

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task 3 — ε-optimality (VSA term project)"
    )
    p.add_argument(
        "--policy-dir",
        type=Path,
        default=DEFAULT_POLICY_DIR,
        help="Directory with policy.csv and N_phi_alpha.csv (default: %(default)s)",
    )
    p.add_argument(
        "--optimal-dir",
        type=Path,
        default=DEFAULT_OPTIMAL_DIR,
        help="Directory with alpha_star.csv (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory (default: %(default)s)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_epsilon_optimality(
        estimation=None,
        optimal=None,
        policy_dir=args.policy_dir,
        optimal_dir=args.optimal_dir,
        output_dir=args.out,
        print_to_console=True,
    )
