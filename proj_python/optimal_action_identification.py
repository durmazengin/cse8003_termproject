"""Task 2 - Optimal Action Identification for the VSA term project.

Identifies the optimal action under the P-model environment:

    alpha* = arg min_k  c_k

where c_k = P(beta = 1 | alpha_k) is the action's penalty probability
(computed in Task 1).

Two run modes
-------------
Direct run (reads ``penalty.csv`` produced by ``estimation.py`` and writes
its own report + CSV to ``./outputs/task2_optimal_action/``, also prints
to the console):

    python optimal_action_identification.py [--input DIR] [--out DIR]

Imported by ``main`` (receives the ``EstimationResult`` returned by
``run_estimation``, returns the optimal-action result, also prints to
the console):

    from optimal_action_identification import run_optimal_action
    result = run_optimal_action(estimation)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from util import CLMN_ALPHA, EstimationResult, HERE, OptimalActionResult


DEFAULT_INPUT = HERE / "outputs" / "task1_estimation"
DEFAULT_OUT = HERE / "outputs" / "task2_optimal_action"

PENALTY_FILE = "penalty.csv"


def _section(title: str) -> str:
    bar = "=" * 72
    return f"\n{bar}\n{title}\n{bar}"


def _load_penalty_from_files(input_dir: Path) -> pd.Series:
    """Read the c_k column from ``penalty.csv`` written by ``estimation.py``."""

    path = input_dir / PENALTY_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {path} (produced by `python estimation.py`). "
            f"Run estimation.py first, or pass --input pointing at its output dir."
        )
    df = pd.read_csv(path, index_col=0, encoding="utf-8")
    if "c_k" not in df.columns:
        raise ValueError(
            f"{path} is missing a 'c_k' column (found {list(df.columns)})."
        )
    series = df["c_k"].astype(float)
    series.index.name = CLMN_ALPHA
    series.name = "c_k"
    return series


def run_optimal_action(
    estimation: Optional[EstimationResult] = None,
    *,
    input_dir: Optional[Union[Path, str]] = None,
    output_dir: Optional[Union[Path, str]] = None,
    print_to_console: bool = True,
) -> OptimalActionResult:
    """Compute alpha* = argmin_k c_k.

    Parameters
    ----------
    estimation : ``EstimationResult`` returned by ``run_estimation``.
        Passed by ``main.run_all``. When ``None`` (direct-run mode), ``c_k`` is
        read from ``<input_dir>/penalty.csv`` instead.
    input_dir : directory containing Task 1's output files. Only used
        when ``estimation`` is ``None``. Defaults to
        ``./outputs/task1_estimation/``.
    output_dir : if given, also write ``ranking.csv``,
        ``alpha_star.csv`` and ``report.txt`` here. When ``None``
        (the path used by ``main.run_all``) nothing is written to disk.
    print_to_console : whether to echo the human-readable report. ``True``
        in both modes per the task spec.

    Returns
    -------
    OptimalActionResult
    """
    if estimation is not None:
        penalty = estimation.penalty.astype(float).copy()
        penalty.index.name = CLMN_ALPHA
        penalty.name = "c_k"
        source = "estimation-result"
    else:
        in_dir = Path(input_dir) if input_dir is not None else DEFAULT_INPUT
        penalty = _load_penalty_from_files(in_dir)
        source = str((in_dir / PENALTY_FILE).resolve())

    if penalty.empty:
        raise ValueError("Penalty vector is empty; nothing to optimize over.")

    c_min = float(penalty.min())
    ties: List[str] = [str(a) for a in penalty.index[np.isclose(penalty.values, c_min)]]
    alpha_star = ties[0]  # deterministic pick: first action tied at the minimum

    ranking = (
        penalty.sort_values(ascending=True)
               .to_frame()
               .assign(rank=lambda d: np.arange(1, len(d) + 1, dtype=int))
    )
    ranking["is_optimal"] = np.isclose(ranking["c_k"].values, c_min)
    ranking.index.name = CLMN_ALPHA

    result = OptimalActionResult(
        alpha_star=alpha_star,
        c_star=c_min,
        ties=ties,
        ranking=ranking,
        penalty=penalty,
        source=source,
    )

    # Human-readable report ------------------------------------------------
    lines: List[str] = []
    lines.append(_section(f"Task 2 - Optimal Action Identification   (source: {source})"))
    lines.append("Penalty probabilities  c_k = P(beta = 1 | alpha_k):")
    lines.append(penalty.round(6).to_string())

    lines.append("")
    lines.append(f"alpha*  = {alpha_star}")
    lines.append(f"c_*     = {c_min:.6f}")
    if len(ties) > 1:
        lines.append(f"Note   : {len(ties)} actions tie at the minimum: {ties}. "
                     f"Picked the first (alphabetical) one.")

    lines.append(_section("Ranking by c_k (ascending)"))
    lines.append(ranking.round(6).to_string())

    report = "\n".join(lines)
    if print_to_console:
        print(report)

    # File output (direct-run mode) ---------------------------------------
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ranking.to_csv(out / "ranking.csv", encoding="utf-8")

        alpha_star_df = pd.DataFrame(
            {"value": [alpha_star, f"{c_min:.10g}"]},
            index=pd.Index(["alpha_star", "c_star"], name="quantity"),
        )
        alpha_star_df.to_csv(out / "alpha_star.csv", encoding="utf-8")

        (out / "report.txt").write_text(report, encoding="utf-8")
        if print_to_console:
            print(f"\nWrote outputs to: {out}")

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task 2 - Optimal Action Identification (VSA term project)"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help="Directory containing estimation.py outputs "
                        "(default: %(default)s)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output directory for files (default: %(default)s)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_optimal_action(
        estimation=None,
        input_dir=args.input,
        output_dir=args.out,
        print_to_console=True,
    )
