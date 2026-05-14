"""Dataset fingerprint (before modeling) for the VSA term project.

Computes and reports the quantities required to validate the submission
against the assignment fingerprint:

    1. Total number of rows
    2. Number of sequences
    3. Mean sequence length
    4. Empirical penalty rate per action
    5. Checksum: sum of the ``step`` column over all rows

Two run modes
-------------
Direct (writes ``report.txt`` and CSV summaries under
``./outputs/dataset_fingerprint/``, also prints to the console):

    python fingerprint.py [--csv PATH] [--out DIR]

Imported by ``main.py`` (returns the result object, also prints to the console):

    from fingerprint import run_fingerprint
    result = run_fingerprint(csv_path)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from util import (
    CLMN_ALPHA,
    CLMN_BETA,
    CLMN_SEQ_ID,
    CLMN_STEP,
    DEFAULT_CSV,
    HERE,
    load_vsa_dataset,
)

DEFAULT_OUT = HERE / "outputs" / "dataset_fingerprint"


def _section(title: str) -> str:
    bar = "=" * 72
    return f"\n{bar}\n{title}\n{bar}"


@dataclass
class FingerprintResult:
    """Structured return value of :func:`run_fingerprint`."""

    n_rows: int
    n_sequences: int
    mean_sequence_length: float
    penalty_rate_per_action: pd.Series  # index: alpha, values: mean(beta)
    checksum: int
    csv_path: str


def run_fingerprint(
    csv_path: Union[Path, str] = DEFAULT_CSV,
    *,
    output_dir: Optional[Union[Path, str]] = None,
    print_to_console: bool = True,
) -> FingerprintResult:
    """Compute dataset fingerprint metrics.

    Parameters
    ----------
    csv_path
        Path to the VSA dataset CSV (same format as Task~1).
    output_dir
        If set, write ``report.txt``, ``fingerprint_summary.csv``, and
        ``penalty_rate_per_action.csv`` under this directory. ``None`` when
        called from ``main.py`` (no files written).
    print_to_console
        Echo the human-readable report to stdout (``True`` in both modes).
    """
    csv_path = Path(csv_path)

    df = load_vsa_dataset(csv_path)

    n_rows = len(df)
    n_sequences = int(df[CLMN_SEQ_ID].nunique())
    lengths = df.groupby(CLMN_SEQ_ID, sort=False).size()
    mean_sequence_length = float(lengths.mean())

    penalty_rate_per_action = (
        df.groupby(CLMN_ALPHA)[CLMN_BETA].mean().sort_index()
    )
    penalty_rate_per_action.index.name = CLMN_ALPHA
    penalty_rate_per_action.name = "empirical_penalty_rate"

    checksum = int(df[CLMN_STEP].sum())

    result = FingerprintResult(
        n_rows=n_rows,
        n_sequences=n_sequences,
        mean_sequence_length=mean_sequence_length,
        penalty_rate_per_action=penalty_rate_per_action.copy(),
        checksum=checksum,
        csv_path=str(csv_path.resolve()),
    )

    lines: list[str] = []
    lines.append(_section(f"Dataset fingerprint   ({csv_path.name})"))
    lines.append(f"CSV path              : {result.csv_path}")
    lines.append(f"Total number of rows  : {n_rows}")
    lines.append(f"Number of sequences   : {n_sequences}")
    lines.append(f"Mean sequence length  : {mean_sequence_length:.6f}")
    lines.append("")
    lines.append("Empirical penalty rate per action  (mean of beta given alpha):")
    lines.append(penalty_rate_per_action.round(6).to_string())
    lines.append("")
    lines.append("Checksum  (sum of step over all rows):")
    lines.append(f"  checksum = {checksum}")

    report = "\n".join(lines)
    if print_to_console:
        print(report)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary = pd.DataFrame(
            [
                {
                    "total_rows": n_rows,
                    "num_sequences": n_sequences,
                    "mean_sequence_length": mean_sequence_length,
                    "checksum": checksum,
                }
            ]
        )
        summary.to_csv(out / "fingerprint_summary.csv", index=False, encoding="utf-8")

        penalty_rate_per_action.to_frame().to_csv(
            out / "penalty_rate_per_action.csv", encoding="utf-8"
        )

        json_blob: Dict[str, Any] = {
            "csv_path": result.csv_path,
            "total_rows": n_rows,
            "num_sequences": n_sequences,
            "mean_sequence_length": mean_sequence_length,
            "checksum": checksum,
            "penalty_rate_per_action": penalty_rate_per_action.round(10).to_dict(),
        }
        (out / "fingerprint.json").write_text(
            json.dumps(json_blob, indent=2), encoding="utf-8"
        )

        (out / "report.txt").write_text(report, encoding="utf-8")
        if print_to_console:
            print(f"\nWrote outputs to: {out}")

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dataset fingerprint (VSA term project)"
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to the dataset CSV (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for files (default: %(default)s)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_fingerprint(args.csv, output_dir=args.out, print_to_console=True)
