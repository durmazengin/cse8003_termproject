"""Shared helpers for the term-project Python scripts.

Defines column-name constants, CSV loading, ``EstimationResult`` (Task~1 output),
and ``OptimalActionResult`` (Task~2 output; used by Task~3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

# `proj_python/` directory (this file's folder)
HERE = Path(__file__).resolve().parent
# Parent is `term_project/` (contains the default CSV)
TERM_PROJECT_DIR = HERE.parent

DEFAULT_CSV = TERM_PROJECT_DIR / "Engin-vsa_dataset_full.csv"

# DataFrame / CSV column names (headerless file column order matches COLUMNS)
CLMN_SEQ_ID = "sequence_id"
CLMN_STEP = "step"
CLMN_PHI_SRC = "phi_src"
CLMN_PHI_DST = "phi_dst"
CLMN_ALPHA = "alpha"
CLMN_BETA = "beta"

COLUMNS = [
    CLMN_SEQ_ID,
    CLMN_STEP,
    CLMN_PHI_SRC,
    CLMN_PHI_DST,
    CLMN_ALPHA,
    CLMN_BETA,
]


@dataclass
class EstimationResult:
    """Structured return value of ``estimation.run_estimation``."""

    counts_phi_alpha: pd.DataFrame                    # N(phi_i, alpha_k)
    counts_phi_alpha_phi: Dict[str, pd.DataFrame]     # N(phi_i, alpha_k, phi_j) per alpha
    counts_alpha_beta: pd.DataFrame                   # N(alpha_k, beta)
    policy: pd.DataFrame                              # P(alpha_k | phi_i)
    transition: Dict[str, pd.DataFrame]               # P(phi_j | phi_i, alpha_k) per alpha
    penalty: pd.Series                                # c_k = P(beta=1 | alpha_k)
    empirical_penalty: pd.Series                      # groupby(CLMN_ALPHA)[CLMN_BETA].mean()
    checks: Dict[str, bool]                           # named pass/fail flags
    states: list
    actions: list
    n_rows: int


@dataclass
class OptimalActionResult:
    """Structured return value of ``optimal_action_identification.run_optimal_action``."""

    alpha_star: str             # the optimal action label (e.g. "α2")
    c_star: float               # its penalty probability (the minimum c_k)
    ties: List[str]             # all actions tied at the minimum (usually just [alpha_star])
    ranking: pd.DataFrame       # actions sorted by c_k (ascending), columns: c_k, rank
    penalty: pd.Series          # the c_k vector echoed back (action -> c_k)
    source: str                 # "estimation-result" or absolute path to penalty.csv


def load_vsa_dataset(csv_path: Union[Path, str]) -> pd.DataFrame:
    """Load the headerless UTF-8 VSA interaction CSV.

    Each row is
    ``(CLMN_SEQ_ID, CLMN_STEP, CLMN_PHI_SRC, CLMN_PHI_DST, CLMN_ALPHA, CLMN_BETA)``
    with integer ``CLMN_SEQ_ID``, ``CLMN_STEP``, and ``CLMN_BETA``; state and
    action columns stay as strings (e.g. Greek labels).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, header=None, names=COLUMNS, encoding="utf-8")
    df[CLMN_SEQ_ID] = df[CLMN_SEQ_ID].astype(int)
    df[CLMN_STEP] = df[CLMN_STEP].astype(int)
    df[CLMN_BETA] = df[CLMN_BETA].astype(int)
    return df
