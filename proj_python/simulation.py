"""Task 4 — Simulation and validation for the VSA term project.

Uses the learned policy, transition model, and penalty probabilities to
generate synthetic interaction sequences with the **same** sequence structure
(length and starting ``phi_src``) as the real dataset, then compares:

    * action distribution
    * penalty rates (global and per action)
    * state visitation (``phi_src`` and ``phi_dst`` marginals)

For each marginal, reports **absolute probability differences**
``|P_data − P_sim|`` (per category / per action / per state as appropriate).

Two run modes
-------------
Direct (reads ``policy.csv``, ``penalty.csv``, ``transition__*.csv`` from the
Task~1 output directory and the real CSV for empirical marginals; writes
under ``./outputs/task4_simulation/``; also prints):

    python simulation.py [--csv PATH] [--policy-dir DIR] [--out DIR]

Imported by ``load_all()`` (passes ``EstimationResult`` in memory; still loads
the real CSV from ``csv_path`` for empirical marginals; also prints):

    from simulation import run_simulation
    result = run_simulation(estimation=..., csv_path=...)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from util import (
    CLMN_ALPHA,
    CLMN_BETA,
    CLMN_PHI_DST,
    CLMN_PHI_SRC,
    CLMN_SEQ_ID,
    CLMN_STEP,
    DEFAULT_CSV,
    EstimationResult,
    HERE,
    load_vsa_dataset,
)

DEFAULT_POLICY_DIR = HERE / "outputs" / "task1_estimation"
DEFAULT_OUT = HERE / "outputs" / "task4_simulation"

PENALTY_FILE = "penalty.csv"
POLICY_FILE = "policy.csv"


@dataclass
class SimulationResult:
    """Structured return value of ``run_simulation``."""

    n_synthetic_rows: int
    abs_diff_action: pd.Series
    abs_diff_phi_src: pd.Series
    abs_diff_phi_dst: pd.Series
    abs_diff_penalty_per_action: pd.Series
    abs_diff_penalty_global: float
    l1_action: float
    l1_phi_src: float
    l1_phi_dst: float
    l1_penalty_per_action: float
    synthetic_df: pd.DataFrame
    source: str
    random_seed: int


def _section(title: str) -> str:
    bar = "=" * 72
    return f"\n{bar}\n{title}\n{bar}"


def _sample_from_row(
    rng: np.random.Generator,
    labels: List[str],
    probs: np.ndarray,
) -> str:
    """Categorical sample; uniform if ``probs`` is all-zero."""
    probs = np.asarray(probs, dtype=float)
    s = probs.sum()
    if s < 1e-15:
        probs = np.ones(len(labels), dtype=float) / len(labels)
    else:
        probs = probs / s
    return str(rng.choice(labels, p=probs))


def _load_model_from_dir(policy_dir: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.Series]:
    p_path = policy_dir / POLICY_FILE
    pen_path = policy_dir / PENALTY_FILE
    if not p_path.exists():
        raise FileNotFoundError(f"Expected {p_path} (Task 1 output). Run estimation.py first.")
    if not pen_path.exists():
        raise FileNotFoundError(f"Expected {pen_path} (Task 1 output). Run estimation.py first.")

    policy = pd.read_csv(p_path, index_col=0, encoding="utf-8")
    policy.index = policy.index.astype(str)
    policy.columns = policy.columns.astype(str)

    pen_df = pd.read_csv(pen_path, index_col=0, encoding="utf-8")
    if "c_k" not in pen_df.columns:
        raise ValueError(f"{pen_path} must contain column 'c_k'.")
    penalty = pen_df["c_k"].astype(float)
    penalty.index = penalty.index.astype(str)
    penalty.name = "c_k"

    transition: Dict[str, pd.DataFrame] = {}
    for path in sorted(policy_dir.glob("transition__*.csv")):
        action = path.stem[len("transition__") :]
        tab = pd.read_csv(path, index_col=0, encoding="utf-8")
        tab.index = tab.index.astype(str)
        tab.columns = tab.columns.astype(str)
        transition[action] = tab

    missing = [a for a in policy.columns if a not in transition]
    if missing:
        raise ValueError(
            f"Missing transition file(s) for action(s) {missing} under {policy_dir}. "
            f"Found transition keys: {list(transition.keys())}."
        )
    return policy, transition, penalty


def _simulate_block(
    real_df: pd.DataFrame,
    policy: pd.DataFrame,
    transition: Dict[str, pd.DataFrame],
    penalty: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate synthetic rows matching real (sequence_id, step) layout."""
    meta = (
        real_df.sort_values([CLMN_SEQ_ID, CLMN_STEP])
        .groupby(CLMN_SEQ_ID, sort=False)
        .agg(length=(CLMN_STEP, "count"), phi0=(CLMN_PHI_SRC, "first"))
    )
    lengths = meta["length"].astype(int).to_numpy()
    phi0s = meta["phi0"].astype(str).to_numpy()
    seq_ids = meta.index.to_numpy()

    states = list(policy.index.astype(str))
    actions = list(policy.columns.astype(str))

    rows: List[dict] = []
    for sid, L, phi in zip(seq_ids, lengths, phi0s):
        phi = str(phi)
        if phi not in policy.index:
            phi = states[0]
        for step in range(1, int(L) + 1):
            prow = policy.loc[phi].to_numpy(dtype=float)
            a = _sample_from_row(rng, actions, prow)
            tmat = transition[a]
            labels = list(tmat.columns.astype(str))
            if phi in tmat.index:
                trow = tmat.loc[phi].to_numpy(dtype=float)
            else:
                trow = np.ones(len(labels), dtype=float) / len(labels)
            phi_next = _sample_from_row(rng, labels, trow)
            p_pen = float(penalty.reindex([a]).fillna(0.0).iloc[0])
            p_pen = min(max(p_pen, 0.0), 1.0)
            beta = int(rng.binomial(1, p_pen))
            rows.append(
                {
                    CLMN_SEQ_ID: int(sid),
                    CLMN_STEP: step,
                    CLMN_PHI_SRC: phi,
                    CLMN_PHI_DST: phi_next,
                    CLMN_ALPHA: a,
                    CLMN_BETA: beta,
                }
            )
            phi = phi_next

    return pd.DataFrame(rows)


def _marginal_action(df: pd.DataFrame) -> pd.Series:
    s = df[CLMN_ALPHA].value_counts(normalize=True, sort=False)
    s.name = "P"
    return s


def _marginal_state(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].value_counts(normalize=True, sort=False)
    s.name = "P"
    return s


def _penalty_per_action(df: pd.DataFrame) -> pd.Series:
    return df.groupby(CLMN_ALPHA)[CLMN_BETA].mean().sort_index()


def _aligned_abs_diff(p_data: pd.Series, p_sim: pd.Series) -> pd.Series:
    idx = sorted(set(p_data.index.astype(str)) | set(p_sim.index.astype(str)))
    d = p_data.reindex(idx, fill_value=0.0).astype(float)
    s = p_sim.reindex(idx, fill_value=0.0).astype(float)
    out = (d - s).abs()
    out.name = "abs_diff"
    return out


def run_simulation(
    estimation: Optional[EstimationResult] = None,
    *,
    csv_path: Union[Path, str, None] = None,
    policy_dir: Optional[Union[Path, str]] = None,
    output_dir: Optional[Union[Path, str]] = None,
    print_to_console: bool = True,
    random_seed: int = 42,
) -> SimulationResult:
    """Generate synthetic sequences and compare marginals to the dataset.

    Parameters
    ----------
    estimation
        Task~1 result from ``load_all()``; if ``None``, load model matrices from
        ``policy_dir``.
    csv_path
        Real dataset CSV for empirical marginals (defaults to ``util.DEFAULT_CSV``).
    policy_dir
        Task~1 output folder when ``estimation`` is ``None``.
    """
    csv_path = Path(csv_path or DEFAULT_CSV)
    real_df = load_vsa_dataset(csv_path)

    if estimation is not None:
        policy = estimation.policy
        transition = estimation.transition
        penalty = estimation.penalty
        source = "estimation-result"
    else:
        pdir = Path(policy_dir) if policy_dir is not None else DEFAULT_POLICY_DIR
        policy, transition, penalty = _load_model_from_dir(pdir)
        source = f"files: {pdir}"

    rng = np.random.default_rng(random_seed)
    sim_df = _simulate_block(real_df, policy, transition, penalty, rng)

    P_a_d = _marginal_action(real_df)
    P_a_s = _marginal_action(sim_df)
    abs_a = _aligned_abs_diff(P_a_d, P_a_s)

    P_ps_d = _marginal_state(real_df, CLMN_PHI_SRC)
    P_ps_s = _marginal_state(sim_df, CLMN_PHI_SRC)
    abs_ps = _aligned_abs_diff(P_ps_d, P_ps_s)

    P_pd_d = _marginal_state(real_df, CLMN_PHI_DST)
    P_pd_s = _marginal_state(sim_df, CLMN_PHI_DST)
    abs_pd = _aligned_abs_diff(P_pd_d, P_pd_s)

    pen_d = _penalty_per_action(real_df)
    pen_s = _penalty_per_action(sim_df)
    abs_pen_a = _aligned_abs_diff(pen_d, pen_s)

    g_d = float(real_df[CLMN_BETA].mean())
    g_s = float(sim_df[CLMN_BETA].mean())
    abs_g = abs(g_d - g_s)

    l1_a = float(abs_a.sum())
    l1_ps = float(abs_ps.sum())
    l1_pd = float(abs_pd.sum())
    l1_pen_a = float(abs_pen_a.sum())

    result = SimulationResult(
        n_synthetic_rows=len(sim_df),
        abs_diff_action=abs_a,
        abs_diff_phi_src=abs_ps,
        abs_diff_phi_dst=abs_pd,
        abs_diff_penalty_per_action=abs_pen_a,
        abs_diff_penalty_global=abs_g,
        l1_action=l1_a,
        l1_phi_src=l1_ps,
        l1_phi_dst=l1_pd,
        l1_penalty_per_action=l1_pen_a,
        synthetic_df=sim_df,
        source=source,
        random_seed=random_seed,
    )

    lines: List[str] = []
    lines.append(_section(f"Task 4 — Simulation   (source: {source}, seed={random_seed})"))
    lines.append(f"Real rows: {len(real_df)}  |  Synthetic rows: {len(sim_df)}")
    lines.append("")
    lines.append("|P_data − P_sim| — action distribution")
    lines.append(abs_a.round(6).to_string())
    lines.append(f"  L1 sum of abs diffs (actions): {l1_a:.6f}")
    lines.append("")
    lines.append("|P_data − P_sim| — phi_src visitation")
    lines.append(abs_ps.round(6).to_string())
    lines.append(f"  L1 sum: {l1_ps:.6f}")
    lines.append("")
    lines.append("|P_data − P_sim| — phi_dst visitation")
    lines.append(abs_pd.round(6).to_string())
    lines.append(f"  L1 sum: {l1_pd:.6f}")
    lines.append("")
    lines.append("|mean_beta_data − mean_beta_sim| per action (penalty rate)")
    lines.append(abs_pen_a.round(6).to_string())
    lines.append(f"  L1 sum (per-action |Δ|): {l1_pen_a:.6f}")
    lines.append("")
    lines.append(f"Global penalty rate  |Δ|:  data={g_d:.6f}  sim={g_s:.6f}  |Δ|={abs_g:.6f}")

    report = "\n".join(lines)
    if print_to_console:
        print(report)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        pd.concat(
            [
                P_a_d.rename("P_data"),
                P_a_s.rename("P_sim"),
                abs_a.rename("abs_diff"),
            ],
            axis=1,
        ).to_csv(out / "marginal_action.csv", encoding="utf-8")

        pd.concat(
            [
                P_ps_d.rename("P_data"),
                P_ps_s.rename("P_sim"),
                abs_ps.rename("abs_diff"),
            ],
            axis=1,
        ).to_csv(out / "marginal_phi_src.csv", encoding="utf-8")

        pd.concat(
            [
                P_pd_d.rename("P_data"),
                P_pd_s.rename("P_sim"),
                abs_pd.rename("abs_diff"),
            ],
            axis=1,
        ).to_csv(out / "marginal_phi_dst.csv", encoding="utf-8")

        pd.concat(
            [
                pen_d.rename("penalty_rate_data"),
                pen_s.rename("penalty_rate_sim"),
                abs_pen_a.rename("abs_diff"),
            ],
            axis=1,
        ).to_csv(out / "penalty_rate_per_action.csv", encoding="utf-8")

        pd.DataFrame(
            [
                {
                    "abs_diff_penalty_global": abs_g,
                    "l1_action": l1_a,
                    "l1_phi_src": l1_ps,
                    "l1_phi_dst": l1_pd,
                    "l1_penalty_per_action": l1_pen_a,
                    "random_seed": random_seed,
                }
            ]
        ).to_csv(out / "simulation_summary.csv", index=False, encoding="utf-8")

        sim_df.to_csv(out / "synthetic_sequences.csv", index=False, encoding="utf-8")
        (out / "report.txt").write_text(report, encoding="utf-8")
        if print_to_console:
            print(f"\nWrote outputs to: {out}")

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 4 — Simulation (VSA term project)")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Real dataset CSV")
    p.add_argument(
        "--policy-dir",
        type=Path,
        default=DEFAULT_POLICY_DIR,
        help="Task 1 output directory (default: %(default)s)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_simulation(
        estimation=None,
        csv_path=args.csv,
        policy_dir=args.policy_dir,
        output_dir=args.out,
        print_to_console=True,
        random_seed=args.seed,
    )
