# Term project — Python implementation

Open this folder in PyCharm. Interpreter: `D:\Documents\00-PhD\shared-venv`.

## Layout

- `util.py` — shared helpers: default CSV path, column name constants,
  `load_vsa_dataset()`, ``EstimationResult`` (Task~1), and ``OptimalActionResult`` (Task~2).
- `main.py` — orchestrator: fingerprint, load CSV, **baseline** Tasks~1--4 on full data;
  optional perturbed passes when ``--perturbation-rate > 0`` (Task~5).
- `fingerprint.py` — **Dataset fingerprint (before modeling).** Row count,
  sequence count, mean sequence length, empirical penalty rate per action,
  and checksum $\sum \text{step}$.
- `estimation.py` — **Task 1: Estimation.** Counts, conditional
  probabilities (policy / transition / penalty), and consistency checks.
- `optimal_action_identification.py` — **Task 2: Optimal Action
  Identification.** Picks $\alpha^{*} = \alpha_{\arg\min_k c_k}$ from the
  penalty vector produced in Task 1.
- `e_optimality_analysis.py` — **Task 3: ε-optimality.** Computes
  $\varepsilon = 1 - \min_i P(\alpha^{*} \mid \phi_i)$ over visited states;
  writes interpretation text for the assignment prompts.
- `simulation.py` — **Task 4: Simulation.** Generates synthetic sequences from
  the learned policy / transitions / $c_k$, compares marginals to the real CSV,
  and reports $|P_{\text{data}} - P_{\text{sim}}|$.
- `requirements.txt` — `numpy`, `pandas`.

## Two run modes for each task module

Per the project convention, every task module supports both modes:

- **Run the module directly** — writes a report + per-table CSVs under
  `./outputs/<task>/`, and also prints the report to the console.
- **Imported by `main.run_all` only** — returns structured results in memory; task
  reports print on the baseline pass when ``perturbation_rate=0``.

## Usage

Default dataset is `../Engin-vsa_dataset_full.csv`.

```text
# Dataset fingerprint alone — writes outputs/dataset_fingerprint/
python fingerprint.py

# Task 1 alone — writes CSVs + report.txt to outputs/task1_estimation/
python estimation.py

# Task 2 alone — reads outputs/task1_estimation/penalty.csv, writes its own
# files to outputs/task2_optimal_action/  (must run estimation.py first)
python optimal_action_identification.py

# Task 3 alone — reads policy.csv + N_phi_alpha.csv (Task 1) and alpha_star.csv (Task 2)
python e_optimality_analysis.py

# Task 4 alone — reads policy/transition/penalty from Task 1 outputs + real CSV for marginals
python simulation.py

# Baseline only — Tasks 1--4 on full data (default)
python main.py

# Task 5 — baseline + 20% row removal, 5 trials
python main.py --perturbation-rate 20 --experiment-count 5
```

Optional flags for **`main.py`**:

- `--perturbation-rate PCT` — percent of rows removed per perturbed trial (default `0`).
- `--experiment-count N` — perturbed trials when rate &gt; 0 (default `1`).
- `--seed` — RNG seed (default `42`).
- `--csv PATH` — dataset CSV.

Other modules:

- `--csv PATH` — dataset CSV path (used by `main.py`, `fingerprint.py`,
  `estimation.py`, and `simulation.py`; default lives in `util.DEFAULT_CSV`).
- `--out DIR` — per-task output directory in direct-run mode
  (defaults: `outputs/dataset_fingerprint/`, `outputs/task1_estimation/`,
  `outputs/task2_optimal_action/`, `outputs/task3_epsilon_optimality/`,
  `outputs/task4_simulation/`).
- `--policy-dir` — Task~1 output folder: used with `--optimal-dir` by
  `e_optimality_analysis.py`, and alone by `simulation.py` (real marginals
  still from `--csv`). Default: `outputs/task1_estimation/`.
- `--optimal-dir` — Task~2 output folder (for `e_optimality_analysis.py` only;
  default: `outputs/task2_optimal_action/`).
- `--seed` — RNG seed for `simulation.py` (default: 42).
- `--in DIR` — for `optimal_action_identification.py` only: where to
  find `penalty.csv` (defaults to `estimation.py`'s output directory).

## Dataset fingerprint outputs

When `fingerprint.py` is run directly it writes to `outputs/dataset_fingerprint/`:

- `fingerprint_summary.csv` — one row: `total_rows`, `num_sequences`,
  `mean_sequence_length`, `checksum`.
- `penalty_rate_per_action.csv` — mean of $\beta$ per action (empirical
  penalty rate).
- `fingerprint.json` — same metrics plus the per-action rates as JSON.
- `report.txt` — full human-readable console output.

## Task 1 outputs

When `estimation.py` is run directly it writes:

- `N_phi_alpha.csv` — raw counts $N(\phi_i, \alpha_k)$.
- `N_phi_alpha_phi__<alpha>.csv` — one per action, the
  $N(\phi_i, \alpha_k, \phi_j)$ slice for that $\alpha_k$.
- `N_alpha_beta.csv` — raw counts $N(\alpha_k, \beta)$.
- `policy.csv` — $P(\alpha_k \mid \phi_i)$.
- `transition__<alpha>.csv` — one per action, $P(\phi_j \mid \phi_i, \alpha_k)$.
- `penalty.csv` — $c_k = P(\beta = 1 \mid \alpha_k)$ side-by-side with
  the empirical penalty rate.
- `report.txt` — full human-readable console output.

## Task 2 outputs

When `optimal_action_identification.py` is run directly it writes to
`outputs/task2_optimal_action/`:

- `ranking.csv` — actions sorted ascending by $c_k$, with rank and an
  `is_optimal` flag.
- `alpha_star.csv` — two rows: `alpha_star` and `c_star`.
- `report.txt` — full human-readable console output.

## Task 3 outputs

When `e_optimality_analysis.py` is run directly it writes to
`outputs/task3_epsilon_optimality/`:

- `epsilon_summary.csv` — one row: `epsilon`, `min_p_alpha_star`, `alpha_star`,
  `visited_state_count`, `argmin_states`.
- `p_alpha_star_per_state.csv` — $P(\alpha^{*} \mid \phi_i)$ for each visited state.
- `interpretation.csv` — qualitative answers plus numeric diagnostics.
- `report.txt` — full human-readable console output.

## Task 4 outputs

When `simulation.py` is run directly it writes to `outputs/task4_simulation/`:

- `marginal_action.csv`, `marginal_phi_src.csv`, `marginal_phi_dst.csv` —
  $P_{\text{data}}$, $P_{\text{sim}}$, and $|P_{\text{data}} - P_{\text{sim}}|$
  per category.
- `penalty_rate_per_action.csv` — empirical penalty rate vs simulated, plus
  absolute difference per action.
- `simulation_summary.csv` — global $|\Delta\text{penalty}|$, L1 sums, seed.
- `synthetic_sequences.csv` — full synthetic dataset (same row count as real CSV).
- `report.txt` — full human-readable console output.
