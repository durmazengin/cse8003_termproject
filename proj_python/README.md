# Term project — Python implementation

Open this folder in PyCharm. Interpreter: `D:\Documents\00-PhD\shared-venv`.

## Layout

- `main.py` — orchestrator. Calls each task module in order and consumes
  the returned results.
- `estimation.py` — **Task 1: Estimation.** Counts, conditional
  probabilities (policy / transition / penalty), and consistency checks.
- `requirements.txt` — `numpy`, `pandas`.

## Two run modes for each task module

Per the project convention, every task module supports both modes:

- **Run the module directly** — writes a report + per-table CSVs under
  `./outputs/<task>/`, and also prints the report to the console.
- **Imported by `main.py`** — returns its structured result to the
  caller (no files written), and still prints the report to the console.

## Usage

Default dataset is `../Engin-vsa_dataset_full.csv`.

```text
# Just Task 1, writing CSVs + report.txt to outputs/task1_estimation/
python estimation.py

# Or via the orchestrator (no files written, results passed in-process)
python main.py
```

Optional flags:

- `--csv PATH` — point at a different dataset CSV (used by `main.py` and
  `estimation.py`).
- `--out DIR` — output directory for `estimation.py`'s direct-run mode
  (default: `./outputs/task1_estimation/`).

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
