# proj_java — VSA term project (Java port)

Plain-Java port of `../proj_python/` (no pandas/NumPy). Package: **`cse8003.vsa.appl`**.

## Class ↔ Python module

| Java | Python |
|------|--------|
| `Util` | `util.py` |
| `Fingerprint` | `fingerprint.py` |
| `Estimation` | `estimation.py` |
| `OptimalActionIdentification` | `optimal_action_identification.py` |
| `EpsilonOptimalityAnalysis` | `e_optimality_analysis.py` |
| `Simulation` | `simulation.py` |
| `Main` | `main.py` |

## Run

From this directory (expects `../Engin-vsa_dataset_full.csv`):

```bash
gradle run
```

Or after generating the wrapper: `./gradlew run` (Unix) / `gradlew.bat run` (Windows).

Run **Main** from the IDE (UTF-8 console) and copy the full log into `../report/experiments_java.txt` if you need a saved Java run for the report.

Default config: 20% perturbation, 10 trials, RNG seed 42 (`java.util.Random`).

## Notes

- JDK 17+ required; **no third-party libraries** for the application (JUnit only for tests).
- Uses **`java.util.Random`** for perturbation row removal and Task 4 simulation. Results will **not** match NumPy/Python run-for-run; Tasks 1–3 on the same data remain deterministic and aligned.
- Optional CSV output dirs from standalone Python runs are not implemented yet.
