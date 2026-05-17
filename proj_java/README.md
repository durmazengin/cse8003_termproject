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

Default config matches Python `load_all()`: 20% perturbation, 10 trials, seed 42.

## Notes

- JDK 17+ required.
- Console output follows the Python pipeline; optional CSV output dirs from standalone Python runs are not implemented yet.
