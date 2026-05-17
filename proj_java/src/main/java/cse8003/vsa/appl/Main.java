package cse8003.vsa.appl;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Orchestrator (Python {@code main.py}). */
public final class Main {

    public record PipelineRunResult(
            String label,
            int nRows,
            String alphaStar,
            double cStar,
            Util.Table2D cKPerState,
            double epsilon,
            double minPAlphaStar,
            double l1Action,
            double l1PhiSrc,
            double l1PhiDst) {}

    private Main() {}

    public static void main(String[] args) throws IOException {
        Util.configureUtf8Stdout();
        loadAll();
    }

    public static List<PipelineRunResult> loadAll() throws IOException {
        return runAll(
                Util.DEFAULT_CSV,
                20,
                10,
                42,
                true);
    }

    public static List<PipelineRunResult> runAll(
            Path csvPath,
            double perturbationRate,
            int experimentCount,
            int randomSeed,
            boolean printToConsole) throws IOException {

        if (perturbationRate < 0 || perturbationRate >= 100) {
            throw new IllegalArgumentException("perturbation_rate must be in [0, 100).");
        }
        if (experimentCount < 1) {
            throw new IllegalArgumentException("experiment_count must be >= 1.");
        }

        csvPath = csvPath.toAbsolutePath().normalize();
        List<Util.VsaRow> dfFull = Util.loadVsaDataset(csvPath);

        Fingerprint.runFingerprint(csvPath, printToConsole);

        PipelineRunResult baseline = runTasks1To4(
                dfFull,
                csvPath,
                csvPath.getFileName() + " (baseline)",
                printToConsole);
        List<PipelineRunResult> results = new ArrayList<>();
        results.add(baseline);
        if (printToConsole) {
            printRunResults(baseline);
        }

        if (perturbationRate > 0) {
            if (experimentCount != PerturbationPlanData.DROP_INDICES.length) {
                throw new IllegalStateException(
                        "experiment_count must be " + PerturbationPlanData.DROP_INDICES.length
                                + " to match Python perturbation_plan (re-run generate_perturbation_plan.py).");
            }
            for (int t = 0; t < experimentCount; t++) {
                DropResult dropped = dropRowsFromPlan(dfFull, t);
                PipelineRunResult perturbed = runTasks1To4(
                        dropped.rows(),
                        csvPath,
                        csvPath.getFileName() + " (trial " + t + ", removed " + dropped.removed() + ")",
                        printToConsole);
                results.add(perturbed);
                if (printToConsole) {
                    printRunResults(perturbed);
                }
            }
            if (printToConsole) {
                printExperimentsSummary(csvPath, dfFull.size(), perturbationRate, experimentCount, baseline, results.subList(1, results.size()));
                printCKPerStateRuns(results);
            }
        } else if (printToConsole) {
            printCKPerStateRuns(List.of(baseline));
        }

        return results;
    }

    private record DropResult(List<Util.VsaRow> rows, int removed) {}

    /** Same row removals as Python (indices from {@link PerturbationPlanData}). */
    private static DropResult dropRowsFromPlan(List<Util.VsaRow> rows, int trial) {
        if (rows.size() != PerturbationPlanData.N_ROWS) {
            throw new IllegalStateException(
                    "Expected " + PerturbationPlanData.N_ROWS + " rows, got " + rows.size());
        }
        int[] drop = PerturbationPlanData.DROP_INDICES[trial];
        Set<Integer> dropSet = new HashSet<>();
        for (int i : drop) {
            dropSet.add(i);
        }
        List<Util.VsaRow> out = new ArrayList<>();
        for (int i = 0; i < rows.size(); i++) {
            if (!dropSet.contains(i)) {
                out.add(rows.get(i));
            }
        }
        return new DropResult(out, drop.length);
    }

    private static PipelineRunResult runTasks1To4(
            List<Util.VsaRow> rows,
            Path csvPath,
            String datasetLabel,
            boolean printToConsole) throws IOException {

        Util.EstimationResult estimation = Estimation.runEstimation(rows, datasetLabel, printToConsole, false);
        Util.OptimalActionResult optimal = OptimalActionIdentification.runOptimalAction(estimation, printToConsole);
        EpsilonOptimalityAnalysis.EpsilonOptimalityResult epsilonRes =
                EpsilonOptimalityAnalysis.runEpsilonOptimality(estimation, optimal, printToConsole);
        Simulation.SimulationResult simRes =
                Simulation.runSimulation(estimation, csvPath, printToConsole, 42);

        return new PipelineRunResult(
                datasetLabel,
                rows.size(),
                optimal.alphaStar(),
                optimal.cStar(),
                estimation.penaltyPerState().copy(),
                epsilonRes.epsilon(),
                epsilonRes.minPAlphaStar(),
                simRes.l1Action(),
                simRes.l1PhiSrc(),
                simRes.l1PhiDst());
    }

    private static void printRunResults(PipelineRunResult run) {
        System.out.println("\n" + "=".repeat(72));
        System.out.println("main: " + run.label() + " — Tasks 1--4 finished");
        System.out.println("=".repeat(72));
        System.out.println("  rows processed     : " + run.nRows());
        System.out.printf("  optimal action     : alpha* = %s  (c* = %.6f)%n", run.alphaStar(), run.cStar());
        System.out.printf(
                "  epsilon (Task 3)   : ε = %.6f  (min P(α*|φ) = %.6f)%n",
                run.epsilon(), run.minPAlphaStar());
        System.out.printf(
                "  simulation L1      : actions=%.4f  phi_src=%.4f  phi_dst=%.4f%n",
                run.l1Action(), run.l1PhiSrc(), run.l1PhiDst());
    }

    private static Map<String, Double> spreadAcrossStates(Util.Table2D cKPerState) {
        Map<String, Double> spread = new LinkedHashMap<>();
        for (String a : cKPerState.colLabels()) {
            spread.put(a, cKPerState.maxColumn(a) - cKPerState.minColumn(a));
        }
        return spread;
    }

    private static void printExperimentsSummary(
            Path csvPath,
            int nFull,
            double perturbationRate,
            int experimentCount,
            PipelineRunResult baseline,
            List<PipelineRunResult> perturbed) {

        StringBuilder lines = new StringBuilder();
        lines.append(Util.section(String.format(
                "Task 5 — Perturbation (%.0f%% rows removed, %d trial(s))",
                perturbationRate, experimentCount)));
        lines.append("Dataset : ").append(csvPath.toAbsolutePath()).append("\n");
        lines.append("Rows    : ").append(nFull).append(" full\n\n");
        lines.append("Baseline (full data, Tasks 1--4)\n");
        lines.append(String.format("  alpha* = %s   c* = %.6f%n", baseline.alphaStar(), baseline.cStar()));
        lines.append(String.format("  epsilon = %.6f%n%n", baseline.epsilon()));
        lines.append("Perturbed runs\n");
        for (PipelineRunResult r : perturbed) {
            lines.append(String.format(
                    "  %s:  alpha*=%s  epsilon=%.6f  L1_act=%.4f%n",
                    r.label(), r.alphaStar(), r.epsilon(), r.l1Action()));
        }
        if (perturbed.size() > 1) {
            double[] epsilons = perturbed.stream().mapToDouble(PipelineRunResult::epsilon).toArray();
            double epsMean = Arrays.stream(epsilons).average().orElse(0.0);
            lines.append(String.format(
                    "%n  epsilon mean=%.6f  std=%.6f  delta_mean_vs_baseline=%+.6f%n",
                    epsMean, stdDev(epsilons), epsMean - baseline.epsilon()));
        } else if (perturbed.size() == 1) {
            lines.append(String.format(
                    "%n  delta_epsilon (perturbed - baseline) = %+.6f%n",
                    perturbed.get(0).epsilon() - baseline.epsilon()));
        }

        Map<String, Double> baseSpread = spreadAcrossStates(baseline.cKPerState());
        List<String> actions = baseline.cKPerState().colLabels();
        lines.append("Spread across states per action  (max_phi c_k - min_phi c_k):\n  Baseline:  ");
        for (String a : actions) {
            lines.append(String.format("%s=%.6f  ", a, baseSpread.get(a)));
        }
        if (!perturbed.isEmpty()) {
            if (perturbed.size() > 1) {
                lines.append("\n  Perturbed: ");
                for (int i = 0; i < actions.size(); i++) {
                    String a = actions.get(i);
                    double[] vals = new double[perturbed.size()];
                    for (int t = 0; t < perturbed.size(); t++) {
                        vals[t] = spreadAcrossStates(perturbed.get(t).cKPerState()).get(a);
                    }
                    lines.append(String.format(
                            "%s=mean %.6f  std %.6f  ", a, Arrays.stream(vals).average().orElse(0.0), stdDev(vals)));
                }
                double[] tableDeltas = perturbed.stream()
                        .mapToDouble(r -> maxAbsTableDelta(baseline.cKPerState(), r.cKPerState()))
                        .toArray();
                lines.append(String.format(
                        "%n%n  |c_k_per_state - baseline| max cell delta:%n"
                                + "    mean=%.6f  std=%.6f  max=%.6f%n",
                        Arrays.stream(tableDeltas).average().orElse(0.0),
                        stdDev(tableDeltas),
                        Arrays.stream(tableDeltas).max().orElse(0.0)));
            } else {
                Map<String, Double> pSpread = spreadAcrossStates(perturbed.get(0).cKPerState());
                lines.append("\n  Perturbed: ");
                for (String a : actions) {
                    lines.append(String.format("%s=%.6f  ", a, pSpread.get(a)));
                }
                lines.append(String.format(
                        "%n%n  |c_k_per_state - baseline| max cell delta: %.6f%n",
                        maxAbsTableDelta(baseline.cKPerState(), perturbed.get(0).cKPerState())));
            }
        }
        System.out.println(lines);
    }

    private static double stdDev(double[] values) {
        if (values.length < 2) {
            return 0.0;
        }
        double mean = Arrays.stream(values).average().orElse(0.0);
        double sumSq = 0.0;
        for (double v : values) {
            double d = v - mean;
            sumSq += d * d;
        }
        return Math.sqrt(sumSq / (values.length - 1));
    }

    private static double maxAbsTableDelta(Util.Table2D reference, Util.Table2D other) {
        double max = 0.0;
        for (String row : reference.rowLabels()) {
            for (String col : reference.colLabels()) {
                double d = Math.abs(reference.get(row, col) - other.get(row, col));
                if (d > max) {
                    max = d;
                }
            }
        }
        return max;
    }

    private static void printCKPerStateRuns(List<PipelineRunResult> runs) {
        StringBuilder lines = new StringBuilder();
        lines.append(Util.section("Task 6 — Model assumption check"));
        lines.append("c_k(phi_i) = P(beta=1 | alpha_k, phi_i)\n");
        for (PipelineRunResult run : runs) {
            lines.append(run.label()).append("  (n=").append(run.nRows()).append(" rows):\n");
            lines.append(run.cKPerState().format(false)).append("\n\n");
        }
        System.out.print(lines);
    }
}
