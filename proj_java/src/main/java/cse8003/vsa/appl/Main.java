package cse8003.vsa.appl;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

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
            double fraction = perturbationRate / 100.0;
            Random rngMaster = new Random(randomSeed);
            for (int t = 0; t < experimentCount; t++) {
                int trialSeed = rngMaster.nextInt(Integer.MAX_VALUE);
                Random rng = new Random(trialSeed);
                DropResult dropped = dropRowsRandomly(dfFull, fraction, rng);
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

    private static DropResult dropRowsRandomly(List<Util.VsaRow> rows, double fractionRemove, Random rng) {
        int n = rows.size();
        if (n <= 1 || fractionRemove <= 0.0) {
            return new DropResult(Util.copyRows(rows), 0);
        }
        int nRemove = Math.min(Math.max(0, (int) Math.round(fractionRemove * n)), n - 1);
        if (nRemove <= 0) {
            return new DropResult(Util.copyRows(rows), 0);
        }
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, rng);
        List<Integer> drop = indices.subList(0, nRemove);
        List<Util.VsaRow> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (!drop.contains(i)) {
                out.add(rows.get(i));
            }
        }
        return new DropResult(out, nRemove);
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
        Map<String, Double> baseSpread = spreadAcrossStates(baseline.cKPerState());
        lines.append("\nSpread across states per action  (max_phi c_k - min_phi c_k):\n  Baseline:  ");
        baseSpread.forEach((a, v) -> lines.append(String.format("%s=%.6f  ", a, v)));
        System.out.println(lines);
    }

    private static void printCKPerStateRuns(List<PipelineRunResult> runs) {
        StringBuilder lines = new StringBuilder();
        lines.append(Util.section("Task 6 — Model assumption check"));
        lines.append("c_k(phi_i) = P(beta=1 | alpha_k, phi_i)\n");
        for (PipelineRunResult run : runs) {
            lines.append(run.label()).append("  (n=").append(run.nRows()).append(" rows):\n");
            lines.append(run.cKPerState()).append("\n\n");
        }
        System.out.print(lines);
    }
}
