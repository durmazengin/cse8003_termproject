package cse8003.vsa.appl;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Task 3 epsilon-optimality (Python {@code e_optimality_analysis.py}). */
public final class EpsilonOptimalityAnalysis {

    public record EpsilonOptimalityResult(
            double epsilon,
            double minPAlphaStar,
            String alphaStar,
            Map<String, Double> pAlphaStarPerVisitedState,
            int visitedStateCount,
            List<String> argminStates,
            Map<String, String> interpretation,
            String source) {}

    private EpsilonOptimalityAnalysis() {}

    public static EpsilonOptimalityResult runEpsilonOptimality(
            Util.EstimationResult estimation,
            Util.OptimalActionResult optimal,
            boolean printToConsole) {

        String alphaStar = optimal.alphaStar();
        if (!estimation.policy().colLabels().contains(alphaStar)) {
            throw new IllegalArgumentException(
                    "alpha* " + alphaStar + " not in policy columns " + estimation.policy().colLabels());
        }

        Map<String, Double> pPerState = new LinkedHashMap<>();
        for (String phi : estimation.states()) {
            if (estimation.countsPhiAlpha().sumRow(phi) > 0) {
                pPerState.put(phi, estimation.policy().get(phi, alphaStar));
            }
        }
        if (pPerState.isEmpty()) {
            throw new IllegalArgumentException("No visited states in count matrix; cannot compute epsilon.");
        }

        double minP = pPerState.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double epsilon = 1.0 - minP;
        List<String> argmin = new ArrayList<>();
        for (var e : pPerState.entrySet()) {
            if (Util.isClose(e.getValue(), minP)) {
                argmin.add(e.getKey());
            }
        }

        Map<String, String> interp = interpret(epsilon, pPerState);
        EpsilonOptimalityResult result = new EpsilonOptimalityResult(
                epsilon,
                minP,
                alphaStar,
                pPerState,
                pPerState.size(),
                argmin,
                interp,
                "estimation-result + optimal-result");

        if (printToConsole) {
            StringBuilder lines = new StringBuilder();
            lines.append(Util.section("Task 3 — ε-optimality   (source: estimation-result + optimal-result)"));
            lines.append("α*  = ").append(alphaStar).append("\n");
            lines.append(String.format("min_i P(α*|φ_i) over visited states = %.6f%n", minP));
            lines.append(String.format("ε   = 1 − min_i P(α*|φ_i) = %.6f%n", epsilon));
            lines.append("Visited states (count): ").append(pPerState.size()).append("\n");
            lines.append("Argmin state(s) (ties): ").append(argmin).append("\n\n");
            lines.append("P(α*|φ_i) on visited states:\n");
            pPerState.entrySet().stream()
                    .sorted(Map.Entry.comparingByKey())
                    .forEach(e -> lines.append(String.format("  %s  %.6f%n", e.getKey(), e.getValue())));
            lines.append("\n").append(Util.section("Interpretation (assignment prompts)"));
            for (var e : interp.entrySet()) {
                if (e.getKey().startsWith("spread_") || e.getKey().startsWith("std_")) {
                    continue;
                }
                lines.append("• ").append(e.getKey()).append("\n  ").append(e.getValue()).append("\n");
            }
            System.out.print(lines);
        }
        return result;
    }

    private static Map<String, String> interpret(double epsilon, Map<String, Double> pSeries) {
        double std = 0.0;
        if (pSeries.size() > 1) {
            double mean = pSeries.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            std = Math.sqrt(
                    pSeries.values().stream().mapToDouble(v -> (v - mean) * (v - mean)).sum()
                            / (pSeries.size() - 1));
        }
        double spread = pSeries.isEmpty()
                ? 0.0
                : pSeries.values().stream().mapToDouble(Double::doubleValue).max().orElse(0.0)
                        - pSeries.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);

        String strong;
        if (epsilon <= 0.05) {
            strong = "Yes — ε is small: even the least favorable (visited) state still places substantial mass on α*.";
        } else if (epsilon <= 0.25) {
            strong = "Moderately — ε is not negligible; some visited states underweight α* compared to the best state.";
        } else {
            strong = "Weakly — ε is large: at least one visited state rarely chooses α* under the empirical policy.";
        }

        String noisy;
        if (epsilon >= 0.3 || std >= 0.2) {
            noisy = "Yes — high ε and/or high spread across states suggests heterogeneous or stochastic-looking action choices.";
        } else if (std >= 0.1) {
            noisy = "Somewhat — moderate spread in P(α*|φ) across visited states.";
        } else {
            noisy = "Not particularly — P(α*|φ) is fairly tight across visited states.";
        }

        String cons;
        if (spread <= 0.08) {
            cons = "Yes — P(α*|φ) is nearly flat across visited states.";
        } else if (spread <= 0.2) {
            cons = "Partially — some variation across states but not extreme.";
        } else {
            cons = "No — behavior differs markedly across visited states (large spread in P(α*|φ)).";
        }

        Map<String, String> out = new LinkedHashMap<>();
        out.put("Is the system strongly optimal?", strong);
        out.put("Is it noisy?", noisy);
        out.put("Is behavior consistent across states?", cons);
        out.put("spread_max_minus_min_P", String.format("%.6f", spread));
        out.put("std_P_across_visited_states", String.format("%.6f", std));
        return out;
    }
}
