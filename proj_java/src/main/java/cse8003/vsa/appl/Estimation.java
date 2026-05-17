package cse8003.vsa.appl;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Task 1 estimation (Python {@code estimation.py}). */
public final class Estimation {

    private static final String PENALTY_CONS_PASS = "c_k matches empirical frequencies";
    private static final String PENALTY_CONS_FAIL = "c_k does not match empirical frequencies";

    private Estimation() {}

    public static Util.EstimationResult runEstimation(
            List<Util.VsaRow> rows,
            String datasetLabel,
            boolean printToConsole,
            boolean printPenaltyPerState) {

        if (datasetLabel == null) {
            datasetLabel = "n=" + rows.size() + " rows";
        }

        List<String> states = Util.sortedStates(rows);
        List<String> actions = Util.sortedActions(rows);
        List<String> betas = List.of("0", "1");

        Util.Table2D nPhiAlpha = Util.Table2D.zeros(states, actions);
        Map<String, Util.Table2D> nPhiAlphaPhi = new LinkedHashMap<>();
        for (String a : actions) {
            nPhiAlphaPhi.put(a, Util.Table2D.zeros(states, states));
        }

        Map<String, Map<String, double[]>> phiAlphaBeta = new LinkedHashMap<>();
        for (Util.VsaRow r : rows) {
            nPhiAlpha.increment(r.phiSrc(), r.alpha());
            nPhiAlphaPhi.get(r.alpha()).increment(r.phiSrc(), r.phiDst());
            phiAlphaBeta
                    .computeIfAbsent(r.phiSrc(), k -> new LinkedHashMap<>())
                    .computeIfAbsent(r.alpha(), k -> new double[2]);
            phiAlphaBeta.get(r.phiSrc()).get(r.alpha())[r.beta()] += 1.0;
        }

        Util.Table2D nAlphaBeta = Util.Table2D.zeros(actions, betas);
        for (Util.VsaRow r : rows) {
            nAlphaBeta.increment(r.alpha(), String.valueOf(r.beta()));
        }

        Util.Table2D policy = nPhiAlpha.copy().divideByRowSums();
        Map<String, Util.Table2D> transition = new LinkedHashMap<>();
        for (String a : actions) {
            transition.put(a, nPhiAlphaPhi.get(a).copy().divideByRowSums());
        }

        Map<String, Double> penalty = new LinkedHashMap<>();
        Map<String, Double> empiricalPenalty = new LinkedHashMap<>(Util.meanBetaByAction(rows));
        for (String a : actions) {
            double total = nAlphaBeta.get(a, "0") + nAlphaBeta.get(a, "1");
            penalty.put(a, total == 0.0 ? 0.0 : nAlphaBeta.get(a, "1") / total);
            empiricalPenalty.putIfAbsent(a, 0.0);
        }

        Util.Table2D penaltyPerState = Util.Table2D.zeros(states, actions);
        for (int ri = 0; ri < states.size(); ri++) {
            String phi = states.get(ri);
            for (int ci = 0; ci < actions.size(); ci++) {
                String a = actions.get(ci);
                double[] c = phiAlphaBeta.getOrDefault(phi, Map.of()).getOrDefault(a, new double[2]);
                double total = c[0] + c[1];
                penaltyPerState.setAt(ri, ci, total == 0.0 ? 0.0 : c[1] / total);
            }
        }

        boolean policyRowsOk = true;
        for (String phi : states) {
            if (nPhiAlpha.sumRow(phi) > 0 && !Util.isClose(policy.sumRow(phi), 1.0)) {
                policyRowsOk = false;
                break;
            }
        }

        boolean transitionRowsOk = true;
        outer:
        for (String a : actions) {
            Util.Table2D n = nPhiAlphaPhi.get(a);
            Util.Table2D t = transition.get(a);
            for (String phi : states) {
                if (n.sumRow(phi) > 0 && !Util.isClose(t.sumRow(phi), 1.0)) {
                    transitionRowsOk = false;
                    break outer;
                }
            }
        }

        boolean penaltyMatches = true;
        for (String a : actions) {
            if (!Util.isClose(penalty.get(a), empiricalPenalty.getOrDefault(a, 0.0))) {
                penaltyMatches = false;
                break;
            }
        }

        Map<String, Boolean> checks = new LinkedHashMap<>();
        checks.put("sum_k P(alpha_k | phi_i) == 1", policyRowsOk);
        checks.put("sum_j P(phi_j | phi_i, alpha_k) == 1", transitionRowsOk);
        checks.put("c_k matches empirical frequencies", penaltyMatches);

        Util.EstimationResult result = new Util.EstimationResult(
                nPhiAlpha,
                nPhiAlphaPhi,
                nAlphaBeta,
                policy,
                transition,
                penalty,
                penaltyPerState,
                empiricalPenalty,
                checks,
                states,
                actions,
                rows.size());

        if (printToConsole) {
            printReport(result, datasetLabel, printPenaltyPerState);
        }
        return result;
    }

    private static void printReport(
            Util.EstimationResult result, String datasetLabel, boolean printPenaltyPerState) {
        StringBuilder lines = new StringBuilder();
        lines.append(Util.section("Task 1 - Estimation   (dataset: " + datasetLabel + ")"));
        lines.append("Rows in dataset : ").append(result.nRows()).append("\n");
        lines.append("States observed : ").append(result.states()).append("\n");
        lines.append("Actions observed: ").append(result.actions()).append("\n");

        lines.append(Util.section("N(phi_i, alpha_k)   -- raw counts"));
        lines.append(result.countsPhiAlpha().format(true)).append("\n");

        for (String a : result.actions()) {
            lines.append(Util.section("N(phi_i, alpha_k=" + a + ", phi_j)   -- raw counts"));
            lines.append(result.countsPhiAlphaPhi().get(a).format(true)).append("\n");
        }

        lines.append(Util.section("N(alpha_k, beta)   -- raw counts"));
        lines.append(result.countsAlphaBeta().format(true)).append("\n");

        lines.append(Util.section("Policy   P(alpha_k | phi_i)"));
        lines.append(formatSumConsTable(result.policy(), result.countsPhiAlpha())).append("\n");

        for (String a : result.actions()) {
            lines.append(Util.section("Transition   P(phi_j | phi_i, alpha_k=" + a + ")"));
            Util.Table2D n = result.countsPhiAlphaPhi().get(a);
            lines.append(formatSumConsTable(result.transition().get(a), n)).append("\n");
        }

        lines.append(Util.section("Penalty probabilities   c_k = P(beta=1 | alpha_k)"));
        lines.append(formatPenaltyTable(result.penalty(), result.empiricalPenalty())).append("\n");

        if (printPenaltyPerState) {
            lines.append(Util.section("Task 6 — Model assumption check"));
            lines.append("c_k(phi_i) = P(beta=1 | alpha_k, phi_i)\n");
            lines.append(result.penaltyPerState().format(false)).append("\n");
        }

        lines.append(Util.section("Consistency checks"));
        for (var e : result.checks().entrySet()) {
            lines.append(String.format("  [%s]  %s%n", e.getValue() ? "PASS" : "FAIL", e.getKey()));
        }
        System.out.print(lines);
    }

    private static String formatSumConsTable(Util.Table2D prob, Util.Table2D rowTotals) {
        List<String> rows = prob.rowLabels();
        List<String> cols = prob.colLabels();
        int idxW = Math.max(4, rows.stream().mapToInt(String::length).max().orElse(0));
        int colW = 10;
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-" + idxW + "s", ""));
        for (String c : cols) {
            sb.append(String.format("%" + (colW + 2) + "s", c));
        }
        sb.append(String.format("%" + (colW + 2) + "s", "SUM"));
        sb.append(String.format("%" + 8 + "s", "Cons."));
        sb.append("\n");
        for (String row : rows) {
            sb.append(String.format("%-" + idxW + "s", row));
            double sum = 0.0;
            for (String c : cols) {
                double v = prob.get(row, c);
                sum += v;
                sb.append(String.format("%" + (colW + 2) + ".6f", v));
            }
            boolean active = rowTotals.sumRow(row) > 0.0;
            String sumCell;
            String consCell;
            if (!active) {
                sumCell = sum == 0.0 ? "0" : String.format("%.6f", sum);
                consCell = "-";
            } else if (Util.isClose(sum, 1.0)) {
                sumCell = "1";
                consCell = "PASS";
            } else {
                sumCell = String.format("%.6f", sum);
                consCell = "FAIL";
            }
            sb.append(String.format("%" + (colW + 2) + "s", sumCell));
            sb.append(String.format("%" + 10 + "s", consCell));
            sb.append("\n");
        }
        return sb.toString().stripTrailing();
    }

    private static String formatPenaltyTable(Map<String, Double> penalty, Map<String, Double> empirical) {
        int valW = 12;
        List<String> actions = new java.util.ArrayList<>(penalty.keySet());
        int idxW = Math.max(5, actions.stream().mapToInt(String::length).max().orElse(0));
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%" + idxW + "s  %" + valW + "s  %" + valW + "s  %s%n", "", "c_k", "empirical", "Cons."));
        sb.append(String.format("%" + idxW + "s  %" + valW + "s  %" + valW + "s%n", "alpha", "", "penalty"));
        for (String a : actions) {
            double ck = penalty.get(a);
            double emp = empirical.getOrDefault(a, 0.0);
            String cons = Util.isClose(ck, emp) ? PENALTY_CONS_PASS : PENALTY_CONS_FAIL;
            sb.append(String.format("%" + idxW + "s  %" + valW + ".6f  %" + valW + ".6f  %s%n", a, ck, emp, cons));
        }
        return sb.toString();
    }
}
