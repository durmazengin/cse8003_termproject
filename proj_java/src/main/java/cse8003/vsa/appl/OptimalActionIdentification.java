package cse8003.vsa.appl;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Task 2 optimal action (Python {@code optimal_action_identification.py}). */
public final class OptimalActionIdentification {

    private OptimalActionIdentification() {}

    public static Util.OptimalActionResult runOptimalAction(
            Util.EstimationResult estimation, boolean printToConsole) {

        Map<String, Double> penalty = new LinkedHashMap<>(estimation.penalty());
        if (penalty.isEmpty()) {
            throw new IllegalArgumentException("Penalty vector is empty; nothing to optimize over.");
        }

        double cMin = penalty.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        List<String> ties = new ArrayList<>();
        for (var e : penalty.entrySet()) {
            if (Util.isClose(e.getValue(), cMin)) {
                ties.add(e.getKey());
            }
        }
        String alphaStar = ties.get(0);

        List<Util.RankRow> ranking = penalty.entrySet().stream()
                .sorted(Map.Entry.comparingByValue())
                .map(e -> new Util.RankRow(
                        e.getKey(),
                        e.getValue(),
                        0,
                        Util.isClose(e.getValue(), cMin)))
                .toList();
        List<Util.RankRow> ranked = new ArrayList<>();
        int rank = 1;
        for (Util.RankRow row : ranking) {
            ranked.add(new Util.RankRow(row.alpha(), row.ck(), rank++, row.optimal()));
        }

        Util.OptimalActionResult result = new Util.OptimalActionResult(
                alphaStar, cMin, ties, ranked, penalty, "estimation-result");

        if (printToConsole) {
            StringBuilder lines = new StringBuilder();
            lines.append(Util.section("Task 2 - Optimal Action Identification   (source: estimation-result)"));
            lines.append("Penalty probabilities  c_k = P(beta = 1 | alpha_k):\n");
            lines.append(Util.formatSeries(Util.CLMN_ALPHA, penalty)).append("\n");
            lines.append(String.format("%nalpha*  = %s%n", alphaStar));
            lines.append(String.format("c_*     = %.6f%n", cMin));
            if (ties.size() > 1) {
                lines.append(String.format(
                        "Note   : %d actions tie at the minimum: %s. Picked the first (alphabetical) one.%n",
                        ties.size(), Util.formatPythonList(ties)));
            }
            lines.append(Util.section("Ranking by c_k (ascending)"));
            lines.append(Util.formatRanking(ranked)).append("\n");
            System.out.print(lines);
        }
        return result;
    }
}
