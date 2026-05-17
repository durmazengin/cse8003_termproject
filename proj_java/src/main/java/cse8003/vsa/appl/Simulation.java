package cse8003.vsa.appl;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

/** Task 4 simulation (Python {@code simulation.py}). */
public final class Simulation {

    public record SimulationResult(
            int nSyntheticRows,
            Map<String, Double> absDiffAction,
            Map<String, Double> absDiffPhiSrc,
            Map<String, Double> absDiffPhiDst,
            Map<String, Double> absDiffPenaltyPerAction,
            double absDiffPenaltyGlobal,
            double l1Action,
            double l1PhiSrc,
            double l1PhiDst,
            List<Util.VsaRow> syntheticRows,
            String source,
            int randomSeed) {}

    private Simulation() {}

    public static SimulationResult runSimulation(
            Util.EstimationResult estimation,
            Path csvPath,
            boolean printToConsole,
            int randomSeed) throws IOException {

        List<Util.VsaRow> realRows = Util.loadVsaDataset(csvPath);
        Random rng = new Random(randomSeed);
        List<Util.VsaRow> simRows = simulateBlock(realRows, estimation, rng);

        Map<String, Double> pActionData = marginalAction(realRows);
        Map<String, Double> pActionSim = marginalAction(simRows);
        Map<String, Double> absAction = alignedAbsDiff(pActionData, pActionSim);

        Map<String, Double> pSrcData = marginalState(realRows, true);
        Map<String, Double> pSrcSim = marginalState(simRows, true);
        Map<String, Double> absSrc = alignedAbsDiff(pSrcData, pSrcSim);

        Map<String, Double> pDstData = marginalState(realRows, false);
        Map<String, Double> pDstSim = marginalState(simRows, false);
        Map<String, Double> absDst = alignedAbsDiff(pDstData, pDstSim);

        Map<String, Double> penData = Util.meanBetaByAction(realRows);
        Map<String, Double> penSim = Util.meanBetaByAction(simRows);
        Map<String, Double> absPen = alignedAbsDiff(penData, penSim);

        double gData = realRows.stream().mapToInt(Util.VsaRow::beta).average().orElse(0.0);
        double gSim = simRows.stream().mapToInt(Util.VsaRow::beta).average().orElse(0.0);
        double absG = Math.abs(gData - gSim);

        double l1A = absAction.values().stream().mapToDouble(Double::doubleValue).sum();
        double l1Src = absSrc.values().stream().mapToDouble(Double::doubleValue).sum();
        double l1Dst = absDst.values().stream().mapToDouble(Double::doubleValue).sum();

        SimulationResult result = new SimulationResult(
                simRows.size(),
                absAction,
                absSrc,
                absDst,
                absPen,
                absG,
                l1A,
                l1Src,
                l1Dst,
                simRows,
                "estimation-result",
                randomSeed);

        if (printToConsole) {
            printReport(result, realRows.size(), gData, gSim);
        }
        return result;
    }

    private static List<Util.VsaRow> simulateBlock(
            List<Util.VsaRow> realRows, Util.EstimationResult model, Random rng) {

        record SeqMeta(int seqId, int length, String phi0) {}

        Map<Integer, SeqMeta> meta = new LinkedHashMap<>();
        realRows.stream()
                .sorted(Comparator.comparingInt(Util.VsaRow::sequenceId).thenComparingInt(Util.VsaRow::step))
                .forEach(r -> {
                    if (!meta.containsKey(r.sequenceId())) {
                        meta.put(r.sequenceId(), new SeqMeta(r.sequenceId(), 0, r.phiSrc()));
                    }
                });
        Map<Integer, Long> lengths = realRows.stream()
                .collect(Collectors.groupingBy(Util.VsaRow::sequenceId, Collectors.counting()));
        for (var e : lengths.entrySet()) {
            SeqMeta old = meta.get(e.getKey());
            meta.put(e.getKey(), new SeqMeta(old.seqId(), e.getValue().intValue(), old.phi0()));
        }

        List<String> states = model.states();
        List<String> actions = model.actions();
        List<Util.VsaRow> out = new ArrayList<>();

        for (SeqMeta seq : meta.values()) {
            String phi = seq.phi0();
            if (!model.policy().rowLabels().contains(phi)) {
                phi = states.get(0);
            }
            for (int step = 1; step <= seq.length(); step++) {
                String a = sampleFromRow(rng, actions, model.policy(), phi);
                Util.Table2D tmat = model.transition().get(a);
                List<String> dstLabels = tmat.colLabels();
                String phiNext;
                if (tmat.rowLabels().contains(phi)) {
                    phiNext = sampleFromRow(rng, dstLabels, tmat, phi);
                } else {
                    double[] uniform = new double[dstLabels.size()];
                    double p = 1.0 / dstLabels.size();
                    for (int i = 0; i < uniform.length; i++) {
                        uniform[i] = p;
                    }
                    phiNext = sampleFromProbs(rng, dstLabels, uniform);
                }
                double pPen = Math.min(1.0, Math.max(0.0, model.penalty().getOrDefault(a, 0.0)));
                int beta = rng.nextDouble() < pPen ? 1 : 0;
                out.add(new Util.VsaRow(seq.seqId(), step, phi, phiNext, a, beta));
                phi = phiNext;
            }
        }
        return out;
    }

    private static String sampleFromRow(
            Random rng, List<String> labels, Util.Table2D table, String row) {
        double[] probs = new double[labels.size()];
        for (int i = 0; i < labels.size(); i++) {
            probs[i] = table.get(row, labels.get(i));
        }
        return sampleFromProbs(rng, labels, probs);
    }

    private static String sampleFromProbs(Random rng, List<String> labels, double[] probs) {
        double sum = 0.0;
        for (double p : probs) {
            sum += p;
        }
        if (sum < 1e-15) {
            double u = 1.0 / labels.size();
            for (int i = 0; i < probs.length; i++) {
                probs[i] = u;
            }
        } else {
            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sum;
            }
        }
        double u = rng.nextDouble();
        double c = 0.0;
        for (int i = 0; i < labels.size(); i++) {
            c += probs[i];
            if (u < c) {
                return labels.get(i);
            }
        }
        return labels.get(labels.size() - 1);
    }

    private static Map<String, Double> marginalAction(List<Util.VsaRow> rows) {
        Map<String, Long> counts = rows.stream()
                .collect(Collectors.groupingBy(Util.VsaRow::alpha, Collectors.counting()));
        long total = rows.size();
        Map<String, Double> out = new LinkedHashMap<>();
        counts.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .forEach(e -> out.put(e.getKey(), (double) e.getValue() / total));
        return out;
    }

    private static Map<String, Double> marginalState(List<Util.VsaRow> rows, boolean src) {
        Map<String, Long> counts = rows.stream()
                .collect(Collectors.groupingBy(r -> src ? r.phiSrc() : r.phiDst(), Collectors.counting()));
        long total = rows.size();
        Map<String, Double> out = new LinkedHashMap<>();
        counts.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .forEach(e -> out.put(e.getKey(), (double) e.getValue() / total));
        return out;
    }

    private static Map<String, Double> alignedAbsDiff(Map<String, Double> data, Map<String, Double> sim) {
        LinkedHashSet<String> keys = new LinkedHashSet<>();
        keys.addAll(data.keySet());
        keys.addAll(sim.keySet());
        Map<String, Double> out = new LinkedHashMap<>();
        for (String k : keys.stream().sorted().collect(Collectors.toList())) {
            out.put(k, Math.abs(data.getOrDefault(k, 0.0) - sim.getOrDefault(k, 0.0)));
        }
        return out;
    }

    private static void printReport(SimulationResult r, int realRows, double gData, double gSim) {
        StringBuilder lines = new StringBuilder();
        lines.append(Util.section("Task 4 — Simulation   (source: estimation-result, seed=" + r.randomSeed() + ")"));
        lines.append("Real rows: ").append(realRows).append("  |  Synthetic rows: ").append(r.nSyntheticRows()).append("\n\n");
        lines.append("|P_data − P_sim| — action distribution\n");
        lines.append(Util.formatSeries(Util.CLMN_ALPHA, r.absDiffAction())).append("\n");
        lines.append(String.format("  L1 sum of abs diffs (actions): %.6f%n%n", r.l1Action()));
        lines.append("|P_data − P_sim| — phi_src visitation\n");
        lines.append(Util.formatSeries(Util.CLMN_PHI_SRC, r.absDiffPhiSrc())).append("\n");
        lines.append(String.format("  L1 sum: %.6f%n%n", r.l1PhiSrc()));
        lines.append("|P_data − P_sim| — phi_dst visitation\n");
        lines.append(Util.formatSeries(Util.CLMN_PHI_DST, r.absDiffPhiDst())).append("\n");
        lines.append(String.format("  L1 sum: %.6f%n%n", r.l1PhiDst()));
        lines.append("|mean_beta_data − mean_beta_sim| per action (penalty rate)\n");
        lines.append(Util.formatSeries(Util.CLMN_ALPHA, r.absDiffPenaltyPerAction())).append("\n");
        double l1Pen = r.absDiffPenaltyPerAction().values().stream().mapToDouble(Double::doubleValue).sum();
        lines.append(String.format("  L1 sum (per-action |Δ|): %.6f%n%n", l1Pen));
        lines.append(String.format("Global penalty rate  |Δ|:  data=%.6f  sim=%.6f  |Δ|=%.6f%n", gData, gSim, r.absDiffPenaltyGlobal()));
        System.out.print(lines);
    }

}
