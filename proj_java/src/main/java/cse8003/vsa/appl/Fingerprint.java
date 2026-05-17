package cse8003.vsa.appl;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** Dataset fingerprint (Python {@code fingerprint.py}). */
public final class Fingerprint {

    public record FingerprintResult(
            int nRows,
            int nSequences,
            double meanSequenceLength,
            Map<String, Double> penaltyRatePerAction,
            long checksum,
            String csvPath) {}

    private Fingerprint() {}

    public static FingerprintResult runFingerprint(Path csvPath, boolean printToConsole) throws IOException {
        csvPath = csvPath.toAbsolutePath().normalize();
        List<Util.VsaRow> rows = Util.loadVsaDataset(csvPath);

        int nRows = rows.size();
        Map<Integer, Long> lengths = rows.stream()
                .collect(Collectors.groupingBy(Util.VsaRow::sequenceId, Collectors.counting()));
        int nSequences = lengths.size();
        double meanLen = lengths.values().stream().mapToLong(Long::longValue).average().orElse(0.0);

        Map<String, Double> penaltyRates = new LinkedHashMap<>();
        Util.meanBetaByAction(rows).entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .forEach(e -> penaltyRates.put(e.getKey(), e.getValue()));

        long checksum = rows.stream().mapToLong(Util.VsaRow::step).sum();

        FingerprintResult result = new FingerprintResult(
                nRows, nSequences, meanLen, penaltyRates, checksum, csvPath.toString());

        if (printToConsole) {
            StringBuilder sb = new StringBuilder();
            sb.append(Util.section("Dataset fingerprint   (" + csvPath.getFileName() + ")"));
            sb.append("CSV path              : ").append(result.csvPath()).append("\n");
            sb.append("Total number of rows  : ").append(nRows).append("\n");
            sb.append("Number of sequences   : ").append(nSequences).append("\n");
            sb.append(String.format("Mean sequence length  : %.6f%n", meanLen));
            sb.append("\nEmpirical penalty rate per action  (mean of beta given alpha):\n");
            sb.append(Util.formatSeries(Util.CLMN_ALPHA, penaltyRates));
            sb.append("\nChecksum  (sum of step over all rows):\n");
            sb.append("  checksum = ").append(checksum);
            System.out.println(sb);
        }
        return result;
    }
}
