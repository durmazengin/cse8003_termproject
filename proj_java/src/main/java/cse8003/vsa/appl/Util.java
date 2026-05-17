package cse8003.vsa.appl;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/** Shared helpers (Python {@code util.py}). */
public final class Util {

    public static final Path HERE = Paths.get(System.getProperty("user.dir")).toAbsolutePath().normalize();
    public static final Path TERM_PROJECT_DIR = HERE.getParent();
    public static final Path DEFAULT_CSV = TERM_PROJECT_DIR.resolve("Engin-vsa_dataset_full.csv");

    public static final String CLMN_SEQ_ID = "sequence_id";
    public static final String CLMN_STEP = "step";
    public static final String CLMN_PHI_SRC = "phi_src";
    public static final String CLMN_PHI_DST = "phi_dst";
    public static final String CLMN_ALPHA = "alpha";
    public static final String CLMN_BETA = "beta";

    private static final double ATOL = 1e-9;

    private Util() {}

    public record VsaRow(int sequenceId, int step, String phiSrc, String phiDst, String alpha, int beta) {}

    public record EstimationResult(
            Table2D countsPhiAlpha,
            Map<String, Table2D> countsPhiAlphaPhi,
            Table2D countsAlphaBeta,
            Table2D policy,
            Map<String, Table2D> transition,
            Map<String, Double> penalty,
            Table2D penaltyPerState,
            Map<String, Double> empiricalPenalty,
            Map<String, Boolean> checks,
            List<String> states,
            List<String> actions,
            int nRows) {}

    public record OptimalActionResult(
            String alphaStar,
            double cStar,
            List<String> ties,
            List<RankRow> ranking,
            Map<String, Double> penalty,
            String source) {}

    public record RankRow(String alpha, double ck, int rank, boolean optimal) {}

    public static boolean isClose(double a, double b) {
        return Math.abs(a - b) <= ATOL;
    }

    public static String section(String title) {
        String bar = "=".repeat(72);
        return "\n" + bar + "\n" + title + "\n" + bar + "\n";
    }

    /** Use UTF-8 for stdout (Greek state/action labels in the dataset). */
    public static void configureUtf8Stdout() {
        System.setOut(new java.io.PrintStream(System.out, true, StandardCharsets.UTF_8));
    }

    public static List<VsaRow> loadVsaDataset(Path csvPath) throws IOException {
        List<VsaRow> rows = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                String[] parts = line.split(",", -1);
                if (parts.length < 6) {
                    throw new IOException("Expected 6 columns, got " + parts.length + ": " + line);
                }
                rows.add(new VsaRow(
                        Integer.parseInt(parts[0].trim()),
                        Integer.parseInt(parts[1].trim()),
                        parts[2].trim(),
                        parts[3].trim(),
                        parts[4].trim(),
                        Integer.parseInt(parts[5].trim())));
            }
        }
        return Collections.unmodifiableList(rows);
    }

    public static List<VsaRow> copyRows(List<VsaRow> rows) {
        return new ArrayList<>(rows);
    }

    public static List<String> sortedStates(List<VsaRow> rows) {
        Set<String> states = new LinkedHashSet<>();
        for (VsaRow r : rows) {
            states.add(r.phiSrc());
            states.add(r.phiDst());
        }
        return states.stream().sorted().collect(Collectors.toList());
    }

    public static List<String> sortedActions(List<VsaRow> rows) {
        return rows.stream().map(VsaRow::alpha).distinct().sorted().collect(Collectors.toList());
    }

    public static Map<String, Double> meanBetaByAction(List<VsaRow> rows) {
        Map<String, long[]> acc = new LinkedHashMap<>();
        for (VsaRow r : rows) {
            long[] pair = acc.computeIfAbsent(r.alpha(), k -> new long[2]);
            pair[0] += r.beta();
            pair[1] += 1;
        }
        Map<String, Double> out = new LinkedHashMap<>();
        for (var e : acc.entrySet()) {
            out.put(e.getKey(), e.getValue()[1] == 0 ? 0.0 : (double) e.getValue()[0] / e.getValue()[1]);
        }
        return out;
    }

    /** Two-dimensional table with labeled rows and columns (pandas DataFrame analogue). */
    public static final class Table2D {
        private final List<String> rowLabels;
        private final List<String> colLabels;
        private final double[][] data;

        public Table2D(List<String> rowLabels, List<String> colLabels) {
            this.rowLabels = List.copyOf(rowLabels);
            this.colLabels = List.copyOf(colLabels);
            this.data = new double[rowLabels.size()][colLabels.size()];
        }

        public static Table2D zeros(List<String> rows, List<String> cols) {
            return new Table2D(rows, cols);
        }

        public List<String> rowLabels() {
            return rowLabels;
        }

        public List<String> colLabels() {
            return colLabels;
        }

        public double get(String row, String col) {
            int ri = rowLabels.indexOf(row);
            int ci = colLabels.indexOf(col);
            if (ri < 0 || ci < 0) {
                return 0.0;
            }
            return data[ri][ci];
        }

        public void increment(String row, String col) {
            int ri = rowLabels.indexOf(row);
            int ci = colLabels.indexOf(col);
            if (ri >= 0 && ci >= 0) {
                data[ri][ci] += 1.0;
            }
        }

        public void set(String row, String col, double value) {
            int ri = rowLabels.indexOf(row);
            int ci = colLabels.indexOf(col);
            if (ri >= 0 && ci >= 0) {
                data[ri][ci] = value;
            }
        }

        public void setAt(int rowIndex, int colIndex, double value) {
            data[rowIndex][colIndex] = value;
        }

        public double sumRow(String row) {
            int ri = rowLabels.indexOf(row);
            if (ri < 0) {
                return 0.0;
            }
            double s = 0.0;
            for (double v : data[ri]) {
                s += v;
            }
            return s;
        }

        public Table2D copy() {
            Table2D c = new Table2D(rowLabels, colLabels);
            for (int i = 0; i < data.length; i++) {
                System.arraycopy(data[i], 0, c.data[i], 0, data[i].length);
            }
            return c;
        }

        public Table2D divideByRowSums() {
            Table2D out = zeros(rowLabels, colLabels);
            for (String row : rowLabels) {
                double total = sumRow(row);
                int ri = rowLabels.indexOf(row);
                if (total <= 0.0) {
                    continue;
                }
                for (int ci = 0; ci < colLabels.size(); ci++) {
                    out.data[ri][ci] = data[ri][ci] / total;
                }
            }
            return out;
        }

        public Map<String, Double> column(String col) {
            int ci = colLabels.indexOf(col);
            Map<String, Double> m = new LinkedHashMap<>();
            if (ci < 0) {
                return m;
            }
            for (int ri = 0; ri < rowLabels.size(); ri++) {
                m.put(rowLabels.get(ri), data[ri][ci]);
            }
            return m;
        }

        public double maxColumn(String col) {
            return column(col).values().stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        }

        public double minColumn(String col) {
            return column(col).values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        }

        /** Pandas-style table text ({@code integerCounts=true} for raw count matrices). */
        public String format(boolean integerCounts) {
            int idxW = Math.max(4, rowLabels.stream().mapToInt(String::length).max().orElse(0));
            int colW = Math.max(6, colLabels.stream().mapToInt(String::length).max().orElse(0));
            colW = Math.max(colW, integerCounts ? 6 : 10);
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("%-" + idxW + "s", ""));
            for (String c : colLabels) {
                sb.append(String.format("%" + (colW + 2) + "s", c));
            }
            sb.append("\n");
            for (int ri = 0; ri < rowLabels.size(); ri++) {
                sb.append(String.format("%-" + idxW + "s", rowLabels.get(ri)));
                for (int ci = 0; ci < colLabels.size(); ci++) {
                    double v = data[ri][ci];
                    if (integerCounts) {
                        sb.append(String.format("%" + (colW + 2) + ".0f", v));
                    } else {
                        sb.append(String.format("%" + (colW + 2) + ".6f", v));
                    }
                }
                sb.append("\n");
            }
            return sb.toString().stripTrailing();
        }

        @Override
        public String toString() {
            return format(false);
        }
    }
}
