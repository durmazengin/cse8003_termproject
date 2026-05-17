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

    /** Python list repr, e.g. {@code ['φ1', 'φ2']}. */
    public static String formatPythonList(List<String> items) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append('\'').append(items.get(i)).append('\'');
        }
        sb.append(']');
        return sb.toString();
    }

    public enum CellKind {
        FLOAT6,
        INT,
        BOOL,
        STRING
    }

    private static String cellToString(Object cell, CellKind kind) {
        if (cell == null) {
            return "";
        }
        return switch (kind) {
            case FLOAT6 -> String.format("%.6f", ((Number) cell).doubleValue());
            case INT -> String.valueOf(((Number) cell).intValue());
            case BOOL -> (Boolean) cell ? "True" : "False";
            case STRING -> cell.toString();
        };
    }

    /** pandas {@code Series.to_string()} for float values (6 decimal places). */
    public static String formatSeries(String indexName, Map<String, Double> values) {
        if (values.isEmpty()) {
            return indexName == null ? "" : indexName;
        }
        List<String> labels = new ArrayList<>(values.keySet());
        List<String> formatted = new ArrayList<>();
        for (String label : labels) {
            formatted.add(String.format("%.6f", values.get(label)));
        }
        int idxW = labels.stream().mapToInt(String::length).max().orElse(0);
        StringBuilder sb = new StringBuilder();
        if (indexName != null) {
            sb.append(indexName).append('\n');
        }
        for (int i = 0; i < labels.size(); i++) {
            sb.append(String.format("%-" + idxW + "s    %s%n", labels.get(i), formatted.get(i)));
        }
        return sb.toString().stripTrailing();
    }

    /**
     * pandas {@code DataFrame.to_string()} layout.
     *
     * @param colIndexNameOnFirstRow {@code true} for count/probability tables (index name on
     *     header row 1); {@code false} for ranking-style frames (index name on header row 2 only)
     */
    public static String formatDataFrame(
            String rowIndexName,
            String colIndexName,
            boolean colIndexNameOnFirstRow,
            List<String> rowLabels,
            List<String> colLabels,
            Object[][] cells,
            CellKind[] columnKinds) {
        int nRows = rowLabels.size();
        int nCols = colLabels.size();
        String[][] str = new String[nRows][nCols];
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                str[i][j] = cellToString(cells[i][j], columnKinds[j]);
            }
        }

        int idxW = rowLabels.stream().mapToInt(String::length).max().orElse(0);
        if (rowIndexName != null) {
            idxW = Math.max(idxW, rowIndexName.length());
        }
        if (colIndexNameOnFirstRow && colIndexName != null) {
            idxW = Math.max(idxW, colIndexName.length());
        }

        int[] colW = new int[nCols];
        for (int j = 0; j < nCols; j++) {
            colW[j] = colLabels.get(j).length();
            for (int i = 0; i < nRows; i++) {
                colW[j] = Math.max(colW[j], str[i][j].length());
            }
        }

        StringBuilder sb = new StringBuilder();
        if (colIndexNameOnFirstRow) {
            sb.append(String.format("%-" + idxW + "s", colIndexName == null ? "" : colIndexName));
        } else {
            sb.append(String.format("%-" + idxW + "s", ""));
        }
        for (int j = 0; j < nCols; j++) {
            sb.append(' ').append(String.format("%" + colW[j] + "s", colLabels.get(j)));
        }
        sb.append('\n');

        if (!colIndexNameOnFirstRow && rowIndexName != null) {
            sb.append(String.format("%-" + idxW + "s", rowIndexName)).append('\n');
        } else if (rowIndexName != null) {
            sb.append(String.format("%-" + idxW + "s", rowIndexName)).append('\n');
        }

        for (int i = 0; i < nRows; i++) {
            sb.append(String.format("%-" + idxW + "s", rowLabels.get(i)));
            for (int j = 0; j < nCols; j++) {
                sb.append(' ').append(String.format("%" + colW[j] + "s", str[i][j]));
            }
            sb.append('\n');
        }
        return sb.toString().stripTrailing();
    }

    /** Ranking table from Task 2 ({@code ranking.round(6).to_string()}). */
    public static String formatRanking(List<RankRow> rows) {
        List<String> rowLabels = rows.stream().map(RankRow::alpha).toList();
        List<String> colLabels = List.of("c_k", "rank", "is_optimal");
        Object[][] cells = new Object[rows.size()][3];
        for (int i = 0; i < rows.size(); i++) {
            RankRow r = rows.get(i);
            cells[i][0] = r.ck();
            cells[i][1] = r.rank();
            cells[i][2] = r.optimal();
        }
        return formatDataFrame(
                CLMN_ALPHA,
                null,
                false,
                rowLabels,
                colLabels,
                cells,
                new CellKind[] {CellKind.FLOAT6, CellKind.INT, CellKind.BOOL});
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

        /** Pandas-style table ({@code integerCounts=true} for raw count matrices). */
        public String format(boolean integerCounts, String rowIndexName, String colIndexName) {
            Object[][] cells = new Object[rowLabels.size()][colLabels.size()];
            CellKind kind = integerCounts ? CellKind.INT : CellKind.FLOAT6;
            CellKind[] kinds = new CellKind[colLabels.size()];
            java.util.Arrays.fill(kinds, kind);
            for (int ri = 0; ri < rowLabels.size(); ri++) {
                for (int ci = 0; ci < colLabels.size(); ci++) {
                    cells[ri][ci] = data[ri][ci];
                }
            }
            return formatDataFrame(
                    rowIndexName, colIndexName, true, rowLabels, colLabels, cells, kinds);
        }

        @Override
        public String toString() {
            return format(false, null, null);
        }
    }
}
