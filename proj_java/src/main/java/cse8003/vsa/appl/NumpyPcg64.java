package cse8003.vsa.appl;

import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.apache.commons.rng.UniformRandomProvider;

/**
 * NumPy {@code PCG64} bit generator (PCG-XSL-RR 128/64), matching
 * {@code numpy.random.default_rng(seed)} random/double draws.
 */
final class NumpyPcg64 implements UniformRandomProvider {

    private static final UInt128 MULTIPLIER =
            UInt128.of(2549297995355413924L, 4865540595714422341L);
    private static final double DOUBLE_UNIT = 1.0 / 9007199254740992.0;

    /** {@code SeedSequence(42).generate_state(4)} — simulation uses seed 42 only. */
    private static final long[] SEED_WORDS_42 = {
        Long.parseUnsignedLong("11465652750463011511"),
        Long.parseUnsignedLong("15382171918060459190"),
        Long.parseUnsignedLong("9018504550953525431"),
        Long.parseUnsignedLong("3703499796004394495")
    };

    private UInt128 state;
    private final UInt128 inc;
    private boolean hasUint32;
    private int uinteger;

    private NumpyPcg64(UInt128 state, UInt128 inc) {
        this.state = state;
        this.inc = inc;
    }

    static NumpyPcg64 fromSeed(long seed) {
        if (seed != 42L) {
            throw new IllegalArgumentException(
                    "Only random seed 42 is used in the VSA pipeline (got " + seed + ").");
        }
        return fromSeedWords(SEED_WORDS_42);
    }

    private static NumpyPcg64 fromSeedWords(long[] words) {
        UInt128 initState = UInt128.of(words[0], words[1]);
        UInt128 initSeq = UInt128.of(words[2], words[3]);
        long incHigh = (initSeq.high() << 1) | (initSeq.low() >>> 63);
        long incLow = (initSeq.low() << 1) | 1L;
        UInt128 inc = UInt128.of(incHigh, incLow);
        UInt128 st = UInt128.ZERO;
        st = st.multiply(MULTIPLIER).add(inc);
        st = st.add(initState);
        st = st.multiply(MULTIPLIER).add(inc);
        return new NumpyPcg64(st, inc);
    }

    @Override
    public void nextBytes(byte[] bytes) {
        for (int i = 0; i < bytes.length; i++) {
            bytes[i] = (byte) nextInt(0, 256);
        }
    }

    @Override
    public int nextInt() {
        return (int) nextLong();
    }

    @Override
    public int nextInt(int bound) {
        return (int) nextLong(bound);
    }

    @Override
    public int nextInt(int origin, int bound) {
        long n = (long) bound - origin;
        if (n <= 0) {
            throw new IllegalArgumentException("bound must be greater than origin");
        }
        return origin + (int) nextLong(n);
    }

    @Override
    public long nextLong() {
        return nextUInt64();
    }

    @Override
    public long nextLong(long bound) {
        if (bound <= 0) {
            throw new IllegalArgumentException("bound must be positive");
        }
        long threshold = (-bound) % bound;
        for (;;) {
            long r = nextUInt64();
            if (Long.compareUnsigned(r, threshold) >= 0) {
                return r % bound;
            }
        }
    }

    @Override
    public double nextDouble() {
        return (nextUInt64() >>> 11) * DOUBLE_UNIT;
    }

    @Override
    public float nextFloat() {
        return (float) nextDouble();
    }

    @Override
    public boolean nextBoolean() {
        return (nextUInt64() & 1L) != 0;
    }

    /** Matches NumPy {@code Generator.binomial(1, p)} (not {@code random() < p}). */
    int binomial1(double p) {
        if (p <= 0.0) {
            return 0;
        }
        if (p >= 1.0) {
            return 1;
        }
        return nextDouble() > (1.0 - p) ? 1 : 0;
    }

    private long nextUInt64() {
        if (hasUint32) {
            hasUint32 = false;
            return uinteger & 0xffffffffL;
        }
        state = state.multiply(MULTIPLIER).add(inc);
        return UInt128.rotateRight(state.high() ^ state.low(), (int) (state.high() >>> 58));
    }

    private static final class UInt128 {
        private static final UInt128 ZERO = new UInt128(0, 0);
        private static final UInt128 ONE = new UInt128(0, 1);

        private final long high;
        private final long low;

        private UInt128(long high, long low) {
            this.high = high;
            this.low = low;
        }

        static UInt128 of(long high, long low) {
            return new UInt128(high, low);
        }

        long high() {
            return high;
        }

        long low() {
            return low;
        }

        UInt128 or(UInt128 other) {
            return new UInt128(high | other.high, low | other.low);
        }

        UInt128 add(UInt128 other) {
            long newLow = low + other.low;
            long carry = Long.compareUnsigned(newLow, low) < 0 ? 1L : 0L;
            return new UInt128(high + other.high + carry, newLow);
        }

        UInt128 multiply(UInt128 other) {
            return fromBigInteger(toBigInteger().multiply(other.toBigInteger()));
        }

        static long rotateRight(long value, int rot) {
            return Long.rotateRight(value, rot);
        }

        private BigInteger toBigInteger() {
            byte[] bytes = new byte[16];
            ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putLong(0, high).putLong(8, low);
            return new BigInteger(1, bytes);
        }

        private static UInt128 fromBigInteger(BigInteger v) {
            byte[] bytes = new byte[16];
            byte[] src = v.toByteArray();
            if (src.length > 16) {
                src = java.util.Arrays.copyOfRange(src, src.length - 16, src.length);
            }
            System.arraycopy(src, 0, bytes, 16 - src.length, src.length);
            ByteBuffer buf = ByteBuffer.wrap(bytes);
            return new UInt128(buf.getLong(0), buf.getLong(8));
        }
    }
}
