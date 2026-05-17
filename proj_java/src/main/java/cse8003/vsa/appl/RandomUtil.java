package cse8003.vsa.appl;

import org.apache.commons.rng.UniformRandomProvider;

/** RNG aligned with NumPy {@code numpy.random.default_rng} (PCG64). */
public final class RandomUtil {

    private RandomUtil() {}

    public static NumpyPcg64 numpyDefaultRng(long seed) {
        return NumpyPcg64.fromSeed(seed);
    }
}
