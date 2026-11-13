"""Share of p-values below 0.05 when the true mean difference is 0.

Usage: python scripts/null_rejection.py [TRIALS]   (default 1000; seeded, so repeatable)

Two nulls: normal differences, and skewed ones (exponential minus 1, mean 0).
"""

import sys

import numpy as np

from retrievalstats.stats import bootstrap, paired_t, randomisation


def main(argv):
    trials = int(argv[0]) if argv else 1000
    rng = np.random.default_rng(12345)
    print("| null | n | t | randomisation | bootstrap |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for null in ("normal", "skewed"):
        for n in (5, 10, 30):
            counts = {"t": 0, "randomisation": 0, "bootstrap": 0}
            for i in range(trials):
                d = rng.normal(0.0, 1.0, n) if null == "normal" else rng.exponential(1.0, n) - 1
                counts["t"] += paired_t(d)[1] < 0.05
                counts["randomisation"] += randomisation(d, samples=2000, seed=i)[1] < 0.05
                counts["bootstrap"] += bootstrap(d, samples=2000, seed=i)[1] < 0.05
            cells = " | ".join(f"{counts[k] / trials:.3f}" for k in counts)
            print(f"| {null} | {n} | {cells} |")


if __name__ == "__main__":
    main(sys.argv[1:])
