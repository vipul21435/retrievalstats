"""Paired significance tests, intervals and corrections for per-query differences.

Every function takes the per-query differences ``d = b - a`` of two runs on the same
queries. All p-values are two-sided.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy import stats as sps

TESTS = ("t", "wilcoxon", "sign", "randomisation", "bootstrap")
CORRECTIONS = ("holm", "bh", "none")
_CHUNK = 4096


def _array(d: Sequence[float]) -> np.ndarray:
    arr = np.asarray(d, dtype=float)
    if arr.ndim != 1:
        raise ValueError("differences must be a one-dimensional sequence")
    return arr


def paired_t(d: Sequence[float]) -> tuple[float, float]:
    """Student's paired t-test. Returns ``(t, p)``."""
    arr = _array(d)
    n = arr.size
    if n < 2:
        return 0.0, 1.0
    mean = arr.mean()
    sd = arr.std(ddof=1)
    if sd == 0:
        return (0.0, 1.0) if mean == 0 else (math.copysign(math.inf, mean), 0.0)
    t = mean / (sd / math.sqrt(n))
    return float(t), float(2 * sps.t.sf(abs(t), n - 1))


def wilcoxon(d: Sequence[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test, zero differences discarded. Returns ``(W, p)``.

    Uses scipy's implementation (exact distribution for small samples without ties,
    normal approximation otherwise). ``W`` is the smaller of the two rank sums.
    """
    arr = _array(d)
    if np.count_nonzero(arr) == 0:
        return 0.0, 1.0
    result = sps.wilcoxon(arr, zero_method="wilcox", alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def sign_test(d: Sequence[float], threshold: float = 0.0) -> tuple[float, float]:
    """Sign test: wins against losses under Binomial(n, 1/2). Returns ``(wins, p)``.

    Differences with absolute value at most ``threshold`` are ties and are dropped.
    """
    wins, _, losses = win_tie_loss(d, threshold)
    if wins + losses == 0:
        return 0.0, 1.0
    return float(wins), float(sps.binomtest(wins, wins + losses, 0.5).pvalue)


def randomisation(
    d: Sequence[float], samples: int = 10_000, seed: int = 0
) -> tuple[float, float, bool]:
    """Fisher's randomisation (paired permutation) test on the mean difference.

    Under the null hypothesis each query's two scores are exchangeable, so the sign of
    each difference is a fair coin. When ``2**n`` (n = non-zero differences) is at most
    ``samples`` every sign pattern is enumerated and the p-value is exact; otherwise
    ``samples`` random patterns are drawn and the observed pattern is counted as one of
    them, ``p = (count + 1) / (samples + 1)``. Returns ``(mean, p, exact)``.
    """
    arr = _array(d)
    observed = arr.mean() if arr.size else 0.0
    nonzero = arr[arr != 0]
    n = nonzero.size
    if n == 0:
        return float(observed), 1.0, True
    target = abs(nonzero.sum())
    tol = 1e-9 * max(1.0, float(np.abs(nonzero).sum()))
    if 2**n <= samples:
        count = 0
        total = 2**n
        bits = np.arange(n)
        for start in range(0, total, _CHUNK):
            patterns = np.arange(start, min(start + _CHUNK, total))[:, None]
            signs = 1 - 2 * ((patterns >> bits) & 1)
            count += int(np.count_nonzero(np.abs(signs @ nonzero) >= target - tol))
        return float(observed), count / total, True
    rng = np.random.default_rng(seed)
    count = 0
    done = 0
    while done < samples:
        size = min(_CHUNK, samples - done)
        signs = rng.choice((-1.0, 1.0), size=(size, n))
        count += int(np.count_nonzero(np.abs(signs @ nonzero) >= target - tol))
        done += size
    return float(observed), (count + 1) / (samples + 1), False


def _bootstrap_means(arr: np.ndarray, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(samples)
    n = arr.size
    for start in range(0, samples, _CHUNK):
        size = min(_CHUNK, samples - start)
        idx = rng.integers(0, n, size=(size, n))
        out[start : start + size] = arr[idx].mean(axis=1)
    return out


def bootstrap(
    d: Sequence[float], samples: int = 10_000, seed: int = 0, confidence: float = 0.95
) -> tuple[float, float, tuple[float, float]]:
    """Paired bootstrap of the mean difference.

    The p-value uses the shift method: the bootstrap means are centred on their own mean
    and compared with the observed mean, ``p = (count + 1) / (samples + 1)``. The interval
    is the percentile interval of the bootstrap means. Returns ``(mean, p, interval)``.
    """
    arr = _array(d)
    if arr.size == 0:
        return 0.0, 1.0, (0.0, 0.0)
    observed = float(arr.mean())
    means = _bootstrap_means(arr, samples, seed)
    shifted = means - means.mean()
    tol = 1e-12 * max(1.0, abs(observed))
    count = int(np.count_nonzero(np.abs(shifted) >= abs(observed) - tol))
    alpha = 1 - confidence
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return observed, (count + 1) / (samples + 1), (float(lo), float(hi))


def t_interval(d: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """Student-t interval for the mean difference."""
    arr = _array(d)
    n = arr.size
    if n == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    if n < 2:
        return mean, mean
    half = sps.t.ppf(0.5 + confidence / 2, n - 1) * arr.std(ddof=1) / math.sqrt(n)
    return mean - float(half), mean + float(half)


def effect_size(d: Sequence[float]) -> float:
    """Standardised mean difference of paired data: mean(d) / sd(d)."""
    arr = _array(d)
    if arr.size < 2:
        return 0.0
    sd = arr.std(ddof=1)
    mean = arr.mean()
    if sd == 0:
        return 0.0 if mean == 0 else math.copysign(math.inf, mean)
    return float(mean / sd)


def win_tie_loss(d: Sequence[float], threshold: float = 0.0) -> tuple[int, int, int]:
    """Count queries where b beats a, ties within ``threshold``, and loses."""
    arr = _array(d)
    wins = int(np.count_nonzero(arr > threshold))
    losses = int(np.count_nonzero(arr < -threshold))
    return wins, arr.size - wins - losses, losses


def t_power(delta: float, sd: float, n: int, alpha: float = 0.05) -> float:
    """Power of the two-sided paired t-test to detect a true mean difference ``delta``."""
    if n < 2 or sd <= 0:
        return 1.0 if delta else alpha
    df = n - 1
    crit = sps.t.ppf(1 - alpha / 2, df)
    nc = delta * math.sqrt(n) / sd
    value = sps.nct.sf(crit, df, nc) + sps.nct.cdf(-crit, df, nc)
    if not math.isfinite(value):
        # scipy's noncentral t can fail far in the tail; the normal limit is accurate there.
        value = sps.norm.sf(crit - nc) + sps.norm.cdf(-crit - nc)
    return float(min(1.0, value))


def minimum_detectable_difference(
    d: Sequence[float], alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest true mean difference the paired t-test detects with the given power.

    Uses the observed standard deviation of the differences and the noncentral t
    distribution, solving ``t_power(delta) = power`` for ``delta``.
    """
    arr = _array(d)
    n = arr.size
    if n < 2:
        return math.inf
    sd = float(arr.std(ddof=1))
    if sd == 0:
        return 0.0
    upper = sd
    while t_power(upper, sd, n, alpha) < power:
        upper *= 2
    return float(optimize.brentq(lambda x: t_power(x, sd, n, alpha) - power, 0.0, upper))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjusted p-values (family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    scaled = (m - np.arange(m)) * p[order]
    adjusted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out.tolist()


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")[::-1]
    ranks = m - np.arange(m)
    scaled = p[order] * m / ranks
    adjusted = np.minimum(np.minimum.accumulate(scaled), 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out.tolist()


def adjust(p_values: Sequence[float], method: str) -> list[float]:
    """Adjust p-values with ``holm``, ``bh`` or ``none``."""
    if method == "holm":
        return holm(p_values)
    if method == "bh":
        return benjamini_hochberg(p_values)
    if method == "none":
        return [float(p) for p in p_values]
    raise ValueError(f"unknown correction {method!r}; choose from {CORRECTIONS}")


@dataclass
class PairedResult:
    """Everything the paired comparison of one measure produced."""

    measure: str
    n: int
    mean_a: float
    mean_b: float
    difference: float
    t_interval: tuple[float, float]
    bootstrap_interval: tuple[float, float]
    effect_size: float
    wins: int
    ties: int
    losses: int
    p_values: dict[str, float]
    mdd: float
    randomisation_exact: bool = False
    adjusted: dict[str, float] | None = None


def paired_comparison(
    measure: str,
    a: Sequence[float],
    b: Sequence[float],
    tests: Sequence[str] = TESTS,
    *,
    samples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    tie_threshold: float = 0.0,
) -> PairedResult:
    """Run every requested test on the per-query scores ``a`` and ``b`` of one measure."""
    for test in tests:
        if test not in TESTS:
            raise ValueError(f"unknown test {test!r}; choose from {TESTS}")
    xa = _array(a)
    xb = _array(b)
    if xa.shape != xb.shape:
        raise ValueError("both runs need a score for every query")
    d = xb - xa
    _, boot_p, boot_ci = bootstrap(d, samples, seed, confidence)
    p_values: dict[str, float] = {}
    exact = False
    for test in tests:
        if test == "t":
            p_values[test] = paired_t(d)[1]
        elif test == "wilcoxon":
            p_values[test] = wilcoxon(d)[1]
        elif test == "sign":
            p_values[test] = sign_test(d, tie_threshold)[1]
        elif test == "randomisation":
            _, p_values[test], exact = randomisation(d, samples, seed)
        else:
            p_values[test] = boot_p
    wins, ties, losses = win_tie_loss(d, tie_threshold)
    return PairedResult(
        measure=measure,
        n=int(d.size),
        mean_a=float(xa.mean()) if xa.size else 0.0,
        mean_b=float(xb.mean()) if xb.size else 0.0,
        difference=float(d.mean()) if d.size else 0.0,
        t_interval=t_interval(d, confidence),
        bootstrap_interval=boot_ci,
        effect_size=effect_size(d),
        wins=wins,
        ties=ties,
        losses=losses,
        p_values=p_values,
        mdd=minimum_detectable_difference(d, 1 - confidence),
        randomisation_exact=exact,
    )


def correct(results: Sequence[PairedResult], method: str) -> None:
    """Fill ``adjusted`` on each result, correcting within each test across the results."""
    tests = list(results[0].p_values) if results else []
    for result in results:
        result.adjusted = {}
    for test in tests:
        adjusted = adjust([r.p_values[test] for r in results], method)
        for result, value in zip(results, adjusted, strict=True):
            assert result.adjusted is not None
            result.adjusted[test] = value
