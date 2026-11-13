import itertools
import math

import numpy as np
import pytest
from scipy import stats as sps

from retrievalstats.stats import (
    adjust,
    benjamini_hochberg,
    bootstrap,
    correct,
    effect_size,
    holm,
    minimum_detectable_difference,
    paired_comparison,
    paired_t,
    randomisation,
    sign_test,
    t_interval,
    t_power,
    wilcoxon,
    win_tie_loss,
)


def brute_force_randomisation(d):
    d = [x for x in d if x != 0]
    target = abs(sum(d))
    hits = 0
    total = 0
    for signs in itertools.product((-1, 1), repeat=len(d)):
        total += 1
        hits += abs(sum(s * x for s, x in zip(signs, d, strict=True))) >= target - 1e-12
    return hits / total


def test_randomisation_hand_worked():
    # Sign flips of (1, 2, 3) give sums 6, 4, 2, 0, 0, -2, -4, -6: two of eight reach 6.
    mean, p, exact = randomisation([1.0, 2.0, 3.0])
    assert exact
    assert mean == 2.0
    assert p == 2 / 8


@pytest.mark.parametrize("seed", range(10))
def test_randomisation_exact_matches_enumeration(seed):
    rng = np.random.default_rng(seed)
    d = np.round(rng.normal(0.05, 0.2, size=rng.integers(1, 11)), 2)
    d[rng.random(d.size) < 0.2] = 0.0
    _, p, exact = randomisation(d, samples=2**12)
    assert exact
    assert p == pytest.approx(brute_force_randomisation(d.tolist()))


def test_randomisation_sampled_is_close_to_exact():
    rng = np.random.default_rng(3)
    d = rng.normal(0.1, 0.3, size=14)
    _, p_exact, exact = randomisation(d, samples=2**14)
    assert exact
    _, p_sampled, exact = randomisation(d, samples=10_000, seed=1)
    assert not exact
    assert p_sampled == pytest.approx(p_exact, abs=0.01)


def test_randomisation_all_zero():
    assert randomisation([0.0, 0.0]) == (0.0, 1.0, True)


def test_paired_t_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.random(50)
    b = a + rng.normal(0.03, 0.1, 50)
    t, p = paired_t(b - a)
    ref = sps.ttest_rel(b, a)
    assert t == pytest.approx(ref.statistic)
    assert p == pytest.approx(ref.pvalue)


def test_paired_t_degenerate():
    assert paired_t([0.1]) == (0.0, 1.0)
    assert paired_t([0.0, 0.0]) == (0.0, 1.0)
    assert paired_t([0.2, 0.2]) == (math.inf, 0.0)


def test_wilcoxon_exact_small_sample():
    # All five differences positive: W- = 0, exact two-sided p = 2 / 2**5.
    w, p = wilcoxon([1.0, 2.0, 3.0, 4.0, 5.0])
    assert w == 0.0
    assert p == pytest.approx(2 / 32)
    assert wilcoxon([0.0, 0.0]) == (0.0, 1.0)


def test_sign_test_hand_worked():
    # Five wins, one loss, two ties: two-sided binomial p = 2 * (1 + 6) / 64.
    wins, p = sign_test([0.1, 0.2, 0.1, 0.3, 0.5, -0.2, 0.0, 0.0])
    assert wins == 5
    assert p == pytest.approx(14 / 64)
    assert sign_test([0.0]) == (0.0, 1.0)
    assert sign_test([0.005, -0.004, 0.2], threshold=0.01)[0] == 1


def test_win_tie_loss_threshold():
    assert win_tie_loss([0.1, -0.1, 0.0, 0.005]) == (2, 1, 1)
    assert win_tie_loss([0.1, -0.1, 0.0, 0.005], threshold=0.01) == (1, 2, 1)


def test_t_interval_matches_scipy():
    d = np.array([0.1, -0.05, 0.2, 0.0, 0.15, 0.3])
    lo, hi = t_interval(d)
    ref = sps.t.interval(0.95, d.size - 1, loc=d.mean(), scale=sps.sem(d))
    assert (lo, hi) == pytest.approx(ref)
    assert t_interval([]) == (0.0, 0.0)
    assert t_interval([0.3]) == (0.3, 0.3)


def test_bootstrap_interval_and_p():
    rng = np.random.default_rng(5)
    d = rng.normal(0.2, 0.1, 40)
    mean, p, (lo, hi) = bootstrap(d, samples=5000, seed=2)
    assert lo < mean < hi
    assert p == pytest.approx(1 / 5001)
    assert bootstrap(d, samples=5000, seed=2) == (mean, p, (lo, hi))
    null = rng.normal(0.0, 0.1, 40)
    assert bootstrap(null, samples=5000)[1] > 0.01
    assert bootstrap([]) == (0.0, 1.0, (0.0, 0.0))


def test_effect_size():
    assert effect_size([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert effect_size([0.0, 0.0]) == 0.0
    assert effect_size([0.5, 0.5]) == math.inf
    assert effect_size([1.0]) == 0.0


def test_holm_and_bh_match_statsmodels():
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(9)
    for _ in range(20):
        p = rng.random(rng.integers(1, 12)) ** 3
        assert holm(p) == pytest.approx(multitest.multipletests(p, method="holm")[1].tolist())
        bh = multitest.multipletests(p, method="fdr_bh")[1].tolist()
        assert benjamini_hochberg(p) == pytest.approx(bh)
        assert benjamini_hochberg(p) == pytest.approx(sps.false_discovery_control(p).tolist())


def test_holm_hand_worked():
    # Sorted p: 0.01, 0.02, 0.03, 0.04 with m = 4: 0.04, 0.06, 0.06, 0.06.
    assert holm([0.03, 0.01, 0.04, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])
    assert benjamini_hochberg([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04] * 4)
    assert holm([]) == [] and benjamini_hochberg([]) == []
    assert adjust([0.2, 0.01], "none") == [0.2, 0.01]
    with pytest.raises(ValueError):
        adjust([0.1], "bonferroni")


def test_minimum_detectable_difference():
    rng = np.random.default_rng(1)
    d = rng.normal(0.0, 0.2, 50)
    mdd = minimum_detectable_difference(d)
    sd = d.std(ddof=1)
    assert t_power(mdd, sd, 50) == pytest.approx(0.8)
    power = pytest.importorskip("statsmodels.stats.power")
    ref = power.TTestPower().solve_power(nobs=50, alpha=0.05, power=0.8) * sd
    assert mdd == pytest.approx(ref, rel=1e-4)
    # Normal approximation for large n.
    big = rng.normal(0.0, 0.2, 2000)
    approx = (sps.norm.ppf(0.975) + sps.norm.ppf(0.8)) * big.std(ddof=1) / math.sqrt(2000)
    assert minimum_detectable_difference(big) == pytest.approx(approx, rel=0.01)
    assert minimum_detectable_difference([0.1]) == math.inf
    assert minimum_detectable_difference([0.1, 0.1]) == 0.0


def test_type_one_error_near_nominal():
    # Seeded simulation under the null: the t and randomisation tests reject about 5 %.
    rng = np.random.default_rng(2024)
    rejections = {"t": 0, "randomisation": 0}
    trials = 300
    for _ in range(trials):
        d = rng.normal(0.0, 0.15, 30)
        rejections["t"] += paired_t(d)[1] < 0.05
        rejections["randomisation"] += randomisation(d, samples=2000, seed=1)[1] < 0.05
    for count in rejections.values():
        assert 0.02 < count / trials < 0.09


def test_paired_comparison_and_correction():
    rng = np.random.default_rng(4)
    a = rng.random(60)
    b = np.clip(a + rng.normal(0.05, 0.1, 60), 0, 1)
    results = [
        paired_comparison("m1", a, b, samples=2000),
        paired_comparison("m2", a, a, samples=2000),
    ]
    first = results[0]
    assert first.n == 60
    assert first.difference == pytest.approx(b.mean() - a.mean())
    assert first.t_interval[0] < first.difference < first.t_interval[1]
    assert first.wins + first.ties + first.losses == 60
    assert set(first.p_values) == {"t", "wilcoxon", "sign", "randomisation", "bootstrap"}
    assert results[1].p_values["t"] == 1.0
    correct(results, "holm")
    assert first.adjusted is not None
    assert first.adjusted["t"] == pytest.approx(min(1.0, 2 * first.p_values["t"]))
    with pytest.raises(ValueError):
        paired_comparison("m", a, b, tests=["z"])
    with pytest.raises(ValueError):
        paired_comparison("m", a, b[:-1])
