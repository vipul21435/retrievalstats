"""Paired comparison of runs, with the diagnostics that go with it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .metrics import Evaluation, Measure, evaluate, parse_measures
from .stats import CORRECTIONS, TESTS, PairedResult, adjust, correct, paired_comparison

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Mapping[str, float]]


@dataclass
class QueryDifference:
    query: str
    a: float
    b: float

    @property
    def difference(self) -> float:
        return self.b - self.a


@dataclass
class Comparison:
    """Result of comparing run B against run A on the queries both were scored on."""

    results: list[PairedResult]
    queries: list[str]
    only_a: list[str]
    only_b: list[str]
    correction: str
    diff_measure: str
    wins: list[QueryDifference]
    losses: list[QueryDifference]
    judged_k: int
    judged_a: float
    judged_b: float
    warnings: list[str] = field(default_factory=list)


def _with_judged(measures: Sequence[Measure], k: int) -> list[Measure]:
    extra = Measure("judged", k)
    return list(measures) if extra in measures else [*measures, extra]


def judged_warnings(name_a: str, a: float, name_b: str, b: float, k: int, gap: float) -> list[str]:
    """Plain-language warnings about unjudged documents in the top ``k``."""
    out = []
    if abs(a - b) >= gap:
        low, high = (name_a, name_b) if a < b else (name_b, name_a)
        out.append(
            f"judged@{k} differs: {name_a} {a:.3f}, {name_b} {b:.3f}. Unjudged documents count "
            f"as non-relevant, so the comparison is biased against {low} relative to {high}."
        )
    if min(a, b) < 0.5:
        out.append(
            f"judged@{k} is below 0.5 for at least one run: most of its top {k} documents have "
            "no judgment, so its scores are lower bounds."
        )
    return out


def compare_runs(
    qrels: Qrels,
    run_a: Run,
    run_b: Run,
    measures: Sequence[str | Measure],
    *,
    names: tuple[str, str] = ("A", "B"),
    tests: Sequence[str] = TESTS,
    correction: str = "holm",
    samples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    tie_threshold: float = 0.0,
    judged_k: int = 10,
    judged_gap: float = 0.05,
    top: int = 5,
    **evaluate_options: object,
) -> Comparison:
    """Compare two runs measure by measure; corrections run across the measures per test."""
    if correction not in CORRECTIONS:
        raise ValueError(f"unknown correction {correction!r}; choose from {CORRECTIONS}")
    parsed = parse_measures(str(m) for m in measures)
    if not parsed:
        raise ValueError("need at least one measure")
    scored = _with_judged(parsed, judged_k)
    eval_a = evaluate(qrels, run_a, scored, **evaluate_options)  # type: ignore[arg-type]
    eval_b = evaluate(qrels, run_b, scored, **evaluate_options)  # type: ignore[arg-type]
    queries = [q for q in eval_a.per_query if q in eval_b.per_query]
    if not queries:
        raise ValueError("the two runs have no scored query in common")
    only_a = [q for q in eval_a.per_query if q not in eval_b.per_query]
    only_b = [q for q in eval_b.per_query if q not in eval_a.per_query]

    results = []
    for measure in parsed:
        name = str(measure)
        a = [eval_a.per_query[q][name] for q in queries]
        b = [eval_b.per_query[q][name] for q in queries]
        results.append(
            paired_comparison(
                name,
                a,
                b,
                tests,
                samples=samples,
                seed=seed,
                confidence=confidence,
                tie_threshold=tie_threshold,
            )
        )
    correct(results, correction)

    diff_measure = str(parsed[0])
    rows = [
        QueryDifference(q, eval_a.per_query[q][diff_measure], eval_b.per_query[q][diff_measure])
        for q in queries
    ]
    wins = sorted((r for r in rows if r.difference > 0), key=lambda r: -r.difference)[:top]
    losses = sorted((r for r in rows if r.difference < 0), key=lambda r: r.difference)[:top]

    judged = f"judged@{judged_k}"
    judged_a = sum(eval_a.per_query[q][judged] for q in queries) / len(queries)
    judged_b = sum(eval_b.per_query[q][judged] for q in queries) / len(queries)
    warnings = judged_warnings(names[0], judged_a, names[1], judged_b, judged_k, judged_gap)
    if only_a or only_b:
        warnings.append(
            f"{len(only_a) + len(only_b)} queries were scored for one run only and are left out "
            "of the paired tests (use --missing zero to score them 0 instead)."
        )
    return Comparison(
        results,
        queries,
        only_a,
        only_b,
        correction,
        diff_measure,
        wins,
        losses,
        judged_k,
        judged_a,
        judged_b,
        warnings,
    )


@dataclass
class Table:
    """Mean scores of several runs, each tested against a baseline."""

    runs: list[str]
    measures: list[str]
    means: dict[str, dict[str, float]]
    baseline: str
    test: str
    correction: str
    p_values: dict[str, dict[str, float]]
    adjusted: dict[str, dict[str, float]]
    queries: int


def run_table(
    qrels: Qrels,
    runs: Mapping[str, Run],
    measures: Sequence[str | Measure],
    *,
    baseline: str | None = None,
    test: str = "t",
    correction: str = "holm",
    samples: int = 10_000,
    seed: int = 0,
    **evaluate_options: object,
) -> Table:
    """Score every run; test each non-baseline run against the baseline on every measure.

    The correction runs over the whole family of (run, measure) comparisons.
    """
    if test not in TESTS:
        raise ValueError(f"unknown test {test!r}; choose from {TESTS}")
    if correction not in CORRECTIONS:
        raise ValueError(f"unknown correction {correction!r}; choose from {CORRECTIONS}")
    names = list(runs)
    if not names:
        raise ValueError("need at least one run")
    baseline = names[0] if baseline is None else baseline
    if baseline not in runs:
        raise ValueError(f"baseline {baseline!r} is not one of the runs")
    parsed = parse_measures(str(m) for m in measures)
    evaluations: dict[str, Evaluation] = {
        name: evaluate(qrels, run, parsed, **evaluate_options)  # type: ignore[arg-type]
        for name, run in runs.items()
    }
    common = [
        q
        for q in evaluations[baseline].per_query
        if all(q in e.per_query for e in evaluations.values())
    ]
    if not common:
        raise ValueError("the runs have no scored query in common")
    labels = [str(m) for m in parsed]
    means = {
        name: {m: sum(e.per_query[q][m] for q in common) / len(common) for m in labels}
        for name, e in evaluations.items()
    }
    keys: list[tuple[str, str]] = []
    raw: list[float] = []
    base = evaluations[baseline]
    for name in names:
        if name == baseline:
            continue
        for m in labels:
            a = [base.per_query[q][m] for q in common]
            b = [evaluations[name].per_query[q][m] for q in common]
            result = paired_comparison(m, a, b, [test], samples=samples, seed=seed)
            keys.append((name, m))
            raw.append(result.p_values[test])
    adjusted_list = adjust(raw, correction)
    p_values: dict[str, dict[str, float]] = {}
    adjusted: dict[str, dict[str, float]] = {}
    for (name, m), p, q in zip(keys, raw, adjusted_list, strict=True):
        p_values.setdefault(name, {})[m] = p
        adjusted.setdefault(name, {})[m] = q
    return Table(names, labels, means, baseline, test, correction, p_values, adjusted, len(common))
