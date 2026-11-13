"""Markdown and JSON rendering of evaluation results."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict

from .compare import Comparison, Table
from .metrics import Evaluation


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]], align: str) -> str:
    """Render a Markdown table; ``align`` has one character per column, ``l`` or ``r``."""
    marks = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _skipped_note(evaluation: Evaluation) -> str:
    parts = []
    labels = {
        "missing_from_run": "in qrels but missing from the run",
        "no_relevant": "without a relevant document",
        "not_in_qrels": "in the run but not in qrels",
    }
    for key, label in labels.items():
        count = len(evaluation.skipped.get(key, []))
        if count:
            parts.append(f"{count} {label}")
    return "; ".join(parts)


def score_markdown(
    name: str, evaluation: Evaluation, *, digits: int = 4, per_query: bool = False
) -> str:
    """Mean of each measure for one run, optionally followed by the per-query table."""
    lines = [f"run: {name}", f"queries scored: {len(evaluation.per_query)}"]
    note = _skipped_note(evaluation)
    if note:
        lines.append(f"not scored: {note}")
    lines.append("")
    means = evaluation.mean()
    rows = [[m, _fmt(means[m], digits)] for m in evaluation.measures]
    lines.append(markdown_table(["measure", "mean"], rows, "lr"))
    if per_query:
        lines.append("")
        rows = [
            [qid] + [_fmt(scores[m], digits) for m in evaluation.measures]
            for qid, scores in evaluation.per_query.items()
        ]
        align = "l" + "r" * len(evaluation.measures)
        lines.append(markdown_table(["query", *evaluation.measures], rows, align))
    return "\n".join(lines)


def score_json(name: str, evaluation: Evaluation, *, per_query: bool = False) -> str:
    """Machine-readable form of :func:`score_markdown`."""
    payload: dict[str, object] = {
        "run": name,
        "settings": evaluation.settings,
        "queries": len(evaluation.per_query),
        "skipped": evaluation.skipped,
        "mean": evaluation.mean(),
    }
    if per_query:
        payload["per_query"] = evaluation.per_query
    return json.dumps(payload, indent=2)


def _clean(value: object) -> object:
    """Make a value JSON-safe: non-finite floats become None, tuples become lists."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _p(value: float, adjusted: float, alpha: float) -> str:
    mark = "*" if adjusted < alpha else ""
    return f"{value:.4f} ({adjusted:.4f}){mark}"


def _num(value: float, digits: int) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else ("inf" if value > 0 else "-inf")


def compare_markdown(
    names: tuple[str, str], comparison: Comparison, *, digits: int = 4, alpha: float = 0.05
) -> str:
    """Two tables (scores and intervals, then p-values) plus per-query differences."""
    a, b = names
    c = comparison
    level = int(round((1 - alpha) * 100))
    lines = [
        f"A: {a}",
        f"B: {b}",
        f"queries paired: {len(c.queries)}",
        "",
        f"Differences are B minus A. Intervals are {level} %. Effect is mean(d) / sd(d).",
        "",
    ]
    headers = ["measure", "A", "B", "B - A", "t interval", "bootstrap interval", "effect"]
    headers += ["wins/ties/losses", "MDD"]
    rows = []
    for r in c.results:
        rows.append(
            [
                r.measure,
                _num(r.mean_a, digits),
                _num(r.mean_b, digits),
                f"{r.difference:+.{digits}f}",
                f"[{r.t_interval[0]:+.{digits}f}, {r.t_interval[1]:+.{digits}f}]",
                f"[{r.bootstrap_interval[0]:+.{digits}f}, {r.bootstrap_interval[1]:+.{digits}f}]",
                _num(r.effect_size, 3),
                f"{r.wins}/{r.ties}/{r.losses}",
                _num(r.mdd, digits),
            ]
        )
    lines.append(markdown_table(headers, rows, "lrrrrrrrr"))
    lines.append("")
    tests = list(c.results[0].p_values)
    family = len(c.results)
    lines.append(
        f"Two-sided p-values, then in brackets after {c.correction} correction across the "
        f"{family} measures within each test; * marks an adjusted p below {alpha}."
    )
    lines.append("")
    rows = [
        [r.measure] + [_p(r.p_values[t], (r.adjusted or r.p_values)[t], alpha) for t in tests]
        for r in c.results
    ]
    lines.append(markdown_table(["measure", *tests], rows, "l" + "r" * len(tests)))
    if c.wins or c.losses:
        lines.append("")
        lines.append(f"Largest per-query differences on {c.diff_measure}:")
        lines.append("")
        rows = [
            [d.query, _num(d.a, digits), _num(d.b, digits), f"{d.difference:+.{digits}f}"]
            for d in c.wins + c.losses
        ]
        lines.append(markdown_table(["query", "A", "B", "B - A"], rows, "lrrr"))
    lines.append("")
    lines.append(f"judged@{c.judged_k}: A {c.judged_a:.3f}, B {c.judged_b:.3f}")
    for warning in c.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


def compare_json(names: tuple[str, str], comparison: Comparison) -> str:
    c = comparison
    payload = {
        "a": names[0],
        "b": names[1],
        "queries": len(c.queries),
        "only_a": c.only_a,
        "only_b": c.only_b,
        "correction": c.correction,
        "results": [asdict(r) for r in c.results],
        "differences": {
            "measure": c.diff_measure,
            "wins": [{"query": d.query, "a": d.a, "b": d.b} for d in c.wins],
            "losses": [{"query": d.query, "a": d.a, "b": d.b} for d in c.losses],
        },
        "judged": {"k": c.judged_k, "a": c.judged_a, "b": c.judged_b},
        "warnings": c.warnings,
    }
    return json.dumps(_clean(payload), indent=2)


def table_markdown(table: Table, *, digits: int = 4, alpha: float = 0.05) -> str:
    lines = [
        f"queries: {table.queries}   baseline: {table.baseline}",
        f"* marks a {table.test}-test difference from the baseline with p below {alpha} after "
        f"{table.correction} correction over all {sum(len(v) for v in table.adjusted.values())} "
        "comparisons.",
        "",
    ]
    rows = []
    for name in table.runs:
        cells = [name]
        for m in table.measures:
            mark = "*" if table.adjusted.get(name, {}).get(m, 1.0) < alpha else ""
            cells.append(f"{table.means[name][m]:.{digits}f}{mark}")
        rows.append(cells)
    lines.append(markdown_table(["run", *table.measures], rows, "l" + "r" * len(table.measures)))
    for warning in table.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


def table_json(table: Table) -> str:
    return json.dumps(_clean(asdict(table)), indent=2)
