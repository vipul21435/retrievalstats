"""Markdown and JSON rendering of evaluation results."""

from __future__ import annotations

import json
from collections.abc import Sequence

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
