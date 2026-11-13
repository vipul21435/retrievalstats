"""Per-query retrieval metrics and their averages.

Conventions (the defaults follow trec_eval):

* Ranking. Documents are ordered by descending score. Tied scores are broken by
  descending document id, the rule trec_eval applies (``ties="trec_eval"``). The other
  rules are ascending document id (``ties="docno"``) and the order in the input file
  (``ties="input"``). The rank column of a TREC run is never used. Under the trec_eval
  rule scores are compared as 32-bit floats, as pytrec_eval does, so scores that differ
  only beyond about seven significant digits tie.
* Relevance. A judged document is relevant when its label is at least
  ``relevance_level`` (default 1). Labels from 0 up to that are judged non-relevant. A
  document missing from the qrels is unjudged and counts as non-relevant everywhere except
  bpref, which skips it, and judged@k, which counts it as unjudged.
* Negative labels. bpref skips them, as if the document were not in the qrels (this is
  what trec_eval and pytrec_eval do); every other trec_eval measure treats them as
  non-relevant with gain 0, which gives the same numbers. judged@k counts them as judged,
  as ir_measures' ``Judged`` does.
* Cut-offs. ``P@k`` divides by ``k`` even when fewer than ``k`` documents were
  retrieved, as trec_eval does. ``judged@k`` divides by the number of documents actually
  retrieved in the top ``k`` (as ir_measures does) and is 0 for an empty ranking.
* Gains for nDCG. ``ndcg`` uses the label as the gain (trec_eval's default, from
  Jarvelin and Kekalainen 2002); ``ndcg_exp`` uses ``2**label - 1``. Negative labels have
  gain 0. The discount is ``log2(rank + 1)``. The ideal ranking sorts every judged
  document of the query by gain, retrieved or not.
* Queries. Only queries in the qrels are scored; queries that appear only in the run are
  ignored. A query in the qrels that is missing from the run is skipped by default
  (trec_eval) or scored 0 on every metric with ``missing="zero"`` (trec_eval ``-c``). A
  query whose judgments contain no relevant document is kept and scores 0 by default
  (trec_eval) or is dropped with ``no_relevant="skip"``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

TIE_RULES = ("trec_eval", "docno", "input")
MISSING_RULES = ("skip", "zero")
NO_RELEVANT_RULES = ("keep", "skip")

_ALIASES = {
    "p": "P",
    "precision": "P",
    "recall": "recall",
    "r": "recall",
    "map": "map",
    "ap": "map",
    "mrr": "mrr",
    "rr": "mrr",
    "recip_rank": "mrr",
    "map_cut": "map",
    "ndcg": "ndcg",
    "ndcg_cut": "ndcg",
    "ndcg_exp": "ndcg_exp",
    "rprec": "rprec",
    "r-prec": "rprec",
    "bpref": "bpref",
    "success": "success",
    "judged": "judged",
}
_NEEDS_CUTOFF = {"P", "recall", "success", "judged"}
_NO_CUTOFF = {"rprec", "bpref"}

DEFAULT_MEASURES = (
    "ndcg@10",
    "map",
    "mrr",
    "P@10",
    "recall@100",
    "rprec",
    "bpref",
    "success@1",
    "judged@10",
)


@dataclass(frozen=True)
class Measure:
    """A metric name with an optional rank cut-off, for example ``ndcg@10``."""

    name: str
    cutoff: int | None = None

    def __str__(self) -> str:
        return self.name if self.cutoff is None else f"{self.name}@{self.cutoff}"


_MEASURE_RE = re.compile(r"^\s*([A-Za-z_\-]+?)(?:[@_](\d+))?\s*$")


def parse_measure(text: str) -> Measure:
    """Parse ``"ndcg@10"``, ``"P@5"``, ``"map"`` and similar into a :class:`Measure`.

    trec_eval spellings such as ``P_10``, ``ndcg_cut_10``, ``recip_rank`` and ``Rprec``
    are accepted too.
    """
    match = _MEASURE_RE.match(text)
    if not match:
        raise ValueError(f"cannot read measure {text!r}")
    base, cutoff_text = match.groups()
    name = _ALIASES.get(base.lower())
    if name is None:
        known = ", ".join(sorted(set(_ALIASES.values())))
        raise ValueError(f"unknown measure {text!r}; known measures: {known}")
    cutoff = int(cutoff_text) if cutoff_text is not None else None
    if cutoff is not None and cutoff < 1:
        raise ValueError(f"cut-off must be at least 1 in {text!r}")
    if name in _NEEDS_CUTOFF and cutoff is None:
        raise ValueError(f"{name} needs a cut-off, for example {name}@10")
    if name in _NO_CUTOFF and cutoff is not None:
        raise ValueError(f"{name} takes no cut-off")
    return Measure(name, cutoff)


def parse_measures(texts: Iterable[str]) -> list[Measure]:
    """Parse several measure names, keeping their order and dropping repeats."""
    seen: dict[str, Measure] = {}
    for text in texts:
        measure = parse_measure(text)
        seen.setdefault(str(measure), measure)
    return list(seen.values())


def rank(scores: Mapping[str, float], ties: str = "trec_eval") -> list[str]:
    """Order document ids by descending score, breaking ties by ``ties``.

    Under ``ties="trec_eval"`` scores are compared after rounding to 32-bit floats, as
    pytrec_eval does, so ``1.0`` and ``1.000000001`` tie. The other rules compare the
    scores at full 64-bit precision.
    """
    if ties not in TIE_RULES:
        raise ValueError(f"unknown tie rule {ties!r}; choose from {TIE_RULES}")
    items = list(scores.items())
    if ties == "trec_eval":
        rounded = np.asarray([score for _, score in items], dtype=np.float32).tolist()
        items = list(zip((doc for doc, _ in items), rounded, strict=True))
        items.sort(key=lambda item: item[0], reverse=True)
    elif ties == "docno":
        items.sort(key=lambda item: item[0])
    items.sort(key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in items]


def _gain(label: int | None, exponential: bool) -> float:
    if label is None or label <= 0:
        return 0.0
    return float(2**label - 1) if exponential else float(label)


def _dcg(gains: Sequence[float], cutoff: int | None) -> float:
    top = gains if cutoff is None else gains[:cutoff]
    return sum(g / math.log2(i + 2) for i, g in enumerate(top) if g)


def query_scores(
    ranking: Sequence[str],
    judgments: Mapping[str, int],
    measures: Sequence[Measure],
    relevance_level: int = 1,
) -> dict[str, float]:
    """Score one ranked list against the judgments of its query."""
    labels = [judgments.get(doc) for doc in ranking]
    relevant = [label is not None and label >= relevance_level for label in labels]
    num_rel = sum(1 for label in judgments.values() if label >= relevance_level)
    num_ret = len(ranking)

    cumulative = []
    count = 0
    for flag in relevant:
        count += flag
        cumulative.append(count)

    def rel_in_top(k: int) -> int:
        k = min(k, num_ret)
        return cumulative[k - 1] if k > 0 else 0

    out: dict[str, float] = {}
    for measure in measures:
        k = measure.cutoff
        depth = num_ret if k is None else min(k, num_ret)
        name = measure.name
        if name == "P":
            value = rel_in_top(k) / k
        elif name == "recall":
            value = rel_in_top(k) / num_rel if num_rel else 0.0
        elif name == "success":
            value = 1.0 if rel_in_top(k) > 0 else 0.0
        elif name == "judged":
            judged = sum(1 for label in labels[:depth] if label is not None)
            value = judged / depth if depth else 0.0
        elif name == "map":
            total = sum(cumulative[i] / (i + 1) for i in range(depth) if relevant[i])
            value = total / num_rel if num_rel else 0.0
        elif name == "mrr":
            first = next((i for i in range(depth) if relevant[i]), None)
            value = 1.0 / (first + 1) if first is not None else 0.0
        elif name == "rprec":
            value = rel_in_top(num_rel) / num_rel if num_rel else 0.0
        elif name == "bpref":
            value = _bpref(labels, judgments, relevance_level, num_rel)
        elif name in ("ndcg", "ndcg_exp"):
            exponential = name == "ndcg_exp"
            gains = [_gain(label, exponential) for label in labels]
            ideal = sorted((_gain(v, exponential) for v in judgments.values()), reverse=True)
            ideal_dcg = _dcg(ideal, k)
            value = _dcg(gains, k) / ideal_dcg if ideal_dcg > 0 else 0.0
        else:  # pragma: no cover - parse_measure only builds known names
            raise ValueError(f"unknown measure {measure}")
        out[str(measure)] = value
    return out


def _bpref(
    labels: Sequence[int | None],
    judgments: Mapping[str, int],
    relevance_level: int,
    num_rel: int,
) -> float:
    """bpref as defined by Buckley and Voorhees (2004), in trec_eval's form.

    For each relevant document retrieved, subtract the share of judged non-relevant
    documents ranked above it, where both counts are capped at R (the number of relevant
    documents). Unjudged documents and negative labels are skipped.
    """
    if not num_rel:
        return 0.0
    num_nonrel = sum(1 for label in judgments.values() if 0 <= label < relevance_level)
    cap = min(num_nonrel, num_rel)
    nonrel_above = 0
    total = 0.0
    for label in labels:
        if label is None or label < 0:
            continue
        if label >= relevance_level:
            total += 1.0 - min(nonrel_above, num_rel) / cap if nonrel_above else 1.0
        else:
            nonrel_above += 1
    return total / num_rel


@dataclass
class Evaluation:
    """Per-query scores of one run, with the settings that produced them."""

    measures: list[str]
    per_query: dict[str, dict[str, float]]
    settings: dict[str, object] = field(default_factory=dict)
    skipped: dict[str, list[str]] = field(default_factory=dict)

    @property
    def queries(self) -> list[str]:
        return list(self.per_query)

    def mean(self) -> dict[str, float]:
        """Arithmetic mean of each measure over the scored queries."""
        n = len(self.per_query)
        if n == 0:
            return {m: 0.0 for m in self.measures}
        return {m: sum(q[m] for q in self.per_query.values()) / n for m in self.measures}

    def values(self, measure: str) -> list[float]:
        """Per-query values of one measure, in query order."""
        return [scores[measure] for scores in self.per_query.values()]


def natural_key(text: str) -> list[object]:
    """Sort key that puts ``q2`` before ``q10``."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def evaluate(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    measures: Iterable[str | Measure] = DEFAULT_MEASURES,
    *,
    relevance_level: int = 1,
    ties: str = "trec_eval",
    missing: str = "skip",
    no_relevant: str = "keep",
) -> Evaluation:
    """Score every query of ``run`` that has judgments in ``qrels``.

    See the module docstring for the conventions; the defaults match trec_eval.
    """
    if missing not in MISSING_RULES:
        raise ValueError(f"unknown rule for missing queries {missing!r}; use {MISSING_RULES}")
    if no_relevant not in NO_RELEVANT_RULES:
        raise ValueError(f"unknown rule {no_relevant!r}; use {NO_RELEVANT_RULES}")
    if relevance_level < 1:
        raise ValueError("relevance_level must be at least 1")
    parsed = [m if isinstance(m, Measure) else parse_measure(m) for m in measures]
    names = [str(m) for m in parsed]

    per_query: dict[str, dict[str, float]] = {}
    skipped: dict[str, list[str]] = {"missing_from_run": [], "no_relevant": []}
    for qid in sorted(qrels, key=natural_key):
        judgments = qrels[qid]
        if not any(label >= relevance_level for label in judgments.values()) and (
            no_relevant == "skip"
        ):
            skipped["no_relevant"].append(qid)
            continue
        if qid not in run:
            if missing == "skip":
                skipped["missing_from_run"].append(qid)
                continue
            per_query[qid] = {name: 0.0 for name in names}
            continue
        ranking = rank(run[qid], ties=ties)
        per_query[qid] = query_scores(ranking, judgments, parsed, relevance_level)

    skipped["not_in_qrels"] = sorted((q for q in run if q not in qrels), key=natural_key)
    settings: dict[str, object] = {
        "relevance_level": relevance_level,
        "ties": ties,
        "missing": missing,
        "no_relevant": no_relevant,
    }
    return Evaluation(names, per_query, settings, skipped)
