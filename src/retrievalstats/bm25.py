"""A small in-memory BM25 so that example runs need no other tool.

Score of document ``D`` for query ``Q`` (Robertson and Zaragoza 2009)::

    sum over query tokens t of
        idf(t) * tf(t, D) * (k1 + 1) / (tf(t, D) + k1 * (1 - b + b * |D| / avgdl))

A token repeated in the query counts once per occurrence. Two idf variants:

* ``lucene`` (default): ``ln(1 + (N - df + 0.5) / (df + 0.5))``, always positive; this is
  the formula Lucene's ``BM25Similarity`` documents.
* ``robertson``: ``ln((N - df + 0.5) / (df + 0.5))``, the Robertson/Sparck Jones weight;
  it is negative for a term in more than half of the documents.

Tokens are the lower-cased runs of word characters (``\\w+``); there is no stemming and no
stopword list.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

IDF_VARIANTS = ("lucene", "robertson")
_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens."""
    return _TOKEN.findall(text.lower())


def idf(df: int, n_docs: int, variant: str = "lucene") -> float:
    """Inverse document frequency of a term found in ``df`` of ``n_docs`` documents."""
    if variant == "lucene":
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    if variant == "robertson":
        return math.log((n_docs - df + 0.5) / (df + 0.5))
    raise ValueError(f"unknown idf variant {variant!r}; choose from {IDF_VARIANTS}")


class BM25:
    """Inverted index over ``{doc_id: text}`` with BM25 scoring."""

    def __init__(
        self,
        documents: Mapping[str, str],
        k1: float = 0.9,
        b: float = 0.4,
        idf_variant: str = "lucene",
    ) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("need k1 >= 0 and 0 <= b <= 1")
        if idf_variant not in IDF_VARIANTS:
            raise ValueError(f"unknown idf variant {idf_variant!r}; choose from {IDF_VARIANTS}")
        self.k1 = k1
        self.b = b
        self.idf_variant = idf_variant
        self.doc_ids = list(documents)
        lengths = []
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for index, doc_id in enumerate(self.doc_ids):
            counts = Counter(tokenize(documents[doc_id]))
            lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                docs, tfs = postings.setdefault(term, ([], []))
                docs.append(index)
                tfs.append(tf)
        self.lengths = np.asarray(lengths, dtype=float)
        self.avgdl = float(self.lengths.mean()) if lengths and self.lengths.mean() > 0 else 1.0
        self.postings = {
            term: (np.asarray(docs), np.asarray(tfs, dtype=float))
            for term, (docs, tfs) in postings.items()
        }

    def idf(self, term: str) -> float:
        docs = self.postings.get(term)
        df = 0 if docs is None else docs[0].size
        return idf(df, len(self.doc_ids), self.idf_variant)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for ``query`` (0 for documents sharing no token)."""
        out = np.zeros(len(self.doc_ids))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
        for term in tokenize(query):
            entry = self.postings.get(term)
            if entry is None:
                continue
            docs, tfs = entry
            out[docs] += self.idf(term) * tfs * (self.k1 + 1) / (tfs + norm[docs])
        return out

    def search(self, query: str, depth: int = 1000) -> dict[str, float]:
        """Top ``depth`` documents that share a token with the query, by descending score.

        Ties at the cut-off are broken by descending document id, the order trec_eval
        reads them back in.
        """
        scores = self.scores(query)
        matched = {term for term in tokenize(query) if term in self.postings}
        if not matched:
            return {}
        candidates = np.unique(np.concatenate([self.postings[t][0] for t in matched]))
        ranked = sorted(candidates.tolist(), key=lambda i: self.doc_ids[i], reverse=True)
        ranked.sort(key=lambda i: scores[i], reverse=True)
        return {self.doc_ids[i]: float(scores[i]) for i in ranked[:depth]}

    def run(self, queries: Mapping[str, str], depth: int = 1000) -> dict[str, dict[str, float]]:
        """Search every query; returns a run ``{qid: {doc_id: score}}``."""
        return {qid: self.search(text, depth) for qid, text in queries.items()}


def _jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_corpus(path: str | Path, fields: Iterable[str] = ("title", "text")) -> dict[str, str]:
    """Read a BEIR ``corpus.jsonl`` (``_id``, ``title``, ``text``) or a ``id<TAB>text`` TSV."""
    path = Path(path)
    fields = list(fields)
    if path.suffix.lower() == ".jsonl":
        return {
            str(row["_id"]): " ".join(str(row.get(f) or "") for f in fields).strip()
            for row in _jsonl(path)
        }
    return _read_tsv(path)


def read_queries(path: str | Path) -> dict[str, str]:
    """Read a BEIR ``queries.jsonl`` (``_id``, ``text``) or a ``id<TAB>text`` TSV."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return {str(row["_id"]): str(row.get("text") or "") for row in _jsonl(path)}
    return _read_tsv(path)


def _read_tsv(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line:
                key, _, text = line.partition("\t")
                out[key] = text
    return out
