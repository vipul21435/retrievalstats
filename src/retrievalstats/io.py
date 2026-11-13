"""Readers and writers for relevance judgments (qrels) and runs.

Supported layouts:

* TREC qrels: ``qid iter docno rel`` (whitespace separated, ``iter`` ignored).
* TREC run: ``qid Q0 docno rank score tag`` (``rank`` and ``tag`` are ignored; the order
  of documents is decided by score, as trec_eval does).
* TSV in the BEIR layout: qrels ``query-id<TAB>corpus-id<TAB>score`` and runs
  ``query-id<TAB>corpus-id<TAB>score``, with or without a header line.
* JSONL: one object per line with ``query-id``, ``corpus-id`` and ``score`` keys
  (``qid``/``docid``/``rel`` are accepted as aliases).
* JSON (runs only): the nested ``{qid: {docid: score}}`` dictionary that BEIR writes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from pathlib import Path

Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]

QRELS_FORMATS = ("auto", "trec", "tsv", "jsonl")
RUN_FORMATS = ("auto", "trec", "tsv", "jsonl", "json")

_QID_KEYS = ("query-id", "query_id", "qid")
_DOC_KEYS = ("corpus-id", "corpus_id", "docid", "doc_id", "docno")
_VALUE_KEYS = ("score", "rel", "relevance", "label")


class FormatError(ValueError):
    """A line of an input file could not be read."""


def _lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if line and not line.startswith("#"):
                yield number, line


def _split(line: str) -> list[str]:
    return line.split("\t") if "\t" in line else line.split()


def _sniff(path: Path, allowed: tuple[str, ...], trec_width: int) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json" and "json" in allowed:
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    for _, line in _lines(path):
        if line.startswith("{"):
            return "jsonl"
        width = len(_split(line))
        if width == trec_width:
            return "trec"
        if width == 3:
            return "tsv"
        raise FormatError(f"{path}: cannot tell the format from a line with {width} fields")
    raise FormatError(f"{path}: file is empty")


def _parse_int(text: str, where: str) -> int:
    try:
        return int(text)
    except ValueError:
        try:
            value = float(text)
        except ValueError:
            raise FormatError(f"{where}: relevance {text!r} is not a number") from None
        if not value.is_integer():
            raise FormatError(f"{where}: relevance {text!r} is not an integer") from None
        return int(value)


def _parse_float(text: str, where: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise FormatError(f"{where}: score {text!r} is not a number") from None
    if math.isnan(value):
        raise FormatError(f"{where}: score is NaN")
    return value


def _pick(obj: Mapping[str, object], keys: tuple[str, ...], where: str) -> object:
    for key in keys:
        if key in obj:
            return obj[key]
    raise FormatError(f"{where}: none of the keys {', '.join(keys)} found")


def _jsonl_rows(path: Path) -> Iterator[tuple[str, str, str, str]]:
    for number, line in _lines(path):
        where = f"{path}:{number}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FormatError(f"{where}: {exc.msg}") from None
        if not isinstance(obj, dict):
            raise FormatError(f"{where}: expected a JSON object")
        qid = str(_pick(obj, _QID_KEYS, where))
        doc = str(_pick(obj, _DOC_KEYS, where))
        value = str(_pick(obj, _VALUE_KEYS, where))
        yield where, qid, doc, value


def _tsv_rows(path: Path) -> Iterator[tuple[str, str, str, str]]:
    first = True
    for number, line in _lines(path):
        where = f"{path}:{number}"
        fields = _split(line)
        if first:
            first = False
            if _looks_like_header(fields):
                continue
        if len(fields) != 3:
            raise FormatError(f"{where}: expected 3 fields, found {len(fields)}")
        yield where, fields[0], fields[1], fields[2]


def _looks_like_header(fields: list[str]) -> bool:
    """The first line is a header only when its third field is not a number at all.

    The data rows accept ``1``, ``1.0`` and ``2.5e-3``, so the same test decides here; a
    line such as ``q1  d1  1.0`` is data, not a header.
    """
    if len(fields) != 3:
        return False
    try:
        float(fields[2])
    except ValueError:
        return True
    return False


def read_qrels(path: str | Path, fmt: str = "auto") -> Qrels:
    """Read relevance judgments into ``{qid: {docno: relevance}}``.

    A (qid, docno) pair judged twice is an error, as it is for trec_eval.
    """
    path = Path(path)
    if fmt not in QRELS_FORMATS:
        raise ValueError(f"unknown qrels format {fmt!r}; choose from {QRELS_FORMATS}")
    if fmt == "auto":
        fmt = _sniff(path, QRELS_FORMATS, trec_width=4)

    rows: Iterator[tuple[str, str, str, str]]
    if fmt == "trec":
        rows = _trec_qrels_rows(path)
    elif fmt == "tsv":
        rows = _tsv_rows(path)
    else:
        rows = _jsonl_rows(path)

    qrels: Qrels = {}
    for where, qid, doc, value in rows:
        judged = qrels.setdefault(qid, {})
        if doc in judged:
            raise FormatError(f"{where}: document {doc!r} judged twice for query {qid!r}")
        judged[doc] = _parse_int(value, where)
    return qrels


def _trec_qrels_rows(path: Path) -> Iterator[tuple[str, str, str, str]]:
    for number, line in _lines(path):
        fields = line.split()
        where = f"{path}:{number}"
        if len(fields) != 4:
            raise FormatError(
                f"{where}: expected 4 fields (qid iter docno rel), found {len(fields)}"
            )
        yield where, fields[0], fields[2], fields[3]


def _trec_run_rows(path: Path) -> Iterator[tuple[str, str, str, str]]:
    for number, line in _lines(path):
        fields = line.split()
        where = f"{path}:{number}"
        if len(fields) != 6:
            raise FormatError(
                f"{where}: expected 6 fields (qid Q0 docno rank score tag), found {len(fields)}"
            )
        yield where, fields[0], fields[2], fields[4]


def read_run(path: str | Path, fmt: str = "auto") -> Run:
    """Read a run into ``{qid: {docno: score}}``, keeping the file order of documents.

    A document listed twice for one query is an error (trec_eval refuses such runs too).
    """
    path = Path(path)
    if fmt not in RUN_FORMATS:
        raise ValueError(f"unknown run format {fmt!r}; choose from {RUN_FORMATS}")
    if fmt == "auto":
        fmt = _sniff(path, RUN_FORMATS, trec_width=6)
    if fmt == "json":
        return _read_json_run(path)

    rows: Iterator[tuple[str, str, str, str]]
    if fmt == "trec":
        rows = _trec_run_rows(path)
    elif fmt == "tsv":
        rows = _tsv_rows(path)
    else:
        rows = _jsonl_rows(path)

    run: Run = {}
    for where, qid, doc, value in rows:
        scores = run.setdefault(qid, {})
        if doc in scores:
            raise FormatError(f"{where}: document {doc!r} listed twice for query {qid!r}")
        scores[doc] = _parse_float(value, where)
    return run


def _read_json_run(path: Path) -> Run:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FormatError(f"{path}: {exc.msg}") from None
    if not isinstance(data, dict):
        raise FormatError(f"{path}: expected an object mapping query ids to document scores")
    run: Run = {}
    for qid, scores in data.items():
        if not isinstance(scores, dict):
            raise FormatError(f"{path}: query {qid!r} does not map to an object")
        run[str(qid)] = {
            str(doc): _parse_float(str(score), f"{path}: query {qid!r}")
            for doc, score in scores.items()
        }
    return run


def write_trec_run(run: Mapping[str, Mapping[str, float]], path: str | Path, tag: str) -> None:
    """Write a run in TREC format.

    Documents are written in the order trec_eval will read them back (descending score,
    ties by descending document id), and scores keep full float precision so that no
    new ties appear on reading.
    """
    with Path(path).open("w", encoding="utf-8") as handle:
        for qid, scores in run.items():
            ranked = sorted(scores.items(), key=lambda item: item[0], reverse=True)
            ranked.sort(key=lambda item: item[1], reverse=True)
            for rank, (doc, score) in enumerate(ranked, start=1):
                handle.write(f"{qid} Q0 {doc} {rank} {score!r} {tag}\n")
