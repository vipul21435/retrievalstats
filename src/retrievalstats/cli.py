"""Command line interface: ``retrievalstats score | compare | table | bm25``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .bm25 import BM25, IDF_VARIANTS, read_corpus, read_queries
from .compare import compare_runs, run_table
from .io import QRELS_FORMATS, RUN_FORMATS, FormatError, read_qrels, read_run, write_trec_run
from .metrics import (
    DEFAULT_MEASURES,
    MISSING_RULES,
    NO_RELEVANT_RULES,
    TIE_RULES,
    evaluate,
    parse_measures,
)
from .report import (
    compare_json,
    compare_markdown,
    score_json,
    score_markdown,
    table_json,
    table_markdown,
)
from .stats import CORRECTIONS, TESTS

COMPARE_MEASURES = ("ndcg@10", "map", "mrr", "P@10", "recall@100")


def _split(values: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out += [part for part in value.replace(",", " ").split() if part]
    return out


def _add_evaluation_options(parser: argparse.ArgumentParser, defaults: Sequence[str]) -> None:
    parser.add_argument(
        "-m",
        "--measure",
        action="append",
        metavar="NAME",
        help=f"measure, repeatable or comma separated (default: {' '.join(defaults)})",
    )
    parser.add_argument(
        "--relevance-level",
        type=int,
        default=1,
        help="smallest label that counts as relevant (default: 1)",
    )
    parser.add_argument(
        "--ties",
        choices=TIE_RULES,
        default="trec_eval",
        help="order of documents with equal scores (default: trec_eval, by descending id)",
    )
    parser.add_argument(
        "--missing",
        choices=MISSING_RULES,
        default="skip",
        help="queries in qrels but not in the run: skip them or score 0 (default: skip)",
    )
    parser.add_argument(
        "--no-relevant",
        choices=NO_RELEVANT_RULES,
        default="keep",
        help="queries whose judgments hold no relevant document: keep (score 0) or skip "
        "(default: keep)",
    )
    parser.add_argument("--qrels-format", choices=QRELS_FORMATS, default="auto")
    parser.add_argument("--run-format", choices=RUN_FORMATS, default="auto")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--digits", type=int, default=4, help="decimals in Markdown output")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _probability(text: str) -> float:
    value = float(text)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError(f"must lie strictly between 0 and 1, got {value}")
    return value


def _add_test_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--correction",
        choices=CORRECTIONS,
        default="holm",
        help="multiple-comparison correction (default: holm)",
    )
    parser.add_argument(
        "--samples",
        type=_positive_int,
        default=10_000,
        help="random sign patterns and bootstrap resamples (default: 10000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--alpha", type=_probability, default=0.05, help="significance level (default: 0.05)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrievalstats",
        description="Retrieval metrics with the paired significance tests that go with them.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score one run against relevance judgments")
    score.add_argument("qrels", type=Path, help="relevance judgments")
    score.add_argument("run", type=Path, help="run to score")
    score.add_argument("--per-query", action="store_true", help="also print per-query values")
    _add_evaluation_options(score, DEFAULT_MEASURES)

    compare = sub.add_parser("compare", help="paired comparison of run B against run A")
    compare.add_argument("qrels", type=Path)
    compare.add_argument("run_a", type=Path, help="baseline run")
    compare.add_argument("run_b", type=Path, help="run compared with the baseline")
    compare.add_argument(
        "--tests",
        default=",".join(TESTS),
        help=f"comma-separated tests (default: {','.join(TESTS)})",
    )
    compare.add_argument(
        "--tie-threshold",
        type=float,
        default=0.0,
        help="|difference| at or below this is a tie for the sign test and win/tie/loss",
    )
    compare.add_argument("--judged-k", type=int, default=10, help="cut-off for judged@k checks")
    compare.add_argument("--top", type=int, default=5, help="largest wins and losses to list")
    _add_evaluation_options(compare, COMPARE_MEASURES)
    _add_test_options(compare)

    table = sub.add_parser("table", help="metric table of several runs against a baseline")
    table.add_argument("qrels", type=Path)
    table.add_argument("runs", type=Path, nargs="+", help="runs; the first is the baseline")
    table.add_argument("--test", choices=TESTS, default="t", help="test against the baseline")
    _add_evaluation_options(table, COMPARE_MEASURES)
    _add_test_options(table)

    bm25 = sub.add_parser("bm25", help="build a BM25 run from a corpus and queries")
    bm25.add_argument("corpus", type=Path, help="BEIR corpus.jsonl or id<TAB>text file")
    bm25.add_argument("queries", type=Path, help="BEIR queries.jsonl or id<TAB>text file")
    bm25.add_argument("-o", "--output", type=Path, required=True, help="TREC run to write")
    bm25.add_argument("--qrels", type=Path, help="only search queries judged in these qrels")
    bm25.add_argument("--k1", type=float, default=0.9, help="term-frequency saturation (0.9)")
    bm25.add_argument("--b", type=float, default=0.4, help="length normalisation (0.4)")
    bm25.add_argument("--idf", choices=IDF_VARIANTS, default="lucene", help="idf variant")
    bm25.add_argument(
        "--fields", default="title,text", help="corpus fields to index (default: title,text)"
    )
    bm25.add_argument("--depth", type=int, default=1000, help="documents kept per query")
    bm25.add_argument("--tag", default="bm25", help="run tag written in the last column")
    return parser


def _evaluate_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "relevance_level": args.relevance_level,
        "ties": args.ties,
        "missing": args.missing,
        "no_relevant": args.no_relevant,
    }


def _measures(args: argparse.Namespace, defaults: Sequence[str]) -> list[str]:
    return _split(args.measure) or list(defaults)


def _cmd_score(args: argparse.Namespace) -> int:
    measures = parse_measures(_measures(args, DEFAULT_MEASURES))
    qrels = read_qrels(args.qrels, args.qrels_format)
    run = read_run(args.run, args.run_format)
    evaluation = evaluate(qrels, run, measures, **_evaluate_options(args))  # type: ignore[arg-type]
    if not evaluation.per_query:
        print("error: no query of the run has judgments in the qrels", file=sys.stderr)
        return 1
    name = args.run.name
    if args.format == "json":
        print(score_json(name, evaluation, per_query=args.per_query))
    else:
        print(score_markdown(name, evaluation, digits=args.digits, per_query=args.per_query))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    qrels = read_qrels(args.qrels, args.qrels_format)
    run_a = read_run(args.run_a, args.run_format)
    run_b = read_run(args.run_b, args.run_format)
    names = (args.run_a.name, args.run_b.name)
    comparison = compare_runs(
        qrels,
        run_a,
        run_b,
        _measures(args, COMPARE_MEASURES),
        names=names,
        tests=_split([args.tests]),
        correction=args.correction,
        samples=args.samples,
        seed=args.seed,
        confidence=1 - args.alpha,
        tie_threshold=args.tie_threshold,
        judged_k=args.judged_k,
        top=args.top,
        **_evaluate_options(args),
    )
    if args.format == "json":
        print(compare_json(names, comparison))
    else:
        print(compare_markdown(names, comparison, digits=args.digits, alpha=args.alpha))
    return 0


def _cmd_table(args: argparse.Namespace) -> int:
    qrels = read_qrels(args.qrels, args.qrels_format)
    runs = {}
    for path in args.runs:
        if path.name in runs:
            raise ValueError(f"two runs share the file name {path.name!r}")
        runs[path.name] = read_run(path, args.run_format)
    table = run_table(
        qrels,
        runs,
        _measures(args, COMPARE_MEASURES),
        test=args.test,
        correction=args.correction,
        samples=args.samples,
        seed=args.seed,
        **_evaluate_options(args),
    )
    if args.format == "json":
        print(table_json(table))
    else:
        print(table_markdown(table, digits=args.digits, alpha=args.alpha))
    return 0


def _cmd_bm25(args: argparse.Namespace) -> int:
    corpus = read_corpus(args.corpus, _split([args.fields]))
    queries = read_queries(args.queries)
    if args.qrels is not None:
        judged = read_qrels(args.qrels)
        queries = {q: text for q, text in queries.items() if q in judged}
    index = BM25(corpus, k1=args.k1, b=args.b, idf_variant=args.idf)
    run = index.run(queries, depth=args.depth)
    write_trec_run(run, args.output, tag=args.tag)
    print(
        f"wrote {args.output}: {len(run)} queries, {len(corpus)} documents, "
        f"k1={args.k1} b={args.b} idf={args.idf}"
    )
    return 0


COMMANDS = {"score": _cmd_score, "compare": _cmd_compare, "table": _cmd_table, "bm25": _cmd_bm25}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (FormatError, ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
