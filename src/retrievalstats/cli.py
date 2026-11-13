"""Command line interface: ``retrievalstats score``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .io import QRELS_FORMATS, RUN_FORMATS, FormatError, read_qrels, read_run
from .metrics import (
    DEFAULT_MEASURES,
    MISSING_RULES,
    NO_RELEVANT_RULES,
    TIE_RULES,
    evaluate,
    parse_measures,
)
from .report import score_json, score_markdown


def _measure_list(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_MEASURES)
    out: list[str] = []
    for value in values:
        out += [part for part in value.replace(",", " ").split() if part]
    return out


def _add_evaluation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--measure",
        action="append",
        metavar="NAME",
        help="measure to report, repeatable or comma separated "
        f"(default: {' '.join(DEFAULT_MEASURES)})",
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
    _add_evaluation_options(score)
    return parser


def _cmd_score(args: argparse.Namespace) -> int:
    measures = parse_measures(_measure_list(args.measure))
    qrels = read_qrels(args.qrels, args.qrels_format)
    run = read_run(args.run, args.run_format)
    evaluation = evaluate(
        qrels,
        run,
        measures,
        relevance_level=args.relevance_level,
        ties=args.ties,
        missing=args.missing,
        no_relevant=args.no_relevant,
    )
    if not evaluation.per_query:
        print("error: no query of the run has judgments in the qrels", file=sys.stderr)
        return 1
    name = args.run.name
    if args.format == "json":
        print(score_json(name, evaluation, per_query=args.per_query))
    else:
        print(score_markdown(name, evaluation, digits=args.digits, per_query=args.per_query))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            return _cmd_score(args)
    except (FormatError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
