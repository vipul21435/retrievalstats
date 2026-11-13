import json
from pathlib import Path

import pytest

from retrievalstats.cli import main

TINY = Path(__file__).resolve().parents[1] / "examples" / "tiny"
QRELS = str(TINY / "qrels.txt")
RUN_A = str(TINY / "run_a.txt")
RUN_B = str(TINY / "run_b.txt")


def test_score_markdown_default_measures(capsys):
    assert main(["score", QRELS, RUN_A]) == 0
    out = capsys.readouterr().out
    assert "queries scored: 4" in out
    assert "1 in qrels but missing from the run" in out
    assert "| ndcg@10 |" in out
    assert "| judged@10 |" in out


def test_score_per_query_values(capsys):
    assert main(["score", QRELS, RUN_B, "-m", "P@5,ndcg@5", "-m", "mrr", "--per-query"]) == 0
    out = capsys.readouterr().out
    # q1: doc01, then the tie at 2.5 puts doc96 ahead of doc02 (trec_eval order).
    assert "| query | P@5 | ndcg@5 | mrr |" in out
    assert "| q1 | 0.6000 | 0.9220 | 1.0000 |" in out
    assert "| q4 | 0.0000 | 0.0000 | 0.0000 |" in out


def test_score_json(capsys):
    assert main(["score", QRELS, RUN_B, "-m", "mrr", "--format", "json", "--per-query"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["queries"] == 4
    assert payload["mean"]["mrr"] == pytest.approx((1 + 1 + 0.5 + 0) / 4)
    assert payload["skipped"]["missing_from_run"] == ["q5"]
    assert payload["per_query"]["q3"]["mrr"] == 0.5


def test_score_missing_zero(capsys):
    assert main(["score", QRELS, RUN_B, "-m", "mrr", "--missing", "zero", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["queries"] == 5
    assert payload["mean"]["mrr"] == pytest.approx(2.5 / 5)


def test_score_ties_option_changes_result(capsys):
    main(["score", QRELS, RUN_B, "-m", "P@2", "--ties", "docno", "--format", "json"])
    docno = json.loads(capsys.readouterr().out)["mean"]["P@2"]
    main(["score", QRELS, RUN_B, "-m", "P@2", "--format", "json"])
    trec = json.loads(capsys.readouterr().out)["mean"]["P@2"]
    assert docno > trec


def test_unknown_measure_is_an_error(capsys):
    assert main(["score", QRELS, RUN_A, "-m", "foo"]) == 2
    assert "unknown measure" in capsys.readouterr().err


def test_bad_file_is_an_error(tmp_path, capsys):
    bad = tmp_path / "run.txt"
    bad.write_text("q1 d1\n")
    assert main(["score", QRELS, str(bad)]) == 2
    assert "fields" in capsys.readouterr().err


def test_no_overlap_is_an_error(tmp_path, capsys):
    other = tmp_path / "run.txt"
    other.write_text("zz Q0 d1 1 1.0 t\n")
    assert main(["score", QRELS, str(other)]) == 1
    assert "no query" in capsys.readouterr().err
