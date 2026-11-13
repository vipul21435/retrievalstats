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


def test_compare_markdown(capsys):
    assert main(["compare", QRELS, RUN_A, RUN_B, "-m", "ndcg@5,map", "--samples", "2000"]) == 0
    out = capsys.readouterr().out
    assert "queries paired: 4" in out
    assert "| measure | t | wilcoxon | sign | randomisation | bootstrap |" in out
    assert "| q1 | 0.5406 | 0.9220 | +0.3815 |" in out
    assert "warning: judged@10 differs" in out


def test_compare_json_and_options(capsys):
    args = ["compare", QRELS, RUN_A, RUN_B, "-m", "mrr", "--tests", "t,sign", "--format", "json"]
    assert main([*args, "--correction", "bh", "--missing", "zero"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["queries"] == 5
    assert set(payload["results"][0]["p_values"]) == {"t", "sign"}
    assert payload["correction"] == "bh"


def test_compare_rejects_unknown_test(capsys):
    assert main(["compare", QRELS, RUN_A, RUN_B, "--tests", "z"]) == 2
    assert "unknown test" in capsys.readouterr().err


def test_table(capsys):
    assert main(["table", QRELS, RUN_A, RUN_B, "-m", "ndcg@5", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline"] == "run_a.txt"
    assert payload["means"]["run_b.txt"]["ndcg@5"] == pytest.approx(0.6258, abs=1e-4)
    assert main(["table", QRELS, RUN_A, RUN_B, "-m", "ndcg@5"]) == 0
    assert "| run_b.txt |" in capsys.readouterr().out
    assert main(["table", QRELS, RUN_A, RUN_A]) == 2


def test_bm25_command(tmp_path, capsys):
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"_id": "d1", "title": "cats", "text": "the cat sat"},
        {"_id": "d2", "title": "dogs", "text": "a dog ran"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"_id": "1", "text": "cat"}) + "\n" + json.dumps({"_id": "2", "text": "dog"})
    )
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\n1\td1\t1\n")
    out = tmp_path / "run.txt"
    assert main(["bm25", str(corpus), str(queries), "-o", str(out), "--qrels", str(qrels)]) == 0
    assert "1 queries" in capsys.readouterr().out
    assert out.read_text().split()[:3] == ["1", "Q0", "d1"]
    assert main(["score", str(qrels), str(out), "-m", "mrr", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["mean"]["mrr"] == 1.0
