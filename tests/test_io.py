import json

import pytest

from retrievalstats.io import FormatError, read_qrels, read_run, write_trec_run


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_trec_qrels(tmp_path):
    path = write(tmp_path, "qrels.txt", "q1 0 d1 1\nq1 0 d2 0\n# comment\n\nq2 0 d3 2\n")
    assert read_qrels(path) == {"q1": {"d1": 1, "d2": 0}, "q2": {"d3": 2}}


def test_beir_tsv_qrels_with_header(tmp_path):
    path = write(tmp_path, "test.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t2\n")
    assert read_qrels(path) == {"q1": {"d1": 1, "d2": 2}}


def test_tsv_qrels_without_header(tmp_path):
    path = write(tmp_path, "qrels.tsv", "q1\td1\t1\n")
    assert read_qrels(path, fmt="tsv") == {"q1": {"d1": 1}}


def test_jsonl_qrels(tmp_path):
    rows = [{"query-id": "q1", "corpus-id": "d1", "score": 1}, {"qid": 7, "docid": "d9", "rel": 0}]
    path = write(tmp_path, "qrels.jsonl", "\n".join(json.dumps(r) for r in rows))
    assert read_qrels(path) == {"q1": {"d1": 1}, "7": {"d9": 0}}


def test_qrels_duplicate_is_an_error(tmp_path):
    path = write(tmp_path, "qrels.txt", "q1 0 d1 1\nq1 0 d1 0\n")
    with pytest.raises(FormatError, match="judged twice"):
        read_qrels(path)


def test_qrels_float_label_must_be_integral(tmp_path):
    assert read_qrels(write(tmp_path, "a.txt", "q1 0 d1 2.0\n")) == {"q1": {"d1": 2}}
    with pytest.raises(FormatError, match="not an integer"):
        read_qrels(write(tmp_path, "b.txt", "q1 0 d1 0.5\n"))


def test_trec_run_ignores_rank_column(tmp_path):
    path = write(tmp_path, "run.txt", "q1 Q0 d1 5 1.5 tag\nq1 Q0 d2 1 0.5 tag\n")
    assert read_run(path) == {"q1": {"d1": 1.5, "d2": 0.5}}


def test_run_keeps_file_order(tmp_path):
    path = write(tmp_path, "run.tsv", "q1\tdz\t1\nq1\tda\t1\nq1\tdm\t1\n")
    assert list(read_run(path)["q1"]) == ["dz", "da", "dm"]


def test_run_duplicate_is_an_error(tmp_path):
    path = write(tmp_path, "run.txt", "q1 Q0 d1 1 1.0 t\nq1 Q0 d1 2 0.5 t\n")
    with pytest.raises(FormatError, match="listed twice"):
        read_run(path)


def test_run_rejects_nan(tmp_path):
    path = write(tmp_path, "run.txt", "q1 Q0 d1 1 nan t\n")
    with pytest.raises(FormatError, match="NaN"):
        read_run(path)


def test_json_run(tmp_path):
    path = write(tmp_path, "results.json", json.dumps({"q1": {"d1": 0.3, "d2": 0.1}}))
    assert read_run(path) == {"q1": {"d1": 0.3, "d2": 0.1}}


def test_jsonl_run(tmp_path):
    path = write(
        tmp_path, "run.jsonl", json.dumps({"query-id": "q1", "corpus-id": "d1", "score": 2})
    )
    assert read_run(path) == {"q1": {"d1": 2.0}}


def test_unknown_width_is_reported(tmp_path):
    path = write(tmp_path, "run.txt", "q1 d1\n")
    with pytest.raises(FormatError, match="2 fields"):
        read_run(path)


def test_wrong_width_in_trec_run(tmp_path):
    path = write(tmp_path, "run.txt", "q1 Q0 d1 1 1.0 t\nq1 Q0 d2 1.0\n")
    with pytest.raises(FormatError, match=":2: expected 6 fields"):
        read_run(path)


def test_write_then_read_round_trip(tmp_path):
    run = {"q1": {"a": 0.1 + 0.2, "b": 0.3, "c": 2.0}, "q2": {"x": -1.0}}
    path = tmp_path / "out.txt"
    write_trec_run(run, path, tag="bm25")
    assert read_run(path) == run
    lines = path.read_text().splitlines()
    assert lines[0].split()[2] == "c"
    assert lines[0].split()[5] == "bm25"
