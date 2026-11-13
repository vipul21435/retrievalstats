import json
import math

import pytest

from retrievalstats.bm25 import BM25, idf, read_corpus, read_queries, tokenize

DOCS = {
    "d1": "the cat sat on the mat",
    "d2": "the dog sat",
    "d3": "cats and dogs and cats",
}


def test_tokenize():
    assert tokenize("Hello, World! x2 don't") == ["hello", "world", "x2", "don", "t"]


def test_idf_variants():
    assert idf(1, 3) == pytest.approx(math.log(1 + 2.5 / 1.5))
    assert idf(1, 3, "robertson") == pytest.approx(math.log(2.5 / 1.5))
    assert idf(3, 4, "robertson") < 0 < idf(3, 4)
    with pytest.raises(ValueError):
        idf(1, 3, "bm15")


def test_hand_worked_score():
    index = BM25(DOCS, k1=1.2, b=0.75)
    avgdl = (6 + 3 + 5) / 3
    # "sat" appears in d1 and d2 once; df = 2.
    w = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected_d1 = w * 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * 6 / avgdl))
    expected_d2 = w * 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * 3 / avgdl))
    scores = index.search("sat")
    assert scores["d1"] == pytest.approx(expected_d1)
    assert scores["d2"] == pytest.approx(expected_d2)
    assert list(scores) == ["d2", "d1"]  # the shorter document wins
    assert "d3" not in scores


def test_b_zero_removes_length_normalisation():
    index = BM25(DOCS, k1=1.2, b=0.0)
    scores = index.search("sat")
    assert scores["d1"] == pytest.approx(scores["d2"])
    assert list(scores) == ["d2", "d1"]  # tie broken by descending id


def test_repeated_query_token_counts_twice():
    index = BM25(DOCS)
    assert index.search("dog dog")["d2"] == pytest.approx(2 * index.search("dog")["d2"])


def test_depth_and_unknown_terms():
    index = BM25(DOCS)
    assert index.search("zebra") == {}
    assert len(index.search("the cats sat", depth=2)) == 2
    run = index.run({"q1": "cat", "q2": "zebra"})
    assert list(run["q1"]) == ["d1"] and run["q2"] == {}


def test_parameters_are_checked():
    with pytest.raises(ValueError):
        BM25(DOCS, b=1.5)
    with pytest.raises(ValueError):
        BM25(DOCS, idf_variant="x")


def test_read_beir_files(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [{"_id": "1", "title": "Cats", "text": "sat"}, {"_id": "2", "title": "", "text": "dog"}]
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert read_corpus(corpus) == {"1": "Cats sat", "2": "dog"}
    assert read_corpus(corpus, fields=["text"]) == {"1": "sat", "2": "dog"}
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"_id": "7", "text": "cat?"}) + "\n")
    assert read_queries(queries) == {"7": "cat?"}
    tsv = tmp_path / "queries.tsv"
    tsv.write_text("7\tcat?\n8\tdog\n")
    assert read_queries(tsv) == {"7": "cat?", "8": "dog"}
    assert read_corpus(tsv) == {"7": "cat?", "8": "dog"}
