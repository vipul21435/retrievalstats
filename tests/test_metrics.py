import math
import random

import pytest

from retrievalstats.metrics import (
    Measure,
    evaluate,
    parse_measure,
    parse_measures,
    query_scores,
    rank,
)

# One query worked by hand. Judged: d1 (2), d3 (1), d4 (1) relevant; d2, d5 non-relevant.
# Ranking: d2, d1, d6 (unjudged), d3, d5. d4 is never retrieved. R = 3.
JUDGMENTS = {"d1": 2, "d2": 0, "d3": 1, "d4": 1, "d5": 0}
RANKING = ["d2", "d1", "d6", "d3", "d5"]


def score(*names, ranking=RANKING, judgments=JUDGMENTS, level=1):
    return query_scores(ranking, judgments, parse_measures(names), relevance_level=level)


def test_precision_recall_success():
    got = score("P@1", "P@2", "P@5", "P@10", "recall@2", "recall@5", "success@1", "success@2")
    assert got["P@1"] == 0.0
    assert got["P@2"] == 1 / 2
    assert got["P@5"] == 2 / 5
    assert got["P@10"] == 2 / 10  # divides by k even with 5 retrieved
    assert got["recall@2"] == pytest.approx(1 / 3)
    assert got["recall@5"] == pytest.approx(2 / 3)
    assert got["success@1"] == 0.0
    assert got["success@2"] == 1.0


def test_map_mrr_rprec():
    got = score("map", "map@2", "mrr", "mrr@1", "rprec")
    assert got["map"] == pytest.approx((1 / 2 + 2 / 4) / 3)
    assert got["map@2"] == pytest.approx((1 / 2) / 3)
    assert got["mrr"] == 1 / 2
    assert got["mrr@1"] == 0.0
    assert got["rprec"] == pytest.approx(1 / 3)  # one relevant in the top R = 3


def test_bpref():
    # d2 is a judged non-relevant above both retrieved relevant documents.
    # Each relevant contributes 1 - min(1, R) / min(N, R) = 1 - 1/2, with N = 2, R = 3.
    assert score("bpref")["bpref"] == pytest.approx((0.5 + 0.5) / 3)


def test_bpref_skips_unjudged_and_handles_no_nonrelevant():
    judgments = {"a": 1, "b": 1}
    assert score("bpref", ranking=["x", "a", "y", "b"], judgments=judgments)["bpref"] == 1.0


def test_negative_labels_are_skipped_by_bpref_and_counted_by_judged():
    # Same numbers pytrec_eval gives: a negative label behaves like a missing judgment in
    # bpref, and like a non-relevant document with gain 0 elsewhere.
    judgments = {"a": -1, "b": 1, "c": 0, "d": 2, "e": -2}
    ranking = ["a", "b", "c", "e", "d", "x"]
    got = score("bpref", "map", "P@5", "judged@6", ranking=ranking, judgments=judgments)
    assert got["bpref"] == 0.5
    assert got["map"] == pytest.approx((1 / 2 + 2 / 5) / 2)
    assert got["P@5"] == 2 / 5
    assert got["judged@6"] == pytest.approx(5 / 6)


def test_relevance_level_must_be_positive():
    with pytest.raises(ValueError):
        evaluate({"q": {"a": 1}}, {"q": {"a": 1.0}}, ["P@1"], relevance_level=0)


def test_judged():
    got = score("judged@2", "judged@5", "judged@10")
    assert got["judged@2"] == 1.0
    assert got["judged@5"] == pytest.approx(4 / 5)
    assert got["judged@10"] == pytest.approx(4 / 5)  # only five were retrieved


def test_ndcg_linear_and_exponential():
    log2 = math.log2
    dcg = 2 / log2(3) + 1 / log2(5)
    ideal = 2 + 1 / log2(3) + 1 / log2(4)
    dcg_exp = 3 / log2(3) + 1 / log2(5)
    ideal_exp = 3 + 1 / log2(3) + 1 / log2(4)
    got = score("ndcg", "ndcg@2", "ndcg_exp", "ndcg_exp@2")
    assert got["ndcg"] == pytest.approx(dcg / ideal)
    assert got["ndcg@2"] == pytest.approx((2 / log2(3)) / (2 + 1 / log2(3)))
    assert got["ndcg_exp"] == pytest.approx(dcg_exp / ideal_exp)
    assert got["ndcg_exp@2"] == pytest.approx((3 / log2(3)) / (3 + 1 / log2(3)))


def test_relevance_level_two():
    got = score("P@5", "map", "rprec", level=2)
    assert got["P@5"] == 1 / 5
    assert got["map"] == 1 / 2  # only d1 is relevant, found at rank 2
    assert got["rprec"] == 0.0


def test_no_relevant_document_scores_zero():
    got = score("P@5", "recall@5", "map", "mrr", "ndcg@5", "rprec", "bpref", judgments={"d1": 0})
    assert set(got.values()) == {0.0}


def test_empty_ranking():
    got = score("P@5", "map", "ndcg@5", "judged@5", "bpref", ranking=[])
    assert set(got.values()) == {0.0}


def test_rank_tie_rules():
    scores = {"b": 1.0, "c": 2.0, "a": 1.0, "d": 1.0}
    assert rank(scores) == ["c", "d", "b", "a"]
    assert rank(scores, ties="docno") == ["c", "a", "b", "d"]
    assert rank(scores, ties="input") == ["c", "b", "a", "d"]
    with pytest.raises(ValueError):
        rank(scores, ties="random")


def test_ties_change_the_score():
    qrels = {"q": {"a": 1}}
    run = {"q": {"a": 1.0, "b": 1.0}}
    assert evaluate(qrels, run, ["mrr"]).mean()["mrr"] == 1 / 2  # trec_eval puts b first
    assert evaluate(qrels, run, ["mrr"], ties="docno").mean()["mrr"] == 1.0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ndcg@10", Measure("ndcg", 10)),
        ("NDCG_cut_10", Measure("ndcg", 10)),
        ("ndcg_exp@5", Measure("ndcg_exp", 5)),
        ("P_5", Measure("P", 5)),
        ("p@5", Measure("P", 5)),
        ("recip_rank", Measure("mrr")),
        ("Rprec", Measure("rprec")),
        ("map", Measure("map")),
        ("map_cut_100", Measure("map", 100)),
        (" success@1 ", Measure("success", 1)),
    ],
)
def test_parse_measure(text, expected):
    assert parse_measure(text) == expected


@pytest.mark.parametrize("text", ["P", "recall", "bpref@10", "rprec@5", "ndcg@0", "foo@3", "@3"])
def test_parse_measure_rejects(text):
    with pytest.raises(ValueError):
        parse_measure(text)


def test_parse_measures_drops_repeats():
    assert [str(m) for m in parse_measures(["P@5", "P_5", "map"])] == ["P@5", "map"]


def test_evaluate_query_rules():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 0}}
    run = {"q1": {"a": 1.0}, "q3": {"c": 1.0}, "q9": {"z": 1.0}}
    default = evaluate(qrels, run, ["P@1"])
    assert default.queries == ["q1", "q3"]
    assert default.mean()["P@1"] == 1 / 2
    assert default.skipped["missing_from_run"] == ["q2"]
    assert default.skipped["not_in_qrels"] == ["q9"]

    zero = evaluate(qrels, run, ["P@1"], missing="zero")
    assert zero.queries == ["q1", "q2", "q3"]
    assert zero.mean()["P@1"] == pytest.approx(1 / 3)

    skip = evaluate(qrels, run, ["P@1"], no_relevant="skip")
    assert skip.queries == ["q1"]
    assert skip.skipped["no_relevant"] == ["q3"]

    with pytest.raises(ValueError):
        evaluate(qrels, run, ["P@1"], missing="drop")
    with pytest.raises(ValueError):
        evaluate(qrels, run, ["P@1"], no_relevant="drop")


def test_queries_sorted_naturally():
    qrels = {q: {"a": 1} for q in ["q10", "q2", "q1"]}
    run = {q: {"a": 1.0} for q in qrels}
    assert evaluate(qrels, run, ["P@1"]).queries == ["q1", "q2", "q10"]


def _random_query(rng):
    docs = [f"d{i}" for i in range(rng.randint(0, 30))]
    judged = rng.sample(docs, k=rng.randint(0, len(docs))) if docs else []
    judgments = {d: rng.choice([0, 0, 1, 2, 3]) for d in judged}
    if not judgments:
        judgments = {"never": 1}
    ranking = rng.sample(docs, k=len(docs))
    return ranking, judgments


ALL = parse_measures(
    ["P@5", "recall@5", "success@5", "judged@5", "map", "map@5", "mrr", "mrr@5"]
    + ["ndcg", "ndcg@5", "ndcg_exp@5", "rprec", "bpref"]
)


def test_every_metric_lies_in_unit_interval():
    rng = random.Random(7)
    for _ in range(500):
        ranking, judgments = _random_query(rng)
        for name, value in query_scores(ranking, judgments, ALL).items():
            assert 0.0 <= value <= 1.0 + 1e-12, (name, value)


def test_appending_below_k_never_changes_a_metric_at_k():
    rng = random.Random(11)
    k = 5
    at_k = [m for m in ALL if m.cutoff == k]
    for _ in range(300):
        ranking, judgments = _random_query(rng)
        if len(ranking) < k:
            continue
        before = query_scores(ranking, judgments, at_k)
        extra = [f"x{i}" for i in range(rng.randint(1, 5))] + list(judgments)
        tail = [d for d in extra if d not in ranking]
        after = query_scores(ranking + tail, judgments, at_k)
        assert before == after


def test_perfect_ranking_scores_one():
    judgments = {"a": 3, "b": 2, "c": 1, "z": 0}
    names = ("ndcg", "ndcg_exp", "map", "mrr", "rprec", "bpref", "ndcg@2")
    got = score(*names, ranking=["a", "b", "c"], judgments=judgments)
    assert all(v == pytest.approx(1.0) for v in got.values()), got
