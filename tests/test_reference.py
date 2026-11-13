"""Agreement with pytrec_eval (trec_eval) and ir_measures on random qrels and runs."""

import random

import pytest

from retrievalstats.metrics import evaluate

pytrec_eval = pytest.importorskip("pytrec_eval")
ir_measures = pytest.importorskip("ir_measures")

CUTOFFS = (1, 3, 5, 10, 20)

# Our name -> trec_eval name.
TREC_NAMES = {"map": "map", "mrr": "recip_rank", "ndcg": "ndcg", "rprec": "Rprec", "bpref": "bpref"}
for k in CUTOFFS:
    TREC_NAMES[f"P@{k}"] = f"P_{k}"
    TREC_NAMES[f"recall@{k}"] = f"recall_{k}"
    TREC_NAMES[f"map@{k}"] = f"map_cut_{k}"
    TREC_NAMES[f"ndcg@{k}"] = f"ndcg_cut_{k}"
    TREC_NAMES[f"success@{k}"] = f"success_{k}"


def random_collection(seed, n_queries=40, negative=False):
    # pytrec-eval-terrier 0.5.11 crashes (segmentation fault) at random on qrels with
    # negative labels, so only the ir_measures comparison uses them.
    rng = random.Random(seed)
    labels = [0, 0, 0, 1, 1, 2, 3] + ([-1, -2] if negative else [])
    qrels, run = {}, {}
    for q in range(n_queries):
        qid = f"q{q}"
        pool = [f"d{q}_{i}" for i in range(rng.randint(1, 40))]
        judged = rng.sample(pool, k=rng.randint(1, len(pool)))
        qrels[qid] = {d: rng.choice(labels) for d in judged}
        if rng.random() < 0.1:
            continue  # query missing from the run
        retrieved = rng.sample(pool, k=rng.randint(0, len(pool)))
        # Few distinct scores, so ties are common.
        run[qid] = {d: float(rng.randint(0, 4)) for d in retrieved}
    run["extra"] = {"x": 1.0}  # query with no judgments
    return qrels, run


def trec_eval_scores(qrels, run, names):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(names))
    return evaluator.evaluate({q: d for q, d in run.items() if d})


@pytest.mark.parametrize("seed", range(5))
def test_every_trec_eval_measure_matches(seed):
    qrels, run = random_collection(seed)
    ours = evaluate(qrels, run, list(TREC_NAMES))
    ref = trec_eval_scores(qrels, run, TREC_NAMES.values())
    # pytrec_eval also skips queries with an empty ranking; ours scores them 0 when the
    # query has an entry in the run. Compare on the queries both score.
    common = [q for q in ours.per_query if q in ref]
    assert len(common) > 30
    for qid in common:
        for name, trec_name in TREC_NAMES.items():
            assert ours.per_query[qid][name] == pytest.approx(ref[qid][trec_name], abs=1e-12), (
                qid,
                name,
            )


@pytest.mark.parametrize("seed", range(3))
def test_exponential_gain_matches_trec_eval_on_mapped_labels(seed):
    # trec_eval uses the label as the gain, so mapping each label l > 0 to 2**l - 1 turns
    # its ndcg into the exponential-gain variant.
    qrels, run = random_collection(seed)
    mapped = {q: {d: (2**v - 1 if v > 0 else v) for d, v in j.items()} for q, j in qrels.items()}
    names = {"ndcg_exp": "ndcg", "ndcg_exp@5": "ndcg_cut_5", "ndcg_exp@10": "ndcg_cut_10"}
    ours = evaluate(qrels, run, list(names))
    ref = trec_eval_scores(mapped, run, names.values())
    for qid in ours.per_query:
        if qid in ref:
            for name, trec_name in names.items():
                assert ours.per_query[qid][name] == pytest.approx(ref[qid][trec_name], abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_judged_matches_ir_measures(seed):
    qrels, run = random_collection(seed, negative=True)
    run = {q: d for q, d in run.items() if d}
    measures = [ir_measures.Judged @ k for k in CUTOFFS]
    # ir_measures orders tied documents its own way; judged@k depends on the order only at
    # the cut-off, so compare on rankings without ties.
    untied = {
        q: {d: s + i * 1e-3 for i, (d, s) in enumerate(docs.items())} for q, docs in run.items()
    }
    ours = evaluate(qrels, untied, [f"judged@{k}" for k in CUTOFFS])
    ref = {}
    for metric in ir_measures.iter_calc(measures, qrels, untied):
        ref[(metric.query_id, str(metric.measure).replace("Judged", "judged"))] = metric.value
    for qid, scores in ours.per_query.items():
        for name, value in scores.items():
            assert value == pytest.approx(ref[(qid, name)], abs=1e-12), (qid, name)


def test_means_match_trec_eval_averages():
    qrels, run = random_collection(99)
    run = {q: d for q, d in run.items() if d}
    ours = evaluate(qrels, run, ["map", "ndcg@10", "P@5", "bpref"]).mean()
    ref = trec_eval_scores(qrels, run, ["map", "ndcg_cut_10", "P_5", "bpref"])
    n = len(ref)
    pairs = {"map": "map", "ndcg@10": "ndcg_cut_10", "P@5": "P_5", "bpref": "bpref"}
    for name, trec_name in pairs.items():
        assert ours[name] == pytest.approx(sum(v[trec_name] for v in ref.values()) / n)


@pytest.mark.parametrize("seed", range(3))
def test_judged_on_tied_runs_matches_ir_measures_with_docno_ties(seed):
    # ir_measures orders tied documents by ascending id, so on tied runs judged@k agrees
    # with it under ties="docno" (the trec_eval rule orders them the other way).
    qrels, run = random_collection(seed, negative=True)
    run = {q: d for q, d in run.items() if d}
    measures = [ir_measures.Judged @ k for k in CUTOFFS]
    ref = {
        (m.query_id, str(m.measure).replace("Judged", "judged")): m.value
        for m in ir_measures.iter_calc(measures, qrels, run)
    }
    ours = evaluate(qrels, run, [f"judged@{k}" for k in CUTOFFS], ties="docno")
    for qid, scores in ours.per_query.items():
        for name, value in scores.items():
            assert value == pytest.approx(ref[(qid, name)], abs=1e-12), (qid, name)


def test_trec_eval_rule_compares_scores_as_float32():
    # pytrec_eval ranks 1.0 and 1.000000001 as tied (equal in 32 bits), so the tie goes
    # to the larger document id.
    qrels = {"q": {"a": 1, "b": 0}}
    run = {"q": {"b": 1.0, "a": 1.000000001}}
    ref = trec_eval_scores(qrels, run, ["recip_rank"])["q"]["recip_rank"]
    assert ref == 0.5
    assert evaluate(qrels, run, ["mrr"]).mean()["mrr"] == ref
    # The other rules keep full precision.
    assert evaluate(qrels, run, ["mrr"], ties="docno").mean()["mrr"] == 1.0


@pytest.mark.parametrize("seed", range(3))
def test_near_ties_match_trec_eval(seed):
    rng = random.Random(seed)
    qrels, run = random_collection(seed)
    run = {
        q: {d: s + rng.choice([0.0, 1e-9, 2e-9, 1e-3]) for d, s in docs.items()}
        for q, docs in run.items()
    }
    names = {"mrr": "recip_rank", "map": "map", "ndcg@5": "ndcg_cut_5", "P@3": "P_3"}
    ours = evaluate(qrels, run, list(names))
    ref = trec_eval_scores(qrels, run, names.values())
    for qid in ours.per_query:
        if qid in ref:
            for name, trec_name in names.items():
                assert ours.per_query[qid][name] == pytest.approx(ref[qid][trec_name], abs=1e-12)
