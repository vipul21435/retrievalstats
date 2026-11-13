"""Largest per-query difference between retrievalstats and pytrec_eval on real runs.

Usage: python scripts/agreement.py QRELS RUN [RUN ...]   (needs pytrec-eval-terrier)
"""

import sys

import pytrec_eval

from retrievalstats.io import read_qrels, read_run
from retrievalstats.metrics import evaluate

PAIRS = {
    "P@10": "P_10",
    "recall@100": "recall_100",
    "map": "map",
    "map@100": "map_cut_100",
    "mrr": "recip_rank",
    "ndcg": "ndcg",
    "ndcg@10": "ndcg_cut_10",
    "rprec": "Rprec",
    "bpref": "bpref",
    "success@1": "success_1",
    "success@10": "success_10",
}


def main(argv):
    qrels = read_qrels(argv[0])
    print("| run | measure | queries | largest absolute difference |")
    print("| --- | --- | ---: | ---: |")
    for path in argv[1:]:
        run = read_run(path)
        ours = evaluate(qrels, run, list(PAIRS))
        ref = pytrec_eval.RelevanceEvaluator(qrels, set(PAIRS.values())).evaluate(run)
        for name, trec_name in PAIRS.items():
            gap = max(abs(ours.per_query[q][name] - ref[q][trec_name]) for q in ref)
            print(f"| {path.rsplit('/', 1)[-1]} | {name} | {len(ref)} | {gap:.1e} |")


if __name__ == "__main__":
    main(sys.argv[1:])
