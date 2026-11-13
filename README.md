# retrievalstats

Score retrieval runs against relevance judgments and say whether one run is really better than another.

Status: early. Metric scoring and the `score` command work; paired significance tests, the `compare`,
`table` and `bm25` commands and the real-data results are not written yet.

## Why this is hard

A difference in mean nDCG over 50 to 300 queries is noisy, so a single number does not tell you whether a
new system is better. Tools also disagree in small details (the gain function, the order of documents with
tied scores, unjudged documents, queries with no relevant document), and those details move the numbers
enough that scores from different tools are not comparable. This package makes each of those choices an
explicit, documented option.

## Install

```
pip install git+https://github.com/vipul21435/retrievalstats
```

Python 3.10 or newer. No runtime dependencies so far.

## Quickstart

The `examples/tiny` folder holds a five-query collection written by hand.

```
$ retrievalstats score examples/tiny/qrels.txt examples/tiny/run_a.txt \
    -m ndcg@10,ndcg_exp@10,map,mrr,P@5,recall@5,rprec,bpref,success@1,judged@5
run: run_a.txt
queries scored: 4
not scored: 1 in qrels but missing from the run

| measure | mean |
| --- | ---: |
| ndcg@10 | 0.4287 |
| ndcg_exp@10 | 0.4144 |
| map | 0.3417 |
| mrr | 0.4583 |
| P@5 | 0.2500 |
| recall@5 | 0.6667 |
| rprec | 0.2083 |
| bpref | 0.2083 |
| success@1 | 0.2500 |
| judged@5 | 0.6917 |
```

`--per-query` adds a row per query and `--format json` prints the same data as JSON.

## What works so far

- Readers for TREC qrels (`qid iter docno rel`) and runs (`qid Q0 docno rank score tag`), BEIR-style TSV
  (`query-id`, `corpus-id`, `score`, header optional), JSONL with the same keys, and the nested
  `{qid: {docid: score}}` JSON that BEIR writes. The format is detected from the file; `--qrels-format`
  and `--run-format` override it.
- Measures: `P@k`, `recall@k`, `map` and `map@k`, `mrr` and `mrr@k`, `ndcg` and `ndcg@k` (label as gain),
  `ndcg_exp` and `ndcg_exp@k` (gain `2**label - 1`), `rprec`, `bpref`, `success@k` and `judged@k` (the
  share of the retrieved top-k documents that have a judgment). trec_eval spellings such as `P_10`,
  `ndcg_cut_10` and `recip_rank` are accepted.
- Conventions, each a flag. The defaults are meant to match trec_eval; agreement with it has not been
  checked yet.
  - `--ties`: documents are ordered by descending score; equal scores are broken by descending document id
    (`trec_eval`, default), ascending id (`docno`) or file order (`input`). The rank column is ignored.
  - `--missing`: a query in the qrels with no line in the run is skipped (`skip`, default) or scores 0 on
    every measure (`zero`).
  - `--no-relevant`: a query whose judgments contain no relevant document is kept and scores 0 (`keep`,
    default) or is dropped (`skip`).
  - `--relevance-level`: the smallest label that counts as relevant (default 1).
  - `P@k` divides by `k` even when fewer than `k` documents were retrieved; `judged@k` divides by the number
    retrieved in the top `k`. Unjudged documents count as non-relevant, except in bpref, which skips them.

## Not implemented yet

Paired tests (t-test, Wilcoxon, sign, randomisation, bootstrap), effect sizes and intervals, Holm and
Benjamini-Hochberg corrections, the per-query difference table and the judged@k warning, the minimum
detectable difference, the BM25 baseline, the `compare`, `table` and `bm25` commands, and any run on real
data.

## Related tools

- [evalstats](https://github.com/vipul21435/evalstats): error bars and paired comparisons for evaluations
  in general.

## Licence

MIT, see [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
