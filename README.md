# retrievalstats

Score retrieval runs against relevance judgments and say whether one run is really better than another, with
per-query paired tests and control for multiple comparisons.

## Why this is hard

A difference in mean nDCG over 50 to 300 queries is noisy: a few queries can move it, and with several
measures and several runs some difference will look significant by chance. Tools also disagree in small
details of the metrics (the gain function, the order of documents with tied scores, unjudged documents,
queries with no relevant document), and those details move the numbers enough that scores from different
tools are not comparable. retrievalstats makes each of those choices an explicit option with a
trec_eval-compatible default, and reports the paired tests, intervals and corrections next to the means.

## Install

```
pip install git+https://github.com/vipul21435/retrievalstats
```

Python 3.10 or newer; numpy and scipy are the only runtime dependencies.

## Quickstart

`examples/tiny` holds a five-query collection written by hand.

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

```
$ retrievalstats compare examples/tiny/qrels.txt examples/tiny/run_a.txt examples/tiny/run_b.txt \
    -m ndcg@5,map --tests t,randomisation --top 2
A: run_a.txt
B: run_b.txt
queries paired: 4

Differences are B minus A. Intervals are 95 %. Effect is mean(d) / sd(d).

| measure | A | B | B - A | t interval | bootstrap interval | effect | wins/ties/losses | MDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ndcg@5 | 0.4287 | 0.6258 | +0.1971 | [-0.0683, +0.4625] | [+0.0655, +0.3288] | 1.182 | 3/1/0 | 0.3549 |
| map | 0.3417 | 0.5222 | +0.1806 | [-0.1002, +0.4614] | [+0.0417, +0.3500] | 1.023 | 3/1/0 | 0.3755 |

Two-sided p-values, then in brackets after holm correction across the 2 measures within each test; * marks an adjusted p below 0.05.

| measure | t | randomisation |
| --- | ---: | ---: |
| ndcg@5 | 0.0991 (0.1981) | 0.2500 (0.5000) |
| map | 0.1332 (0.1981) | 0.2500 (0.5000) |

Largest per-query differences on ndcg@5:

| query | A | B | B - A |
| --- | ---: | ---: | ---: |
| q1 | 0.5406 | 0.9220 | +0.3815 |
| q2 | 0.6742 | 0.9502 | +0.2761 |

judged@10: A 0.658, B 0.804
warning: judged@10 differs: run_a.txt 0.658, run_b.txt 0.804. Unjudged documents count as non-relevant, so the comparison is biased against run_a.txt relative to run_b.txt.
warning: only 4 paired queries: below 30 the bootstrap p-value is too small and its interval too narrow (it rejected a true null in 16 % of simulated samples at n = 5 and 10 % at n = 10); rely on the t or randomisation test.
```

With four queries neither test finds a difference, and the minimum detectable difference (MDD) says why:
at this variance a true gain of 0.35 nDCG@5 would be detected in 80 % of samples, and smaller gains less
often. The other commands are
`retrievalstats table QRELS RUN...` (one row per run, each tested against the first) and
`retrievalstats bm25 CORPUS QUERIES -o RUN`. Every command except `bm25` takes `--format json`.
`python examples/compare_runs.py` does the same through the Python API.

## How it works

### Input

TREC qrels (`qid iter docno rel`) and runs (`qid Q0 docno rank score tag`); BEIR-style TSV (`query-id`,
`corpus-id`, `score`, header optional); JSONL with the same keys; and the nested `{qid: {docid: score}}`
JSON that BEIR writes. The format is detected from the file (`--qrels-format`, `--run-format` override it).
A document judged twice, or listed twice in one query of a run, is an error.

### Metrics and their conventions

| name | definition |
| --- | --- |
| `P@k` | relevant documents in the top k, divided by k (even when fewer than k were retrieved) |
| `recall@k` | relevant in the top k divided by R, the number of relevant documents judged for the query |
| `map`, `map@k` | sum over retrieved relevant documents (within k) of the precision at their rank, divided by R |
| `mrr`, `mrr@k` | 1 / rank of the first relevant document, 0 if none (within k) |
| `ndcg`, `ndcg@k` | DCG with gain = label and discount log2(rank + 1), divided by the DCG of the ideal ordering of all judged documents (Jarvelin and Kekalainen 2002) |
| `ndcg_exp`, `ndcg_exp@k` | the same with gain 2^label - 1 |
| `rprec` | precision at rank R |
| `bpref` | (1/R) times the sum over retrieved relevant r of (1 - min(n_r, R) / min(N, R)), where n_r is the number of judged non-relevant documents ranked above r and N the number judged non-relevant; unjudged documents are skipped (Buckley and Voorhees 2004, in the form trec_eval computes) |
| `success@k` | 1 if a relevant document is in the top k |
| `judged@k` | share of the retrieved top-k documents that have a judgment, dividing by min(k, retrieved) as ir_measures' `Judged` does |

trec_eval spellings (`P_10`, `ndcg_cut_10`, `map_cut_100`, `recip_rank`, `Rprec`) are accepted.
The conventions, each a flag, with defaults that follow trec_eval:

- `--ties`: documents are ordered by descending score. Equal scores are ordered by descending document
  id, the rule trec_eval and pytrec_eval apply (`trec_eval`, default; the agreement tests below use runs
  full of ties). The alternatives are ascending id (`docno`) and file order (`input`). The rank column of
  a TREC run is never read. For a query whose one relevant document ties with an unjudged one, the rule
  alone moves MRR between 0.5 and 1.0 (`tests/test_metrics.py::test_ties_change_the_score`). Under the
  `trec_eval` rule scores are compared as 32-bit floats, as pytrec_eval does: 1.0 and 1.000000001 tie,
  so a relevant document scored 1.000000001 next to a non-relevant one scored 1.0 gets MRR 0.5 when the
  non-relevant document has the larger id. `docno` and `input` compare the full 64-bit scores.
- `--missing`: a query in the qrels with no line in the run is skipped (`skip`, default) or scores 0 on
  every measure (`zero`, trec_eval's `-c`). Queries only in the run are always ignored.
- `--no-relevant`: a query whose judgments contain no relevant document is kept and scores 0 (`keep`,
  default) or dropped (`skip`).
- `--relevance-level`: the smallest label that counts as relevant (default 1).
- Negative labels: bpref skips them, as if the document were unjudged. Every other trec_eval measure
  treats them as non-relevant with gain 0. judged@k counts them as judged, as ir_measures does.

The two conventions left open in the first version were settled by running the references.
ir_measures 0.4.3 gives `Judged@10 = 1.0` for a three-document run with every document judged, so it
divides by the number retrieved. pytrec_eval gave the same numbers for labels -1, -2 and -3 as with the
document removed from the qrels. The random agreement tests use labels 0 to 3 only, because
pytrec-eval-terrier 0.5.11 repeatedly crashed with a segmentation fault on qrels holding negative labels.

### Paired tests

For queries scored in both runs, let d_i = B_i - A_i. All p-values are two-sided.

- **t-test**: t = mean(d) / (sd(d) / sqrt(n)) with n - 1 degrees of freedom.
- **Wilcoxon signed-rank**: zero differences dropped; scipy's implementation (exact distribution for
  small samples without ties, normal approximation otherwise).
- **Sign test**: wins against losses under Binomial(n0, 1/2), n0 = non-zero differences; `--tie-threshold`
  treats |d| at or below the threshold as a tie.
- **Fisher randomisation** (paired permutation): under the null each sign of d_i is a fair coin; p is the
  share of sign patterns whose |sum| is at least the observed |sum|. When 2^n0 is at most `--samples`, every
  pattern is enumerated and p is exact; otherwise `--samples` random patterns are drawn and the observed one
  counts as one of them, p = (count + 1) / (samples + 1).
- **Paired bootstrap**: `--samples` resamples of d with replacement; the p-value uses the shift method
  (bootstrap means centred on their own mean and compared with the observed mean), and the interval is the
  percentile interval.

These are the five tests compared by Smucker, Allan and Carterette (2007) and by Urbano, Lima and Hanjalic
(2019). Urbano et al. found that "the sign and Wilcoxon tests are unreliable for hypotheses about mean
effectiveness", that the t-test and the permutation test keep the type I error at the nominal level, and
that the bootstrap-shift test "does not behave well unless the sample size is large"; they recommend the
t-test for hypotheses about mean effectiveness. retrievalstats reports all five so that any disagreement is
visible; `table` uses the t-test by default.

Next to the p-values: the mean difference with a Student-t interval and the bootstrap percentile interval;
the effect size mean(d) / sd(d); win, tie and loss counts; and the **minimum detectable difference**, the
smallest true mean difference the two-sided paired t-test would detect with 80 % power at the observed
sd(d). It solves the noncentral t power function for delta; the tests check it against statsmodels'
`TTestPower`.

### Multiple comparisons

`--correction holm` (default) controls the family-wise error rate with Holm's step-down procedure
(Holm 1979): the sorted p-values p_(j) become the maximum over i <= j of (m - i + 1) p_(i), capped at 1.
`--correction bh` controls the false discovery rate (Benjamini and Hochberg 1995): p_(j) becomes the
minimum over i >= j of m p_(i) / i. `compare` corrects across the measures within each test; `table`
corrects across every (run, measure) comparison against the baseline. Fuhr (2017) lists testing several
hypotheses without correction among the common mistakes in IR evaluation.

### Diagnostics

- The per-query difference table lists the largest wins and losses on the first measure (`--top`).
- judged@k warning: unjudged documents count as non-relevant, so a run that retrieves more unjudged
  documents (often a new system that did not contribute to the pool) is scored unfairly low. `compare`
  warns when the two runs' mean judged@k differ by 0.05 or more, and when either is below 0.5.

### BM25

`retrievalstats bm25` indexes a corpus in memory and scores it with
idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |D| / avgdl)), summed over query tokens (Robertson and
Zaragoza 2009). The default idf is `ln(1 + (N - df + 0.5) / (df + 0.5))`, the formula Lucene's
`BM25Similarity` documents; `--idf robertson` uses `ln((N - df + 0.5) / (df + 0.5))`, which is negative for
terms in more than half of the documents. Tokens are lower-cased runs of word characters, with no stemming
and no stopword list. Defaults are k1 = 0.9 and b = 0.4.

## Results on real public data

Data: BEIR SciFact (Thakur et al. 2021; Wadden et al. 2020): 5,183 abstracts and the 300 test queries with
their 339 judgments, every judgment labelled 1. The SciFact repository states that the claims and annotations
are released under CC BY 4.0 and the abstracts, from S2ORC, under ODC-By 1.0; the BEIR copy on the Hugging Face
hub is tagged cc-by-sa-4.0. No data is committed here; `scripts/fetch_scifact.sh` downloads the BEIR zip and
checks its SHA-256.

```
scripts/fetch_scifact.sh data
D=data/scifact
mkdir -p runs
retrievalstats bm25 $D/corpus.jsonl $D/queries.jsonl --qrels $D/qrels/test.tsv \
    -o runs/bm25-title-k0.9-b0.4.txt --tag title-k0.9-b0.4
retrievalstats bm25 $D/corpus.jsonl $D/queries.jsonl --qrels $D/qrels/test.tsv \
    --k1 1.2 --b 0.75 -o runs/bm25-title-k1.2-b0.75.txt --tag title-k1.2-b0.75
retrievalstats bm25 $D/corpus.jsonl $D/queries.jsonl --qrels $D/qrels/test.tsv \
    --fields text -o runs/bm25-text-k0.9-b0.4.txt --tag text-k0.9-b0.4
retrievalstats table $D/qrels/test.tsv runs/bm25-title-k0.9-b0.4.txt runs/bm25-title-k1.2-b0.75.txt \
    runs/bm25-text-k0.9-b0.4.txt -m ndcg@10,ndcg_exp@10,map,mrr,P@10,recall@100,rprec,bpref,success@1,judged@10
```

Building the three runs took about 11 seconds here.

| run | ndcg@10 | ndcg_exp@10 | map | mrr | P@10 | recall@100 | rprec | bpref | success@1 | judged@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25-title-k0.9-b0.4.txt | 0.6618 | 0.6618 | 0.6240 | 0.6348 | 0.0860 | 0.8852 | 0.5415 | 0.9650 | 0.5433 | 0.0860 |
| bm25-title-k1.2-b0.75.txt | 0.6634 | 0.6634 | 0.6236 | 0.6351 | 0.0870 | 0.8826 | 0.5315 | 0.9639 | 0.5367 | 0.0870 |
| bm25-text-k0.9-b0.4.txt | 0.6451 | 0.6451 | 0.6070 | 0.6172 | 0.0840 | 0.8762 | 0.5179 | 0.9583 | 0.5167 | 0.0840 |

None of the 20 t-test comparisons against the first run is significant after Holm correction over all 20.
Because SciFact judges only relevant documents, all with label 1, `ndcg_exp` equals `ndcg` and judged@10
equals P@10, and bpref has no judged non-relevant documents to count and reduces to recall over the whole
ranking.

**Agreement with pytrec_eval.** `uv run --extra dev python scripts/agreement.py $D/qrels/test.tsv runs/*.txt`
compares 11 measures (P@10, recall@100, map, map@100, mrr, ndcg, ndcg@10, rprec, bpref, success@1,
success@10) per query on all three runs. In all 33 run and measure pairs the largest absolute per-query
difference is at most 1.1e-16, so every metric agrees to four decimals and beyond. pytrec_eval has no
judged@k and no exponential gain. `tests/test_reference.py` checks `ndcg_exp` against pytrec_eval's `ndcg`
on qrels whose labels l are mapped to 2^l - 1, on random runs with many tied scores. It checks judged@k
against ir_measures on random runs with the ties removed, and on tied runs with `--ties docno`: ir_measures
orders tied documents by ascending id, so on a tied run judged@k under the default trec_eval rule can differ
from ir_measures' `Judged`.

**Which differences are significant.** `retrievalstats compare $D/qrels/test.tsv runs/bm25-title-k0.9-b0.4.txt
runs/bm25-text-k0.9-b0.4.txt -m ndcg@10,map,mrr,P@10,recall@100` (seed 0, 10,000 samples) tests the effect of
dropping the title field. Each cell is the raw p, then in brackets the Holm-adjusted p across the five
measures within each test; * marks an adjusted p below 0.05.

| measure | B - A | t | wilcoxon | sign | randomisation | bootstrap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ndcg@10 | -0.0167 | 0.0028 (0.0140)* | 0.0137 (0.0687) | 0.3817 (1.0000) | 0.0023 (0.0115)* | 0.0035 (0.0175)* |
| map | -0.0170 | 0.0083 (0.0303)* | 0.1530 (0.1968) | 0.9188 (1.0000) | 0.0082 (0.0288)* | 0.0084 (0.0276)* |
| mrr | -0.0176 | 0.0076 (0.0303)* | 0.0917 (0.1968) | 1.0000 (1.0000) | 0.0072 (0.0288)* | 0.0069 (0.0276)* |
| P@10 | -0.0020 | 0.0141 (0.0303)* | 0.0196 (0.0785) | 0.0312 (0.1562) | 0.0312 (0.0625) | 0.0113 (0.0276)* |
| recall@100 | -0.0090 | 0.0743 (0.0743) | 0.0656 (0.1968) | 0.1250 (0.5000) | 0.1250 (0.1250) | 0.0549 (0.0549) |

The t, randomisation and bootstrap tests agree that dropping titles costs about 0.017 nDCG@10 (t interval
[-0.0276, -0.0058]), just above the MDD of 0.0156; the sign test sees 20 wins against 27 losses and finds
nothing. Only six queries change P@10, so the randomisation test enumerated all 2^6 sign patterns and gives
the exact p = 2/64 = 0.0312.

The same command with `runs/bm25-title-k1.2-b0.75.txt` as B (other k1 and b) shows the opposite split:

| measure | B - A | wins/ties/losses | t | wilcoxon | sign | randomisation | bootstrap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ndcg@10 | +0.0016 | 32/257/11 | 0.7432 (1.0000) | 0.0917 (0.2750) | 0.0019 (0.0057)* | 0.7482 (1.0000) | 0.7441 (1.0000) |
| map | -0.0005 | 66/206/28 | 0.9413 (1.0000) | 0.0013 (0.0063)* | 0.0001 (0.0006)* | 0.9384 (1.0000) | 0.9359 (1.0000) |
| mrr | +0.0003 | 58/215/27 | 0.9664 (1.0000) | 0.0068 (0.0272)* | 0.0010 (0.0041)* | 0.9689 (1.0000) | 0.9652 (1.0000) |

The mean differences are near zero, but the second setting wins more queries by small margins and loses
fewer by larger ones, so the sign and Wilcoxon tests reject while the tests of the mean do not. This matches
Urbano et al.'s finding that the sign and Wilcoxon tests are unreliable for hypotheses about the mean.

## Limitations

- Only the trec_eval measures above, plus judged@k and exponential-gain nDCG. There is no infAP, ERR, RBP,
  statAP or condensed-list ("judged only") evaluation.
- Negative labels are checked against pytrec_eval only on hand-made cases, because pytrec-eval-terrier 0.5.11
  crashes on them.
- Corrections are Holm and Benjamini-Hochberg only; there is no Tukey HSD or other procedure that models all
  pairs of systems jointly.
- On small samples the bootstrap-shift p-value is too small (Urbano et al. 2019) and the percentile
  interval, printed by default, is too narrow. On the four-query example the bootstrap gives p = 0.0031 for
  nDCG@5 where the exact randomisation test gives 0.25, and the interval [+0.0655, +0.3288] where the t
  interval is [-0.0683, +0.4625]. `compare` prints a warning below 30 paired queries. Share of p-values
  below 0.05 when the true mean difference is 0, from `python scripts/null_rejection.py` (1,000 seeded
  samples per row, 2,000 resamples or sign patterns each; "skewed" draws exponential minus 1):

  | null | n | t | randomisation | bootstrap |
  | --- | ---: | ---: | ---: | ---: |
  | normal | 5 | 0.055 | 0.000 | 0.164 |
  | normal | 10 | 0.045 | 0.042 | 0.099 |
  | normal | 30 | 0.044 | 0.044 | 0.053 |
  | skewed | 5 | 0.116 | 0.000 | 0.235 |
  | skewed | 10 | 0.092 | 0.086 | 0.141 |
  | skewed | 30 | 0.089 | 0.086 | 0.103 |

  At n = 5 the randomisation test enumerates all 32 sign patterns, so its smallest p is 2/32 and it never
  rejects at 0.05. Under the skewed null even the tests of the mean reject well above 5 % at these sizes.
- The MDD assumes the paired t-test and treats the observed sd(d) as the true one.
- Under the default tie rule, scores closer than 32-bit precision count as tied (as in pytrec_eval), so
  runs written with 17 significant digits, as `bm25` writes them, can score differently under `--ties
  docno` or `--ties input` even when no two scores are exactly equal.
- BM25 is a small in-memory implementation without stemming or stopwords, run only on SciFact here; it is
  for building example runs, not a search engine.
- The random agreement tests cover cut-offs 1, 3, 5, 10 and 20 and labels 0 to 3; the real-data check covers
  SciFact, whose judgments are binary.

## Related tools

- [trec_eval](https://github.com/usnistgov/trec_eval): the standard evaluator for TREC runs; retrievalstats
  follows its conventions by default.
- [pytrec_eval](https://github.com/cvangysel/pytrec_eval) (Van Gysel and de Rijke 2018): Python interface to
  trec_eval, used here as a test-only reference.
- [ir_measures](https://ir-measur.es/): one interface over several evaluation tools; its `Judged` measure is
  the reference for judged@k.
- [evalstats](https://github.com/vipul21435/evalstats): error bars and paired comparisons for evaluations in
  general.
- [recallmcp](https://github.com/vipul21435/recallmcp): Vipul Raj Jha's retrieval service.

## References

- K. Jarvelin, J. Kekalainen. Cumulated gain-based evaluation of IR techniques. ACM TOIS 20(4), 422-446, 2002.
  https://doi.org/10.1145/582415.582418
- C. Buckley, E. M. Voorhees. Retrieval evaluation with incomplete information. SIGIR 2004, 25-32.
  https://doi.org/10.1145/1008992.1009000
- M. D. Smucker, J. Allan, B. Carterette. A comparison of statistical significance tests for information
  retrieval evaluation. CIKM 2007, 623-632. https://doi.org/10.1145/1321440.1321528
- J. Urbano, H. Lima, A. Hanjalic. Statistical significance testing in information retrieval: an empirical
  analysis of type I, type II and type III errors. SIGIR 2019, 505-514. https://doi.org/10.1145/3331184.3331259
  (preprint https://arxiv.org/abs/1905.11096)
- N. Fuhr. Some common mistakes in IR evaluation, and how they can be avoided. SIGIR Forum 51(3), 32-41,
  December 2017. https://doi.org/10.1145/3190580.3190586
- N. Thakur, N. Reimers, A. Ruckle, A. Srivastava, I. Gurevych. BEIR: a heterogenous benchmark for zero-shot
  evaluation of information retrieval models. NeurIPS 2021 Datasets and Benchmarks Track.
  https://arxiv.org/abs/2104.08663
- D. Wadden, S. Lin, K. Lo, L. L. Wang, M. van Zuylen, A. Cohan, H. Hajishirzi. Fact or fiction: verifying
  scientific claims. EMNLP 2020. https://github.com/allenai/scifact
- S. Robertson, H. Zaragoza. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in
  Information Retrieval, 2009. https://doi.org/10.1561/1500000019
- S. Holm. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6(2),
  65-70, 1979.
- Y. Benjamini, Y. Hochberg. Controlling the false discovery rate: a practical and powerful approach to multiple
  testing. JRSS B 57(1), 289-300, 1995. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- C. Van Gysel, M. de Rijke. Pytrec_eval: an extremely fast Python interface to trec_eval. SIGIR 2018.

## Licence

MIT, see [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
