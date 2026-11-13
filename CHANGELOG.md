# Changelog

## 0.1.0

- Readers for TREC qrels and runs, BEIR-style TSV, JSONL and the BEIR results JSON.
- Per-query and mean P@k, recall@k, MAP (and MAP@k), MRR (and MRR@k), nDCG and nDCG@k with linear and
  exponential gain, R-precision, bpref, success@k and judged@k, with trec_eval-compatible defaults for tied
  scores, missing queries, queries without relevant documents and negative labels.
- Paired t, Wilcoxon signed-rank, sign, Fisher randomisation (exact for small samples) and paired bootstrap
  tests; t and bootstrap intervals, effect size, win/tie/loss counts, minimum detectable difference.
- Holm and Benjamini-Hochberg corrections.
- Per-query difference table and judged@k warnings.
- A small BM25 with Lucene and Robertson idf variants.
- Commands `score`, `compare`, `table` and `bm25`, with Markdown and JSON output.
- Agreement tests against pytrec_eval and ir_measures; SciFact fetch and agreement scripts.
