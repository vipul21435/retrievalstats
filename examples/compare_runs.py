"""Score two runs, compare them with every paired test, and print a short report.

Run from the repository root:  python examples/compare_runs.py
"""

from pathlib import Path

from retrievalstats.compare import compare_runs
from retrievalstats.io import read_qrels, read_run
from retrievalstats.metrics import evaluate

HERE = Path(__file__).resolve().parent / "tiny"


def main() -> None:
    qrels = read_qrels(HERE / "qrels.txt")
    run_a = read_run(HERE / "run_a.txt")
    run_b = read_run(HERE / "run_b.txt")

    for name, run in (("run_a", run_a), ("run_b", run_b)):
        means = evaluate(qrels, run, ["ndcg@5", "map", "bpref"]).mean()
        print(name, {k: round(v, 4) for k, v in means.items()})

    comparison = compare_runs(qrels, run_a, run_b, ["ndcg@5", "map"], samples=5000)
    for result in comparison.results:
        print(
            f"{result.measure}: B - A = {result.difference:+.4f}, "
            f"t interval [{result.t_interval[0]:+.4f}, {result.t_interval[1]:+.4f}], "
            f"randomisation p = {result.p_values['randomisation']:.4f} "
            f"({'exact' if result.randomisation_exact else 'sampled'}), "
            f"Holm-adjusted t p = {result.adjusted['t']:.4f}"
        )
    for warning in comparison.warnings:
        print("warning:", warning)


if __name__ == "__main__":
    main()
