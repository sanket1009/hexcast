"""
compare_retrieval.py
====================
Paired comparison of two A1 runs (B minus A) on the same gold set. No model calls.

    PYTHONPATH=. .venv/bin/python scripts/compare_retrieval.py \\
        evals/results/retrieval_<A>.json evals/results/retrieval_<B>.json

Prints, per metric, the mean per-query difference with a bootstrap 95% CI and
win/loss/tie counts. A CI that spans 0 means the gold set cannot tell the two
apart. Both runs must use the same gold file and cutoff.
"""

import json
import sys
from pathlib import Path

from src.evals.retrieval import paired_comparison


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    runs = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
    a, b = (r["manifest"] for r in runs)
    for key in ("gold_hash", "cutoff"):
        if a[key] != b[key]:
            print(f"Runs differ in {key}: {a[key]} vs {b[key]}. Not comparable.")
            return 1

    def label(m: dict) -> str:
        name = m["embedding_model"].split("/")[-1]
        if m.get("reranker"):
            name += " + " + m["reranker"].split("/")[-1]
        return name

    print(f"A: {label(a)}\nB: {label(b)}\ncutoff {a['cutoff']}, gold {a['gold']}\n")
    result = paired_comparison(runs[0]["per_query"], runs[1]["per_query"])
    print(f"  {'metric':10} {'B - A':>8}  {'95% CI':>18}  wins/losses/ties")
    for metric, r in result.items():
        name = "mrr" if metric == "rr" else metric
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        print(
            f"  {name:10} {r['mean_diff']:+8.3f}  {ci:>18}  "
            f"{r['wins']}/{r['losses']}/{r['ties']} (n={r['n']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
