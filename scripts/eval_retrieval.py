"""
eval_retrieval.py
=================
A1 retrieval eval on the gold set, plus the max_distance sweep. No API cost.

    PYTHONPATH=. .venv/bin/python scripts/eval_retrieval.py

Options:
    --gold        gold set (default evals/datasets/retrieval_gold_v1.json)
    --cutoff      cutoff for the per-query table (default: rag.max_distance)
    --no-mlflow   skip MLflow logging
    --embedding-model NAME
                  score another sentence-transformers model. It gets its own
                  Chroma collection (built from the PDFs on first use), so the
                  production store is untouched. Query/document prompts come
                  from rag.embedding_prompts.
    --reranker MODEL
                  cross-encoder that reorders the top 30 candidates (e.g.
                  cross-encoder/ettin-reranker-150m-v1). The distance cutoff
                  still decides what is kept; the reranker sets the order.
    --instruction TEXT
                  custom query task instruction (Qwen3-Embedding format
                  "Instruct: TEXT\nQuery:"). Only queries change, so the
                  stored document vectors are reused.

    --save-rankings PATH
                  also freeze the page rankings for the CI gate (A9), e.g.
                  tests/fixtures/evals/retrieval_rankings_v1.json

Writes evals/results/retrieval_<time>.json (metrics, sweep, per query) and
evals/results/retrieval_sweep_<time>.png (recall vs negative rejection).
"""

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config and Chroma paths are relative

from src.evals import retrieval as a1  # noqa: E402
from src.evals.harness import RESULTS_DIR, _git, _hash  # noqa: E402
from src.rag.ingest_docs import ingest_documents  # noqa: E402
from src.rag.retriever import PolicyRetriever  # noqa: E402
from src.rag.vectorstore import (  # noqa: E402
    build_vectorstore,
    collection_name,
    encode_kwargs,
)

CUTOFFS = [round(0.30 + 0.02 * i, 2) for i in range(16)]  # 0.30 .. 0.60
CHUNKS_PER_QUERY = 30  # enough chunks to fill 10 unique pages
EXPERIMENT = "chbindex-evals"


def load_config() -> dict:
    return {
        **yaml.safe_load(open("config/pipeline.yaml")),
        **yaml.safe_load(open("config/cities/los_angeles.yaml")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="evals/datasets/retrieval_gold_v1.json")
    parser.add_argument("--cutoff", type=float)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--embedding-model")
    parser.add_argument("--instruction")
    parser.add_argument("--reranker")
    parser.add_argument("--save-rankings")
    args = parser.parse_args()

    config = load_config()
    rag = config["rag"]
    cutoff = args.cutoff if args.cutoff is not None else rag["max_distance"]
    gold = a1.load_gold(args.gold)

    if args.embedding_model and args.embedding_model != rag["embedding_model"]:
        rag["embedding_model"] = args.embedding_model
        slug = args.embedding_model.split("/")[-1].lower()
        rag["collection_suffix"] = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")

    if args.reranker:
        rag["reranker"] = {**(rag.get("reranker") or {}), "model": args.reranker}

    if args.instruction:  # Qwen3-Embedding query format; documents unchanged
        prompts = rag.setdefault("embedding_prompts", {})
        prompts[rag["embedding_model"]] = {
            **prompts.get(rag["embedding_model"], {}),
            "query_text": f"Instruct: {args.instruction}\nQuery:",
        }

    try:
        retriever = PolicyRetriever(config)
    except RuntimeError:  # no collection for this model yet
        print(f"Building collection for {rag['embedding_model']} (first use)...")
        build_vectorstore(config, ingest_documents(config))
        retriever = PolicyRetriever(config)
    retriever.max_distance = 2.0  # cutoff off: every cutoff is applied afterwards

    def search(query: str) -> list[tuple[str, int, float]]:
        return [
            (c.source, c.page, c.distance)
            for c in retriever.retrieve(query, top_k=CHUNKS_PER_QUERY)
        ]

    rankings = a1.retrieve_all(gold, search)
    metrics, rows = a1.evaluate(gold, rankings, cutoff)
    curve = a1.sweep(gold, rankings, CUTOFFS)
    drivers = a1.by_driver(rows)

    label = rag["embedding_model"].split("/")[-1]
    if (rag.get("reranker") or {}).get("model"):
        label += " + " + rag["reranker"]["model"].split("/")[-1]
    _print_report(label, cutoff, metrics, rows, curve, drivers)

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = {
        "gold": args.gold,
        "gold_hash": _hash(Path(args.gold).read_text()),
        "n_queries": len(gold),
        "n_negatives": sum(q.is_negative for q in gold),
        "cutoff": cutoff,
        "embedding_model": rag["embedding_model"],
        "query_encoding": json.dumps(encode_kwargs(rag, "query")),
        "document_encoding": json.dumps(encode_kwargs(rag, "document")),
        "collection": collection_name(rag, config["full_fips"]),
        "reranker": (rag.get("reranker") or {}).get("model") or "",
        "chunk_size": rag["chunk_size"],
        "chunk_overlap": rag["chunk_overlap"],
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "evaluated_at": stamp,
    }
    if args.save_rankings:
        from src.evals.gate import save_rankings

        frozen = save_rankings(
            Path(args.save_rankings), args.gold, cutoff, rankings, manifest
        )
        print(f"Rankings frozen: {frozen}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"retrieval_{stamp[:19].replace(':', '')}"
    out = RESULTS_DIR / f"{stem}.json"
    out.write_text(
        json.dumps(
            {
                "manifest": manifest,
                "metrics": metrics,
                "by_driver": drivers,
                "sweep": curve,
                "per_query": rows,
            },
            indent=2,
        )
    )
    plot = _plot_sweep(curve, cutoff, RESULTS_DIR / f"retrieval_sweep_{stem[10:]}.png")
    print(f"\nSaved: {out}\n       {plot}")

    if not args.no_mlflow:
        run_id = _log_to_mlflow(config, manifest, metrics, curve, rows, plot)
        print(f"MLflow run: {run_id}")
    return 0


def _print_report(manifest_model, cutoff, metrics, rows, curve, drivers) -> None:
    print(f"\nA1 retrieval, {manifest_model}, cutoff {cutoff}")
    for name, value in metrics.items():
        print(f"  {name:20} {value}")
    print("\nBy driver (positives):")
    for driver, m in drivers.items():
        print(
            f"  {driver:18} recall@3 {m['recall@3']:.3f}  empty {m['empty_rate']:.3f}  n={m['n']}"
        )

    print("\nPer query (best = distance of the closest relevant page, cutoff ignored):")
    for r in rows:
        if r["negative"]:
            verdict = "rejected" if r["rejected"] else f"LEAKED {r['n_within']} pages"
            print(f"  NEG {r['id']:18} {verdict:18} closest {r['closest']}")
        else:
            print(
                f"  {r['id']:22} r@3 {r['recall@3']:.2f}  rr {r['rr']:.2f}  "
                f"best {r['best_relevant_distance']}  | {r['top3'] or 'NOTHING'}"
            )

    print("\nCutoff sweep:")
    print("  cutoff  recall@3  mrr     empty   neg_rejection")
    for c in curve:
        print(
            f"  {c['cutoff']:.2f}    {c['recall@3']:.3f}     {c['mrr']:.3f}   "
            f"{c['empty_rate']:.3f}   {c.get('negative_rejection', float('nan')):.3f}"
        )


def _plot_sweep(curve, current: float, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [c["cutoff"] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, [c["recall@3"] for c in curve], marker="o", label="recall@3 (positives)")
    ax.plot(x, [c["mrr"] for c in curve], marker="o", label="MRR (positives)")
    ax.plot(
        x,
        [c["negative_rejection"] for c in curve],
        marker="s",
        label="negative rejection",
    )
    ax.plot(
        x,
        [c["empty_rate"] for c in curve],
        linestyle="--",
        label="empty rate (positives)",
    )
    ax.axvline(current, color="grey", linewidth=1)
    ax.text(current, 1.02, f" current {current}", fontsize=8, color="grey")
    ax.set_xlabel("max_distance (cosine)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Retrieval cutoff sweep: higher cutoff finds more, rejects less")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _log_to_mlflow(config, manifest, metrics, curve, rows, plot) -> str:
    import mlflow
    import pandas as pd

    tracking_uri = config["ml"]["mlflow_tracking_uri"]
    mlflow.set_tracking_uri(Path(tracking_uri).resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(
        run_name=" ".join(
            ["retrieval A1", manifest["embedding_model"].split("/")[-1]]
            + (
                [f"+ {manifest['reranker'].split('/')[-1]}"]
                if manifest["reranker"]
                else []
            )
        )
    ) as run:
        mlflow.log_params(manifest)
        mlflow.log_metrics(
            {f"retrieval/{k.replace('@', '_at_')}": v for k, v in metrics.items()}
        )
        for c in curve:
            mlflow.log_metrics(
                {
                    f"sweep/{k.replace('@', '_at_')}": v
                    for k, v in c.items()
                    if k != "cutoff"
                },
                step=int(round(c["cutoff"] * 100)),
            )
        mlflow.log_table(pd.DataFrame(rows), artifact_file="per_query.json")
        mlflow.log_artifact(str(plot))
        return run.info.run_id


if __name__ == "__main__":
    sys.exit(main())
