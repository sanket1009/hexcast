"""
build_corpus.py
===============
Rebuild the policy corpus for a city and check what each driver query finds.

Step 1: chunk and embed every PDF in policy_docs_dir, then sync ChromaDB
        (new chunks upserted, chunks from removed PDFs pruned).
Step 2: for each driver query in pipeline.yaml, print the nearest chunks
        with their cosine distance, so you can see whether each query lands
        on the right document before spending model calls.

No API calls. Run after adding or removing PDFs.

Usage (from the project folder):
    python3 scripts/build_corpus.py              # rebuild, then check
    python3 scripts/build_corpus.py --check-only # skip the rebuild
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config and Chroma paths are relative

from src.rag.briefing import CANDIDATES_PER_DRIVER, pick_excerpts  # noqa: E402
from src.rag.ingest_docs import ingest_documents  # noqa: E402
from src.rag.retriever import PolicyRetriever  # noqa: E402
from src.rag.vectorstore import build_vectorstore  # noqa: E402

SHOW = 5  # chunks to print per query, ignoring the distance cutoff


def load_config() -> dict:
    return {
        **yaml.safe_load(open("config/pipeline.yaml")),
        **yaml.safe_load(open("config/cities/los_angeles.yaml")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = load_config()
    rag = config["rag"]

    if not args.check_only:
        chunks = ingest_documents(config)
        per_doc = Counter(c["metadata"]["source"] for c in chunks)
        print("\nChunks per document:")
        for source, n in sorted(per_doc.items()):
            print(f"  {n:5d}  {source}")
        build_vectorstore(config, chunks)

    retriever = PolicyRetriever(config)
    cutoff = rag["max_distance"]
    group = rag["default_vulnerable_group"]

    for driver, template in rag["driver_queries"].items():
        query = template.format(group=group)
        print(f"\n=== {driver}\n{query}")
        # Ask for more than top_k and bypass the cutoff so near misses show too
        retriever.max_distance = 2.0
        for c in retriever.retrieve(query, top_k=SHOW):
            mark = "KEEP" if c.distance <= cutoff else "drop"
            snippet = " ".join(c.text.split())[:110]
            print(f"  {c.distance:.3f} {mark}  {c.citation}\n         {snippet}")
        retriever.max_distance = cutoff

        # What a briefing gets: within the cutoff, one chunk per page
        candidates = retriever.retrieve(query, top_k=CANDIDATES_PER_DRIVER)
        chosen = pick_excerpts([candidates], rag["briefing_excerpts_per_driver"])
        print("  Briefing gets: " + "; ".join(f"{c.citation} ({c.distance:.3f})" for c in chosen))

    print(f"\nKEEP = within max_distance {cutoff}. A briefing uses up to "
          f"{rag['briefing_excerpts_per_driver']} kept chunks per driver, one per page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
