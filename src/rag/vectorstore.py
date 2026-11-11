"""
vectorstore.py
==============
Store embedded policy chunks in a persistent ChromaDB collection.

Step 2 of the RAG layer. ingest_docs.py produces chunks with
embeddings; this module writes them to disk so retriever.py can
search them without re-embedding anything.

Design:
    - Embedded PersistentClient: no server process, no network port
    - One collection per city, named by FIPS code
    - We supply our own embeddings; Chroma never embeds anything
    - Cosine distance
    - Sync semantics: upsert current chunks, prune chunks that no
      longer exist in the source PDFs

Usage:
    from src.rag.ingest_docs import ingest_documents
    from src.rag.vectorstore import build_vectorstore, get_collection

    chunks = ingest_documents(config)
    build_vectorstore(config, chunks)

    collection = get_collection(config)   # used by retriever.py
"""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def build_vectorstore(config: dict[str, Any], chunks: list[dict[str, Any]]) -> int:
    """
    Sync the city's policy collection with the given chunks.

    Upserts every chunk, then deletes any stored chunk whose ID is not
    in this run. Re-running with the same PDFs is a no-op.

    Returns
    -------
    int
        Number of chunks in the collection after the sync.
    """
    if not chunks:
        # Safety: a broken extraction run should not wipe the store
        logger.warning("No chunks to store. Vector store left unchanged.")
        return 0

    collection = get_collection(config, create=True)
    batch_size = config["rag"]["upsert_batch_size"]

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    n_pruned = _prune_stale(
        collection,
        keep_ids={c["id"] for c in chunks},
        batch_size=batch_size,
    )

    total = collection.count()
    logger.info(
        f"Vector store synced: collection '{collection.name}' has {total} chunks "
        f"({len(chunks)} upserted, {n_pruned} stale removed)"
    )
    return total


def get_collection(config: dict[str, Any], create: bool = False):
    """
    Open the city's policy collection.

    Parameters
    ----------
    create : bool
        True when building (create if missing). False when reading,
        so retriever.py fails clearly if the store was never built.
    """
    rag_cfg = config["rag"]
    city_fips = config["full_fips"]
    model_name = rag_cfg["embedding_model"]
    name = collection_name(rag_cfg, city_fips)

    client = _get_client(rag_cfg["chroma_dir"])

    if create:
        collection = client.get_or_create_collection(
            name=name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": model_name, "city_fips": city_fips},
        )
    else:
        try:
            collection = client.get_collection(name=name, embedding_function=None)
        except Exception as e:
            raise RuntimeError(
                f"Collection '{name}' not found in {rag_cfg['chroma_dir']}. "
                f"Run build_vectorstore() first."
            ) from e

    _check_embedding_model(collection, model_name)
    return collection


def collection_name(rag_cfg: dict[str, Any], city_fips: str) -> str:
    """One collection per embedding model.

    The configured model keeps the plain name; a model under comparison
    (scripts/eval_retrieval.py --embedding-model) gets its own suffix, so
    the production store is never overwritten by an experiment.
    """
    base = f"{rag_cfg['collection_prefix']}_{city_fips}"
    suffix = rag_cfg.get("collection_suffix")
    return f"{base}_{suffix}" if suffix else base


def encode_kwargs(rag_cfg: dict[str, Any], role: str) -> dict[str, str]:
    """sentence-transformers encode() arguments for 'query' or 'document' text.

    Per model in rag.embedding_prompts:
        <role>: name of a prompt stored with the model (prompt_name)
        <role>_text: literal prompt text, e.g. a custom task instruction
    Instruction-tuned models (e.g. Qwen3-Embedding) prefix queries with a task
    instruction and embed documents plain. Unlisted models encode plain text.
    """
    prompts = rag_cfg.get("embedding_prompts", {}).get(rag_cfg["embedding_model"], {})
    if text := prompts.get(f"{role}_text"):
        return {"prompt": text}
    if name := prompts.get(role):
        return {"prompt_name": name}
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_client(chroma_dir: str) -> chromadb.ClientAPI:
    """Embedded, on-disk Chroma client with telemetry off."""
    path = Path(chroma_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )


def _check_embedding_model(collection, expected: str) -> None:
    """Fail loudly if the store was built with a different embedding model."""
    stored = (collection.metadata or {}).get("embedding_model")
    if stored and stored != expected:
        raise ValueError(
            f"Collection '{collection.name}' was built with '{stored}' but config "
            f"says '{expected}'. Vectors from different models are not comparable. "
            f"Delete {collection.name} and rebuild."
        )


def _prune_stale(collection, keep_ids: set[str], batch_size: int) -> int:
    """Delete stored chunks that are not in the current run."""
    existing = set(collection.get(include=[])["ids"])
    stale = sorted(existing - keep_ids)

    for start in range(0, len(stale), batch_size):
        collection.delete(ids=stale[start : start + batch_size])

    return len(stale)
