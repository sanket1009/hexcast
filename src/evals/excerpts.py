"""
excerpts.py
===========
Read stored policy chunks back from Chroma by ID, without loading the
embedding model. Briefings keep only chunk IDs (retrieved_chunk_ids); the
judge and the robustness tests need the text the model saw.
"""

from typing import Any

from src.rag.retriever import RetrievedChunk
from src.rag.vectorstore import get_collection


def fetch_chunks(config: dict[str, Any], ids: list[str]) -> dict[str, RetrievedChunk]:
    """Chunks by ID, in no particular order. Raises if any ID is missing.

    A missing ID means the corpus was rebuilt with different chunks since the
    briefings were written, so the text would not be what the model saw.
    """
    wanted = list(dict.fromkeys(ids))
    if not wanted:
        return {}
    got = get_collection(config).get(ids=wanted, include=["documents", "metadatas"])
    chunks = {
        cid: RetrievedChunk(
            id=cid, text=doc, source=meta["source"], page=meta["page"], distance=0.0
        )
        for cid, doc, meta in zip(
            got["ids"], got["documents"], got["metadatas"], strict=True
        )
    }
    missing = [i for i in wanted if i not in chunks]
    if missing:
        raise KeyError(
            f"{len(missing)} chunk IDs not in the policy collection "
            f"(corpus rebuilt since the briefings?): {missing[:3]}"
        )
    return chunks
