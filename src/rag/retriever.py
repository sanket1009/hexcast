"""
retriever.py
============
Find the policy chunks most relevant to a question.

Step 3 of the RAG layer. Embeds a query with the same model used in
ingest_docs.py, runs a nearest-neighbor search in ChromaDB, and drops
results too distant to be relevant.

Design:
    - Class, so the embedding model and collection load once and are
      reused across many queries (briefing.py runs one per tract)
    - Distance cutoff: return fewer chunks rather than irrelevant ones
    - Optional reranker (rag.reranker.model): a cross-encoder reads query and
      chunk together and reorders the candidates that passed the cutoff. The
      embedding cutoff stays the relevance gate; the reranker only reorders.
    - Typed results: briefing.py never touches Chroma's raw format

Usage:
    from src.rag.retriever import PolicyRetriever

    retriever = PolicyRetriever(config)
    chunks = retriever.retrieve("cooling center access for elderly residents")
    for c in chunks:
        print(c.citation, c.distance)
"""

from dataclasses import dataclass, replace
from typing import Any

from sentence_transformers import CrossEncoder, SentenceTransformer

from src.rag.vectorstore import encode_kwargs, get_collection
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    """One policy chunk returned by a search."""

    id: str
    text: str
    source: str
    page: int
    distance: float
    rerank_score: float | None = None  # set when a reranker ordered the list

    @property
    def citation(self) -> str:
        """Human-readable citation, e.g. 'heat_plan.pdf, p. 14'."""
        return f"{self.source}, p. {self.page}"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class PolicyRetriever:
    """Semantic search over one city's policy document collection."""

    def __init__(self, config: dict[str, Any]):
        rag_cfg = config["rag"]
        self.top_k = rag_cfg["top_k_chunks"]
        self.max_distance = rag_cfg["max_distance"]

        # Raises a clear error if build_vectorstore() was never run
        self.collection = get_collection(config)

        logger.info(f"Loading embedding model: {rag_cfg['embedding_model']}")
        self.model = SentenceTransformer(rag_cfg["embedding_model"])
        self.query_args = encode_kwargs(rag_cfg, "query")

        rerank_cfg = rag_cfg.get("reranker") or {}
        self.reranker = None
        self.rerank_candidates = rerank_cfg.get("candidates", 30)
        if rerank_cfg.get("model"):
            logger.info(f"Loading reranker: {rerank_cfg['model']}")
            self.reranker = CrossEncoder(rerank_cfg["model"])

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Return up to top_k chunks closest in meaning to the query.

        Chunks with cosine distance above max_distance are dropped, so
        the result can be shorter than top_k, or empty.
        """
        if not query or not query.strip():
            raise ValueError("Query text is empty.")

        # Chroma cannot return more results than the collection holds
        want = top_k or self.top_k
        # With a reranker, fetch a wider pool so it has something to reorder
        pool = max(want, self.rerank_candidates) if self.reranker else want
        k = min(pool, self.collection.count())
        if k == 0:
            logger.warning(f"Collection '{self.collection.name}' is empty.")
            return []

        query_vec = self.model.encode(
            query, **self.query_args, normalize_embeddings=True
        ).tolist()

        result = self.collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns one inner list per query; we sent one query
        candidates = [
            RetrievedChunk(
                id=chunk_id,
                text=doc,
                source=meta["source"],
                page=meta["page"],
                distance=dist,
            )
            for chunk_id, doc, meta, dist in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=False,
            )
        ]

        kept = [c for c in candidates if c.distance <= self.max_distance]

        dropped = len(candidates) - len(kept)
        if dropped:
            logger.info(
                f"Dropped {dropped} of {len(candidates)} chunks above "
                f"max_distance={self.max_distance}"
            )

        if self.reranker and kept:
            kept = self._rerank(query, kept)
        return kept[:want]

    def _rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Order chunks by cross-encoder relevance, highest first.

        The reranker gets the plain query: the embedding task instruction is a
        convention of the embedding model, not part of the question.
        """
        scores = self.reranker.predict([(query, c.text) for c in chunks])
        scored = [
            replace(c, rerank_score=float(s))
            for c, s in zip(chunks, scores, strict=True)
        ]
        return sorted(scored, key=lambda c: c.rerank_score, reverse=True)
