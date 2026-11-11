"""
Reranker in PolicyRetriever: the cutoff still filters, the cross-encoder only
reorders, and top_k is applied after reranking. Fake collection and models.
"""

import numpy as np

from src.rag import retriever as retriever_mod


class FakeCollection:
    name = "fake"

    def count(self):
        return 4

    def query(self, query_embeddings, n_results, include):
        rows = [
            ("c1", "describes who is at risk", "a.pdf", 1, 0.20),
            ("c2", "recommends cooling centers", "a.pdf", 2, 0.30),
            ("c3", "recommends outreach", "b.pdf", 5, 0.40),
            ("c4", "off topic", "z.pdf", 9, 0.90),
        ][:n_results]
        return {
            "ids": [[r[0] for r in rows]],
            "documents": [[r[1] for r in rows]],
            "metadatas": [[{"source": r[2], "page": r[3]} for r in rows]],
            "distances": [[r[4] for r in rows]],
        }


class FakeEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text, **kwargs):
        return np.zeros(3)


class FakeCrossEncoder:
    """Scores 'recommends' passages above descriptions."""

    def __init__(self, *args, **kwargs):
        self.pairs = []

    def predict(self, pairs):
        self.pairs = pairs
        return [1.0 if "recommends" in text else 0.0 for _, text in pairs]


def _retriever(monkeypatch, reranker_model):
    monkeypatch.setattr(
        retriever_mod, "get_collection", lambda config: FakeCollection()
    )
    monkeypatch.setattr(retriever_mod, "SentenceTransformer", FakeEmbedder)
    monkeypatch.setattr(retriever_mod, "CrossEncoder", FakeCrossEncoder)
    config = {
        "rag": {
            "top_k_chunks": 3,
            "max_distance": 0.5,
            "embedding_model": "m",
            "reranker": {"model": reranker_model, "candidates": 4},
        }
    }
    return retriever_mod.PolicyRetriever(config)


def test_without_reranker_order_is_by_distance(monkeypatch):
    r = _retriever(monkeypatch, None)
    assert [c.id for c in r.retrieve("q")] == ["c1", "c2", "c3"]


def test_reranker_reorders_but_cutoff_still_filters(monkeypatch):
    r = _retriever(monkeypatch, "fake-ce")
    got = r.retrieve("q", top_k=2)
    assert [c.id for c in got] == ["c2", "c3"]  # actions first, top_k after rerank
    assert all(c.rerank_score is not None for c in got)
    texts = [text for _, text in r.reranker.pairs]
    assert "off topic" not in texts  # c4 failed the cutoff before reranking
