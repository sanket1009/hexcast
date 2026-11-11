"""
Embedding model settings: per-model Chroma collections and query/document
prompts. Pure config logic, no model or Chroma calls.
"""

from src.rag.vectorstore import collection_name, encode_prompt

RAG = {
    "collection_prefix": "policy_docs",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_prompts": {"Qwen/Qwen3-Embedding-0.6B": {"query": "query"}},
}


def test_configured_model_keeps_the_production_collection():
    assert collection_name(RAG, "06037") == "policy_docs_06037"


def test_compared_model_gets_its_own_collection():
    rag = {**RAG, "collection_suffix": "qwen3-embedding-0-6b"}
    assert collection_name(rag, "06037") == "policy_docs_06037_qwen3-embedding-0-6b"


def test_unlisted_model_encodes_plain_text():
    assert encode_prompt(RAG, "query") is None
    assert encode_prompt(RAG, "document") is None


def test_instruction_model_prompts_queries_only():
    rag = {**RAG, "embedding_model": "Qwen/Qwen3-Embedding-0.6B"}
    assert encode_prompt(rag, "query") == "query"
    assert encode_prompt(rag, "document") is None
