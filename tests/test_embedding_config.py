"""
Embedding model settings: per-model Chroma collections and query/document
prompts. Pure config logic, no model or Chroma calls.
"""

from src.rag.vectorstore import collection_name, encode_kwargs

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
    assert encode_kwargs(RAG, "query") == {}
    assert encode_kwargs(RAG, "document") == {}


def test_instruction_model_prompts_queries_only():
    rag = {**RAG, "embedding_model": "Qwen/Qwen3-Embedding-0.6B"}
    assert encode_kwargs(rag, "query") == {"prompt_name": "query"}
    assert encode_kwargs(rag, "document") == {}


def test_literal_instruction_overrides_the_named_prompt():
    rag = {
        **RAG,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_prompts": {
            "Qwen/Qwen3-Embedding-0.6B": {
                "query": "query",
                "query_text": "Instruct: find actions\nQuery:",
            }
        },
    }
    assert encode_kwargs(rag, "query") == {"prompt": "Instruct: find actions\nQuery:"}
