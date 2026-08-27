"""
RAG pipeline modules for CHBIndex.

Generates plain-English policy briefings for high-risk tracts:
    ingest_docs.py  — Load policy PDFs into ChromaDB vector store
    vectorstore.py  — ChromaDB interface for document storage
    retriever.py    — Semantic search for relevant policy chunks
    briefing.py     — Structured LLM briefing via Anthropic API
"""
