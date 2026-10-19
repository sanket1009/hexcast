"""
ingest_docs.py
==============
Load policy PDFs, split them into overlapping chunks, and embed each
chunk with a local sentence-transformers model.

Step 1 of the RAG layer. Turns long policy documents into small,
searchable pieces. vectorstore.py stores them in ChromaDB and
retriever.py searches them.

Input:
    PDFs in config["policy_docs_dir"]

Output:
    list[dict], one dict per chunk:
        id, text, embedding, metadata (source, page, chunk_index, city_fips)

Usage:
    from src.rag.ingest_docs import ingest_documents
    chunks = ingest_documents(config)
"""

import hashlib
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def ingest_documents(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract, chunk, and embed every PDF in the city's policy docs folder.

    Returns
    -------
    list[dict]
        One dict per chunk with keys: id, text, embedding, metadata.
        Empty list if no PDFs are found.
    """
    rag_cfg = config["rag"]
    docs_dir = Path(config["policy_docs_dir"])
    city_fips = config["full_fips"]

    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning(f"No PDFs found in {docs_dir}. Nothing to ingest.")
        return []

    logger.info(f"Found {len(pdf_paths)} policy documents in {docs_dir}")

    # Step 1 and 2: extract text page by page, then chunk each page
    chunks: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        pages = _extract_pages(pdf_path)

        for page_num, page_text in pages:
            page_chunks = _split_text(
                page_text,
                chunk_size=rag_cfg["chunk_size"],
                overlap=rag_cfg["chunk_overlap"],
            )

            for i, text in enumerate(page_chunks):
                # Drop tiny fragments like headers, footers, page numbers
                if len(text) < rag_cfg["min_chunk_chars"]:
                    continue

                chunks.append({
                    "id": _make_chunk_id(city_fips, pdf_path.name, page_num, i),
                    "text": text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_num,
                        "chunk_index": i,
                        "city_fips": city_fips,
                    },
                })

    if not chunks:
        logger.warning("PDFs were found but produced no usable text.")
        return []

    logger.info(f"Created {len(chunks)} chunks from {len(pdf_paths)} documents")

    # Step 3: embed all chunks in one batch call
    embeddings = _embed_texts(
        [c["text"] for c in chunks],
        model_name=rag_cfg["embedding_model"],
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    logger.info(
        f"Embedded {len(chunks)} chunks "
        f"({len(embeddings[0])} dimensions each)"
    )

    return chunks


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Extract text from each page of a PDF.

    Returns a list of (page_number, text) with 1-based page numbers so
    citations match what a reader sees in the PDF viewer.

    Scanned PDFs (images of text) return no text. Those need OCR,
    which is out of scope here, so we log a warning and skip them.
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_num, text))

    if not pages:
        logger.warning(
            f"{pdf_path.name}: no extractable text. "
            f"Likely a scanned PDF that needs OCR. Skipping."
        )
    else:
        logger.info(f"{pdf_path.name}: extracted {len(pages)} pages")

    return pages


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks, preferring sentence boundaries.

    Each chunk is at most chunk_size characters. Consecutive chunks
    share `overlap` characters so no sentence is lost at a cut line.
    """
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Collapse line breaks and repeated spaces left over from PDF layout
    text = " ".join(text.split())

    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # If we are mid-text, back up to the last full sentence,
        # but only if that keeps the chunk at least half full
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Step back by `overlap` so the next chunk repeats the tail
        start = max(end - overlap, start + 1)

    return chunks


def _make_chunk_id(city_fips: str, source: str, page: int, index: int) -> str:
    """
    Build a deterministic ID for a chunk.

    Same input always gives the same ID, so weekly re-runs overwrite
    existing chunks instead of creating duplicates (idempotency).
    """
    raw = f"{city_fips}|{source}|{page}|{index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    """
    Embed texts with a local sentence-transformers model.

    normalize_embeddings=True scales every vector to length 1, so
    cosine similarity becomes a simple dot product. ChromaDB will
    use cosine distance for search.
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.tolist()