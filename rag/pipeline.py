import os
import numpy as np
from pathlib import Path
from openai import OpenAI

from rag.config import settings
from rag.ingestion.registry import extract_pages
from rag.ingestion.chunker import chunk_pages
from rag.storage import document_store as ds
from rag.storage import vector_store as vs

_client = OpenAI(api_key=settings.openai_api_key)


def _embed(texts: list[str]) -> list[list[float]]:
    response = _client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [obj.embedding for obj in response.data]


def ingest(source: str) -> str:
    """
    Full pipeline: extract → chunk → embed → store.
    Returns the doc_id. Skips silently if already indexed.
    """
    filename = Path(source).name if not source.startswith("http") else source
    doc_type = "url" if source.startswith("http") else Path(source).suffix.lstrip(".")

    # Deduplication — hash the file content (or URL string for web)
    if source.startswith("http"):
        import hashlib
        doc_id = hashlib.sha256(source.encode()).hexdigest()
    else:
        doc_id = ds.hash_file(source)

    existing = ds.get_document(doc_id)
    if existing and existing["status"] == "ready":
        print(f"  already indexed: {filename}")
        return doc_id

    doc_id = ds.add_document(filename, source, doc_type, doc_id=doc_id)
    ds.update_status(doc_id, "indexing")

    # Extract and chunk
    pages = extract_pages(source)
    chunks = chunk_pages(pages, doc_id)

    if not chunks:
        ds.update_status(doc_id, "error")
        raise ValueError(f"No content extracted from: {source}")

    # Embed in batches of 100
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        all_embeddings.extend(_embed([c.text for c in batch]))

    # Assign FAISS IDs starting after current index size
    base_id = vs.count()
    faiss_ids = np.arange(base_id, base_id + len(chunks), dtype="int64")
    vectors = np.array(all_embeddings, dtype="float32")
    vs.add(vectors, faiss_ids)

    # Save chunk metadata to SQLite
    chunk_rows = [
        {
            "id": f"{doc_id}-{c.chunk_index}",
            "doc_id": doc_id,
            "page": c.page,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "faiss_index": int(faiss_ids[i]),
        }
        for i, c in enumerate(chunks)
    ]
    ds.add_chunks(chunk_rows)
    ds.update_status(doc_id, "ready", page_count=len(pages), chunk_count=len(chunks))

    return doc_id
