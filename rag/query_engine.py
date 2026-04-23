from typing import Iterator
from openai import OpenAI

from rag.config import settings
from rag.storage import vector_store as vs
from rag.storage import document_store as ds
from rag.generation.generator import generate, generate_stream

_client = OpenAI(api_key=settings.openai_api_key)


def _embed_query(query: str) -> list[float]:
    response = _client.embeddings.create(
        model=settings.embedding_model,
        input=[query],
    )
    return response.data[0].embedding


def _retrieve(query: str, doc_ids: list[str] | None = None) -> list[dict]:
    scores, faiss_ids = vs.search(_embed_query(query), top_k=settings.top_k_retrieval)
    results = []
    for score, fid in zip(scores, faiss_ids):
        if fid == -1:
            continue
        # find chunk by faiss_index
        with ds._engine().connect() as conn:
            from sqlalchemy import text
            row = conn.execute(
                text("SELECT * FROM chunks WHERE faiss_index = :fid"), {"fid": int(fid)}
            ).mappings().fetchone()
        if row is None:
            continue
        chunk = dict(row)
        if doc_ids and chunk["doc_id"] not in doc_ids:
            continue
        chunk["score"] = float(score)
        results.append(chunk)
    return results[:settings.top_k_rerank]


def query(text: str, doc_ids: list[str] | None = None) -> dict:
    chunks = _retrieve(text, doc_ids)
    answer = generate(text, chunks)
    return {"answer": answer, "chunks": chunks}


def query_stream(text: str, doc_ids: list[str] | None = None) -> Iterator[str]:
    chunks = _retrieve(text, doc_ids)
    return generate_stream(text, chunks)
