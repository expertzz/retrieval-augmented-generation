import hashlib
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    create_engine, text,
    Table, Column, MetaData,
    String, Integer, DateTime, Text,
)

from rag.config import settings

metadata = MetaData()

documents = Table(
    "documents", metadata,
    Column("id", String, primary_key=True),       # SHA-256 of file content
    Column("filename", String, nullable=False),
    Column("source_path", String, nullable=False),
    Column("doc_type", String, nullable=False),    # pdf, docx, txt, url
    Column("page_count", Integer, default=0),
    Column("chunk_count", Integer, default=0),
    Column("ingested_at", DateTime),
    Column("status", String, default="pending"),   # pending, indexing, ready, error
)

chunks = Table(
    "chunks", metadata,
    Column("id", String, primary_key=True),
    Column("doc_id", String, nullable=False),
    Column("page", Integer, nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("faiss_index", Integer, nullable=False), # row position in FAISS index
)


def _engine():
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{settings.db_path}")
    metadata.create_all(engine)
    return engine


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def add_document(filename: str, source_path: str, doc_type: str, doc_id: str | None = None) -> str:
    engine = _engine()
    doc_id = doc_id or hash_file(source_path)
    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT id FROM documents WHERE id = :id"), {"id": doc_id}
        ).fetchone()
        if exists:
            return doc_id
        conn.execute(documents.insert().values(
            id=doc_id,
            filename=filename,
            source_path=source_path,
            doc_type=doc_type,
            ingested_at=datetime.now(timezone.utc),
            status="pending",
        ))
    return doc_id


def update_status(doc_id: str, status: str, page_count: int = 0, chunk_count: int = 0) -> None:
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(
            documents.update()
            .where(documents.c.id == doc_id)
            .values(status=status, page_count=page_count, chunk_count=chunk_count)
        )


def add_chunks(chunk_rows: list[dict]) -> None:
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(chunks.insert(), chunk_rows)


def list_documents() -> list[dict]:
    engine = _engine()
    with engine.connect() as conn:
        rows = conn.execute(documents.select()).mappings().all()
        return [dict(r) for r in rows]


def get_document(doc_id: str) -> dict | None:
    engine = _engine()
    with engine.connect() as conn:
        row = conn.execute(
            documents.select().where(documents.c.id == doc_id)
        ).mappings().fetchone()
        return dict(row) if row else None


def delete_document(doc_id: str) -> None:
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(chunks.delete().where(chunks.c.doc_id == doc_id))
        conn.execute(documents.delete().where(documents.c.id == doc_id))


def get_chunks_for_doc(doc_id: str) -> list[dict]:
    engine = _engine()
    with engine.connect() as conn:
        rows = conn.execute(
            chunks.select().where(chunks.c.doc_id == doc_id).order_by(chunks.c.chunk_index)
        ).mappings().all()
        return [dict(r) for r in rows]
