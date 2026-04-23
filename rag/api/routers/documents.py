import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from rag.pipeline import ingest
from rag.storage import document_store as ds
from rag.storage import vector_store as vs
from rag.api.schemas import DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])


def _run_ingest(path: str, cleanup: bool = False) -> None:
    try:
        ingest(path)
    finally:
        if cleanup:
            Path(path).unlink(missing_ok=True)


@router.post("", status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(file.file, tmp)
    tmp.close()
    background_tasks.add_task(_run_ingest, tmp.name, cleanup=True)
    return {"message": "ingestion started", "filename": file.filename}


@router.post("/url", status_code=202)
async def ingest_url(background_tasks: BackgroundTasks, url: str):
    background_tasks.add_task(_run_ingest, url)
    return {"message": "ingestion started", "url": url}


@router.get("", response_model=list[DocumentOut])
def list_documents():
    return ds.list_documents()


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str):
    doc = ds.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    doc = ds.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = ds.get_chunks_for_doc(doc_id)
    if chunks:
        import numpy as np
        faiss_ids = np.array([c["faiss_index"] for c in chunks], dtype="int64")
        vs.remove(faiss_ids)
    ds.delete_document(doc_id)
