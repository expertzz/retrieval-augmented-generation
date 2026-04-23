import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from rag.query_engine import query, query_stream
from rag.api.schemas import QueryRequest, QueryResponse, ChunkOut

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query_sync(request: QueryRequest):
    result = query(request.query, request.doc_ids)
    return QueryResponse(
        answer=result["answer"],
        chunks=[ChunkOut(**{k: c[k] for k in ("doc_id", "page", "text", "score")}) for c in result["chunks"]],
    )


@router.post("/stream")
def query_stream_endpoint(request: QueryRequest):
    def event_generator():
        for token in query_stream(request.query, request.doc_ids):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
