from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    filename: str
    doc_type: str
    page_count: int
    chunk_count: int
    status: str


class QueryRequest(BaseModel):
    query: str
    doc_ids: list[str] | None = None
    stream: bool = False


class ChunkOut(BaseModel):
    doc_id: str
    page: int
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    chunks: list[ChunkOut]
