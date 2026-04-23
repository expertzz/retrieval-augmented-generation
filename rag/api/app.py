from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.routers import documents, query

app = FastAPI(title="RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api")
app.include_router(query.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
