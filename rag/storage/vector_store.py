import numpy as np
import faiss

from rag.config import settings


def _index_path() -> str:
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    return str(settings.index_dir / "vectors.faiss")


def _build_empty(dim: int) -> faiss.IndexIDMap2:
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(base)


def add(vectors: np.ndarray, ids: np.ndarray) -> None:
    """Add vectors with explicit integer IDs (FAISS row indices)."""
    path = _index_path()
    vectors = vectors.astype("float32")
    faiss.normalize_L2(vectors)

    try:
        index = faiss.read_index(path)
    except Exception:
        index = _build_empty(vectors.shape[1])

    index.add_with_ids(vectors, ids)
    faiss.write_index(index, path)


def search(query_vector: list[float], top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, ids) for the top_k nearest vectors."""
    index = faiss.read_index(_index_path())
    vector = np.array([query_vector], dtype="float32")
    faiss.normalize_L2(vector)
    scores, ids = index.search(vector, top_k)
    return scores[0], ids[0]


def remove(ids: np.ndarray) -> None:
    """Delete vectors by their IDs — enables per-document deletion."""
    path = _index_path()
    index = faiss.read_index(path)
    index.remove_ids(ids)
    faiss.write_index(index, path)


def count() -> int:
    try:
        return faiss.read_index(_index_path()).ntotal
    except Exception:
        return 0
