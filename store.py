import json
import numpy as np
import faiss


INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


def build(chunks: list[dict]) -> None:
    vectors = np.array([c["embedding"] for c in chunks], dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)

    metadata = [{"page": c["page"], "text": c["text"]} for c in chunks]
    with open(CHUNKS_FILE, "w") as f:
        json.dump(metadata, f)


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE) as f:
        metadata = json.load(f)

    vector = np.array([query_embedding], dtype="float32")
    faiss.normalize_L2(vector)

    scores, indices = index.search(vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({"score": float(score), **metadata[idx]})
    return results
