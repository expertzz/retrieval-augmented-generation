from embed import client
from store import search


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    query_embedding = response.data[0].embedding
    return search(query_embedding, top_k)
