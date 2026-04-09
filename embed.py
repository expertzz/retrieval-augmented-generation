import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    for chunk, embedding_obj in zip(chunks, response.data):
        chunk["embedding"] = embedding_obj.embedding
    return chunks
