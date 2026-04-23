from typing import Iterator
import anthropic
from rag.config import settings

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using only the provided context.
If the answer is not in the context, say so clearly.
When referencing information, cite the source as [Page N]."""


def _build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[Page {c['page']}]: {c['text']}" for c in chunks)


def generate(query: str, chunks: list[dict]) -> str:
    response = _client.messages.create(
        model=settings.generation_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{_build_context(chunks)}\n\nQuestion: {query}"}],
    )
    return response.content[0].text


def generate_stream(query: str, chunks: list[dict]) -> Iterator[str]:
    with _client.messages.stream(
        model=settings.generation_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{_build_context(chunks)}\n\nQuestion: {query}"}],
    ) as stream:
        for text in stream.text_stream:
            yield text
