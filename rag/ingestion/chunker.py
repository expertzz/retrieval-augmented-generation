from nltk.tokenize import sent_tokenize
from rag.config import settings
from rag.ingestion.base import PageData, ChunkData


def chunk_pages(pages: list[PageData], doc_id: str) -> list[ChunkData]:
    max_size = settings.chunk_size
    overlap = settings.chunk_overlap
    chunks = []
    chunk_index = 0

    for page in pages:
        sentences = sent_tokenize(page.text)
        buffer = ""

        for sentence in sentences:
            if len(sentence) > max_size:
                if buffer:
                    chunks.append(ChunkData(doc_id=doc_id, page=page.page, chunk_index=chunk_index, text=buffer))
                    chunk_index += 1
                    buffer = ""
                start = 0
                while start < len(sentence):
                    chunks.append(ChunkData(
                        doc_id=doc_id,
                        page=page.page,
                        chunk_index=chunk_index,
                        text=sentence[start:start + max_size],
                    ))
                    chunk_index += 1
                    start += max_size - overlap
            elif len(buffer) + len(sentence) + 1 > max_size:
                chunks.append(ChunkData(doc_id=doc_id, page=page.page, chunk_index=chunk_index, text=buffer))
                chunk_index += 1
                buffer = sentence
            else:
                buffer = (buffer + " " + sentence).strip()

        if buffer:
            chunks.append(ChunkData(doc_id=doc_id, page=page.page, chunk_index=chunk_index, text=buffer))
            chunk_index += 1

    return chunks
