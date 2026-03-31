from pypdf import PdfReader
from nltk.tokenize import sent_tokenize


def extract_text(pdf_path: str) -> list[dict]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(pages: list[dict], max_chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    for page in pages:
        sentences = sent_tokenize(page["text"])
        buffer = ""
        for sentence in sentences:
            if len(sentence) > max_chunk_size:
                if buffer:
                    chunks.append({"page": page["page"], "text": buffer})
                    buffer = ""
                start = 0
                while start < len(sentence):
                    chunks.append({"page": page["page"], "text": sentence[start:start + max_chunk_size]})
                    start += max_chunk_size - overlap
            elif len(buffer) + len(sentence) + 1 > max_chunk_size:
                chunks.append({"page": page["page"], "text": buffer})
                buffer = sentence
            else:
                buffer = (buffer + " " + sentence).strip()
        if buffer:
            chunks.append({"page": page["page"], "text": buffer})
    return chunks


def ingest(pdf_path: str) -> list[dict]:
    pages = extract_text(pdf_path)
    chunks = chunk_text(pages)
    return chunks


if __name__ == "__main__":
    import sys
    
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    chunks = ingest(path)
    print(f"Extracted {len(chunks)} chunks from '{path}'")
    for c in chunks:
        print(f"\n--- Page {c['page']} ---")
        print(c["text"][:50])
