from pathlib import Path
from pypdf import PdfReader


def extract_text(pdf_path: str) -> list[dict]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        text = text.strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({"page": page["page"], "start": start, "text": chunk})
            start += chunk_size - overlap
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
    for c in chunks[:3]:
        print(f"\n--- Page {c['page']}, offset {c['start']} ---")
        print(c["text"][:200])
