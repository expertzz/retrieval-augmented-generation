from docx import Document
from rag.ingestion.base import PageData


class DocxIngestor:
    def can_handle(self, source: str) -> bool:
        return source.lower().endswith(".docx")

    def extract_pages(self, source: str) -> list[PageData]:
        doc = Document(source)
        # Word has no real pages — group paragraphs into chunks of 10
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        pages = []
        group_size = 10
        for i in range(0, len(paragraphs), group_size):
            group = paragraphs[i:i + group_size]
            pages.append(PageData(page=i // group_size + 1, text="\n\n".join(group)))
        return pages
