from pypdf import PdfReader
from rag.ingestion.base import PageData


class PdfIngestor:
    def can_handle(self, source: str) -> bool:
        return source.lower().endswith(".pdf")

    def extract_pages(self, source: str) -> list[PageData]:
        reader = PdfReader(source)
        pages = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(PageData(page=i + 1, text=text))
        return pages
