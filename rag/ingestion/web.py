import trafilatura
from rag.ingestion.base import PageData


class WebIngestor:
    def can_handle(self, source: str) -> bool:
        return source.startswith("http://") or source.startswith("https://")

    def extract_pages(self, source: str) -> list[PageData]:
        downloaded = trafilatura.fetch_url(source)
        text = trafilatura.extract(downloaded) or ""
        if not text:
            return []
        # Split into pseudo-pages of 3000 chars
        pages, buffer, page_num = [], "", 1
        for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
            if len(buffer) + len(para) > 3000:
                if buffer:
                    pages.append(PageData(page=page_num, text=buffer, source_url=source))
                    page_num += 1
                buffer = para
            else:
                buffer = (buffer + "\n\n" + para).strip()
        if buffer:
            pages.append(PageData(page=page_num, text=buffer, source_url=source))
        return pages
