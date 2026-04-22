from rag.ingestion.base import PageData


class TxtIngestor:
    def can_handle(self, source: str) -> bool:
        return source.lower().endswith((".txt", ".md"))

    def extract_pages(self, source: str) -> list[PageData]:
        with open(source, encoding="utf-8") as f:
            text = f.read()
        # Split on double newlines, group into pseudo-pages of 3000 chars
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        pages, buffer, page_num = [], "", 1
        for para in paragraphs:
            if len(buffer) + len(para) > 3000:
                if buffer:
                    pages.append(PageData(page=page_num, text=buffer))
                    page_num += 1
                buffer = para
            else:
                buffer = (buffer + "\n\n" + para).strip()
        if buffer:
            pages.append(PageData(page=page_num, text=buffer))
        return pages
