from rag.ingestion.base import Ingestor, PageData
from rag.ingestion.pdf import PdfIngestor
from rag.ingestion.docx import DocxIngestor
from rag.ingestion.txt import TxtIngestor
from rag.ingestion.web import WebIngestor

_INGESTORS: list[Ingestor] = [
    PdfIngestor(),
    DocxIngestor(),
    TxtIngestor(),
    WebIngestor(),
]


def get_ingestor(source: str) -> Ingestor:
    for ingestor in _INGESTORS:
        if ingestor.can_handle(source):
            return ingestor
    raise ValueError(f"No ingestor found for: {source}")


def extract_pages(source: str) -> list[PageData]:
    return get_ingestor(source).extract_pages(source)
