from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class PageData:
    page: int
    text: str
    doc_title: str = ""
    source_url: str = ""
    section_heading: str = ""


@dataclass
class ChunkData:
    doc_id: str
    page: int
    chunk_index: int
    text: str
    faiss_index: int = 0  # filled in by pipeline after embedding


class Ingestor(Protocol):
    def can_handle(self, source: str) -> bool: ...
    def extract_pages(self, source: str) -> list[PageData]: ...
