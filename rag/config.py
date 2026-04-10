from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API keys
    openai_api_key: str
    anthropic_api_key: str
    cohere_api_key: str | None = None

    # Storage paths
    db_path: Path = Path("data/rag.db")
    index_dir: Path = Path("data/indices")

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    # Models
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "claude-opus-4-6"

    # Features
    use_hybrid: bool = True
    use_reranker: bool = True


settings = Settings()
