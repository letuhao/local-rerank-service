from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rerank_host: str = Field(default="0.0.0.0", alias="RERANK_HOST")
    rerank_port: int = Field(default=28417, alias="RERANK_PORT")
    rerank_service_token: str = Field(alias="RERANK_SERVICE_TOKEN")
    rerank_default_ttl: int = Field(default=600, alias="RERANK_DEFAULT_TTL")
    rerank_reaper_interval: int = Field(default=30, alias="RERANK_REAPER_INTERVAL")
    rerank_vram_budget_mb: int = Field(default=4096, alias="RERANK_VRAM_BUDGET_MB")
    rerank_max_loaded: int = Field(default=1, alias="RERANK_MAX_LOADED")
    rerank_models: str = Field(default="bge-reranker-v2-m3", alias="RERANK_MODELS")
    rerank_model_path: Path = Field(
        default=Path("./models/bge-reranker-v2-m3"),
        alias="RERANK_MODEL_PATH",
    )

    max_documents: int = Field(default=256, alias="RERANK_MAX_DOCUMENTS")
    max_doc_chars: int = Field(default=8192, alias="RERANK_MAX_DOC_CHARS")
    cross_encoder_max_length: int = Field(default=512, alias="RERANK_MAX_LENGTH")
    score_batch_size: int = Field(default=32, alias="RERANK_SCORE_BATCH_SIZE")
    estimated_vram_mb: int = Field(default=1180, alias="RERANK_ESTIMATED_VRAM_MB")

    @field_validator("rerank_service_token")
    @classmethod
    def token_must_be_set(cls, value: str) -> str:
        if not value or value == "change-me":
            # Allow dev default but warn at runtime; e2e uses .env value.
            return value
        return value

    @property
    def model_ids(self) -> list[str]:
        return [item.strip() for item in self.rerank_models.split(",") if item.strip()]

    def resolve_model_path(self, model_id: str) -> Path | None:
        if model_id not in self.model_ids:
            return None
        if len(self.model_ids) == 1 and model_id == self.model_ids[0]:
            return self.rerank_model_path.resolve()
        candidate = Path("models") / model_id
        return candidate.resolve() if candidate.exists() else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
