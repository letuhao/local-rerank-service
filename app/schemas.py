from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ModelState = Literal["unloaded", "loading", "loaded", "unloading", "error"]


class ErrorResponse(BaseModel):
    error: str
    retry_after_ms: int | None = None


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int | None = None
    return_documents: bool = False


class RerankResultItem(BaseModel):
    index: int
    relevance_score: float
    document: str | None = None


class RerankMeta(BaseModel):
    cold_start: bool
    loaded_ms: int
    scored: int


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResultItem]
    meta: RerankMeta


class ModelStatus(BaseModel):
    id: str
    state: ModelState
    vram_mb: int = 0
    last_used_at: str | None = None
    ttl_seconds: int
    expires_at: str | None = None
    keep_warm: bool = False


class ModelsListResponse(BaseModel):
    models: list[ModelStatus]


class LoadOptions(BaseModel):
    ttl_seconds: int | None = None
    keep_warm: bool | None = None


class LoadResponse(BaseModel):
    id: str
    state: ModelState
    vram_mb: int
    loaded_ms: int
    expires_at: str | None = None


class UnloadResponse(BaseModel):
    id: str
    state: ModelState


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool = True
