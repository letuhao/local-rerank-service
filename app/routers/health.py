from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth import require_token
from app.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(_: None = Depends(require_token)) -> HealthResponse:
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
async def ready(_: None = Depends(require_token)) -> ReadyResponse:
    return ReadyResponse()
