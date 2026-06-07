from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.auth import require_token
from app.model_manager import ModelManager, ModelState
from app.schemas import LoadOptions, LoadResponse, ModelStatus, ModelsListResponse, UnloadResponse

router = APIRouter(prefix="/v1/models", tags=["models"])


def get_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


@router.get("", response_model=ModelsListResponse)
async def list_models(
    _: None = Depends(require_token),
    manager: ModelManager = Depends(get_manager),
) -> ModelsListResponse:
    models = [ModelStatus(**manager.to_status(record)) for record in manager.list_models()]
    return ModelsListResponse(models=models)


@router.get("/{model_id}", response_model=ModelStatus)
async def get_model(
    model_id: str,
    _: None = Depends(require_token),
    manager: ModelManager = Depends(get_manager),
) -> ModelStatus:
    record = manager.get_record(model_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "model_not_found"})
    return ModelStatus(**manager.to_status(record))


@router.post("/{model_id}/load", response_model=LoadResponse)
async def load_model(
    model_id: str,
    options: LoadOptions | None = None,
    _: None = Depends(require_token),
    manager: ModelManager = Depends(get_manager),
) -> LoadResponse:
    if manager.get_record(model_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "model_not_found"})

    opts = options or LoadOptions()
    try:
        record, loaded_ms, _ = await manager.load(
            model_id,
            ttl_seconds=opts.ttl_seconds,
            keep_warm=opts.keep_warm,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": "out_of_memory"},
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal"},
        ) from exc

    status_payload = manager.to_status(record)
    return LoadResponse(
        id=record.model_id,
        state=record.state.value,
        vram_mb=record.vram_mb,
        loaded_ms=loaded_ms,
        expires_at=status_payload["expires_at"],
    )


@router.post("/{model_id}/unload", response_model=UnloadResponse)
async def unload_model(
    model_id: str,
    _: None = Depends(require_token),
    manager: ModelManager = Depends(get_manager),
) -> UnloadResponse:
    if manager.get_record(model_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "model_not_found"})

    record = manager.get_record(model_id)
    assert record is not None
    if record.state == ModelState.UNLOADED:
        return UnloadResponse(id=record.model_id, state=record.state.value)
    if record.state == ModelState.LOADED and record.ref_count > 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "model_loading"},
        )

    try:
        record = await manager.unload(model_id)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "model_loading"},
        ) from exc

    return UnloadResponse(id=record.model_id, state=record.state.value)
