from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.auth import require_token
from app.config import Settings, get_settings
from app.model_manager import ModelManager
from app.schemas import RerankMeta, RerankRequest, RerankResponse, RerankResultItem

router = APIRouter(prefix="/v1", tags=["rerank"])


def get_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


@router.post("/rerank", response_model=RerankResponse)
async def rerank(
    body: RerankRequest,
    _: None = Depends(require_token),
    manager: ModelManager = Depends(get_manager),
    settings: Settings = Depends(get_settings),
) -> RerankResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": "validation"})
    if not body.documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": "validation"})
    if len(body.documents) > settings.max_documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": "validation"})

    documents = [doc[: settings.max_doc_chars] for doc in body.documents]

    if manager.get_record(body.model) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "model_not_found"})

    try:
        results, meta = await manager.rerank(
            body.model,
            query,
            documents,
            top_n=body.top_n,
            return_documents=body.return_documents,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "model_not_found"},
        ) from exc
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
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal"},
        ) from exc

    return RerankResponse(
        model=body.model,
        results=[RerankResultItem(**item) for item in results],
        meta=RerankMeta(**meta),
    )
