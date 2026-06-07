from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.config import get_settings
from app.model_manager import ModelManager
from app.routers import health, models, rerank

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    manager = ModelManager(settings)
    await manager.start()
    app.state.model_manager = manager
    logger.info(
        "rerank service ready on %s:%s (models=%s)",
        settings.rerank_host,
        settings.rerank_port,
        settings.model_ids,
    )
    yield
    await manager.stop()


app = FastAPI(title="local-rerank-service", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(models.router)
app.include_router(rerank.router)


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.rerank_host,
        port=settings.rerank_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
