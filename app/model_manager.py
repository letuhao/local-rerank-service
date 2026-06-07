from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from app.config import Settings

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class ModelRecord:
    model_id: str
    path: Path
    state: ModelState = ModelState.UNLOADED
    encoder: CrossEncoder | None = None
    last_used_at: datetime | None = None
    ttl_seconds: int = 600
    keep_warm: bool = False
    vram_mb: int = 0
    ref_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._records: dict[str, ModelRecord] = {}
        self._global_lock = asyncio.Lock()
        self._reaper_task: asyncio.Task[None] | None = None

        for model_id in settings.model_ids:
            path = settings.resolve_model_path(model_id)
            if path is None:
                logger.warning("model %s has no resolvable path", model_id)
                continue
            self._records[model_id] = ModelRecord(
                model_id=model_id,
                path=path,
                ttl_seconds=settings.rerank_default_ttl,
            )

    async def start(self) -> None:
        self._reaper_task = asyncio.create_task(self._reaper_loop(), name="model-reaper")

    async def stop(self) -> None:
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
        async with self._global_lock:
            for record in self._records.values():
                await self._unload_record(record)

    def list_models(self) -> list[ModelRecord]:
        return list(self._records.values())

    def get_record(self, model_id: str) -> ModelRecord | None:
        return self._records.get(model_id)

    def to_status(self, record: ModelRecord) -> dict[str, Any]:
        expires_at: str | None = None
        if record.state == ModelState.LOADED and record.last_used_at and not record.keep_warm:
            expires_at = (
                record.last_used_at + timedelta(seconds=record.ttl_seconds)
            ).isoformat().replace("+00:00", "Z")

        last_used = (
            record.last_used_at.isoformat().replace("+00:00", "Z")
            if record.last_used_at
            else None
        )
        return {
            "id": record.model_id,
            "state": record.state.value,
            "vram_mb": record.vram_mb if record.state == ModelState.LOADED else 0,
            "last_used_at": last_used,
            "ttl_seconds": record.ttl_seconds,
            "expires_at": expires_at,
            "keep_warm": record.keep_warm,
        }

    async def load(
        self,
        model_id: str,
        *,
        ttl_seconds: int | None = None,
        keep_warm: bool | None = None,
    ) -> tuple[ModelRecord, int, bool]:
        record = self._require_record(model_id)
        if ttl_seconds is not None:
            record.ttl_seconds = ttl_seconds
        if keep_warm is not None:
            record.keep_warm = keep_warm

        async with record.lock:
            if record.state == ModelState.LOADED:
                return record, 0, False

            await self._ensure_capacity_for(record)
            started = time.perf_counter()
            cold_start = True
            record.state = ModelState.LOADING
            try:
                encoder = await asyncio.to_thread(self._load_encoder, record.path)
                record.encoder = encoder
                record.state = ModelState.LOADED
                record.vram_mb = self._estimate_vram_mb()
                record.last_used_at = datetime.now(UTC)
                loaded_ms = int((time.perf_counter() - started) * 1000)
                return record, loaded_ms, cold_start
            except RuntimeError as exc:
                record.state = ModelState.ERROR
                if "out of memory" in str(exc).lower():
                    raise
                raise RuntimeError("load failed") from exc
            except Exception as exc:
                record.state = ModelState.ERROR
                raise RuntimeError("load failed") from exc

    async def unload(self, model_id: str) -> ModelRecord:
        record = self._require_record(model_id)
        async with record.lock:
            if record.ref_count > 0:
                raise RuntimeError("model in use")
            await self._unload_record(record)
            return record

    async def rerank(
        self,
        model_id: str,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        return_documents: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        record = self._require_record(model_id)
        loaded_ms = 0
        cold_start = False

        async with record.lock:
            record.ref_count += 1
            try:
                if record.state != ModelState.LOADED:
                    await self._ensure_capacity_for(record)
                    started = time.perf_counter()
                    cold_start = True
                    record.state = ModelState.LOADING
                    record.encoder = await asyncio.to_thread(
                        self._load_encoder, record.path
                    )
                    record.state = ModelState.LOADED
                    record.vram_mb = self._estimate_vram_mb()
                    loaded_ms = int((time.perf_counter() - started) * 1000)

                record.last_used_at = datetime.now(UTC)
                assert record.encoder is not None
                pairs = [[query, doc] for doc in documents]
                raw_scores = await asyncio.to_thread(
                    record.encoder.predict,
                    pairs,
                    batch_size=self._settings.score_batch_size,
                )
                scores = self._normalize_scores(raw_scores)
                ranked = sorted(
                    enumerate(scores),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if top_n is not None:
                    ranked = ranked[:top_n]

                results: list[dict[str, Any]] = []
                for index, score in ranked:
                    item: dict[str, Any] = {
                        "index": index,
                        "relevance_score": float(score),
                    }
                    if return_documents:
                        item["document"] = documents[index]
                    results.append(item)

                meta = {
                    "cold_start": cold_start,
                    "loaded_ms": loaded_ms,
                    "scored": len(documents),
                }
                return results, meta
            finally:
                record.ref_count -= 1

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(self._settings.rerank_reaper_interval)
            try:
                await self._reap_expired()
            except Exception:
                logger.exception("reaper tick failed")

    async def _reap_expired(self) -> None:
        now = datetime.now(UTC)
        async with self._global_lock:
            for record in self._records.values():
                if record.keep_warm or record.state != ModelState.LOADED:
                    continue
                if record.ref_count > 0:
                    continue
                if record.last_used_at is None:
                    continue
                idle_seconds = (now - record.last_used_at).total_seconds()
                if idle_seconds >= record.ttl_seconds:
                    async with record.lock:
                        if record.ref_count == 0 and record.state == ModelState.LOADED:
                            await self._unload_record(record)
                            logger.info("reaped idle model %s", record.model_id)

    async def _ensure_capacity_for(self, target: ModelRecord) -> None:
        async with self._global_lock:
            loaded = [
                r
                for r in self._records.values()
                if r.state == ModelState.LOADED and r.model_id != target.model_id
            ]
            loaded.sort(key=lambda r: r.last_used_at or datetime.min.replace(tzinfo=UTC))

            while len(loaded) >= self._settings.rerank_max_loaded:
                evicted = await self._evict_one(loaded)
                if evicted is None:
                    raise RuntimeError("out of memory")
                loaded = [
                    r
                    for r in self._records.values()
                    if r.state == ModelState.LOADED and r.model_id != target.model_id
                ]

            projected_vram = sum(r.vram_mb for r in loaded) + self._settings.estimated_vram_mb
            while projected_vram > self._settings.rerank_vram_budget_mb and loaded:
                evicted = await self._evict_one(loaded)
                if evicted is None:
                    raise RuntimeError("out of memory")
                loaded = [
                    r
                    for r in self._records.values()
                    if r.state == ModelState.LOADED and r.model_id != target.model_id
                ]
                projected_vram = (
                    sum(r.vram_mb for r in loaded) + self._settings.estimated_vram_mb
                )

            if projected_vram > self._settings.rerank_vram_budget_mb:
                raise RuntimeError("out of memory")

    async def _evict_one(self, loaded: list[ModelRecord]) -> ModelRecord | None:
        candidates = [r for r in loaded if not r.keep_warm and r.ref_count == 0]
        if not candidates:
            return None
        candidates.sort(key=lambda r: r.last_used_at or datetime.min.replace(tzinfo=UTC))
        victim = candidates[0]
        async with victim.lock:
            if victim.ref_count == 0 and victim.state == ModelState.LOADED:
                await self._unload_record(victim)
                return victim
        return None

    async def _unload_record(self, record: ModelRecord) -> None:
        if record.state == ModelState.UNLOADED:
            return
        record.state = ModelState.UNLOADING
        record.encoder = None
        record.vram_mb = 0
        record.last_used_at = None
        await asyncio.to_thread(self._release_gpu_memory)
        record.state = ModelState.UNLOADED

    def _load_encoder(self, path: Path) -> CrossEncoder:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        automodel_args: dict[str, Any] = {}
        if device == "cuda":
            automodel_args["torch_dtype"] = torch.float16
        return CrossEncoder(
            str(path),
            max_length=self._settings.cross_encoder_max_length,
            device=device,
            automodel_args=automodel_args,
        )

    def _release_gpu_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _estimate_vram_mb(self) -> int:
        if torch.cuda.is_available():
            return max(1, int(torch.cuda.memory_allocated() / (1024 * 1024)))
        return 0

    @staticmethod
    def _normalize_scores(raw_scores: Any) -> list[float]:
        values = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return []
        if values.min() >= 0.0 and values.max() <= 1.0:
            return values.tolist()
        sigmoid = 1.0 / (1.0 + np.exp(-values))
        return sigmoid.tolist()

    def _require_record(self, model_id: str) -> ModelRecord:
        record = self._records.get(model_id)
        if record is None:
            raise KeyError(model_id)
        return record
