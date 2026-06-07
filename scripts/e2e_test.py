"""End-to-end smoke test against a running rerank service."""

from __future__ import annotations

import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("RERANK_E2E_URL", f"http://127.0.0.1:{os.getenv('RERANK_PORT', '28417')}")
TOKEN = os.getenv("RERANK_SERVICE_TOKEN", "change-me")
MODEL = os.getenv("RERANK_MODELS", "bge-reranker-v2-m3").split(",")[0].strip()

QUERY = "主角第一次开启神武印记的经过"
DOCS = [
    "张若尘睁开双眼，识海深处一道神武印记缓缓浮现，与他血脉相连。",
    "他在武市买了一把剑，准备参加年终比武。",
    "量子计算机在低温实验室里稳定运行。",
]


def headers(token: str | None = TOKEN) -> dict[str, str]:
    if token is None:
        return {}
    return {"Authorization": f"Bearer {token}"}


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise SystemExit(1)


def main() -> None:
    print(f"e2e target: {BASE_URL}")
    timeout = httpx.Timeout(120.0, connect=10.0)

    with httpx.Client(base_url=BASE_URL, timeout=timeout) as client:
        r = client.get("/health")
        check("health without auth -> 401", r.status_code == 401)

        r = client.get("/health", headers=headers("bad-token"))
        check("health bad token -> 401", r.status_code == 401)

        r = client.get("/health", headers=headers())
        check("health ok", r.status_code == 200, r.text)

        r = client.get("/ready", headers=headers())
        check("ready ok", r.status_code == 200)

        r = client.get("/v1/models", headers=headers())
        check("list models", r.status_code == 200)
        models = r.json()["models"]
        check("default model registered", any(m["id"] == MODEL for m in models))

        r = client.post(
            "/v1/rerank",
            headers=headers(),
            json={"model": MODEL, "query": "", "documents": DOCS},
        )
        check("empty query -> 400", r.status_code == 400)

        r = client.post(
            "/v1/rerank",
            headers=headers(),
            json={"model": "unknown-model", "query": QUERY, "documents": DOCS},
        )
        check("unknown model -> 404", r.status_code == 404)

        print("[....] cold-start rerank (may take ~30-60s on first load)...")
        started = time.perf_counter()
        r = client.post(
            "/v1/rerank",
            headers=headers(),
            json={
                "model": MODEL,
                "query": QUERY,
                "documents": DOCS,
                "top_n": 3,
                "return_documents": False,
            },
        )
        elapsed = time.perf_counter() - started
        check("rerank cold/warm -> 200", r.status_code == 200, f"{elapsed:.1f}s {r.text[:200]}")
        payload = r.json()
        results = payload["results"]
        meta = payload["meta"]
        check("results sorted desc", all(
            results[i]["relevance_score"] >= results[i + 1]["relevance_score"]
            for i in range(len(results) - 1)
        ))
        top_index = results[0]["index"]
        top_score = results[0]["relevance_score"]
        junk_score = next(item["relevance_score"] for item in results if item["index"] == 2)
        check(
            "relevant doc beats junk",
            top_index == 0 and top_score > junk_score,
            f"top_index={top_index} top={top_score:.4f} junk={junk_score:.4f}",
        )
        print(
            f"      meta: cold_start={meta['cold_start']} loaded_ms={meta['loaded_ms']} "
            f"scored={meta['scored']}"
        )

        r = client.get(f"/v1/models/{MODEL}", headers=headers())
        check("model loaded after rerank", r.status_code == 200)
        check("model state loaded", r.json()["state"] == "loaded")

        print("[....] warm rerank...")
        started = time.perf_counter()
        r = client.post(
            "/v1/rerank",
            headers=headers(),
            json={"model": MODEL, "query": QUERY, "documents": DOCS, "top_n": 3},
        )
        warm_elapsed = time.perf_counter() - started
        check("warm rerank -> 200", r.status_code == 200, f"{warm_elapsed:.1f}s")
        warm_meta = r.json()["meta"]
        check("warm cold_start false", warm_meta["cold_start"] is False)
        check("warm loaded_ms zero", warm_meta["loaded_ms"] == 0)

        r = client.post(f"/v1/models/{MODEL}/unload", headers=headers())
        check("unload -> 200", r.status_code == 200)
        check("unload state", r.json()["state"] == "unloaded")

        r = client.post(f"/v1/models/{MODEL}/load", headers=headers(), json={})
        check("explicit load -> 200", r.status_code == 200)
        check("load state", r.json()["state"] == "loaded")

    print("\nAll e2e checks passed.")


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError as exc:
        print(f"Cannot connect to {BASE_URL}: {exc}", file=sys.stderr)
        print("Start the server first: .\\.venv\\Scripts\\python.exe -m app.main", file=sys.stderr)
        raise SystemExit(1) from exc
