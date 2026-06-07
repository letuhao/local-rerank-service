"""Quick sanity check after environment setup."""

from __future__ import annotations

import sys


def main() -> None:
    print(f"python: {sys.version.split()[0]} @ {sys.executable}")

    import torch

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    import sentence_transformers

    print(f"sentence-transformers: {sentence_transformers.__version__}")

    import fastapi
    import uvicorn

    print(f"fastapi: {fastapi.__version__}")
    print(f"uvicorn: {uvicorn.__version__}")

    from sentence_transformers import CrossEncoder

    print(f"CrossEncoder: {CrossEncoder}")

    print("environment ok")


if __name__ == "__main__":
    main()
