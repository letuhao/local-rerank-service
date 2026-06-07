# Create / refresh the project-local Python environment (Windows PowerShell).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> Creating .venv with Python 3.12 (uv)..."
uv venv --python 3.12 .venv

Write-Host "==> Installing dependencies (PyTorch CUDA 12.4 + app libs)..."
uv sync

Write-Host "==> Verifying GPU + imports..."
.\.venv\Scripts\python.exe .\scripts\verify_env.py

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
