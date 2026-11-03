$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Chưa có .venv. Hãy cài môi trường theo README.md trước.'
}
# Deterministic question answering does not start or require a language model.
& $pythonExe -m uvicorn financial_rag.api:app --host 127.0.0.1 --port 8000
