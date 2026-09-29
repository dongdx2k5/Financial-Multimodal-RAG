$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$ollamaExe = 'C:\DXD\Ollama\ollama.exe'
if (-not (Test-Path -LiteralPath $ollamaExe)) {
    $ollamaExe = (Get-Command ollama -ErrorAction Stop).Source
}
try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 | Out-Null }
catch {
    $env:OLLAMA_MODELS = 'C:\DXD\OllamaModels'
    $env:OLLAMA_HOST = '127.0.0.1:11434'
    $env:OLLAMA_NO_CLOUD = '1'
    Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden
    Start-Sleep -Seconds 3
}
& '.\.venv\Scripts\python.exe' -m uvicorn financial_rag.api:app --host 127.0.0.1 --port 8000
