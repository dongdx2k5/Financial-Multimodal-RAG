param(
    [ValidateSet("vietocr", "tesseract")][string]$OcrBackend = "vietocr",
    [int]$Port = 8001,
    [switch]$Tables
)
$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$labPython = Join-Path $repoRoot "work\ocr-lab\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $labPython)) { throw "Run scripts/setup-ocr-lab.ps1 first." }
if ($Port -lt 1024 -or $Port -gt 65535) { throw "Port must be between 1024 and 65535." }
$labSettings = @{
    PYTHONUTF8 = "1"
    DATA_DIR = (Join-Path $repoRoot "work\ocr-lab\app-data")
    OCR_ENABLED = "true"
    OCR_BACKEND = $OcrBackend
    OCR_FALLBACK_TO_TESSERACT = "false"
    VIETOCR_WEIGHTS = (Join-Path $repoRoot "work\ocr-lab\models\vgg_seq2seq.pth")
    OCR_DETECTOR_PATH = (Join-Path $repoRoot "work\ocr-lab\models\ch_PP-OCRv5_det_mobile.onnx")
    TABLE_BACKEND = $(if ($Tables) { "slanet_plus" } else { "none" })
    TABLE_MODEL_PATH = (Join-Path $repoRoot "work\ocr-lab\models\slanet-plus.onnx")
}
$previousValues = @{}
Push-Location $repoRoot
try {
    foreach ($entry in $labSettings.GetEnumerator()) {
        $previousValues[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, "Process")
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
    Write-Host "Experimental OCR app: http://127.0.0.1:$Port ; isolated data in work/ocr-lab/app-data"
    & $labPython -m uvicorn financial_rag.api:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw "OCR lab server exited with code $LASTEXITCODE." }
} finally {
    foreach ($entry in $previousValues.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
    Pop-Location
}
