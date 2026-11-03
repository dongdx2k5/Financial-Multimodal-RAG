param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$labRoot = Join-Path $repoRoot "work\ocr-lab"
$labPython = Join-Path $labRoot ".venv\Scripts\python.exe"
$modelRoot = Join-Path $labRoot "models"

function Assert-Exit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE)." }
}

function Get-Model([string]$Url, [string]$Name, [long]$Size, [string]$Sha256 = "") {
    $target = [IO.Path]::GetFullPath((Join-Path $modelRoot $Name))
    if (-not $target.StartsWith($modelRoot + [IO.Path]::DirectorySeparatorChar)) {
        throw "Model target is outside the OCR lab."
    }
    if (-not (Test-Path -LiteralPath $target)) {
        $partial = $target + ".part"
        & curl.exe --fail --location --continue-at - --max-time 1800 --output $partial $Url
        Assert-Exit "Download $Name (partial file retained for retry)"
        if ((Get-Item -LiteralPath $partial).Length -ne $Size) { throw "Unexpected size: $Name" }
        if ($Sha256 -and (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $Sha256) {
            throw "Checksum mismatch: $Name"
        }
        Move-Item -LiteralPath $partial -Destination $target
    }
    if ((Get-Item -LiteralPath $target).Length -ne $Size) {
        throw "Existing $Name is incomplete. Use a separate clean model directory; it was not overwritten."
    }
    if ($Sha256 -and (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $Sha256) {
        throw "Checksum mismatch: $Name"
    }
    Get-FileHash -LiteralPath $target -Algorithm SHA256 | Format-List
}

Push-Location $repoRoot
$oldUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = "1" # Upstream setup.py reads its Vietnamese README using the default encoding.
    New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null
    if (-not (Test-Path -LiteralPath $labPython)) {
        & $Python -m venv (Join-Path $labRoot ".venv")
        Assert-Exit "Create isolated environment"
    }
    & $labPython -m pip install --upgrade pip
    Assert-Exit "pip"
    & $labPython -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
    Assert-Exit "CPU PyTorch"
    # Pinned upstream source avoids old PyPI VietOCR dependencies conflicting with NumPy 2.
    & $labPython -m pip install -e ".[dev]" rapidocr==3.10.0 rapid-table==3.0.2 onnxruntime==1.31.1 `
        "vietocr @ https://github.com/pbcquoc/vietocr/archive/fe8c3a7fc714aec57ab81cec844eb3adf0c1636c.zip"
    Assert-Exit "OCR dependencies"
    Get-Model "https://vocr.vn/data/vietocr/vgg_seq2seq.pth" "vgg_seq2seq.pth" 89575371
    Get-Model "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx" `
        "ch_PP-OCRv5_det_mobile.onnx" 4819576 "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"
    Get-Model "https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/v2.0.0/slanet-plus.onnx" `
        "slanet-plus.onnx" 7758305 "d57a942af6a2f57d6a4a0372573c696a2379bf5857c45e2ac69993f3b334514b"
    & $labPython -m pip check
    Assert-Exit "Dependency validation"
    & $labPython -c "import pymupdf, torch, rapidocr, rapid_table; print('OCR runtime imports OK')"
    Assert-Exit "Runtime validation (Windows policy may block native DLLs)"
    Write-Host "OCR lab ready. The app .env and existing index were NOT changed."
} finally {
    $env:PYTHONUTF8 = $oldUtf8
    Pop-Location
}
