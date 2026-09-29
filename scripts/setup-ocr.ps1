param([switch]$IncludeBest)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$variants = @('fast')
if ($IncludeBest) { $variants += 'best' }
foreach ($variant in $variants) {
    $folder = if ($variant -eq 'fast') { 'tessdata' } else { 'tessdata-best' }
    $destination = Join-Path $projectRoot "data/$folder"
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    foreach ($language in @('vie', 'eng')) {
        $target = Join-Path $destination "$language.traineddata"
        if (-not (Test-Path -LiteralPath $target)) {
            Invoke-WebRequest "https://raw.githubusercontent.com/tesseract-ocr/tessdata_$variant/main/$language.traineddata" -OutFile $target
        }
    }
    Write-Output "Installed $variant OCR language files: $destination"
}
Write-Output 'Set OCR_ENABLED=true and TESSDATA_PREFIX in .env. Optionally set TESSDATA_BEST_PREFIX.'
