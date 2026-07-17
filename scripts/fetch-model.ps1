# Fetch the all-MiniLM-L6-v2 ONNX export that burn-import compiles at build time.
# The file is large (~86 MB) and gitignored, so each machine fetches it once.

$Dest = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../crates/embed-burn/artifacts/model.onnx"))
$Url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"

$ParentDir = Split-Path -Parent $Dest
if (-not (Test-Path $ParentDir)) {
    New-Item -ItemType Directory -Force -Path $ParentDir | Out-Null
}

if (Test-Path $Dest) {
    Write-Host "already present: $Dest"
    exit 0
}

Write-Host "downloading $Url"
try {
    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
    Write-Host "saved to $Dest"
} catch {
    Write-Error "Failed to download model: $_"
    exit 1
}
