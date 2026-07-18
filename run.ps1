<#
.SYNOPSIS
    One-command reproduction of rust-inference-bench on Windows.
.DESCRIPTION
    Runs the full pipeline: parity -> CPU -> GPU -> footprint -> plots.
    Auto-detects host OS, CPU BLAS, and GPU backends.
    Pins threads (RAYON_NUM_THREADS=1) and builds in release mode for reproducibility.
.PARAMETER Gpu
    GPU backend (auto, macos, cuda, wgpu, off). Default is auto.
.PARAMETER CpuOnly
    Skip GPU phase (alias for -Gpu off).
.PARAMETER Blas
    CPU BLAS backend (auto, accelerate, mkl, openblas). Default is auto.
.PARAMETER Trials
    Benchmark trials per phase (default: 10).
.PARAMETER NoFootprint
    Skip cold-start / RSS / binary / build phase.
.PARAMETER NoPlots
    Skip SVG plot generation.
.EXAMPLE
    ./run.ps1 -Gpu cuda -Trials 5
#>
param(
    [string]$Gpu = "auto",

    [Alias("cpu-only")]
    [switch]$CpuOnly,

    [string]$Blas = "auto",

    [string]$Trials = "",

    [Alias("no-footprint")]
    [switch]$NoFootprint,

    [Alias("no-plots")]
    [switch]$NoPlots
)

$Root = $PSScriptRoot
Push-Location $Root

# Set default GPU behavior if CPU-only requested
if ($CpuOnly) {
    $Gpu = "off"
}

# Logging helper functions
function Log-Info {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Log-Warn {
    param([string]$Message)
    Write-Host " warn: $Message" -ForegroundColor Yellow
}

function Log-ErrorAndExit {
    param([string]$Message)
    Write-Host "error: $Message" -ForegroundColor Red
    Pop-Location -ErrorAction SilentlyContinue
    exit 1
}

# Detect operating system and architecture
$OS = if ([Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([Runtime.InteropServices.OSPlatform]::Windows)) {
    "Windows"
} elseif ([Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([Runtime.InteropServices.OSPlatform]::OSX)) {
    "Darwin"
} else {
    "Linux"
}
$Arch = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
$Exe = if ($OS -eq "Windows") { ".exe" } else { "" }

# Auto-configure target features on ARM64
$RustFlagsList = @()
if ($Arch -match "Arm64" -or $Arch -match "aarch64") {
    $RustFlagsList += "+fp16"
    $RustFlagsList += "+fhm"
}

if ($RustFlagsList.Count -gt 0) {
    $FeatureStr = $RustFlagsList -join ","
    if (-not $env:RUSTFLAGS) {
        $env:RUSTFLAGS = "-C target-feature=$FeatureStr"
    } else {
        $env:RUSTFLAGS += " -C target-feature=$FeatureStr"
    }
}

# Preflight tool checks
Log-Info "preflight ($OS/$Arch)"

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Log-ErrorAndExit "cargo not found - install Rust: https://rustup.rs"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Log-ErrorAndExit "uv not found - install uv: https://github.com/astral-sh/uv"
}

# Dynamic patching for esaxx-rs build.rs to prevent LNK2038 linkage conflicts on Windows
if ($OS -eq "Windows") {
    $esaxx_build_rs_glob = "$env:USERPROFILE\.cargo\registry\src\*\esaxx-rs-*\build.rs"
    $esaxx_files = Get-Item $esaxx_build_rs_glob -ErrorAction SilentlyContinue
    foreach ($file in $esaxx_files) {
        $content = Get-Content $file.FullName -Raw
        if ($content -like "*static_crt(true)*") {
            Log-Info "Found esaxx-rs build.rs forcing static CRT linkage. Patching..."
            $content = $content -replace "static_crt\(true\)", "static_crt(false)"
            Set-Content $file.FullName $content -NoNewline
            Log-Info "Successfully patched: $($file.FullName)"
        }
    }
}

# Resolve CPU BLAS backend
$BlasFeatures = ""
if ($Blas -eq "auto") {
    if ($OS -eq "Darwin") { $Blas = "accelerate" } else { $Blas = "none" }
}

switch ($Blas) {
    "accelerate" {
        if ($OS -ne "Darwin") {
            Log-Warn "accelerate is macOS-only; on $OS it has no effect"
        }
    }
    "none" {}
    "mkl" { $BlasFeatures = "blas-mkl" }
    "openblas" { $BlasFeatures = "blas-openblas" }
    default { Log-ErrorAndExit "unknown --blas: $Blas (auto|accelerate|mkl|openblas)" }
}

$BlasLogMsg = "BLAS backend: $Blas"
if ($BlasFeatures) {
    $BlasLogMsg += "  (features: $BlasFeatures)"
}
Log-Info $BlasLogMsg

# Detect GPU backend
function Detect-Gpu {
    if ($OS -eq "Darwin") { return "macos" }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { return "cuda" }
    if ($OS -eq "Windows") { return "wgpu" } # Assume wgpu via DX12/Vulkan on Windows
    if (Get-Command vulkaninfo -ErrorAction SilentlyContinue) { return "wgpu" }
    return "off"
}

if ($Gpu -eq "auto") {
    $Gpu = Detect-Gpu
    Log-Info "GPU backend: $Gpu (auto-detected)"
} else {
    Log-Info "GPU backend: $Gpu"
}

$GpuFeatures = ""
switch ($Gpu) {
    "off" {}
    "macos" { $GpuFeatures = "gpu-macos" }
    "cuda" { $GpuFeatures = "gpu-cuda" }
    "wgpu" {
        $GpuFeatures = "gpu-wgpu"
        Log-Warn "gpu-wgpu accelerates Burn only; Candle stays on CPU in the GPU table"
    }
    default { Log-ErrorAndExit "unknown --gpu: $Gpu (auto|macos|cuda|wgpu|off)" }
}

function Join-Feats {
    param([string[]]$Features)
    $Active = $Features | Where-Object { $_ -ne "" }
    return $Active -join ","
}

# Setup results folder
$Machine = (& uv run python scripts/machine-key.py).Trim()
$env:BENCH_RESULTS_DIR = "results/$Machine"
$BenchResultsDir = "results/$Machine"

if (-not (Test-Path "$BenchResultsDir/plots")) {
    New-Item -ItemType Directory -Force -Path "$BenchResultsDir/plots" | Out-Null
}
Log-Info "results folder: $BenchResultsDir"

# Migrate any legacy flat results to new folder layout
function Migrate-Legacy {
    $Moved = 0
    if (Test-Path "results/*.json") {
        foreach ($f in Get-ChildItem "results/*.json") {
            try {
                $Key = (& uv run python scripts/machine-key.py $f.FullName 2>$null).Trim()
                if ($Key) {
                    $TargetDir = "results/$Key"
                    if (-not (Test-Path $TargetDir)) {
                        New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
                    }
                    Move-Item -Path $f.FullName -Destination $TargetDir -Force
                    $Moved++
                }
            } catch {}
        }
    }
    
    if (Test-Path "results/plots/*.svg") {
        if (-not (Test-Path "$BenchResultsDir/plots")) {
            New-Item -ItemType Directory -Force -Path "$BenchResultsDir/plots" | Out-Null
        }
        Move-Item -Path "results/plots/*.svg" -Destination "$BenchResultsDir/plots/" -Force -ErrorAction SilentlyContinue
        
        if (Test-Path "results/plots/gpu/*.svg") {
            if (-not (Test-Path "$BenchResultsDir/plots/gpu")) {
                New-Item -ItemType Directory -Force -Path "$BenchResultsDir/plots/gpu" | Out-Null
            }
            Move-Item -Path "results/plots/gpu/*.svg" -Destination "$BenchResultsDir/plots/gpu/" -Force -ErrorAction SilentlyContinue
            Remove-Item -Path "results/plots/gpu" -Force -ErrorAction SilentlyContinue
        }
        Remove-Item -Path "results/plots" -Force -ErrorAction SilentlyContinue
    }
    
    if ($Moved -gt 0) {
        Log-Info "migrated $Moved legacy result file(s) into per-machine folders"
    }
}
Migrate-Legacy

# Configure thread pinning and trials
$env:RAYON_NUM_THREADS = "1"
if ($Trials) {
    $env:BENCH_TRIALS = $Trials
}
$TrialsMsg = "pinned RAYON_NUM_THREADS=1"
if ($Trials) {
    $TrialsMsg += ", trials=$Trials"
}
Log-Info $TrialsMsg

# AC power safety check
if ($OS -eq "Darwin") {
    if (Get-Command pmset -ErrorAction SilentlyContinue) {
        $Batt = & pmset -g batt 2>$null
        if ($Batt -notmatch "AC Power") {
            Log-Warn "on battery - plug in for stable numbers"
        }
    }
} elseif ($OS -eq "Windows") {
    try {
        $Power = Get-CimInstance -ClassName Win32_Battery -ErrorAction SilentlyContinue
        if ($Power -and ($Power.BatteryStatus -eq 1)) {
            Log-Warn "on battery - plug in for stable numbers"
        }
    } catch {}
}

# 1. Fetch model
Log-Info "fetching model (one-time)"
& "$PSScriptRoot/scripts/fetch-model.ps1"

# 2. Upfront Compilation Phase
Log-Info "Compilation Phase: Building all targets"

# A. Build parity check (debug)
Log-Info "Compiling parity check (debug)"
$ParityFeats = Join-Feats -Features $BlasFeatures
$BuildArgs = @("build", "-p", "pure-rust-framework")
if ($ParityFeats) {
    $BuildArgs += "--features"
    $BuildArgs += $ParityFeats
}
& cargo @BuildArgs
if ($LASTEXITCODE -ne 0) {
    Log-ErrorAndExit "parity check build failed"
}

# B. Build CPU benchmark (release)
Log-Info "Compiling CPU benchmark (release)"
$CpuFeats = Join-Feats -Features $BlasFeatures
$BuildCpuArgs = @("build", "--release", "-p", "pure-rust-framework", "--bin", "bench")
if ($CpuFeats) {
    $BuildCpuArgs += "--features"
    $BuildCpuArgs += $CpuFeats
}
& cargo @BuildCpuArgs
if ($LASTEXITCODE -ne 0) {
    Log-ErrorAndExit "CPU benchmark build failed"
}
Copy-Item "./target/release/bench$Exe" "./target/release/bench-cpu$Exe" -Force

# C. Build GPU benchmark (release)
$GpuBuildSucceeded = $false
if ($Gpu -ne "off") {
    Log-Info "Compiling GPU benchmark (release, $Gpu)"
    $GpuFeats = Join-Feats -Features $GpuFeatures, $BlasFeatures
    $BuildGpuArgs = @("build", "--release", "-p", "pure-rust-framework", "--bin", "bench")
    if ($GpuFeats) {
        $BuildGpuArgs += "--features"
        $BuildGpuArgs += $GpuFeats
    }
    & cargo @BuildGpuArgs
    if ($LASTEXITCODE -ne 0) {
        Log-Warn "GPU build failed - continuing with CPU results only"
    } else {
        $GpuBuildSucceeded = $true
        Copy-Item "./target/release/bench$Exe" "./target/release/bench-gpu$Exe" -Force
    }
}

# 3. Parity Check (Phase 0)
Log-Info "parity check (Phase 0)"
& "./target/debug/pure-rust-framework$Exe"
if ($LASTEXITCODE -ne 0) {
    Log-ErrorAndExit "parity check failed - engines disagree; not benchmarking"
}

# 4. CPU Benchmark (Phase 1)
Log-Info "CPU benchmark (Phase 1)"
& "./target/release/bench-cpu$Exe" run cpu
if ($LASTEXITCODE -ne 0) {
    Log-ErrorAndExit "CPU benchmark phase failed"
}

# 5. GPU Benchmark (Phase 2)
if ($Gpu -ne "off") {
    if ($GpuBuildSucceeded) {
        Log-Info "GPU benchmark (Phase 2, $Gpu)"
        & "./target/release/bench-gpu$Exe" run gpu
        if ($LASTEXITCODE -ne 0) {
            Log-Warn "GPU phase failed (backend unavailable at runtime?) - continuing with CPU results"
        }
    } else {
        Log-Warn "GPU benchmark skipped because build failed"
    }
} else {
    Log-Info "GPU benchmark: skipped"
}

# 6. Secondary metrics (Footprint)
if (-not $NoFootprint) {
    Log-Info "footprint: cold start / RSS / binary / build"
    $env:BENCH_FEATURES = Join-Feats -Features $BlasFeatures
    & "$PSScriptRoot/scripts/secondary.ps1"
} else {
    Log-Info "footprint: skipped"
}

# 7. Generate plots
function Get-LatestResult {
    param([string]$Prefix)
    $Files = Get-ChildItem "$BenchResultsDir/$Prefix-*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($Files) {
        return $Files[0].FullName
    }
    return $null
}

if (-not $NoPlots) {
    Log-Info "plots"
    $CpuJson = Get-LatestResult "cpu"
    if ($CpuJson) {
        & uv run python scripts/plot.py $CpuJson "$BenchResultsDir/plots"
    } else {
        Log-Warn "no CPU results to plot"
    }
    
    if ($Gpu -ne "off") {
        $GpuJson = Get-LatestResult "gpu"
        if ($GpuJson) {
            & uv run python scripts/plot.py $GpuJson "$BenchResultsDir/plots/gpu"
        } else {
            Log-Warn "no GPU results to plot"
        }
    }

    Log-Info "converting SVGs to PNGs"
    & uv run --with pymupdf python scripts/convert_svgs.py
} else {
    Log-Info "plots: skipped"
}

# Summary and finish
Log-Info "done - artifacts in $BenchResultsDir/"
$LatestFiles = Get-ChildItem "$BenchResultsDir/*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 5
foreach ($File in $LatestFiles) {
    Write-Host "  $BenchResultsDir/$($File.Name)"
}
Write-Host "  plots: $BenchResultsDir/plots/latency.{svg,png}, throughput.{svg,png}, slowdown.{svg,png}"

Pop-Location
