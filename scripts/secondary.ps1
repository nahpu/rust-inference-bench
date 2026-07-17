# Secondary metrics: cold start, peak RSS, binary size, clean build time.
# Cross-platform: Windows, macOS, and Linux support.

$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Push-Location $Root

$Samples = if ($env:SAMPLES) { [int]$env:SAMPLES } else { 7 }

$IsWin = [Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([Runtime.InteropServices.OSPlatform]::Windows)
$Exe = if ($IsWin) { ".exe" } else { "" }

$FeatArgs = @()
if ($env:BENCH_FEATURES) {
    $FeatArgs += "--features"
    $FeatArgs += $env:BENCH_FEATURES
}

$Engines = @("candle-cpu:single_candle", "burn-ndarray:single_burn")

Write-Host "== building single-engine binaries (release) =="
& cargo build --release @FeatArgs --bin single_candle --bin single_burn *>$null

$TempDir = if ($env:TEMP) { $env:TEMP } else { [System.IO.Path]::GetTempPath() }
$env:TEMP_DIR = $TempDir

Write-Host "== cold start + peak RSS ($Samples runs each) =="
foreach ($Spec in $Engines) {
    $parts = $Spec -split ":"
    $name = $parts[0]
    $binaryName = $parts[1]
    $BinPath = Join-Path $Root "target/release/$binaryName$Exe"

    $OutPath = Join-Path $TempDir "cvb_$binaryName.txt"
    if (Test-Path $OutPath) { Remove-Item $OutPath -Force }

    for ($i = 0; $i -lt $Samples; $i++) {
        $TempOut = [System.IO.Path]::GetTempFileName()
        $Process = Start-Process -FilePath $BinPath -NoNewWindow -PassThru -RedirectStandardOutput $TempOut -Wait

        $ColdMs = (Get-Content $TempOut -Raw)
        if ($ColdMs) { $ColdMs = $ColdMs.Trim() } else { $ColdMs = "0" }
        Remove-Item $TempOut -ErrorAction SilentlyContinue

        $PeakRSS = "na"
        try {
            $PeakRSS = $Process.PeakWorkingSet64
            if (-not $PeakRSS) { $PeakRSS = "na" }
        } catch {
            $PeakRSS = "na"
        }

        Add-Content -Path $OutPath -Value "$ColdMs $PeakRSS"
    }
}

Write-Host "== binary size (stripped if strip is available) =="
$HaveStrip = $false
if (Get-Command strip -ErrorAction SilentlyContinue) {
    $HaveStrip = $true
}

$BinJson = ""
foreach ($Spec in $Engines) {
    $parts = $Spec -split ":"
    $binaryName = $parts[1]
    $Target = Join-Path $Root "target/release/$binaryName$Exe"

    if ($HaveStrip) {
        $StripTarget = Join-Path $TempDir "cvb_strip_$binaryName$Exe"
        & strip -o $StripTarget $Target 2>$null
        if ($LASTEXITCODE -eq 0) {
            $Target = $StripTarget
        }
    }

    $Size = (Get-Item $Target).Length
    $BinJson += "$binaryName $Size`n"
}

Write-Host "== clean build time (full clean before each; shared deps counted in each) =="
$BuildJson = ""
foreach ($Spec in $Engines) {
    $parts = $Spec -split ":"
    $binaryName = $parts[1]

    & cargo clean *>$null
    $t0 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    & cargo build --release @FeatArgs --bin $binaryName *>$null
    $t1 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()

    $DiffSeconds = "{0:F1}" -f (($t1 - $t0) / 1000.0)
    $BuildJson += "$binaryName $DiffSeconds`n"
}

Write-Host "== aggregate -> JSON + summary =="
$SpecsJson = ($Engines -join "`n") + "`n"

# Write Python helper script to temp file
$PyScript = @"
import sys, json, statistics as st, subprocess, os, platform

specs = [l for l in sys.argv[1].splitlines() if l]
bin_size = dict(l.split() for l in sys.argv[2].splitlines() if l)
build_s  = dict(l.split() for l in sys.argv[3].splitlines() if l)

def load(path):
    cold, rss = [], []
    for line in open(path):
        c, r = line.split()
        cold.append(float(c))
        if r != "na":
            rss.append(float(r))
    return cold, rss

def cmd(*a):
    try: return subprocess.run(a, capture_output=True, text=True).stdout.strip()
    except Exception: return ""

def get_windows_cpu():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        brand = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        return brand.strip()
    except Exception:
        return ""

def cpu_brand():
    s = cmd("sysctl", "-n", "machdep.cpu.brand_string")
    if s: return s
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    if platform.system() == "Windows":
        win_brand = get_windows_cpu()
        if win_brand:
            return win_brand
    return platform.processor() or platform.machine()

def is_on_ac():
    if platform.system() == "Windows":
        try:
            import ctypes
            class SYSTEM_POWER_STATUS(ctypes.Structure):
                _fields_ = [
                    ('ACLineStatus', ctypes.c_byte),
                    ('BatteryFlag', ctypes.c_byte),
                    ('BatteryLifePercent', ctypes.c_byte),
                    ('Reserved1', ctypes.c_byte),
                    ('BatteryLifeTime', ctypes.c_ulong),
                    ('BatteryFullLifeTime', ctypes.c_ulong)
                ]
            status = SYSTEM_POWER_STATUS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
                return status.ACLineStatus == 1
        except Exception:
            pass
        return False
    return "AC Power" in cmd("pmset", "-g", "batt")

mb = lambda b: round(int(b)/1_000_000, 1)

cold_start, peak_rss, binary, clean = {}, {}, {}, {}
temp_dir = os.environ.get("TEMP_DIR", "/tmp")

for spec in specs:
    name, binary_name = spec.split(":")
    cc, cr = load(os.path.join(temp_dir, f"cvb_{binary_name}.txt"))
    cold_start[name] = {"median": round(st.median(cc),1), "min": round(min(cc),1)}
    peak_rss[name]   = {"median": round(st.median(cr)/1_000_000,1)} if cr else {"median": None}
    binary[name]     = mb(bin_size[binary_name])
    clean[name]      = float(build_s[binary_name])

report = {
  "environment": {
    "cpu": cpu_brand(),
    "os": f"{platform.system()} {platform.machine()}",
    "rustc": cmd("rustc","--version"),
    "rayon_threads": os.environ.get("RAYON_NUM_THREADS","unset"),
    "on_ac_power": is_on_ac(),
  },
  "cold_start_ms": cold_start,
  "peak_rss_mb": peak_rss,
  "binary_size_mb": binary,
  "clean_build_s": clean,
}

import datetime
ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
host = cmd("hostname") or platform.node() or "host"
outdir = os.environ.get("BENCH_RESULTS_DIR", "results")
os.makedirs(outdir, exist_ok=True)
path = f"{outdir}/secondary-{ts}-{host}.json"
json.dump(report, open(path,"w"), indent=2)

names = [s.split(":")[0] for s in specs]
def rss_cell(n):
    v = peak_rss[n]["median"]
    return "n/a" if v is None else v
hdr = f"{'metric':<16}" + "".join(f"{n:>14}" for n in names)
print("\n" + hdr)
print(f"{'cold start (ms)':<16}" + "".join(f"{cold_start[n]['median']:>14}" for n in names))
print(f"{'peak RSS (MB)':<16}"   + "".join(f"{rss_cell(n):>14}" for n in names))
print(f"{'binary (MB)':<16}"     + "".join(f"{binary[n]:>14}" for n in names))
print(f"{'clean build (s)':<16}" + "".join(f"{clean[n]:>14}" for n in names))
print(f"\nwrote {path}")
"@

$PyFile = Join-Path $TempDir "cvb_aggregate.py"
Set-Content -Path $PyFile -Value $PyScript -Encoding UTF8

# Invoke using uv
& uv run python $PyFile $SpecsJson $BinJson $BuildJson

# Clean up
Remove-Item $PyFile -ErrorAction SilentlyContinue
foreach ($Spec in $Engines) {
    $parts = $Spec -split ":"
    $binaryName = $parts[1]
    Remove-Item (Join-Path $TempDir "cvb_$binaryName.txt") -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $TempDir "cvb_strip_$binaryName$Exe") -ErrorAction SilentlyContinue
}

Pop-Location
