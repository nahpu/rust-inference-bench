#!/usr/bin/env bash
# Secondary metrics: cold start, peak RSS, binary size, clean build time.
# Cross-platform (worklog 12): macOS + Linux measure everything; Windows/Git-Bash
# measures cold start / binary / build and reports RSS as "n/a" (no portable
# peak-RSS probe without /usr/bin/time). Run pinned + on AC, from the repo root:
#   RAYON_NUM_THREADS=1 scripts/secondary.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SAMPLES="${SAMPLES:-7}"
OS="$(uname -s)"

# Python interpreter (Windows installs often expose `python`, not `python3`).
# run.sh passes a resolved PYTHON; fall back to detection when run standalone.
PY="${PYTHON:-}"
[ -z "$PY" ] && PY="$(command -v python3 || command -v python || true)"
[ -z "$PY" ] && { echo "error: need python3/python for aggregation" >&2; exit 1; }

# Executable suffix (Windows/Git-Bash builds are foo.exe).
EXE=""
case "$OS" in MINGW*|MSYS*|CYGWIN*) EXE=".exe" ;; esac

now() { "$PY" -c 'import time; print(time.time())'; }

# Optional cargo features (run.sh passes BLAS overrides here so the footprint
# build matches the benchmarked build). Empty on macOS / default builds.
FEAT_ARGS=()
if [ -n "${BENCH_FEATURES:-}" ]; then
    FEAT_ARGS=(--features "$BENCH_FEATURES")
fi

# engine display-name : single-engine binary
ENGINES=("candle-cpu:single_candle" "burn-ndarray:single_burn" "ort-cpu:single_ort")

# --- pick a peak-RSS probe for this OS -------------------------------------
# macOS BSD `time -l` reports RSS in bytes; GNU `time -v` reports it in kbytes.
# Anything else (Windows/Git-Bash, no /usr/bin/time) -> RSS unavailable.
TIME_BIN=""
TIME_FLAG=""
RSS_UNIT=""
case "$OS" in
    Darwin) TIME_BIN="/usr/bin/time"; TIME_FLAG="-l"; RSS_UNIT="bytes" ;;
    Linux)  if [ -x /usr/bin/time ]; then TIME_BIN="/usr/bin/time"; TIME_FLAG="-v"; RSS_UNIT="kbytes"; fi ;;
esac
if [ -z "$TIME_BIN" ]; then
    echo "note: no peak-RSS probe on this platform ($OS) — RSS will be reported as n/a" >&2
fi

parse_rss() {  # $1=stderr-file -> RSS in bytes, or empty if unavailable
    local err="$1"
    case "$RSS_UNIT" in
        bytes)  grep 'maximum resident set size' "$err" | awk '{print $1}' ;;
        kbytes) awk '/Maximum resident set size/ {print $NF * 1024}' "$err" ;;
        *)      echo "" ;;
    esac
}

echo "== building single-engine binaries (release) =="
cargo build --release ${FEAT_ARGS[@]+"${FEAT_ARGS[@]}"} --bin single_candle --bin single_burn --bin single_ort >/dev/null 2>&1

measure_runtime() {  # $1=binary -> appends "coldms rssbytes" lines to stdout
    local bin="$1"
    for _ in $(seq "$SAMPLES"); do
        local err cold rss
        err="$(mktemp)"
        if [ -n "$TIME_BIN" ]; then
            cold="$("$TIME_BIN" "$TIME_FLAG" "./target/release/${bin}${EXE}" 2>"$err")"
            rss="$(parse_rss "$err")"
        else
            cold="$("./target/release/${bin}${EXE}")"
            rss=""
        fi
        rm -f "$err"
        echo "$cold ${rss:-na}"
    done
}

echo "== cold start + peak RSS ($SAMPLES runs each) =="
for spec in "${ENGINES[@]}"; do
    bin="${spec##*:}"
    measure_runtime "$bin" > "/tmp/cvb_${bin}.txt"
done

echo "== binary size (stripped if strip is available) =="
HAVE_STRIP=0; command -v strip >/dev/null 2>&1 && HAVE_STRIP=1
BIN_JSON=""
for spec in "${ENGINES[@]}"; do
    bin="${spec##*:}"
    target="./target/release/${bin}${EXE}"
    if [ "$HAVE_STRIP" -eq 1 ]; then
        strip -o "/tmp/cvb_strip_${bin}${EXE}" "$target" 2>/dev/null && target="/tmp/cvb_strip_${bin}${EXE}"
    fi
    # wc -c is portable (stat's size flag differs across BSD/GNU).
    BIN_JSON="${BIN_JSON}${bin} $(wc -c < "$target" | tr -d ' ')
"
done

echo "== clean build time (full clean before each; shared deps counted in each) =="
BUILD_JSON=""
for spec in "${ENGINES[@]}"; do
    bin="${spec##*:}"
    cargo clean >/dev/null 2>&1
    t0=$(now); cargo build --release ${FEAT_ARGS[@]+"${FEAT_ARGS[@]}"} --bin "$bin" >/dev/null 2>&1; t1=$(now)
    BUILD_JSON="${BUILD_JSON}${bin} $(python3 -c "print(f'{$t1-$t0:.1f}')")
"
done

echo "== aggregate -> JSON + summary =="
SPECS_JSON="$(printf '%s\n' "${ENGINES[@]}")"

"$PY" - "$SPECS_JSON" "$BIN_JSON" "$BUILD_JSON" <<'PY'
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

def cpu_brand():
    s = cmd("sysctl", "-n", "machdep.cpu.brand_string")
    if s: return s
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()

mb = lambda b: round(int(b)/1_000_000, 1)

cold_start, peak_rss, binary, clean = {}, {}, {}, {}
for spec in specs:
    name, binary_name = spec.split(":")
    cc, cr = load(f"/tmp/cvb_{binary_name}.txt")
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
    "on_ac_power": "AC Power" in cmd("pmset","-g","batt"),
  },
  "cold_start_ms": cold_start,
  "peak_rss_mb": peak_rss,
  "binary_size_mb": binary,
  "clean_build_s": clean,
}
ts = cmd("date","-u","+%Y%m%dT%H%M%SZ") or "run"
host = cmd("hostname","-s") or cmd("hostname") or "host"
# run.sh sets BENCH_RESULTS_DIR to a per-machine folder; standalone -> flat results/.
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
PY
