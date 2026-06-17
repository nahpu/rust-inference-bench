#!/usr/bin/env bash
# Secondary metrics: cold start, peak RSS, binary size, clean build time.
# Run pinned + on AC, from the repo root:
#   RAYON_NUM_THREADS=1 scripts/secondary.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SAMPLES=7
now() { python3 -c 'import time; print(time.time())'; }

# engine display-name : single-engine binary
ENGINES=("candle-cpu:single_candle" "burn-ndarray:single_burn" "ort-cpu:single_ort")

echo "== building single-engine binaries (release) =="
cargo build --release --bin single_candle --bin single_burn --bin single_ort >/dev/null 2>&1

measure_runtime() {  # $1=binary  -> appends "coldms rssbytes" lines to stdout
    local bin="$1"
    for _ in $(seq "$SAMPLES"); do
        local err cold rss
        err="$(mktemp)"
        cold="$(/usr/bin/time -l "./target/release/$bin" 2>"$err")"
        rss="$(grep 'maximum resident set size' "$err" | awk '{print $1}')"
        rm -f "$err"
        echo "$cold $rss"
    done
}

echo "== cold start + peak RSS ($SAMPLES runs each) =="
for spec in "${ENGINES[@]}"; do
    bin="${spec##*:}"
    measure_runtime "$bin" > "/tmp/cvb_${bin}.txt"
done

echo "== binary size (stripped) =="
BIN_JSON=""
for spec in "${ENGINES[@]}"; do
    bin="${spec##*:}"
    strip -o "/tmp/cvb_strip_${bin}" "./target/release/${bin}"
    BIN_JSON="${BIN_JSON}${bin} $(stat -f%z "/tmp/cvb_strip_${bin}")
"
done

echo "== clean build time (full clean before each; shared deps counted in each) =="
BUILD_JSON=""
for spec in "${ENGINES[@]}"; do
    bin="${spec##*:}"
    cargo clean >/dev/null 2>&1
    t0=$(now); cargo build --release --bin "$bin" >/dev/null 2>&1; t1=$(now)
    BUILD_JSON="${BUILD_JSON}${bin} $(python3 -c "print(f'{$t1-$t0:.1f}')")
"
done

echo "== aggregate -> JSON + summary =="
SPECS_JSON="$(printf '%s\n' "${ENGINES[@]}")"

python3 - "$SPECS_JSON" "$BIN_JSON" "$BUILD_JSON" <<'PY'
import sys, json, statistics as st, subprocess, os

specs = [l for l in sys.argv[1].splitlines() if l]
bin_size = dict(l.split() for l in sys.argv[2].splitlines() if l)
build_s  = dict(l.split() for l in sys.argv[3].splitlines() if l)

def load(path):
    cold, rss = [], []
    for line in open(path):
        c, r = line.split()
        cold.append(float(c)); rss.append(float(r))
    return cold, rss

def cmd(*a):
    try: return subprocess.run(a, capture_output=True, text=True).stdout.strip()
    except Exception: return ""

mb = lambda b: round(int(b)/1_000_000, 1)

cold_start, peak_rss, binary, clean = {}, {}, {}, {}
for spec in specs:
    name, binary_name = spec.split(":")
    cc, cr = load(f"/tmp/cvb_{binary_name}.txt")
    cold_start[name] = {"median": round(st.median(cc),1), "min": round(min(cc),1)}
    peak_rss[name]   = {"median": round(st.median(cr)/1_000_000,1)}
    binary[name]     = mb(bin_size[binary_name])
    clean[name]      = float(build_s[binary_name])

report = {
  "environment": {
    "cpu": cmd("sysctl","-n","machdep.cpu.brand_string"),
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
host = cmd("hostname","-s") or "host"
os.makedirs("results", exist_ok=True)
path = f"results/secondary-{ts}-{host}.json"
json.dump(report, open(path,"w"), indent=2)

names = [s.split(":")[0] for s in specs]
hdr = f"{'metric':<16}" + "".join(f"{n:>14}" for n in names)
print("\n" + hdr)
print(f"{'cold start (ms)':<16}" + "".join(f"{cold_start[n]['median']:>14}" for n in names))
print(f"{'peak RSS (MB)':<16}"   + "".join(f"{peak_rss[n]['median']:>14}" for n in names))
print(f"{'binary (MB)':<16}"     + "".join(f"{binary[n]:>14}" for n in names))
print(f"{'clean build (s)':<16}" + "".join(f"{clean[n]:>14}" for n in names))
print(f"\nwrote {path}")
PY