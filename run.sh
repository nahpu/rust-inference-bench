#!/usr/bin/env bash
# One-command reproduction of the rust-inference-bench (worklog 12).
#
# Runs the full pipeline the way we did — parity -> CPU -> GPU -> footprint ->
# plots — auto-detecting the host OS, CPU BLAS backend, and GPU backend so a
# colleague on macOS / Linux / Windows(Git-Bash) can regenerate the same JSON +
# SVG artifacts in results/. Every phase pins threads (RAYON_NUM_THREADS=1) and
# builds --release for a fair, reproducible comparison.
#
# Usage:
#   ./run.sh [options]
#
# Options:
#   --gpu <auto|macos|cuda|wgpu|off>  GPU backend (default: auto-detect)
#   --cpu-only                        skip the GPU phase (alias for --gpu off)
#   --blas <auto|accelerate|mkl|openblas>
#                                     CPU BLAS backend (default: auto per-OS;
#                                     macOS=accelerate, others=none/pure kernels)
#   --trials <N>                      benchmark trials per phase (default: 10)
#   --no-footprint                    skip cold-start / RSS / binary / build phase
#   --no-plots                        skip SVG plot generation
#   -h, --help                        show this help
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ---- defaults -------------------------------------------------------------
GPU="auto"
BLAS="auto"
TRIALS=""
DO_FOOTPRINT=1
DO_PLOTS=1

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)         GPU="$2"; shift 2 ;;
        --cpu-only)    GPU="off"; shift ;;
        --blas)        BLAS="$2"; shift 2 ;;
        --trials)      TRIALS="$2"; shift 2 ;;
        --no-footprint) DO_FOOTPRINT=0; shift ;;
        --no-plots)    DO_PLOTS=0; shift ;;
        -h|--help)     usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

OS="$(uname -s)"
ARCH="$(uname -m)"

# ---- preflight ------------------------------------------------------------
log "preflight ($OS/$ARCH)"
command -v cargo >/dev/null 2>&1 || die "cargo not found — install Rust: https://rustup.rs"
# Windows installs often expose `python`, not `python3` — accept either.
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || die "python3/python not found — needed for plots + footprint aggregation"
export PYTHON="$PY"
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    die "need curl or wget to fetch the model"
fi

# ---- resolve BLAS backend -------------------------------------------------
# macOS auto -> Accelerate (compiled in automatically by the crates' target
# deps; no feature needed). Non-macOS auto -> pure kernels (no system BLAS, so
# it always builds). Overrides map to runner features.
BLAS_FEATURES=""
if [ "$BLAS" = "auto" ]; then
    if [ "$OS" = "Darwin" ]; then BLAS="accelerate"; else BLAS="none"; fi
fi
case "$BLAS" in
    accelerate)
        [ "$OS" = "Darwin" ] || warn "accelerate is macOS-only; on $OS it has no effect"
        ;;
    none) ;;
    mkl)      BLAS_FEATURES="blas-mkl" ;;
    openblas) BLAS_FEATURES="blas-openblas" ;;
    *) die "unknown --blas: $BLAS (auto|accelerate|mkl|openblas)" ;;
esac
log "BLAS backend: $BLAS${BLAS_FEATURES:+  (features: $BLAS_FEATURES)}"

# ---- resolve GPU backend --------------------------------------------------
detect_gpu() {
    case "$OS" in
        Darwin) echo "macos"; return ;;
    esac
    if command -v nvidia-smi >/dev/null 2>&1; then echo "cuda"; return; fi
    if command -v vulkaninfo >/dev/null 2>&1; then echo "wgpu"; return; fi
    echo "off"
}
if [ "$GPU" = "auto" ]; then
    GPU="$(detect_gpu)"
    log "GPU backend: $GPU (auto-detected)"
else
    log "GPU backend: $GPU"
fi

GPU_FEATURES=""
case "$GPU" in
    off) ;;
    macos) GPU_FEATURES="gpu-macos" ;;
    cuda)  GPU_FEATURES="gpu-cuda" ;;
    wgpu)  GPU_FEATURES="gpu-wgpu"
           warn "gpu-wgpu accelerates Burn only; Candle + ORT stay on CPU in the GPU table" ;;
    *) die "unknown --gpu: $GPU (auto|macos|cuda|wgpu|off)" ;;
esac

# join two space-separated feature lists into one comma-separated --features arg
join_feats() {
    local out=""
    for f in "$@"; do [ -n "$f" ] && out="${out:+$out,}$f"; done
    echo "$out"
}

# ---- per-machine results folder -------------------------------------------
# Every result lands in results/<os>-<arch>-<cpu> so runs from different
# machines (and people) can be committed side-by-side without colliding.
MACHINE="$("$PY" scripts/machine-key.py)"
export BENCH_RESULTS_DIR="results/$MACHINE"
mkdir -p "$BENCH_RESULTS_DIR/plots"
log "results folder: $BENCH_RESULTS_DIR"

# One-time migration: move any legacy flat results/*.json into the per-machine
# folder they belong to (routed by each file's own environment block), and the
# old flat plots/ under the current machine. Idempotent — a no-op once migrated.
migrate_legacy() {
    local moved=0 f key
    for f in results/*.json; do
        [ -e "$f" ] || continue
        key="$("$PY" scripts/machine-key.py "$f" 2>/dev/null)" || continue
        [ -n "$key" ] || continue
        mkdir -p "results/$key"
        mv "$f" "results/$key/" && moved=$((moved + 1))
    done
    if ls results/plots/*.svg >/dev/null 2>&1; then
        mkdir -p "$BENCH_RESULTS_DIR/plots"
        mv results/plots/*.svg "$BENCH_RESULTS_DIR/plots/" 2>/dev/null || true
        if ls results/plots/gpu/*.svg >/dev/null 2>&1; then
            mkdir -p "$BENCH_RESULTS_DIR/plots/gpu"
            mv results/plots/gpu/*.svg "$BENCH_RESULTS_DIR/plots/gpu/" 2>/dev/null || true
            rmdir results/plots/gpu 2>/dev/null || true
        fi
        rmdir results/plots 2>/dev/null || true
    fi
    [ "$moved" -gt 0 ] && log "migrated $moved legacy result file(s) into per-machine folders"
}
migrate_legacy

# ---- pinning + trial count ------------------------------------------------
export RAYON_NUM_THREADS=1
[ -n "$TRIALS" ] && export BENCH_TRIALS="$TRIALS"
log "pinned RAYON_NUM_THREADS=1${TRIALS:+, trials=$TRIALS}"
# AC-power reminder (best-effort, macOS only).
if [ "$OS" = "Darwin" ] && command -v pmset >/dev/null 2>&1; then
    pmset -g batt 2>/dev/null | grep -q "AC Power" || warn "on battery — plug in for stable numbers"
fi

# ---- 1. fetch model -------------------------------------------------------
log "fetching model (one-time)"
bash scripts/fetch-model.sh

# ---- 2. parity (Phase 0) --------------------------------------------------
# Gates the whole run: if the engines disagree, the speed numbers are meaningless.
log "parity check (Phase 0)"
PARITY_FEATS="$(join_feats "$BLAS_FEATURES")"
cargo run -p runner --bin runner ${PARITY_FEATS:+--features "$PARITY_FEATS"} \
    || die "parity check failed — engines disagree; not benchmarking"

# ---- 3. CPU benchmark (Phase 1) -------------------------------------------
log "CPU benchmark (Phase 1)"
CPU_FEATS="$(join_feats "$BLAS_FEATURES")"
cargo run --release -p runner --bin bench ${CPU_FEATS:+--features "$CPU_FEATS"} -- cpu

# ---- 4. GPU benchmark (Phase 2) -------------------------------------------
if [ "$GPU" != "off" ]; then
    log "GPU benchmark (Phase 2, $GPU)"
    GPU_FEATS="$(join_feats "$GPU_FEATURES" "$BLAS_FEATURES")"
    if ! cargo run --release -p runner --bin bench --features "$GPU_FEATS" -- gpu; then
        warn "GPU phase failed (backend unavailable at runtime?) — continuing with CPU results"
    fi
else
    log "GPU benchmark: skipped"
fi

# ---- 5. footprint (secondary metrics) -------------------------------------
if [ "$DO_FOOTPRINT" -eq 1 ]; then
    log "footprint: cold start / RSS / binary / build"
    BENCH_FEATURES="$(join_feats "$BLAS_FEATURES")" bash scripts/secondary.sh
else
    log "footprint: skipped"
fi

# ---- 6. plots -------------------------------------------------------------
if [ "$DO_PLOTS" -eq 1 ]; then
    log "plots"
    latest() { ls -t "$BENCH_RESULTS_DIR"/"$1"-*.json 2>/dev/null | head -1; }
    cpu_json="$(latest cpu)"
    [ -n "$cpu_json" ] && "$PY" scripts/plot.py "$cpu_json" "$BENCH_RESULTS_DIR/plots" || warn "no CPU results to plot"
    if [ "$GPU" != "off" ]; then
        gpu_json="$(latest gpu)"
        [ -n "$gpu_json" ] && "$PY" scripts/plot.py "$gpu_json" "$BENCH_RESULTS_DIR/plots/gpu" || warn "no GPU results to plot"
    fi
    log "converting SVGs to PNGs"
    if command -v uv >/dev/null 2>&1; then
        uv run --with pymupdf python scripts/convert_svgs.py
    else
        "$PY" scripts/convert_svgs.py || true
    fi
else
    log "plots: skipped"
fi

# ---- summary --------------------------------------------------------------
log "done — artifacts in $BENCH_RESULTS_DIR/"
ls -t "$BENCH_RESULTS_DIR"/*.json 2>/dev/null | head -5 | sed 's/^/  /'
echo "  plots: $BENCH_RESULTS_DIR/plots/{latency,throughput,slowdown}.{svg,png}"
