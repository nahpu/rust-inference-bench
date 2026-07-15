# rust-inference-bench

A reproducible benchmark comparing Rust ML inference engines — **Candle**,
**Burn**, and **ONNX Runtime** (`ort`) — on a sentence-embedding workload, to
inform the NAHPU framework choice.

## Scope (intentional constraints)

- **Desktop only.** No mobile / on-device targets. The Flutter (FFI) integration
  layer adds a *framework-agnostic constant*, so it does not change the relative
  ranking — a standalone Rust benchmark is sufficient for the decision. (Mobile
  would change the picture via per-platform backend availability; out of scope.)
- **Inference only.** No training / fine-tuning.
- **One model:** `all-MiniLM-L6-v2` (22M params, 384-dim sentence embeddings).
- **CPU + GPU.** CPU baseline (Candle CPU vs Burn `ndarray` vs ORT CPU), then GPU
  (Candle Metal vs Burn `wgpu` vs ORT CoreML).

## Why a benchmark at all

No public source does a rigorous, same-model/same-hardware head-to-head of these
engines — they only report each against PyTorch separately. This repo produces
that comparison for NAHPU's actual workload. ORT was added as a genuinely
independent (C++ runtime) data point alongside the two Rust-native frameworks.

## Architecture

Every engine lives behind the `InferenceEngine` trait in `embed-core`. Nothing
outside the per-engine crates touches a framework type — only `&str` in,
`Vec<f32>` out cross the boundary. This keeps the comparison fair and makes
swapping engines one new impl rather than a rewrite.

```
crates/
  embed-core/     trait + types + cosine util + corpus loader (no framework deps)
  embed-candle/   CandleEngine    (safetensors; CPU + Metal)
  embed-burn/     BurnEngine      (ONNX via burn-import; ndarray + wgpu, NOT burn-candle)
  embed-ort/      OrtEngine       (ONNX Runtime; CPU + CoreML)
runner/
  src/main.rs            Phase 0 parity check (pairwise cosine across engines)
  src/bin/bench.rs       N-engine interleaved latency + throughput harness
  src/bin/single_*.rs    one-engine binaries for cold start / RSS / binary size
data/
  corpus.sample.txt  synthetic specimen-like text (no real NAHPU data)
  models/            (gitignored) model weights fetched locally
```

All three load the **same** all-MiniLM weights (Candle from safetensors; Burn and
ORT from the identical ONNX export), so embeddings are numerically identical.

## Phases

0. **Parity** — all engines produce equivalent embeddings (min cosine > 0.999).
1. **CPU** — Candle CPU vs Burn `ndarray` vs ORT CPU. The desktop baseline.
2. **GPU** — Candle Metal vs Burn `wgpu` vs ORT CoreML.
3. **Secondary metrics** — peak memory (RSS), binary size, cold start, build time.

## Metrics & fairness rules

- Report **distributions** (median + IQR), never a single number; interleave
  engines per trial so they share identical conditions.
- **Warm up** before measuring; measure cold start separately.
- Hold constant: model weights, tokenizer (`tokenizers` crate, excluded from
  timing), input corpus, precision (f32), thread count, release build flags.
- Pin threads when benching: `RAYON_NUM_THREADS=1` (Candle/Burn Rayon + ORT
  intra-op); Candle/Burn share Apple Accelerate BLAS; ORT uses its own kernels.
- **Do not** use Burn's `burn-candle` backend, or `mistral.rs` — both are built on
  Candle, which would make the comparison Candle-vs-Candle.

## Running

### One command (any machine)

```sh
./run.sh
```

`run.sh` reproduces the whole pipeline — parity → CPU → GPU → footprint → plots —
auto-detecting the OS, CPU BLAS backend, and GPU backend, and writing the same
`results/*.json` + `results/plots/*.svg` we produced. It pins threads
(`RAYON_NUM_THREADS=1`) and builds `--release` for every measured phase.

```sh
./run.sh --cpu-only            # skip GPU
./run.sh --gpu cuda            # force a backend (auto|macos|cuda|wgpu|off)
./run.sh --blas openblas       # opt-in CPU BLAS on Linux (auto|accelerate|mkl|openblas)
./run.sh --trials 20           # more trials = tighter IQR
./run.sh --no-footprint --no-plots
```

**Prerequisites:** Rust (`rustup`), `python3`, and `curl`/`wget`. ONNX Runtime is
pulled prebuilt by the `ort` crate — no system install. GPU backends need their
platform SDK/driver present at runtime (Metal on macOS, CUDA toolkit for
`--gpu cuda`, a Vulkan driver for `--gpu wgpu`); if the chosen backend isn't
available the GPU phase is skipped with a warning and CPU results still land.

### Cross-platform notes

- **CPU BLAS is auto-selected per OS:** macOS links Apple Accelerate; other
  platforms default to the frameworks' built-in kernels so `cargo build` works
  with no system BLAS. `--blas mkl` (Candle, Intel) / `--blas openblas` (Burn)
  opt into a system BLAS symmetrically.
- **GPU auto-detect:** macOS → Metal/CoreML; `nvidia-smi` present → CUDA;
  `vulkaninfo` present → wgpu; otherwise the GPU phase is skipped.
- **Windows:** run under WSL (treated as Linux) or Git-Bash. `run.sh` accepts
  `python` or `python3`, handles the `.exe` suffix, and needs no Unix `date`.
  Footprint measures cold start / binary / build; peak RSS is `n/a` (no portable
  probe without `/usr/bin/time`). A GPU-less box auto-selects `--gpu off`.
- For the ORT cross-platform execution-provider matrix (DirectML / OpenVINO /
  NNAPI on non-Mac hardware) see [GUIDE-cross-platform.md](GUIDE-cross-platform.md).

### Results layout

`run.sh` writes every artifact into a per-machine folder keyed by
`<os>-<arch>-<cpu>`, so runs from different machines (and people) can be committed
side-by-side without colliding:

```
results/
  macos-arm64-apple-m4/
    cpu-<stamp>-<host>.json
    gpu-<stamp>-<host>.json
    secondary-<stamp>-<host>.json
    plots/{latency,throughput,slowdown}.svg
    plots/gpu/…
  linux-x86_64-core-i7-13700k/
    …
```

The key is computed by `scripts/machine-key.py` (the single source of truth —
also used to route legacy files). `run.sh` migrates any old flat `results/*.json`
into the right per-machine folder on first run. A manual `cargo run` without
`BENCH_RESULTS_DIR` set falls back to the flat `results/` root.

### Manual (individual phases)

```sh
scripts/fetch-model.sh                     # one-time ONNX fetch for embed-burn build
cargo run -p runner --bin runner           # Phase 0 parity
RAYON_NUM_THREADS=1 cargo run --release -p runner --bin bench cpu
cargo run --release -p runner --bin bench --features gpu-macos -- gpu   # or gpu-cuda / gpu-wgpu
RAYON_NUM_THREADS=1 scripts/secondary.sh   # footprint
python3 scripts/plot.py results/cpu-*.json results/plots
```

## Status

- **Phase 0 (parity): done** — Candle / Burn / ORT agree to min cosine **1.000000**.
- **Phase 1 (CPU): done** — Candle fastest overall; **ORT fastest for a single
  short embed** (1.85 ms). See [REPORT.md](REPORT.md).
- **Phase 2 (GPU): done** — Candle Metal dominates (3.7–7.5×); ORT-CoreML beats
  Burn wgpu on throughput.
- **Secondary metrics: done** — Candle wins cold start, RSS, binary; ORT builds fastest.
- **Recommendation: Candle** for the NAHPU desktop embedding workload.
