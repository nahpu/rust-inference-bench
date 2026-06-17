# candle-vs-burn-bench

A reproducible benchmark comparing Rust ML inference engines — **Candle**,
**Burn**, and **ONNX Runtime** (`ort`) — on a sentence-embedding workload, to
inform the NAHPU framework choice. (Repo name predates the ORT addition.)

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

```sh
scripts/fetch-model.sh                     # one-time ONNX fetch for embed-burn build

# Phase 0 parity
cargo run -p runner --bin runner

# CPU benchmark + plots
RAYON_NUM_THREADS=1 cargo run --release -p runner --bin bench cpu
python3 scripts/plot.py results/cpu-*.json results/plots

# GPU benchmark + plots
cargo run --release -p runner --bin bench --features gpu -- gpu
python3 scripts/plot.py results/gpu-*.json results/plots/gpu

# Footprint (cold start / RSS / binary / build)
RAYON_NUM_THREADS=1 scripts/secondary.sh
```

## Status

- **Phase 0 (parity): done** — Candle / Burn / ORT agree to min cosine **1.000000**.
- **Phase 1 (CPU): done** — Candle fastest overall; **ORT fastest for a single
  short embed** (1.85 ms). See [REPORT.md](REPORT.md).
- **Phase 2 (GPU): done** — Candle Metal dominates (3.7–7.5×); ORT-CoreML beats
  Burn wgpu on throughput.
- **Secondary metrics: done** — Candle wins cold start, RSS, binary; ORT builds fastest.
- **Recommendation: Candle** for the NAHPU desktop embedding workload.
