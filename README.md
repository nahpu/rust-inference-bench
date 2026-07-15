# rust-inference-bench

A reproducible benchmark comparing Rust ML inference engines — **Candle**,
**Burn**, and **ONNX Runtime** (`ort`) — on a sentence-embedding workload
(`all-MiniLM-L6-v2`), to inform the NAHPU framework choice.

**Recommendation: Candle** for the NAHPU desktop embedding workload — fastest
overall throughput (CPU + GPU), cold start, memory, and binary size. ORT's one
edge is the lowest single-short-embed CPU latency. Full numbers in
[REPORT.md](REPORT.md).

## Quick start

```sh
./run.sh          # parity → CPU → GPU → footprint → plots, auto-detecting the machine
```

Needs Rust, `python3` (or `python`), and `curl`/`wget`. Results land in a
per-machine folder under `results/<os>-<arch>-<cpu>/`. Common flags:
`--cpu-only`, `--gpu auto|macos|cuda|wgpu|off`, `--blas auto|mkl|openblas`,
`--trials N`.

## Documentation

Everything else lives in the [**wiki**](../../wiki):

- [Design and Methodology](../../wiki/Design-and-Methodology) — scope, phases, fairness rules
- [Architecture](../../wiki/Architecture) — the `InferenceEngine` trait + crate layout
- [Running and Reproducing](../../wiki/Running-and-Reproducing) — full `run.sh` usage, cross-platform notes, results layout
- [Cross-platform EP matrix](GUIDE-cross-platform.md) — ORT DirectML / OpenVINO / NNAPI
