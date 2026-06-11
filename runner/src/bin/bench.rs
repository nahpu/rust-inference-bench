//! Phase 1 benchmark harness.
//!
//! Two scenarios (per the design, mapping MLPerf single-stream + offline to
//! NAHPU usage):
//!   - `latency_single`  — interactive search: one embed at a time (batch=1),
//!     reported as p50/p95/p99 over short/medium/long inputs.
//!   - `throughput`      — bulk indexing: embed a batch, swept over batch sizes,
//!     reported as sentences/second.
//!
//! Methodology: warm up past the turbo window, collect many samples, report the
//! distribution + relative margin of error (DoD bar: < 5%). Pin threads before
//! running, e.g. `RAYON_NUM_THREADS=1 cargo run --release -p runner --bin bench`.
//! Emits results/<timestamp>.json (the single source of truth for tables/plots).

use std::process::Command;
use std::time::Instant;

use anyhow::Result;
use embed_burn::BurnEngine;
use embed_candle::CandleEngine;
use embed_core::InferenceEngine;
use serde::Serialize;

const WARMUP: usize = 20;
const LATENCY_SAMPLES: usize = 200;
const THROUGHPUT_REPS: usize = 20;
const BATCH_SIZES: &[usize] = &[1, 8, 16, 32, 64];

// Framework versions (kept in sync with the Cargo.toml dep specs).
const CANDLE_VERSION: &str = "0.9";
const BURN_VERSION: &str = "0.21";

#[derive(Serialize)]
struct Report {
    timestamp: String,
    environment: Environment,
    results: Vec<Measurement>,
}

#[derive(Serialize)]
struct Environment {
    cpu: String,
    cores: usize,
    os: String,
    rustc: String,
    rayon_threads: String,
    candle_version: String,
    burn_version: String,
}

#[derive(Serialize)]
#[serde(tag = "scenario")]
enum Measurement {
    #[serde(rename = "latency_single")]
    Latency {
        engine: String,
        seq_label: String,
        approx_tokens: usize,
        samples: usize,
        mean_ms: f64,
        p50_ms: f64,
        p95_ms: f64,
        p99_ms: f64,
        rel_moe_pct: f64,
    },
    #[serde(rename = "throughput")]
    Throughput {
        engine: String,
        batch: usize,
        reps: usize,
        sentences_per_sec: f64,
    },
}

fn main() -> Result<()> {
    let engines = load_engines()?;
    if engines.is_empty() {
        anyhow::bail!("no working engines");
    }

    // Inputs of increasing length for the latency scenario.
    let inputs = [
        ("short", "Dark brown iris."),
        (
            "medium",
            "Adult male with reddish-brown dorsal fur and pale ventral coloration.",
        ),
        (
            "long",
            "Adult male with reddish-brown dorsal fur and pale ventral coloration, \
             collected near a montane stream at 1850 meters elevation; iris dark brown, \
             bill black with a yellow gape, pelage soft and dense, hind foot length 24 \
             millimeters, with worn molars indicating an older individual found in \
             primary forest understory shortly after dawn.",
        ),
    ];

    let mut results = Vec::new();

    for (name, engine) in &engines {
        eprintln!("== {name} ==");
        // Scenario A: single-stream latency.
        for (label, text) in &inputs {
            let m = bench_latency(engine.as_ref(), name, label, text)?;
            if let Measurement::Latency {
                p50_ms,
                p95_ms,
                rel_moe_pct,
                ..
            } = &m
            {
                eprintln!(
                    "  latency[{label:>6}] p50={p50_ms:.3}ms p95={p95_ms:.3}ms moe={rel_moe_pct:.1}%"
                );
            }
            results.push(m);
        }
        // Scenario B: offline throughput.
        for &batch in BATCH_SIZES {
            let m = bench_throughput(engine.as_ref(), name, batch)?;
            if let Measurement::Throughput {
                sentences_per_sec,
                ..
            } = &m
            {
                eprintln!("  thrpt[b={batch:>2}] {sentences_per_sec:.0} sent/s");
            }
            results.push(m);
        }
    }

    let report = Report {
        timestamp: cmd("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]).unwrap_or_default(),
        environment: capture_env(),
        results,
    };

    std::fs::create_dir_all("results")?;
    let stamp = cmd("date", &["-u", "+%Y%m%dT%H%M%SZ"]).unwrap_or_else(|| "run".to_string());
    let host = cmd("hostname", &["-s"]).unwrap_or_else(|| "host".to_string());
    let path = format!("results/{stamp}-{host}.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
    eprintln!("\nwrote {path}");
    Ok(())
}

fn load_engines() -> Result<Vec<(String, Box<dyn InferenceEngine>)>> {
    let model_dir = "data/models/all-MiniLM-L6-v2";
    let mut v: Vec<(String, Box<dyn InferenceEngine>)> = Vec::new();
    let candle = CandleEngine::load(model_dir)?;
    if candle.embed("probe").is_ok() {
        v.push((candle.name().to_string(), Box::new(candle)));
    }
    let burn = BurnEngine::load(model_dir)?;
    if burn.embed("probe").is_ok() {
        v.push((burn.name().to_string(), Box::new(burn)));
    }
    Ok(v)
}

fn bench_latency(
    engine: &dyn InferenceEngine,
    name: &str,
    label: &str,
    text: &str,
) -> Result<Measurement> {
    for _ in 0..WARMUP {
        engine.embed(text)?;
    }
    let mut samples = Vec::with_capacity(LATENCY_SAMPLES);
    for _ in 0..LATENCY_SAMPLES {
        let t = Instant::now();
        engine.embed(text)?;
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
    let std = var.sqrt();
    let moe = 1.96 * std / (samples.len() as f64).sqrt();
    let approx_tokens = text.split_whitespace().count();

    Ok(Measurement::Latency {
        engine: name.to_string(),
        seq_label: label.to_string(),
        approx_tokens,
        samples: samples.len(),
        mean_ms: mean,
        p50_ms: percentile(&samples, 50.0),
        p95_ms: percentile(&samples, 95.0),
        p99_ms: percentile(&samples, 99.0),
        rel_moe_pct: if mean > 0.0 { moe / mean * 100.0 } else { 0.0 },
    })
}

fn bench_throughput(engine: &dyn InferenceEngine, name: &str, batch: usize) -> Result<Measurement> {
    let corpus = embed_core::load_corpus("data/corpus.sample.txt").unwrap_or_default();
    let base = if corpus.is_empty() {
        vec!["a specimen note".to_string()]
    } else {
        corpus
    };
    let texts: Vec<String> = (0..batch).map(|i| base[i % base.len()].clone()).collect();

    for _ in 0..WARMUP.min(5) {
        engine.embed_batch(&texts)?;
    }
    let t = Instant::now();
    for _ in 0..THROUGHPUT_REPS {
        engine.embed_batch(&texts)?;
    }
    let elapsed = t.elapsed().as_secs_f64();
    let sps = (batch * THROUGHPUT_REPS) as f64 / elapsed;

    Ok(Measurement::Throughput {
        engine: name.to_string(),
        batch,
        reps: THROUGHPUT_REPS,
        sentences_per_sec: sps,
    })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (rank - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

fn capture_env() -> Environment {
    Environment {
        cpu: cmd("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default(),
        cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        rustc: cmd("rustc", &["--version"]).unwrap_or_default(),
        rayon_threads: std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string()),
        candle_version: CANDLE_VERSION.to_string(),
        burn_version: BURN_VERSION.to_string(),
    }
}

fn cmd(prog: &str, args: &[&str]) -> Option<String> {
    Command::new(prog)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}
