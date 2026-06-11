//! Phase 1 benchmark harness (interleaved, paired).
//!
//! Two scenarios (design: MLPerf single-stream + offline mapped to NAHPU usage):
//!   - `latency_single` — interactive search, one embed at a time (batch=1),
//!     over short/medium/long inputs.
//!   - `throughput`     — bulk indexing, embed a batch, swept over batch sizes.
//!
//! Why interleaved: an unpinned laptop (Apple Silicon P/E cores, power mgmt)
//! drifts run-to-run more than 5%, so absolute single-run reproducibility is not
//! achievable here. Instead we run N trials; within each trial both engines are
//! measured back-to-back (order alternates per trial) so they share identical
//! conditions. We then report median +/- IQR across trials, and the per-trial
//! speedup ratio. The decision rule: engines are *distinguishable* when the IQR
//! of the speedup ratio excludes 1.0 (effect size exceeds run-to-run spread).
//!
//! Run pinned + on AC: `RAYON_NUM_THREADS=1 cargo run --release -p runner --bin bench`.

use std::process::Command;
use std::time::Instant;

use anyhow::Result;
use embed_burn::BurnEngine;
use embed_candle::CandleEngine;
use embed_core::InferenceEngine;
use serde::Serialize;

const TRIALS: usize = 10;
const WARMUP: usize = 5;
const LATENCY_SAMPLES: usize = 80;
const THROUGHPUT_REPS: usize = 10;
const BATCH_SIZES: &[usize] = &[1, 8, 16, 32, 64];

const CANDLE: &str = "candle-cpu";
const BURN: &str = "burn-ndarray";
const CANDLE_VERSION: &str = "0.9";
const BURN_VERSION: &str = "0.21";

#[derive(Serialize)]
struct Report {
    timestamp: String,
    environment: Environment,
    config: Config,
    latency: Vec<LatencyRec>,
    throughput: Vec<ThroughputRec>,
}

#[derive(Serialize)]
struct Config {
    trials: usize,
    latency_samples: usize,
    throughput_reps: usize,
    batch_sizes: Vec<usize>,
}

#[derive(Serialize)]
struct Environment {
    cpu: String,
    cores: usize,
    os: String,
    rustc: String,
    rayon_threads: String,
    on_ac_power: bool,
    candle_version: String,
    burn_version: String,
}

/// Aggregate of one metric across trials.
#[derive(Serialize)]
struct Agg {
    median: f64,
    p25: f64,
    p75: f64,
    iqr: f64,
    n: usize,
}

#[derive(Serialize)]
struct LatencyRec {
    seq_label: String,
    approx_tokens: usize,
    candle_ms: Agg,
    burn_ms: Agg,
    /// Per-trial burn_ms / candle_ms (>1 means Candle is faster).
    candle_speedup_x: Agg,
    distinguishable: bool,
    faster: String,
}

#[derive(Serialize)]
struct ThroughputRec {
    batch: usize,
    candle_sps: Agg,
    burn_sps: Agg,
    /// Per-trial candle_sps / burn_sps (>1 means Candle is faster).
    candle_speedup_x: Agg,
    distinguishable: bool,
    faster: String,
}

fn main() -> Result<()> {
    let (candle, burn) = load_engines()?;

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
    let corpus = embed_core::load_corpus("data/corpus.sample.txt").unwrap_or_default();

    // Per-trial samples, keyed by scenario; index 0 = candle, 1 = burn.
    let mut lat: Vec<[Vec<f64>; 2]> = (0..inputs.len()).map(|_| [vec![], vec![]]).collect();
    let mut thr: Vec<[Vec<f64>; 2]> = (0..BATCH_SIZES.len()).map(|_| [vec![], vec![]]).collect();

    for trial in 0..TRIALS {
        eprintln!("trial {}/{}", trial + 1, TRIALS);
        // Alternate which engine is measured first to cancel order/drift bias.
        let candle_first = trial % 2 == 0;

        for (i, (_label, text)) in inputs.iter().enumerate() {
            let (c, b) = measure_pair(candle_first, || latency_once(&candle, text), || {
                latency_once(&burn, text)
            })?;
            lat[i][0].push(c);
            lat[i][1].push(b);
        }
        for (i, &batch) in BATCH_SIZES.iter().enumerate() {
            let (c, b) = measure_pair(
                candle_first,
                || throughput_once(&candle, batch, &corpus),
                || throughput_once(&burn, batch, &corpus),
            )?;
            thr[i][0].push(c);
            thr[i][1].push(b);
        }
    }

    let latency: Vec<LatencyRec> = inputs
        .iter()
        .enumerate()
        .map(|(i, (label, text))| {
            let speedup: Vec<f64> = zip_ratio(&lat[i][1], &lat[i][0]); // burn/candle
            let s = agg(speedup);
            let (dist, faster) = decide(&s);
            LatencyRec {
                seq_label: label.to_string(),
                approx_tokens: text.split_whitespace().count(),
                candle_ms: agg(lat[i][0].clone()),
                burn_ms: agg(lat[i][1].clone()),
                candle_speedup_x: s,
                distinguishable: dist,
                faster,
            }
        })
        .collect();

    let throughput: Vec<ThroughputRec> = BATCH_SIZES
        .iter()
        .enumerate()
        .map(|(i, &batch)| {
            let speedup: Vec<f64> = zip_ratio(&thr[i][0], &thr[i][1]); // candle/burn
            let s = agg(speedup);
            let (dist, faster) = decide(&s);
            ThroughputRec {
                batch,
                candle_sps: agg(thr[i][0].clone()),
                burn_sps: agg(thr[i][1].clone()),
                candle_speedup_x: s,
                distinguishable: dist,
                faster,
            }
        })
        .collect();

    print_summary(&latency, &throughput);

    let report = Report {
        timestamp: cmd("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]).unwrap_or_default(),
        environment: capture_env(),
        config: Config {
            trials: TRIALS,
            latency_samples: LATENCY_SAMPLES,
            throughput_reps: THROUGHPUT_REPS,
            batch_sizes: BATCH_SIZES.to_vec(),
        },
        latency,
        throughput,
    };

    std::fs::create_dir_all("results")?;
    let stamp = cmd("date", &["-u", "+%Y%m%dT%H%M%SZ"]).unwrap_or_else(|| "run".to_string());
    let host = cmd("hostname", &["-s"]).unwrap_or_else(|| "host".to_string());
    let path = format!("results/{stamp}-{host}.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
    eprintln!("\nwrote {path}");
    Ok(())
}

/// Measure two closures adjacently, honoring the alternating order.
fn measure_pair(
    candle_first: bool,
    candle: impl Fn() -> Result<f64>,
    burn: impl Fn() -> Result<f64>,
) -> Result<(f64, f64)> {
    if candle_first {
        let c = candle()?;
        let b = burn()?;
        Ok((c, b))
    } else {
        let b = burn()?;
        let c = candle()?;
        Ok((c, b))
    }
}

/// One latency measurement: median of LATENCY_SAMPLES single embeds (ms).
fn latency_once(engine: &dyn InferenceEngine, text: &str) -> Result<f64> {
    for _ in 0..WARMUP {
        engine.embed(text)?;
    }
    let mut s = Vec::with_capacity(LATENCY_SAMPLES);
    for _ in 0..LATENCY_SAMPLES {
        let t = Instant::now();
        engine.embed(text)?;
        s.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(percentile(&s, 50.0))
}

/// One throughput measurement: sentences/second over THROUGHPUT_REPS batches.
fn throughput_once(engine: &dyn InferenceEngine, batch: usize, corpus: &[String]) -> Result<f64> {
    let fallback = [String::from("a specimen note")];
    let base: &[String] = if corpus.is_empty() { &fallback } else { corpus };
    let texts: Vec<String> = (0..batch).map(|i| base[i % base.len()].clone()).collect();
    for _ in 0..WARMUP.min(3) {
        engine.embed_batch(&texts)?;
    }
    let t = Instant::now();
    for _ in 0..THROUGHPUT_REPS {
        engine.embed_batch(&texts)?;
    }
    Ok((batch * THROUGHPUT_REPS) as f64 / t.elapsed().as_secs_f64())
}

fn zip_ratio(num: &[f64], den: &[f64]) -> Vec<f64> {
    num.iter()
        .zip(den)
        .filter(|(_, &d)| d > 0.0)
        .map(|(&n, &d)| n / d)
        .collect()
}

fn agg(mut v: Vec<f64>) -> Agg {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p25 = percentile(&v, 25.0);
    let p75 = percentile(&v, 75.0);
    Agg {
        median: percentile(&v, 50.0),
        p25,
        p75,
        iqr: p75 - p25,
        n: v.len(),
    }
}

/// Distinguishable when the speedup IQR excludes 1.0 (effect > spread).
fn decide(speedup: &Agg) -> (bool, String) {
    if speedup.p25 > 1.0 {
        (true, CANDLE.to_string())
    } else if speedup.p75 < 1.0 {
        (true, BURN.to_string())
    } else {
        (false, "tie".to_string())
    }
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

fn print_summary(latency: &[LatencyRec], throughput: &[ThroughputRec]) {
    eprintln!("\n=== LATENCY p50 (ms), median [IQR] ===");
    for r in latency {
        eprintln!(
            "  {:6}  candle {:6.2} [{:.2}]   burn {:6.2} [{:.2}]   {:.2}x -> {}",
            r.seq_label,
            r.candle_ms.median,
            r.candle_ms.iqr,
            r.burn_ms.median,
            r.burn_ms.iqr,
            r.candle_speedup_x.median,
            if r.distinguishable { &r.faster } else { "tie" }
        );
    }
    eprintln!("\n=== THROUGHPUT (sent/s), median [IQR] ===");
    for r in throughput {
        eprintln!(
            "  b={:<3} candle {:6.0} [{:.0}]   burn {:6.0} [{:.0}]   {:.2}x -> {}",
            r.batch,
            r.candle_sps.median,
            r.candle_sps.iqr,
            r.burn_sps.median,
            r.burn_sps.iqr,
            r.candle_speedup_x.median,
            if r.distinguishable { &r.faster } else { "tie" }
        );
    }
}

fn load_engines() -> Result<(CandleEngine, BurnEngine)> {
    let model_dir = "data/models/all-MiniLM-L6-v2";
    let candle = CandleEngine::load(model_dir)?;
    let burn = BurnEngine::load(model_dir)?;
    candle.embed("probe")?;
    burn.embed("probe")?;
    Ok((candle, burn))
}

fn capture_env() -> Environment {
    let on_ac = cmd("pmset", &["-g", "batt"])
        .map(|s| s.contains("AC Power"))
        .unwrap_or(false);
    Environment {
        cpu: cmd("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default(),
        cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        rustc: cmd("rustc", &["--version"]).unwrap_or_default(),
        rayon_threads: std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string()),
        on_ac_power: on_ac,
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
