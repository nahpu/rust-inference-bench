//! Phase 1 benchmark harness (interleaved, N-engine).
//!
//! Compares an arbitrary set of embedding engines (currently candle, burn, ort)
//! over two scenarios (design: MLPerf single-stream + offline, mapped to NAHPU):
//!   - `latency_single` — interactive search, one embed at a time (batch=1),
//!     over short/medium/long inputs.
//!   - `throughput`     — bulk indexing, embed a batch, swept over batch sizes.
//!
//! Why interleaved: an unpinned laptop (Apple Silicon P/E cores, power mgmt)
//! drifts run-to-run more than 5%, so absolute single-run reproducibility is not
//! achievable here. Instead we run N trials; within each trial *every* engine is
//! measured back-to-back, and the starting engine rotates per trial so order/
//! drift bias cancels. We then report each engine's median +/- IQR across
//! trials, plus the per-trial pairwise speedup ratio. Decision rule: a pair is
//! *distinguishable* when the IQR of its speedup ratio excludes 1.0 (effect size
//! exceeds run-to-run spread).
//!
//! Run pinned + on AC: `RAYON_NUM_THREADS=1 cargo run --release -p runner --bin bench`.
//! GPU peers: `cargo run --release -p runner --bin bench --features gpu -- gpu`.

use std::collections::BTreeMap;
use std::process::Command;
use std::time::Instant;

use anyhow::Result;
use embed_burn::BurnEngine;
#[cfg(feature = "gpu")]
use embed_burn::BurnWgpuEngine;
use embed_candle::CandleEngine;
use embed_core::InferenceEngine;
use embed_ort::OrtEngine;
use serde::Serialize;

const TRIALS: usize = 10;
const WARMUP: usize = 5;
const LATENCY_SAMPLES: usize = 80;
const THROUGHPUT_REPS: usize = 10;
const BATCH_SIZES: &[usize] = &[1, 8, 16, 32, 64];

const CANDLE_VERSION: &str = "0.9";
const BURN_VERSION: &str = "0.21";
const ORT_VERSION: &str = "2.0.0-rc.10";

#[derive(Serialize)]
struct Report {
    timestamp: String,
    mode: String,
    engines: Vec<String>,
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
    ort_version: String,
}

/// Aggregate of one metric across trials.
#[derive(Serialize, Clone)]
struct Agg {
    median: f64,
    p25: f64,
    p75: f64,
    iqr: f64,
    n: usize,
}

/// Per-trial speedup ratio between two engines, defined so >1 favours `a`.
#[derive(Serialize)]
struct PairRec {
    a: String,
    b: String,
    speedup_x: Agg,
    distinguishable: bool,
    faster: String,
}

#[derive(Serialize)]
struct LatencyRec {
    seq_label: String,
    approx_tokens: usize,
    /// engine name -> latency aggregate (ms, lower is better).
    ms: BTreeMap<String, Agg>,
    fastest: String,
    pairwise: Vec<PairRec>,
}

#[derive(Serialize)]
struct ThroughputRec {
    batch: usize,
    /// engine name -> throughput aggregate (sentences/s, higher is better).
    sps: BTreeMap<String, Agg>,
    fastest: String,
    pairwise: Vec<PairRec>,
}

struct EngineHandle {
    name: String,
    engine: Box<dyn InferenceEngine>,
}

fn main() -> Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "cpu".to_string());
    let engines = load_engines(&mode)?;
    let names: Vec<String> = engines.iter().map(|e| e.name.clone()).collect();
    let e = engines.len();
    eprintln!("mode={mode}: {}", names.join(" vs "));

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

    // Per-trial samples: [scenario][engine] -> Vec<f64> over trials.
    let mut lat: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; e]; inputs.len()];
    let mut thr: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; e]; BATCH_SIZES.len()];

    for trial in 0..TRIALS {
        eprintln!("trial {}/{}", trial + 1, TRIALS);
        // Rotate which engine goes first each trial to cancel order/drift bias.
        let order: Vec<usize> = (0..e).map(|k| (trial + k) % e).collect();

        for (i, (_label, text)) in inputs.iter().enumerate() {
            for &idx in &order {
                let v = latency_once(engines[idx].engine.as_ref(), text)?;
                lat[i][idx].push(v);
            }
        }
        for (i, &batch) in BATCH_SIZES.iter().enumerate() {
            for &idx in &order {
                let v = throughput_once(engines[idx].engine.as_ref(), batch, &corpus)?;
                thr[i][idx].push(v);
            }
        }
    }

    let latency: Vec<LatencyRec> = inputs
        .iter()
        .enumerate()
        .map(|(i, (label, text))| {
            let ms: BTreeMap<String, Agg> = names
                .iter()
                .enumerate()
                .map(|(k, n)| (n.clone(), agg(lat[i][k].clone())))
                .collect();
            LatencyRec {
                seq_label: label.to_string(),
                approx_tokens: text.split_whitespace().count(),
                fastest: fastest_by(&ms, Metric::Latency),
                pairwise: pairwise(&names, &lat[i], Metric::Latency),
                ms,
            }
        })
        .collect();

    let throughput: Vec<ThroughputRec> = BATCH_SIZES
        .iter()
        .enumerate()
        .map(|(i, &batch)| {
            let sps: BTreeMap<String, Agg> = names
                .iter()
                .enumerate()
                .map(|(k, n)| (n.clone(), agg(thr[i][k].clone())))
                .collect();
            ThroughputRec {
                batch,
                fastest: fastest_by(&sps, Metric::Throughput),
                pairwise: pairwise(&names, &thr[i], Metric::Throughput),
                sps,
            }
        })
        .collect();

    print_summary(&names, &latency, &throughput);

    let report = Report {
        timestamp: cmd("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]).unwrap_or_default(),
        mode: mode.clone(),
        engines: names,
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
    let path = format!("results/{mode}-{stamp}-{host}.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
    eprintln!("\nwrote {path}");
    Ok(())
}

#[derive(Clone, Copy)]
enum Metric {
    /// Lower is better (ms).
    Latency,
    /// Higher is better (sentences/s).
    Throughput,
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

/// The engine with the best median (min for latency, max for throughput).
fn fastest_by(stats: &BTreeMap<String, Agg>, metric: Metric) -> String {
    stats
        .iter()
        .min_by(|a, b| {
            let (x, y) = (a.1.median, b.1.median);
            match metric {
                Metric::Latency => x.partial_cmp(&y).unwrap(),
                Metric::Throughput => y.partial_cmp(&x).unwrap(),
            }
        })
        .map(|(n, _)| n.clone())
        .unwrap_or_default()
}

/// Per-trial pairwise speedup ratios for every unordered engine pair.
fn pairwise(names: &[String], samples: &[Vec<f64>], metric: Metric) -> Vec<PairRec> {
    let mut out = Vec::new();
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            // Define ratio so >1 means `a` (engine i) is faster.
            let ratio = match metric {
                // latency: smaller is faster -> b_ms / a_ms
                Metric::Latency => zip_ratio(&samples[j], &samples[i]),
                // throughput: larger is faster -> a_sps / b_sps
                Metric::Throughput => zip_ratio(&samples[i], &samples[j]),
            };
            let s = agg(ratio);
            let (dist, faster) = decide(&s, &names[i], &names[j]);
            out.push(PairRec {
                a: names[i].clone(),
                b: names[j].clone(),
                speedup_x: s,
                distinguishable: dist,
                faster,
            });
        }
    }
    out
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
/// Speedup ratio is defined so >1 favours `a`.
fn decide(speedup: &Agg, a: &str, b: &str) -> (bool, String) {
    if speedup.p25 > 1.0 {
        (true, a.to_string())
    } else if speedup.p75 < 1.0 {
        (true, b.to_string())
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

fn print_summary(names: &[String], latency: &[LatencyRec], throughput: &[ThroughputRec]) {
    eprintln!("\n=== LATENCY p50 (ms), median [IQR] — * = fastest ===");
    for r in latency {
        eprint!("  {:6}", r.seq_label);
        for n in names {
            let a = &r.ms[n];
            let star = if *n == r.fastest { "*" } else { " " };
            eprint!("  {n} {:6.2}[{:.2}]{star}", a.median, a.iqr);
        }
        eprintln!();
    }
    eprintln!("\n=== THROUGHPUT (sent/s), median [IQR] — * = fastest ===");
    for r in throughput {
        eprint!("  b={:<3}", r.batch);
        for n in names {
            let a = &r.sps[n];
            let star = if *n == r.fastest { "*" } else { " " };
            eprint!("  {n} {:6.0}[{:.0}]{star}", a.median, a.iqr);
        }
        eprintln!();
    }
}

fn load_engines(mode: &str) -> Result<Vec<EngineHandle>> {
    let dir = "data/models/all-MiniLM-L6-v2";
    let mut engines: Vec<EngineHandle> = Vec::new();
    match mode {
        "gpu" => {
            #[cfg(feature = "gpu")]
            {
                let c = CandleEngine::load_metal(dir)?;
                let b = BurnWgpuEngine::load_wgpu(dir)?;
                let o = OrtEngine::load_coreml(dir)?;
                engines.push(EngineHandle {
                    name: c.name().to_string(),
                    engine: Box::new(c),
                });
                engines.push(EngineHandle {
                    name: b.name().to_string(),
                    engine: Box::new(b),
                });
                engines.push(EngineHandle {
                    name: o.name().to_string(),
                    engine: Box::new(o),
                });
            }
            #[cfg(not(feature = "gpu"))]
            {
                anyhow::bail!("rebuild with `--features gpu` to run GPU mode");
            }
        }
        _ => {
            let c = CandleEngine::load(dir)?;
            let b = BurnEngine::load(dir)?;
            let o = OrtEngine::load(dir)?;
            engines.push(EngineHandle {
                name: c.name().to_string(),
                engine: Box::new(c),
            });
            engines.push(EngineHandle {
                name: b.name().to_string(),
                engine: Box::new(b),
            });
            engines.push(EngineHandle {
                name: o.name().to_string(),
                engine: Box::new(o),
            });
        }
    }
    // Probe each engine once so the first real measurement isn't a cold path.
    for h in &engines {
        h.engine.embed("probe")?;
    }
    Ok(engines)
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
        ort_version: ORT_VERSION.to_string(),
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
