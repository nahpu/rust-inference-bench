//! Single-engine binary for secondary metrics: prints cold-start time
//! (process start -> model loaded -> first embedding done, in ms). serving
//! as the artifact for peak-RSS and binary-size measurement.

use std::time::Instant;

use embed_candle::CandleEngine;
use embed_core::InferenceEngine;

fn main() -> anyhow::Result<()> {
    let t = Instant::now();
    let engine = CandleEngine::load("data/models/all-MiniLM-L6-v2")?;
    let _ = engine.embed("cold start probe")?;
    println!("{:.1}", t.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}
