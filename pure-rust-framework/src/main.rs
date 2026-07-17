//! Phase 0 parity check: confirm every engine produces equivalent embeddings for
//! the same input. Until this passes, any speed comparison is apples-to-oranges,
//! so this runs first.
//!
//! This version is pure Rust and compares Candle and Burn.
//!
//! Usage: `cargo run -p pure-rust-framework -- [corpus_path] [model_dir]`

use anyhow::Result;
use embed_burn::BurnEngine;
use embed_candle::CandleEngine;
use embed_core::{cosine_similarity, load_corpus, Embedding, InferenceEngine};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let corpus_path = args
        .next()
        .unwrap_or_else(|| "data/corpus.sample.txt".to_string());
    let model_dir = args
        .next()
        .unwrap_or_else(|| "data/models/all-MiniLM-L6-v2".to_string());

    let corpus = load_corpus(&corpus_path)?;
    println!("Loaded {} sentences from {corpus_path}", corpus.len());

    let engines: Vec<Box<dyn InferenceEngine>> = vec![
        Box::new(CandleEngine::load(&model_dir)?),
        Box::new(BurnEngine::load(&model_dir)?),
    ];

    // Embed the whole corpus with each engine.
    let mut outputs: Vec<(String, Vec<Embedding>)> = Vec::new();
    for engine in &engines {
        match engine.embed_batch(&corpus) {
            Ok(v) => outputs.push((engine.name().to_string(), v)),
            Err(e) => println!("  [skip] {}: {e}", engine.name()),
        }
    }

    if outputs.len() < 2 {
        println!(
            "\nNeed >=2 working engines for a parity check. \
             Implement the forward passes (Phase 0), then re-run."
        );
        return Ok(());
    }

    // Compare every pair sentence-by-sentence; gate on the worst pair.
    let mut overall_min = f32::MAX;
    println!();
    for i in 0..outputs.len() {
        for j in (i + 1)..outputs.len() {
            let (name_a, a) = &outputs[i];
            let (name_b, b) = &outputs[j];
            let min_sim = a
                .iter()
                .zip(b)
                .map(|(x, y)| cosine_similarity(x, y))
                .fold(f32::MAX, f32::min);
            overall_min = overall_min.min(min_sim);
            println!("{name_a} vs {name_b}: min cosine similarity = {min_sim:.6}");
        }
    }

    println!(
        "\nParity {} (worst pair min cosine = {overall_min:.6})",
        if overall_min > 0.999 { "PASS" } else { "FAIL" }
    );
    Ok(())
}
