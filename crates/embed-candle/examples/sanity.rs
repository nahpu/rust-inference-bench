//! Sanity check for the Candle engine: confirms embeddings carry semantic
//! structure (related sentences score higher than unrelated ones) and have the
//! expected dimensionality. Run: `cargo run -p embed-candle --example sanity`.

use embed_core::{cosine_similarity, InferenceEngine, EMBED_DIM};
use embed_candle::CandleEngine;

fn main() -> anyhow::Result<()> {
    let engine = CandleEngine::load("data/models/all-MiniLM-L6-v2")?;

    let a = "Adult male with reddish-brown dorsal fur.";
    let b = "The specimen has rusty brown colored fur.";
    let c = "Collected near a montane stream at high elevation.";

    let (ea, eb, ec) = (engine.embed(a)?, engine.embed(b)?, engine.embed(c)?);

    assert_eq!(ea.len(), EMBED_DIM, "unexpected embedding dimension");

    let related = cosine_similarity(&ea, &eb);
    let unrelated = cosine_similarity(&ea, &ec);

    println!("dim            = {}", ea.len());
    println!("related   (a~b) = {related:.4}");
    println!("unrelated (a~c) = {unrelated:.4}");
    println!(
        "semantic ordering {}",
        if related > unrelated { "OK" } else { "WRONG" }
    );
    Ok(())
}
