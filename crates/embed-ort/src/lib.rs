//! ONNX Runtime implementation of [`InferenceEngine`].
//!
//! Loads the *same* all-MiniLM ONNX export that `embed-burn` compiles at build
//! time, but runs it through ONNX Runtime (the `ort` crate) rather than
//! generating Rust. Pooling + normalization mirror the Candle and Burn engines
//! exactly (attention-masked mean pooling + L2 norm), so all three are directly
//! comparable.
//!
//! Execution providers:
//!   - default (CPU) — the desktop baseline, matched against candle-cpu / burn-ndarray.
//!   - CoreML (feature `coreml`) — GPU/ANE on macOS, the Phase 2 GPU peer.

use std::sync::Mutex;

use anyhow::{Context, Result};
use embed_core::{Embedding, InferenceEngine, EMBED_DIM};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const REVISION: &str = "main";
/// The ONNX export path inside the HF repo (same file embed-burn fetches).
const ONNX_FILE: &str = "onnx/model.onnx";

/// ONNX Runtime-backed embedding engine.
///
/// `Session::run` takes `&mut self`, so we keep it behind a `Mutex` to satisfy
/// the `&self` [`InferenceEngine`] contract. The benchmark is single-threaded
/// per engine, so the lock is uncontended.
pub struct OrtEngine {
    name: String,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl OrtEngine {
    /// Load on the default CPU execution provider (the desktop baseline).
    pub fn load(model_dir: &str) -> Result<Self> {
        Self::load_with("ort-cpu", model_dir, |b| Ok(b))
    }

    /// Load on the CoreML execution provider (GPU/ANE on macOS). Phase 2.
    #[cfg(feature = "coreml")]
    pub fn load_coreml(model_dir: &str) -> Result<Self> {
        use ort::execution_providers::coreml::{CoreMLComputeUnits, CoreMLModelFormat};
        use ort::execution_providers::CoreMLExecutionProvider;
        // MLProgram supports more operators than the default NeuralNetwork format
        // (which fails at runtime on this BERT export with dynamic shapes).
        Self::load_with("ort-coreml", model_dir, |b| {
            b.with_execution_providers([CoreMLExecutionProvider::default()
                .with_model_format(CoreMLModelFormat::MLProgram)
                .with_compute_units(CoreMLComputeUnits::All)
                .build()])
                .map_err(Into::into)
        })
    }

    /// Shared loader. `configure` registers any execution providers before the
    /// session is committed. `model_dir` is unused: weights are fetched (and
    /// cached) from the HF Hub at the pinned revision so the bench is self-contained.
    fn load_with(
        name: &str,
        _model_dir: &str,
        configure: impl FnOnce(
            ort::session::builder::SessionBuilder,
        ) -> Result<ort::session::builder::SessionBuilder>,
    ) -> Result<Self> {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        let repo = Api::new()?.repo(Repo::with_revision(
            MODEL_ID.to_string(),
            RepoType::Model,
            REVISION.to_string(),
        ));
        let onnx_path = repo.get(ONNX_FILE).context("fetch onnx/model.onnx")?;
        let tokenizer_path = repo.get("tokenizer.json").context("fetch tokenizer.json")?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        tokenizer
            .with_padding(Some(tokenizers::PaddingParams::default()))
            .with_truncation(Some(tokenizers::TruncationParams::default()))
            .map_err(anyhow::Error::msg)?;

        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?;
        let builder = configure(builder)?;
        let session = builder
            .commit_from_file(&onnx_path)
            .with_context(|| format!("load ONNX session from {}", onnx_path.display()))?;

        Ok(Self {
            name: name.to_string(),
            session: Mutex::new(session),
            tokenizer,
        })
    }

    fn forward(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(anyhow::Error::msg)?;

        let n = encodings.len();
        let seq = encodings[0].len();
        let mut ids = Vec::with_capacity(n * seq);
        let mut mask = Vec::with_capacity(n * seq);
        for enc in &encodings {
            ids.extend(enc.get_ids().iter().map(|&x| x as i64));
            mask.extend(enc.get_attention_mask().iter().map(|&x| x as i64));
        }
        let token_type = vec![0i64; n * seq];

        let input_ids = Tensor::from_array(([n, seq], ids.clone()))?;
        let attention_mask = Tensor::from_array(([n, seq], mask.clone()))?;
        let token_type_ids = Tensor::from_array(([n, seq], token_type))?;

        // Run + copy out the result while the lock is held; `SessionOutputs`
        // borrows the session, so we own the tensor before releasing it.
        // last_hidden_state: [n, seq, hidden]
        let (hidden, hsize) = {
            let mut session = self.session.lock().expect("ort session poisoned");
            let outputs = session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            ])?;
            let (shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
            (data.to_vec(), shape[2] as usize)
        };
        debug_assert_eq!(hsize, EMBED_DIM);

        // Attention-masked mean pooling + L2 norm, per sentence.
        let mut out = Vec::with_capacity(n);
        for s in 0..n {
            let mut acc = vec![0f32; hsize];
            let mut count = 0f32;
            for t in 0..seq {
                let m = mask[s * seq + t] as f32;
                if m == 0.0 {
                    continue;
                }
                count += m;
                let base = (s * seq + t) * hsize;
                for h in 0..hsize {
                    acc[h] += hidden[base + h] * m;
                }
            }
            let denom = if count == 0.0 { 1.0 } else { count };
            for v in acc.iter_mut() {
                *v /= denom;
            }
            let norm = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in acc.iter_mut() {
                    *v /= norm;
                }
            }
            out.push(acc);
        }
        Ok(out)
    }
}

impl InferenceEngine for OrtEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn embed(&self, text: &str) -> Result<Embedding> {
        Ok(self.forward(&[text.to_string()])?.remove(0))
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.forward(texts)
    }
}
