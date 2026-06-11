//! Candle implementation of [`InferenceEngine`].
//!
//! Loads `all-MiniLM-L6-v2` (BERT) on the CPU backend, runs the forward pass,
//! applies attention-masked mean pooling + L2 normalization to match the
//! sentence-transformers reference. Every `candle_*` / `tokenizers` type stays
//! confined to this crate so the framework boundary stays clean.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use embed_core::{Embedding, InferenceEngine};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
/// Pin the revision for reproducibility. TODO(phase-1): replace with the exact
/// commit SHA and record it in the results JSON.
const REVISION: &str = "main";

/// Candle-backed embedding engine (CPU backend, per the desktop-only scope).
pub struct CandleEngine {
    name: String,
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleEngine {
    /// Load weights + tokenizer. `model_dir` is currently unused: weights are
    /// fetched (and cached) from the HF Hub at the pinned revision so the
    /// benchmark is self-contained.
    pub fn load(_model_dir: &str) -> Result<Self> {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        let device = Device::Cpu;
        let repo = Api::new()?.repo(Repo::with_revision(
            MODEL_ID.to_string(),
            RepoType::Model,
            REVISION.to_string(),
        ));

        let config_path = repo.get("config.json").context("fetch config.json")?;
        let tokenizer_path = repo.get("tokenizer.json").context("fetch tokenizer.json")?;
        let weights_path = repo
            .get("model.safetensors")
            .context("fetch model.safetensors")?;

        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(config_path)?).context("parse config")?;
        let mut tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        // Fixed-length padding keeps batched tensors rectangular; truncation
        // bounds the sequence length we benchmark.
        tokenizer
            .with_padding(Some(tokenizers::PaddingParams::default()))
            .with_truncation(Some(tokenizers::TruncationParams::default()))
            .map_err(anyhow::Error::msg)?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };
        let model = BertModel::load(vb, &config).context("load BERT")?;

        Ok(Self {
            name: "candle-cpu".to_string(),
            model,
            tokenizer,
            device,
        })
    }

    /// Run the model over a batch of pre-tokenized strings and return one
    /// normalized embedding per input. Shared by `embed` and `embed_batch`.
    fn forward(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(anyhow::Error::msg)?;

        let mut ids = Vec::with_capacity(encodings.len());
        let mut masks = Vec::with_capacity(encodings.len());
        for enc in &encodings {
            ids.push(enc.get_ids().to_vec());
            masks.push(enc.get_attention_mask().to_vec());
        }

        let seq_len = ids[0].len();
        let n = ids.len();
        let input_ids = Tensor::from_vec(ids.concat(), (n, seq_len), &self.device)?;
        let attention_mask = Tensor::from_vec(masks.concat(), (n, seq_len), &self.device)?;
        let token_type_ids = input_ids.zeros_like()?;

        // [n, seq_len, hidden]
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Attention-masked mean pooling: sum(hidden * mask) / sum(mask).
        let mask_f = attention_mask.to_dtype(DType::F32)?; // [n, seq_len]
        let mask_exp = mask_f.unsqueeze(2)?; // [n, seq_len, 1]
        let summed = hidden.broadcast_mul(&mask_exp)?.sum(1)?; // [n, hidden]
        let counts = mask_f.sum(1)?.unsqueeze(1)?; // [n, 1]
        let mean = summed.broadcast_div(&counts)?;

        // L2 normalize.
        let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = mean.broadcast_div(&norm)?;

        Ok(normalized.to_vec2::<f32>()?)
    }
}

impl InferenceEngine for CandleEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn embed(&self, text: &str) -> Result<Embedding> {
        let mut out = self.forward(&[text.to_string()])?;
        Ok(out.remove(0))
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.forward(texts)
    }
}
