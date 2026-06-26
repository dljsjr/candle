//! Gemma 4 (E-models) quantized GGUF implementation — SKELETON.
//!
//! Mirrors the parity-validated float [`crate::models::gemma4::text`] with `QMatMul`, loading the
//! official `gemma4` GGUF (the arch llama.cpp ships). Scaffolded from [`super::quantized_gemma3`];
//! the E-model machinery added on top is: PLE (per-layer embeddings), shared-KV (layers
//! `>= block_count - shared_kv_layers`), double-wide MLP on shared layers, a per-layer
//! `layer_scalar`, per-layer head_dim (global vs sliding), proportional rope on global layers,
//! `final_logit_softcapping`, and a gelu-tanh MLP (NOT silu).
//!
//! Based on the HuggingFace `transformers` gemma4 implementation (Apache-2.0) and eddyb's #3608.
//! See `.sandpiper/docs/gemma4-candle-gguf-plan.md` for the full tensor/metadata map.
//!
//! STATUS: skeleton — `from_gguf` loads the full tensor set; `forward` is a port-in-progress.

#![allow(dead_code)]

use crate::quantized_nn::RmsNorm;
use candle::quantized::gguf_file;
use candle::quantized::QTensor;
use candle::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};

const MAX_SEQ_LEN: usize = 131072;

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

/// SwiGLU-style MLP but with **gelu_pytorch_tanh** (the gemma activation); `quantized_gemma3`'s
/// template uses silu, which is wrong for gemma-4.
#[derive(Debug, Clone)]
struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(xs)?.gelu()?; // candle's gelu() is the tanh approximation
        let up = self.up.forward(xs)?;
        self.down.forward(&(gate * up)?)
    }
}

/// Per-layer-input (PLE) projection bits — present on every layer (`inp_gate`/`proj`/`post_norm`).
#[derive(Debug, Clone)]
struct PerLayerInput {
    input_gate: QMatMul,
    projection: QMatMul,
    post_norm: RmsNorm,
}

/// On non-shared layers K/V is computed from the layer's own weights; shared layers reuse the
/// donor layer's K/V (the GGUF doesn't even carry k/v/k_norm for them).
#[derive(Debug, Clone)]
enum KvSource {
    Compute {
        wk: QMatMul,
        wv: QMatMul,
        k_norm: RmsNorm,
    },
    Shared,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    // TODO: global layers use *proportional* rope (partial_rotary_factor 0.25 + NoPE zero-padding);
    // this is the plain form (correct for the sliding layers). Port from gemma4/text.rs.
    fn new(dim: usize, freq_base: f32, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / freq_base.powf(i as f32 / dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self {
            sin: idx_theta.sin()?,
            cos: idx_theta.cos()?,
        })
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    wq: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    kv: KvSource,

    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm, // GGUF `ffn_norm`
    post_feedforward_layernorm: RmsNorm, // GGUF `post_ffw_norm`

    mlp: Mlp,
    per_layer: PerLayerInput,
    post_per_layer_input_norm: RmsNorm, // GGUF `post_norm`
    layer_scalar: Tensor,               // GGUF `layer_output_scale`

    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    is_global: bool,
    is_shared: bool,
    sliding_window: Option<usize>,

    rotary: RotaryEmbedding,
    kv_cache: Option<(Tensor, Tensor)>,
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    per_layer_token_embeddings: Embedding, // GGUF `per_layer_token_embd` (PLE table)
    per_layer_model_projection: QMatMul,   // GGUF `per_layer_model_proj` (F16)
    per_layer_projection_norm: RmsNorm,    // GGUF `per_layer_proj_norm`

    embedding_length: usize,
    per_layer_input_dim: usize, // `embedding_length_per_layer_input` (256)

    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul, // tied to `token_embd`
    final_logit_softcapping: f64,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| {
            let key = format!("gemma4.{s}");
            match ct.metadata.get(&key) {
                None => candle::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };

        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let per_layer_input_dim = md_get("embedding_length_per_layer_input")?.to_u32()? as usize;
        let key_length = md_get("attention.key_length")?.to_u32()? as usize; // 512 (global)
        let key_length_swa = md_get("attention.key_length_swa")?.to_u32()? as usize; // 256 (sliding)
        let shared_kv_layers = md_get("attention.shared_kv_layers")?.to_u32()? as usize;
        let rms_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window = md_get("attention.sliding_window")?.to_u32()? as usize;
        let rope_freq_base = md_get("rope.freq_base")?.to_f32()?; // 1e6 (global)
        let rope_freq_base_swa = md_get("rope.freq_base_swa")?.to_f32()?; // 1e4 (sliding)
        let final_logit_softcapping = md_get("final_logit_softcapping")?.to_f32()? as f64;

        let first_shared = block_count.saturating_sub(shared_kv_layers); // 35 - 20 = 15

        // --- non-layer tensors ---
        let tok_embeddings = ct
            .tensor(reader, "token_embd.weight", device)?
            .dequantize(device)?;
        let tok_embeddings = Embedding::new(tok_embeddings, embedding_length);

        let plte = ct
            .tensor(reader, "per_layer_token_embd.weight", device)?
            .dequantize(device)?;
        let per_layer_token_embeddings = Embedding::new(plte, per_layer_input_dim * block_count);

        let per_layer_model_projection =
            QMatMul::from_qtensor(ct.tensor(reader, "per_layer_model_proj.weight", device)?)?;
        let per_layer_projection_norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "per_layer_proj_norm.weight", device)?,
            rms_eps,
        )?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_eps,
        )?;
        // Tied output (no `output.weight` in the gemma-4 GGUF).
        let output = QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?;

        // --- per-layer ---
        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let p = format!("blk.{i}");
            let is_shared = i >= first_shared;
            // global = full-attention layers (sliding_window_pattern==false). E2B: every 5th.
            // TODO: read `gemma4.attention.sliding_window_pattern` for robustness across variants.
            let is_global = (i + 1) % 5 == 0;
            let head_dim = if is_global { key_length } else { key_length_swa };

            let wq = QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_q.weight"), device)?)?;
            let wo = QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{p}.attn_output.weight"), device)?,
            )?;
            let q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.attn_q_norm.weight"), device)?,
                rms_eps,
            )?;

            let kv = if is_shared {
                KvSource::Shared
            } else {
                KvSource::Compute {
                    wk: QMatMul::from_qtensor(
                        ct.tensor(reader, &format!("{p}.attn_k.weight"), device)?,
                    )?,
                    wv: QMatMul::from_qtensor(
                        ct.tensor(reader, &format!("{p}.attn_v.weight"), device)?,
                    )?,
                    k_norm: RmsNorm::from_qtensor(
                        ct.tensor(reader, &format!("{p}.attn_k_norm.weight"), device)?,
                        rms_eps,
                    )?,
                }
            };

            let input_layernorm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.attn_norm.weight"), device)?,
                rms_eps,
            )?;
            let post_attention_layernorm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.post_attention_norm.weight"), device)?,
                rms_eps,
            )?;
            let pre_feedforward_layernorm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.ffn_norm.weight"), device)?,
                rms_eps,
            )?;
            let post_feedforward_layernorm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.post_ffw_norm.weight"), device)?,
                rms_eps,
            )?;

            let mlp = Mlp {
                gate: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.ffn_gate.weight"), device)?,
                )?,
                up: QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.ffn_up.weight"), device)?)?,
                down: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.ffn_down.weight"), device)?,
                )?,
            };

            let per_layer = PerLayerInput {
                input_gate: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.inp_gate.weight"), device)?,
                )?,
                projection: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.proj.weight"), device)?,
                )?,
                post_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.post_norm.weight"), device)?,
                    rms_eps,
                )?,
            };
            let post_per_layer_input_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.post_norm.weight"), device)?,
                rms_eps,
            )?;
            let layer_scalar = ct
                .tensor(reader, &format!("{p}.layer_output_scale.weight"), device)?
                .dequantize(device)?;

            let freq = if is_global {
                rope_freq_base
            } else {
                rope_freq_base_swa
            };
            let rotary = RotaryEmbedding::new(head_dim, freq, device)?;

            layers.push(LayerWeights {
                wq,
                wo,
                q_norm,
                kv,
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
                mlp,
                per_layer,
                post_per_layer_input_norm,
                layer_scalar,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                is_global,
                is_shared,
                sliding_window: if is_global { None } else { Some(sliding_window) },
                rotary,
                kv_cache: None,
            });
        }

        Ok(Self {
            tok_embeddings,
            per_layer_token_embeddings,
            per_layer_model_projection,
            per_layer_projection_norm,
            embedding_length,
            per_layer_input_dim,
            layers,
            norm,
            output,
            final_logit_softcapping,
        })
    }

    pub fn forward(&mut self, _x: &Tensor, _index_pos: usize) -> Result<Tensor> {
        // TODO: port the (parity-validated) forward from `crate::models::gemma4::text`:
        //   1. embed: tok_embeddings(x) * sqrt(embedding_length). Build the PLE per-layer inputs
        //      = (per_layer_token_embeddings(x)  +  proj_norm(per_layer_model_projection(embeds)))
        //        * (1/sqrt(2)), reshaped to [.., block_count, per_layer_input_dim].
        //   2. per layer i:
        //        residual = h
        //        h = input_layernorm(h); attn:
        //          q = q_norm(wq(h)); for !is_shared compute k=k_norm(wk(h)), v=wv(h) and (if it's
        //          the donor — last non-shared layer of this type) stash into shared_kv_states by
        //          layer type; for is_shared reuse the stashed donor K/V. rope per head_dim/freq,
        //          GQA repeat, sliding-vs-global mask, softmax (f32), o = wo(...).
        //        h = post_attention_layernorm(attn) + residual
        //        residual = h; h = post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
        //          + residual            (mlp is double-wide on is_shared — already in the weights)
        //        PLE mix: g = input_gate(per_layer_input[i]); h = post_per_layer_input_norm(
        //          projection(h * activation(g)) ) ...  (see gemma4/text.rs `pli_mixer` for exact form)
        //        h = h * layer_scalar
        //   3. h = norm(h); logits = output(h[last]); softcap: tanh(logits / cap) * cap.
        todo!("port gemma4/text.rs forward (PLE + shared-KV + softcap)")
    }
}
