//! Gemma 4 (E-models) quantized GGUF implementation.
//!
//! Mirrors the parity-validated float [`crate::models::gemma4::text`] with `QMatMul`, loading the
//! official `gemma4` GGUF (the arch llama.cpp ships). Scaffolded from [`super::quantized_gemma3`];
//! the E-model machinery added on top is: PLE (per-layer embeddings), shared-KV (layers
//! `>= block_count - shared_kv_layers`), double-wide MLP on shared layers, a per-layer
//! `layer_scalar`, per-layer head_dim (global vs sliding), proportional rope on global layers,
//! `final_logit_softcapping`, and a gelu-tanh MLP (NOT silu).
//!
//! The two embedding tables (`token_embd`, `per_layer_token_embd`) stay packed as `QTensor` and
//! are looked up via [`QTensor::embedding`] — dequantizing the PLE table (~2.3B params at Q6_K)
//! would cost ~9 GB in f32 and defeat the point of the quantized model.
//!
//! Based on the HuggingFace `transformers` gemma4 implementation (Apache-2.0) and eddyb's #3608.
//! See `.sandpiper/docs/gemma4-candle-gguf-plan.md` for the full tensor/metadata map.

use crate::quantized_nn::RmsNorm;
use candle::quantized::gguf_file;
use candle::quantized::QTensor;
use candle::{DType, Device, Result, Tensor};
use candle_nn::Module;
use std::sync::Arc;

const MAX_SEQ_LEN: usize = 131072;
/// The fraction of global-layer head dims that rotate (the rest are NoPE / identity). Matches the
/// float path's `Gemma4TextConfig::partial_rotary_factor` default; the GGUF carries no key for it.
const PARTIAL_ROTARY_FACTOR: f64 = 0.25;

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_arc(Arc::new(qtensor))
    }

    fn from_arc(qtensor: Arc<QTensor>) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_arc(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

/// SwiGLU-style MLP but with **gelu_pytorch_tanh** (the gemma activation); `quantized_gemma3`'s
/// template uses silu, which is wrong for gemma-4. Double-wide on shared layers is implicit in
/// the loaded tensor shapes.
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

/// Per-layer-input (PLE) mixer bits — present on every layer (`inp_gate`/`proj`/`post_norm`).
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

/// The donor K/V stashed during a forward pass, one slot per layer type. Shared sliding layers
/// reuse the last non-shared sliding layer's K/V; shared full layers the last non-shared full one.
#[derive(Default)]
struct SharedKvStates {
    for_full: Option<(Tensor, Tensor)>,
    for_sliding: Option<(Tensor, Tensor)>,
}

/// Pure RMS normalization without learned weight (used for V norm), as in the float path.
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    let v_f32 = v.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(candle::D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32.broadcast_div(&rms)?.to_dtype(original_dtype)
}

/// One sin/cos table serving both rope variants: `rope_angles == half_dim` is the standard rope
/// (sliding layers); `rope_angles < half_dim` is the proportional rope (global layers) — the
/// remaining dims get inv_freq 0 → cos=1/sin=0 → identity (NoPE padding), as in the float path's
/// `ProportionalRotaryEmbedding`.
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_angles: usize, freq_base: f32, device: &Device) -> Result<Self> {
        let half_dim = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq.push(1f32 / freq_base.powf((2 * i) as f32 / head_dim as f32));
        }
        inv_freq.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));
        let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?;
        let t = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn cos_sin(&self, index_pos: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        Ok((cos, sin))
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
    pre_feedforward_layernorm: RmsNorm,  // GGUF `ffn_norm`
    post_feedforward_layernorm: RmsNorm, // GGUF `post_ffw_norm`

    mlp: Mlp,
    per_layer: PerLayerInput,
    layer_scalar: Tensor, // GGUF `layer_output_scale`

    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rms_eps: f64,
    is_global: bool,
    /// This layer is the last non-shared layer of its type: stash its K/V for the shared layers.
    store_shared_kv: bool,

    rotary: RotaryEmbedding,
    kv_cache: Option<(Tensor, Tensor)>,

    span_attn: tracing::Span,
    span_mlp: tracing::Span,
}

impl LayerWeights {
    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        shared_kv_states: &mut SharedKvStates,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.wq.forward(x)?;
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let (cos, sin) = self.rotary.cos_sin(index_pos, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;

        let (k, v) = match &self.kv {
            KvSource::Compute { wk, wv, k_norm } => {
                let k = wk.forward(x)?;
                let v = wv.forward(x)?;
                let k = k
                    .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                    .transpose(1, 2)?;
                let v = v
                    .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                    .transpose(1, 2)?;
                let k = k_norm.forward(&k.contiguous()?)?;
                let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
                // V norm (RMS without learned weight)
                let v = v_norm(&v, self.rms_eps)?;

                let (k, v) = match &self.kv_cache {
                    Some((k_cache, v_cache)) if index_pos > 0 => {
                        let k = Tensor::cat(&[k_cache, &k], 2)?;
                        let v = Tensor::cat(&[v_cache, &v], 2)?;
                        (k, v)
                    }
                    _ => (k, v),
                };
                self.kv_cache = Some((k.clone(), v.clone()));

                if self.store_shared_kv {
                    let kv = (k.clone(), v.clone());
                    if self.is_global {
                        shared_kv_states.for_full = Some(kv);
                    } else {
                        shared_kv_states.for_sliding = Some(kv);
                    }
                }
                (k, v)
            }
            KvSource::Shared => {
                let stash = if self.is_global {
                    &shared_kv_states.for_full
                } else {
                    &shared_kv_states.for_sliding
                };
                match stash {
                    Some((k, v)) => (k.clone(), v.clone()),
                    None => candle::bail!("shared-KV layer reached before its donor layer"),
                }
            }
        };

        let k = crate::utils::repeat_kv(k, self.n_head / self.n_kv_head)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.n_head / self.n_kv_head)?.contiguous()?;

        // No 1/sqrt(head_dim) scaling: gemma-4 q/k-norms make it superfluous and the reference
        // uses scale 1.0 (parity-validated in the float path).
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = match mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        // The reference runs the attention softmax in f32 (a no-op here where everything is f32
        // already, but kept explicit to match the float path).
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?
            .to_dtype(v.dtype())?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;
        self.wo.forward(&attn_output)
    }
}

/// Additive causal mask over absolute positions. Query row `i` sits at absolute position
/// `index_pos + i`; key column `j` covers the whole cache `0..index_pos + tgt_len`. At
/// `index_pos == 0` this is exactly the float path's `prepare_decoder_attention_mask`; unlike the
/// float path (which relies on `RotatingKvCache` eviction), the sliding constraint is also applied
/// to cached positions, since our concat cache never evicts.
fn causal_mask(
    b_sz: usize,
    tgt_len: usize,
    index_pos: usize,
    sliding_window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let total = index_pos + tgt_len;
    let mask: Vec<f32> = (0..tgt_len)
        .flat_map(|i| {
            let q = index_pos + i;
            (0..total).map(move |j| {
                let masked = j > q || sliding_window.is_some_and(|w| j + w < q);
                if masked {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (tgt_len, total), device)?.expand((b_sz, 1, tgt_len, total))
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Arc<QTensor>, // GGUF `token_embd` (also the tied output)
    per_layer_token_embeddings: Arc<QTensor>, // GGUF `per_layer_token_embd` (the PLE table)
    per_layer_model_projection: QMatMul, // GGUF `per_layer_model_proj` (F16)
    per_layer_projection_norm: RmsNorm, // GGUF `per_layer_proj_norm`

    embedding_length: usize,
    per_layer_input_dim: usize, // `embedding_length_per_layer_input` (256)
    sliding_window: usize,

    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul, // tied to `token_embd`
    final_logit_softcapping: Option<f64>,

    span: tracing::Span,
    span_output: tracing::Span,
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
        let final_logit_softcapping = md_get("final_logit_softcapping")
            .and_then(|v| v.to_f32())
            .ok()
            .map(|v| v as f64);

        // True = sliding, false = full/global attention, one entry per layer.
        let is_sliding = md_get("attention.sliding_window_pattern")?
            .to_vec()?
            .iter()
            .map(|v| v.to_bool())
            .collect::<Result<Vec<bool>>>()?;
        if is_sliding.len() != block_count {
            candle::bail!(
                "sliding_window_pattern has {} entries for {block_count} layers",
                is_sliding.len()
            )
        }

        let first_shared = block_count.saturating_sub(shared_kv_layers); // 35 - 20 = 15
                                                                         // The donor of each layer type: the last non-shared layer of that type.
        let donor_sliding = (0..first_shared).filter(|&i| is_sliding[i]).max();
        let donor_full = (0..first_shared).filter(|&i| !is_sliding[i]).max();

        // --- non-layer tensors ---
        // Both embedding tables stay packed; rows dequantize on lookup via QTensor::embedding.
        let tok_embeddings = Arc::new(ct.tensor(reader, "token_embd.weight", device)?);
        let per_layer_token_embeddings =
            Arc::new(ct.tensor(reader, "per_layer_token_embd.weight", device)?);

        let per_layer_model_projection =
            QMatMul::from_qtensor(ct.tensor(reader, "per_layer_model_proj.weight", device)?)?;
        let per_layer_projection_norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "per_layer_proj_norm.weight", device)?,
            rms_eps,
        )?;
        let norm =
            RmsNorm::from_qtensor(ct.tensor(reader, "output_norm.weight", device)?, rms_eps)?;
        // Tied output (no `output.weight` in the gemma-4 GGUF).
        let output = QMatMul::from_arc(tok_embeddings.clone())?;

        // One rope table per layer type; layers share them (Tensor clones are shallow).
        let global_rope_angles = (PARTIAL_ROTARY_FACTOR * key_length as f64 / 2.0) as usize;
        let rotary_global =
            RotaryEmbedding::new(key_length, global_rope_angles, rope_freq_base, device)?;
        let rotary_sliding = RotaryEmbedding::new(
            key_length_swa,
            key_length_swa / 2,
            rope_freq_base_swa,
            device,
        )?;

        // --- per-layer ---
        let mut layers = Vec::with_capacity(block_count);
        for (i, &layer_is_sliding) in is_sliding.iter().enumerate() {
            let p = format!("blk.{i}");
            let is_shared = i >= first_shared;
            let is_global = !layer_is_sliding;
            let head_dim = if is_global {
                key_length
            } else {
                key_length_swa
            };

            let wq =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_q.weight"), device)?)?;
            let wo = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{p}.attn_output.weight"),
                device,
            )?)?;
            let q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.attn_q_norm.weight"), device)?,
                rms_eps,
            )?;

            let kv = if is_shared {
                KvSource::Shared
            } else {
                KvSource::Compute {
                    wk: QMatMul::from_qtensor(ct.tensor(
                        reader,
                        &format!("{p}.attn_k.weight"),
                        device,
                    )?)?,
                    wv: QMatMul::from_qtensor(ct.tensor(
                        reader,
                        &format!("{p}.attn_v.weight"),
                        device,
                    )?)?,
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
                gate: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_gate.weight"),
                    device,
                )?)?,
                up: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_up.weight"),
                    device,
                )?)?,
                down: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_down.weight"),
                    device,
                )?)?,
            };

            let per_layer = PerLayerInput {
                input_gate: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.inp_gate.weight"),
                    device,
                )?)?,
                projection: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.proj.weight"),
                    device,
                )?)?,
                post_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.post_norm.weight"), device)?,
                    rms_eps,
                )?,
            };
            let layer_scalar = ct
                .tensor(reader, &format!("{p}.layer_output_scale.weight"), device)?
                .dequantize(device)?;

            let rotary = if is_global {
                rotary_global.clone()
            } else {
                rotary_sliding.clone()
            };
            let store_shared_kv =
                shared_kv_layers > 0 && (Some(i) == donor_full || Some(i) == donor_sliding);

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
                layer_scalar,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rms_eps,
                is_global,
                store_shared_kv,
                rotary,
                kv_cache: None,
                span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                span_mlp: tracing::span!(tracing::Level::TRACE, "attn-mlp"),
            });
        }

        Ok(Self {
            tok_embeddings,
            per_layer_token_embeddings,
            per_layer_model_projection,
            per_layer_projection_norm,
            embedding_length,
            per_layer_input_dim,
            sliding_window,
            layers,
            norm,
            output,
            final_logit_softcapping,
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }

    /// The full and sliding additive masks for this step (None where no position is masked).
    fn masks(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        device: &Device,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let full = if seq_len > 1 {
            Some(causal_mask(b_sz, seq_len, index_pos, None, device)?)
        } else {
            None
        };
        let sliding = if seq_len > 1 || index_pos + seq_len > self.sliding_window {
            Some(causal_mask(
                b_sz,
                seq_len,
                index_pos,
                Some(self.sliding_window),
                device,
            )?)
        } else {
            None
        };
        Ok((full, sliding))
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len) = x.dims2()?;

        let xs = self.tok_embeddings.embedding(x)?;
        let xs = (xs * (self.embedding_length as f64).sqrt())?;

        // PLE per-layer inputs: (context-aware projection + per-layer token embedding) / sqrt(2),
        // shaped (b, seq, n_layers, per_layer_input_dim). Mirrors the float forward_embeds.
        let n_layers = self.layers.len();
        let per_layer_projection = (self.per_layer_model_projection.forward(&xs)?
            * (1.0 / (self.embedding_length as f64).sqrt()))?;
        let per_layer_projection = per_layer_projection
            .reshape((b_sz, seq_len, n_layers, self.per_layer_input_dim))?
            .apply(&self.per_layer_projection_norm)?;
        let per_layer_embeds = (self.per_layer_token_embeddings.embedding(x)?
            * (self.per_layer_input_dim as f64).sqrt())?
        .reshape((b_sz, seq_len, n_layers, self.per_layer_input_dim))?;
        let per_layer_inputs =
            ((per_layer_projection + per_layer_embeds)? * (1.0 / 2.0f64.sqrt()))?;

        let (full_mask, sliding_mask) = self.masks(b_sz, seq_len, index_pos, x.device())?;

        let mut shared_kv_states = SharedKvStates::default();
        let mut xs = xs;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let mask = if layer.is_global {
                full_mask.as_ref()
            } else {
                sliding_mask.as_ref()
            };

            let residual = &xs;
            let x1 = layer.input_layernorm.forward(&xs)?;
            let x1 = layer.forward_attn(&x1, mask, index_pos, &mut shared_kv_states)?;
            let x1 = layer.post_attention_layernorm.forward(&x1)?;
            let xs_attn = (x1 + residual)?;

            let _mlp_enter = layer.span_mlp.enter();
            let residual = &xs_attn;
            let x2 = layer.pre_feedforward_layernorm.forward(&xs_attn)?;
            let x2 = layer.mlp.forward(&x2)?;
            let x2 = layer.post_feedforward_layernorm.forward(&x2)?;
            let xs_mlp = (residual + x2)?;
            drop(_mlp_enter);

            // PLE mix: gate the hidden state, multiply with this layer's input, project back up.
            let per_layer_input = per_layer_inputs.get_on_dim(2, i)?;
            let residual = &xs_mlp;
            let m = layer.per_layer.input_gate.forward(&xs_mlp)?.gelu()?;
            let m = (m * per_layer_input)?;
            let m = layer.per_layer.projection.forward(&m)?;
            let m = layer.per_layer.post_norm.forward(&m)?;
            let xs_mixed = (residual + m)?;

            xs = xs_mixed.broadcast_mul(&layer.layer_scalar)?;
        }

        let _enter = self.span_output.enter();
        let logits = xs.narrow(1, seq_len - 1, 1)?.apply(&self.norm)?;
        let logits = self.output.forward(&logits)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(((logits / sc)?.tanh()? * sc)?),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_cache = None
        }
    }
}
