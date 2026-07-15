//! Gemma 4 (E-models) quantized GGUF implementation.
//!
//! Mirrors the parity-validated float [`crate::models::gemma4::text`] with `QMatMul`, loading the
//! official `gemma4` GGUF (the arch llama.cpp ships). Scaffolded from [`super::quantized_gemma3`];
//! the E-model machinery added on top is: PLE (per-layer embeddings), shared-KV (layers
//! `>= block_count - shared_kv_layers`), double-wide MLP on shared layers, a per-layer
//! `layer_scalar`, per-layer head_dim (global vs sliding), proportional rope on global layers,
//! `final_logit_softcapping`, and a gelu-tanh MLP (NOT silu).
//!
//! The dense variants (12B/31B) carry none of the PLE/shared-KV/double-wide machinery but add
//! `attention_k_eq_v` (global layers have no `attn_v` tensor — V shares K's projection) and a
//! per-layer `head_count_kv` array. Both shapes load from the same metadata/tensor probing.
//!
//! The two embedding tables (`token_embd`, `per_layer_token_embd`) stay packed as `QTensor` and
//! are looked up via [`QTensor::embedding`] — dequantizing the PLE table (~2.3B params at Q6_K)
//! would cost ~9 GB in f32 and defeat the point of the quantized model.
//!
//! KV-cache state lives outside the weights ([`Gemma4KvCache`]): [`ModelWeights::forward`] runs
//! a single internal lane, while [`ModelWeights::forward_with_cache`] lets a caller own any
//! number of independent lanes over one set of weights (multi-session inference; forking a lane
//! is an O(1) clone).
//!
//! Based on the HuggingFace `transformers` gemma4 implementation (Apache-2.0) and eddyb's #3608.
//! See `.sandpiper/docs/gemma4-candle-gguf-plan.md` for the full tensor/metadata map.

use crate::quantized_nn::RmsNorm;
use candle::quantized::gguf_file;
use candle::quantized::{QStorage, QTensor};
use candle::{DType, Device, Result, Tensor};
use candle_nn::Module;
use std::borrow::Cow;
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

/// Per-layer-input (PLE) mixer bits — on E-models, present on every layer (`inp_gate`/`proj`/
/// `post_norm`); absent on the dense variants (12B/31B).
#[derive(Debug, Clone)]
struct PerLayerInput {
    input_gate: QMatMul,
    projection: QMatMul,
    post_norm: RmsNorm,
}

/// The PLE model-level pieces (E-models only): the per-layer token-embedding table and the
/// context-aware projection that together form each layer's per-layer input.
#[derive(Debug, Clone)]
struct PerLayerEmbeddings {
    token_embeddings: Arc<QTensor>, // GGUF `per_layer_token_embd` (packed; rows dequant on lookup)
    model_projection: QMatMul,      // GGUF `per_layer_model_proj` (F16)
    projection_norm: RmsNorm,       // GGUF `per_layer_proj_norm`
    input_dim: usize,               // `embedding_length_per_layer_input` (256 on E-models)
}

/// On non-shared layers K/V is computed from the layer's own weights; shared layers reuse the
/// donor layer's K/V (the GGUF doesn't even carry k/v/k_norm for them). `wv: None` is the dense
/// variants' `attention_k_eq_v` on global layers: no `attn_v` tensor, V shares K's projection.
#[derive(Debug, Clone)]
enum KvSource {
    Compute {
        wk: QMatMul,
        wv: Option<QMatMul>,
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

// ── MoE (26B-A4B) — mirrors the float `gemma4::text` MoE, quantized ────────────────────────────

/// Split a stacked-expert 3D `QTensor` `[E, rows, cols]` into one `QMatMul` per expert. Block
/// quantization is row-contiguous, so each expert is a contiguous slice of the raw bytes; the
/// experts stay packed (dequantizing the 26B's expert slabs would cost ~60 GB f32). The CUDA-only
/// `moe_gemm_gguf`/`indexed_moe_forward` primitives can replace this on GPU later.
fn split_experts(t: &QTensor, device: &Device) -> Result<Vec<QMatMul>> {
    let (num_experts, rows, cols) = t.shape().dims3()?;
    let dtype = t.dtype();
    let data = t.data()?;
    if !data.len().is_multiple_of(num_experts) {
        candle::bail!(
            "expert tensor bytes ({}) not divisible by expert count ({num_experts})",
            data.len()
        )
    }
    let bytes_per_expert = data.len() / num_experts;
    (0..num_experts)
        .map(|e| {
            let slice = &data[e * bytes_per_expert..(e + 1) * bytes_per_expert];
            let storage = QStorage::from_data(Cow::Borrowed(slice), device, dtype)?;
            QMatMul::from_qtensor(QTensor::new(storage, (rows, cols))?)
        })
        .collect()
}

/// Expert router: weightless RMS norm → per-dim scale → `hidden^-0.5` → scores → f32 softmax →
/// top-k → renormalize → per-expert scale. Top-k selection runs host-side (E=128, n small).
/// GGUF mapping: proj = `ffn_gate_inp.weight` [E, H] (F32), scale = `ffn_gate_inp.scale` [H],
/// per-expert scale = `ffn_down_exps.scale` [E].
#[derive(Debug, Clone)]
struct Router {
    proj: Tensor,  // [num_experts, hidden] f32
    scale: Tensor, // [hidden] f32
    per_expert_scale: Vec<f32>,
    eps: f64,
    hidden_size: usize,
    top_k: usize,
}

impl Router {
    /// `xs`: flat `[n, hidden]`. Returns `n` vecs of `top_k` `(expert, weight)` picks.
    fn forward(&self, xs: &Tensor) -> Result<Vec<Vec<(usize, f32)>>> {
        let h = v_norm(xs, self.eps)?;
        let h = h.broadcast_mul(&self.scale)?;
        let h = (h * (self.hidden_size as f64).powf(-0.5))?;
        let scores = h.matmul(&self.proj.t()?)?.to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let probs: Vec<Vec<f32>> = probs.to_vec2()?;
        Ok(probs
            .iter()
            .map(|row| {
                let mut order: Vec<usize> = (0..row.len()).collect();
                order.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));
                let top = &order[..self.top_k];
                let sum: f32 = top.iter().map(|&e| row[e]).sum();
                top.iter()
                    .map(|&e| (e, row[e] / sum * self.per_expert_scale[e]))
                    .collect()
            })
            .collect())
    }
}

/// Per-expert weights: `gate_up[e]` `[2I, H]` (gate/up fused; output chunks in half) and
/// `down[e]` `[H, I]`. Per-token top-k loop — fine for one-shot validation and 1-token decode.
#[derive(Debug, Clone)]
struct Experts {
    gate_up: Vec<QMatMul>,
    down: Vec<QMatMul>,
}

impl Experts {
    fn forward(&self, xs: &Tensor, routes: &[Vec<(usize, f32)>]) -> Result<Tensor> {
        let mut out_rows = Vec::with_capacity(routes.len());
        for (n, picks) in routes.iter().enumerate() {
            let x = xs.narrow(0, n, 1)?; // [1, hidden]
            let mut acc: Option<Tensor> = None;
            for &(e, w) in picks {
                let y = self.gate_up[e].forward(&x)?; // [1, 2I]
                let i = y.dim(1)? / 2;
                let gate = y.narrow(1, 0, i)?.gelu()?;
                let up = y.narrow(1, i, i)?;
                let z = self.down[e].forward(&(gate * up)?)?; // [1, hidden]
                let z = (z * w as f64)?;
                acc = Some(match acc {
                    None => z,
                    Some(a) => (a + z)?,
                });
            }
            out_rows.push(acc.expect("top_k >= 1"));
        }
        Tensor::cat(&out_rows, 0)
    }
}

/// The MoE runs in PARALLEL with the dense MLP, both fed from the pre-MLP residual:
/// `post_ffw_norm_1(mlp) + post_ffw_norm_2(experts(pre_ffw_norm_2(residual)))`.
#[derive(Debug, Clone)]
struct MoeBlock {
    router: Router,
    experts: Experts,
    post_ffw_norm_1: RmsNorm,
    post_ffw_norm_2: RmsNorm,
    pre_ffw_norm_2: RmsNorm,
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
    moe: Option<MoeBlock>,
    per_layer: Option<PerLayerInput>,
    layer_scalar: Tensor, // GGUF `layer_output_scale`

    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rms_eps: f64,
    is_global: bool,
    /// iSWA ring bound for sliding layers (`None` on global layers): the cache lane retains only
    /// the last `window` positions, since the model defines these layers to attend nothing older.
    window: Option<usize>,
    /// This layer is the last non-shared layer of its type: stash its K/V for the shared layers.
    store_shared_kv: bool,

    rotary: RotaryEmbedding,

    span_attn: tracing::Span,
    span_mlp: tracing::Span,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        cache_slot: &mut Option<(Tensor, Tensor)>,
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
                // `attention_k_eq_v` (dense variants, global layers): V shares K's projection
                // output — taken RAW, before k_norm/rope (only v_norm applies), as in the float path.
                let v = match wv {
                    Some(wv) => wv.forward(x)?,
                    None => k.clone(),
                };
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

                let (k, v) = match cache_slot.as_ref() {
                    Some((k_cache, v_cache)) => {
                        let k = Tensor::cat(&[k_cache, &k], 2)?;
                        let v = Tensor::cat(&[v_cache, &v], 2)?;
                        (k, v)
                    }
                    None => (k, v),
                };
                // iSWA ring: retain only the last `window` positions on sliding layers — a query
                // at position q attends the cached q-w..q-1 plus itself, exactly the set the
                // additive mask admits, so every dropped position was already masked for all
                // future queries. Global layers retain everything.
                *cache_slot = Some(match self.window {
                    Some(w) if k.dim(2)? > w => {
                        let len = k.dim(2)?;
                        (
                            k.narrow(2, len - w, w)?.contiguous()?,
                            v.narrow(2, len - w, w)?.contiguous()?,
                        )
                    }
                    _ => (k.clone(), v.clone()),
                });

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
/// `index_pos + i`; the `kv_len` key columns are the last `kv_len` positions ending at the final
/// query (column `c` is absolute position `index_pos + tgt_len - kv_len + c`) — the whole
/// history on global layers, only the ring-retained tail on sliding layers. At `index_pos == 0`
/// with full `kv_len` this is exactly the float path's `prepare_decoder_attention_mask`. The
/// window rule admits `q - w..=q` (window + self); ring retention of the last `w` positions
/// preserves exactly that set across steps.
fn causal_mask(
    b_sz: usize,
    tgt_len: usize,
    index_pos: usize,
    kv_len: usize,
    sliding_window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let key_start = index_pos + tgt_len - kv_len;
    let mask: Vec<f32> = (0..tgt_len)
        .flat_map(|i| {
            let q = index_pos + i;
            (0..kv_len).map(move |c| {
                let j = key_start + c;
                let masked = j > q || sliding_window.is_some_and(|w| j + w < q);
                if masked {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (tgt_len, kv_len), device)?.expand((b_sz, 1, tgt_len, kv_len))
}

/// Externalized per-layer KV cache for [`ModelWeights`].
///
/// Holds one `(k, v)` pair per layer that computes its own K/V (shared-KV layers reuse their
/// donor's entries and stay `None`), plus the stream position the cache has consumed up to.
/// Callers that only use [`ModelWeights::forward`] never touch this type — the model owns an
/// internal lane. Owning caches externally via [`ModelWeights::forward_with_cache`] lets one set
/// of weights serve multiple independent sequences.
///
/// Cloning is O(1): tensors are `Arc`-backed and the cache is append-only (no in-place tensor
/// mutation), so a clone shares storage with its source and the two diverge naturally on their
/// next appends.
#[derive(Debug, Clone, Default)]
pub struct Gemma4KvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
    pos: usize,
}

impl Gemma4KvCache {
    /// The next stream position this cache expects (= number of positions consumed so far).
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Override the position bookkeeping. Only needed by callers that edit [`Self::layers_mut`]
    /// directly and must keep the position consistent with their surgery.
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos
    }

    /// Drop all cached K/V and rewind to position 0.
    pub fn reset(&mut self) {
        self.pos = 0;
        for slot in self.layers.iter_mut() {
            *slot = None
        }
    }

    /// Per-layer cache entries, indexed by layer. `None` on shared-KV layers (and everywhere
    /// before the first forward). K and V are shaped `(b, n_kv_head, seq, head_dim)`.
    pub fn layers(&self) -> &[Option<(Tensor, Tensor)>] {
        &self.layers
    }

    /// Mutable per-layer access for cache surgery (external eviction policies etc.). Callers are
    /// responsible for keeping [`Self::set_pos`] consistent with what they leave behind.
    pub fn layers_mut(&mut self) -> &mut [Option<(Tensor, Tensor)>] {
        &mut self.layers
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Arc<QTensor>, // GGUF `token_embd` (also the tied output)
    ple: Option<PerLayerEmbeddings>, // E-models only; dense variants (12B/31B) have no PLE

    embedding_length: usize,
    sliding_window: usize,

    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul, // tied to `token_embd`
    final_logit_softcapping: Option<f64>,

    /// The internal lane used by [`Self::forward`]; external lanes go through
    /// [`Self::forward_with_cache`].
    cache: Gemma4KvCache,

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
        let block_count = md_get("block_count")?.to_u32()? as usize;
        // Scalar on the E-models; a per-layer array on the dense variants (8 sliding / 1 global).
        // Array elements are written as i32 by the converter, so accept either signedness.
        let as_count = |x: &gguf_file::Value| -> Result<usize> {
            match x.to_u32() {
                Ok(n) => Ok(n as usize),
                Err(_) => Ok(x.to_i32()? as usize),
            }
        };
        let head_count_kv: Vec<usize> = {
            let v = md_get("attention.head_count_kv")?;
            match v.to_vec() {
                Ok(arr) => arr.iter().map(as_count).collect::<Result<_>>()?,
                Err(_) => vec![as_count(v)?; block_count],
            }
        };
        if head_count_kv.len() != block_count {
            candle::bail!(
                "head_count_kv has {} entries for {block_count} layers",
                head_count_kv.len()
            )
        }
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        // 0 (or absent) = no PLE — the dense variants (12B/31B).
        let per_layer_input_dim = md_get("embedding_length_per_layer_input")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        // MoE (26B-A4B): expert metadata present = MoE model.
        let num_experts = md_get("expert_count")
            .and_then(|v| v.to_u32())
            .ok()
            .map(|v| v as usize);
        let top_k_experts = md_get("expert_used_count")
            .and_then(|v| v.to_u32())
            .ok()
            .map(|v| v as usize);
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
        let ple = if per_layer_input_dim > 0 {
            Some(PerLayerEmbeddings {
                token_embeddings: Arc::new(ct.tensor(
                    reader,
                    "per_layer_token_embd.weight",
                    device,
                )?),
                model_projection: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    "per_layer_model_proj.weight",
                    device,
                )?)?,
                projection_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, "per_layer_proj_norm.weight", device)?,
                    rms_eps,
                )?,
                input_dim: per_layer_input_dim,
            })
        } else {
            None
        };
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
                // `attention_k_eq_v` layers (dense variants, global attention) carry no attn_v
                // tensor: detect by presence rather than a config key (the GGUF has none).
                let v_name = format!("{p}.attn_v.weight");
                let wv = if ct.tensor_infos.contains_key(&v_name) {
                    Some(QMatMul::from_qtensor(ct.tensor(reader, &v_name, device)?)?)
                } else {
                    None
                };
                KvSource::Compute {
                    wk: QMatMul::from_qtensor(ct.tensor(
                        reader,
                        &format!("{p}.attn_k.weight"),
                        device,
                    )?)?,
                    wv,
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

            let per_layer = if per_layer_input_dim > 0 {
                Some(PerLayerInput {
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
                })
            } else {
                None
            };
            let layer_scalar = ct
                .tensor(reader, &format!("{p}.layer_output_scale.weight"), device)?
                .dequantize(device)?;

            let moe = if let (Some(num_experts), Some(top_k)) = (num_experts, top_k_experts) {
                let gate_up_3d =
                    ct.tensor(reader, &format!("{p}.ffn_gate_up_exps.weight"), device)?;
                let down_3d = ct.tensor(reader, &format!("{p}.ffn_down_exps.weight"), device)?;
                let proj = ct
                    .tensor(reader, &format!("{p}.ffn_gate_inp.weight"), device)?
                    .dequantize(device)?;
                let scale = ct
                    .tensor(reader, &format!("{p}.ffn_gate_inp.scale"), device)?
                    .dequantize(device)?;
                let per_expert_scale = ct
                    .tensor(reader, &format!("{p}.ffn_down_exps.scale"), device)?
                    .dequantize(device)?
                    .to_vec1::<f32>()?;
                if per_expert_scale.len() != num_experts {
                    candle::bail!(
                        "per-expert scale has {} entries for {num_experts} experts",
                        per_expert_scale.len()
                    )
                }
                Some(MoeBlock {
                    router: Router {
                        proj,
                        scale,
                        per_expert_scale,
                        eps: rms_eps,
                        hidden_size: embedding_length,
                        top_k,
                    },
                    experts: Experts {
                        gate_up: split_experts(&gate_up_3d, device)?,
                        down: split_experts(&down_3d, device)?,
                    },
                    post_ffw_norm_1: RmsNorm::from_qtensor(
                        ct.tensor(reader, &format!("{p}.post_ffw_norm_1.weight"), device)?,
                        rms_eps,
                    )?,
                    post_ffw_norm_2: RmsNorm::from_qtensor(
                        ct.tensor(reader, &format!("{p}.post_ffw_norm_2.weight"), device)?,
                        rms_eps,
                    )?,
                    pre_ffw_norm_2: RmsNorm::from_qtensor(
                        ct.tensor(reader, &format!("{p}.pre_ffw_norm_2.weight"), device)?,
                        rms_eps,
                    )?,
                })
            } else {
                None
            };

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
                moe,
                per_layer,
                layer_scalar,
                n_head: head_count,
                n_kv_head: head_count_kv[i],
                head_dim,
                rms_eps,
                is_global,
                window: (!is_global).then_some(sliding_window),
                store_shared_kv,
                rotary,
                span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                span_mlp: tracing::span!(tracing::Level::TRACE, "attn-mlp"),
            });
        }

        Ok(Self {
            tok_embeddings,
            ple,
            embedding_length,
            sliding_window,
            layers,
            norm,
            output,
            final_logit_softcapping,
            cache: Gemma4KvCache::default(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }

    /// The full and sliding additive masks for this step (None where no position is masked).
    /// Single-token steps never need a mask: no future position exists, and on sliding layers
    /// the ring retention guarantees every cached position is inside the window.
    fn masks(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        device: &Device,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let full = if seq_len > 1 {
            Some(causal_mask(
                b_sz,
                seq_len,
                index_pos,
                index_pos + seq_len,
                None,
                device,
            )?)
        } else {
            None
        };
        let sliding = if seq_len > 1 {
            let kv_len = index_pos.min(self.sliding_window) + seq_len;
            Some(causal_mask(
                b_sz,
                seq_len,
                index_pos,
                kv_len,
                Some(self.sliding_window),
                device,
            )?)
        } else {
            None
        };
        Ok((full, sliding))
    }

    /// Forward on the model's internal cache lane (single-sequence use). For multiple sequences
    /// over one set of weights, use [`Self::forward_with_cache`] with caller-owned lanes.
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let mut cache = std::mem::take(&mut self.cache);
        let res = self.forward_with_cache(x, index_pos, &mut cache);
        self.cache = cache;
        res
    }

    /// Forward against a caller-owned cache lane. `index_pos == 0` resets the lane (a fresh
    /// prefill); any other `index_pos` must equal [`Gemma4KvCache::pos`] — decoding at a drifted
    /// position is a caller bug and fails loudly rather than silently mis-attending.
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &mut Gemma4KvCache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len) = x.dims2()?;

        if cache.layers.is_empty() {
            cache.layers = vec![None; self.layers.len()];
        } else if cache.layers.len() != self.layers.len() {
            candle::bail!(
                "cache has {} layers but the model has {}",
                cache.layers.len(),
                self.layers.len()
            )
        }
        if index_pos == 0 {
            cache.reset();
        } else if index_pos != cache.pos {
            candle::bail!(
                "cache position drift: forward at index_pos {index_pos} but the cache is at {}",
                cache.pos
            )
        }

        let xs = self.tok_embeddings.embedding(x)?;
        let xs = (xs * (self.embedding_length as f64).sqrt())?;

        // PLE per-layer inputs (E-models only): (context-aware projection + per-layer token
        // embedding) / sqrt(2), shaped (b, seq, n_layers, input_dim). Mirrors the float
        // forward_embeds; None on the dense variants.
        let n_layers = self.layers.len();
        let per_layer_inputs = self
            .ple
            .as_ref()
            .map(|ple| -> Result<Tensor> {
                let per_layer_projection = (ple.model_projection.forward(&xs)?
                    * (1.0 / (self.embedding_length as f64).sqrt()))?;
                let per_layer_projection = per_layer_projection
                    .reshape((b_sz, seq_len, n_layers, ple.input_dim))?
                    .apply(&ple.projection_norm)?;
                let per_layer_embeds = (ple.token_embeddings.embedding(x)?
                    * (ple.input_dim as f64).sqrt())?
                .reshape((b_sz, seq_len, n_layers, ple.input_dim))?;
                (per_layer_projection + per_layer_embeds)? * (1.0 / 2.0f64.sqrt())
            })
            .transpose()?;

        let (full_mask, sliding_mask) = self.masks(b_sz, seq_len, index_pos, x.device())?;

        let mut shared_kv_states = SharedKvStates::default();
        let mut xs = xs;
        for (i, (layer, cache_slot)) in self
            .layers
            .iter()
            .zip(cache.layers.iter_mut())
            .enumerate()
        {
            let mask = if layer.is_global {
                full_mask.as_ref()
            } else {
                sliding_mask.as_ref()
            };

            let residual = &xs;
            let x1 = layer.input_layernorm.forward(&xs)?;
            let x1 = layer.forward_attn(&x1, mask, index_pos, cache_slot, &mut shared_kv_states)?;
            let x1 = layer.post_attention_layernorm.forward(&x1)?;
            let xs_attn = (x1 + residual)?;

            let _mlp_enter = layer.span_mlp.enter();
            let residual = &xs_attn;
            let x2 = layer.pre_feedforward_layernorm.forward(&xs_attn)?;
            let x2 = layer.mlp.forward(&x2)?;
            // MoE runs in parallel with the dense MLP, both branches fed from the pre-MLP
            // residual, combined before the shared post-feedforward norm.
            let x2 = match &layer.moe {
                None => x2,
                Some(moe) => {
                    let h1 = moe.post_ffw_norm_1.forward(&x2)?;
                    let flat = residual.reshape((b_sz * seq_len, self.embedding_length))?;
                    let routes = moe.router.forward(&flat)?;
                    let h2 = moe.pre_ffw_norm_2.forward(&flat)?;
                    let h2 = moe.experts.forward(&h2, &routes)?;
                    let h2 = h2.reshape((b_sz, seq_len, self.embedding_length))?;
                    let h2 = moe.post_ffw_norm_2.forward(&h2)?;
                    (h1 + h2)?
                }
            };
            let x2 = layer.post_feedforward_layernorm.forward(&x2)?;
            let xs_mlp = (residual + x2)?;
            drop(_mlp_enter);

            // PLE mix: gate the hidden state, multiply with this layer's input, project back up.
            let xs_mixed = match (&layer.per_layer, &per_layer_inputs) {
                (Some(per_layer), Some(per_layer_inputs)) => {
                    let per_layer_input = per_layer_inputs.get_on_dim(2, i)?;
                    let residual = &xs_mlp;
                    let m = per_layer.input_gate.forward(&xs_mlp)?.gelu()?;
                    let m = (m * per_layer_input)?;
                    let m = per_layer.projection.forward(&m)?;
                    let m = per_layer.post_norm.forward(&m)?;
                    (residual + m)?
                }
                (None, None) => xs_mlp,
                _ => candle::bail!("PLE state mismatch between model and layer {i}"),
            };

            xs = xs_mixed.broadcast_mul(&layer.layer_scalar)?;
        }
        cache.pos = index_pos + seq_len;

        let _enter = self.span_output.enter();
        let logits = xs.narrow(1, seq_len - 1, 1)?.apply(&self.norm)?;
        let logits = self.output.forward(&logits)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(((logits / sc)?.tanh()? * sc)?),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        self.cache.reset()
    }

    /// Whether layer `idx` uses sliding-window (iSWA) attention rather than full attention.
    /// Sliding layers' cache lanes are ring-bounded at [`Self::sliding_window`] positions.
    pub fn is_sliding(&self, idx: usize) -> bool {
        self.layers.get(idx).is_some_and(|l| !l.is_global)
    }

    /// The iSWA attention window of the sliding layers, in positions.
    pub fn sliding_window(&self) -> usize {
        self.sliding_window
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::IndexOp;

    fn grid(mask: &Tensor) -> Result<Vec<Vec<f32>>> {
        mask.i((0, 0))?.to_vec2::<f32>()
    }

    const F: f32 = f32::NEG_INFINITY;

    #[test]
    fn full_mask_fresh_prefill_is_lower_triangular() -> Result<()> {
        let m = causal_mask(1, 3, 0, 3, None, &Device::Cpu)?;
        assert_eq!(
            grid(&m)?,
            vec![vec![0., F, F], vec![0., 0., F], vec![0., 0., 0.]]
        );
        Ok(())
    }

    #[test]
    fn full_mask_continued_chunk_sees_all_cached_positions() -> Result<()> {
        // index_pos 2, two queries (q=2, q=3), full history (kv_len 4).
        let m = causal_mask(1, 2, 2, 4, None, &Device::Cpu)?;
        assert_eq!(grid(&m)?, vec![vec![0., 0., 0., F], vec![0., 0., 0., 0.]]);
        Ok(())
    }

    #[test]
    fn sliding_mask_admits_window_plus_self() -> Result<()> {
        // w=2, fresh prefill of 4: query q admits absolute q-2..=q.
        let m = causal_mask(1, 4, 0, 4, Some(2), &Device::Cpu)?;
        assert_eq!(
            grid(&m)?,
            vec![
                vec![0., F, F, F],
                vec![0., 0., F, F],
                vec![0., 0., 0., F],
                vec![F, 0., 0., 0.],
            ]
        );
        Ok(())
    }

    #[test]
    fn sliding_mask_ring_trimmed_kv_maps_columns_to_absolute_positions() -> Result<()> {
        // w=3, index_pos 5, two queries (q=5, q=6); ring retained 3 cached positions, so the 5
        // kv columns are absolute positions 2..=6.
        let m = causal_mask(1, 2, 5, 5, Some(3), &Device::Cpu)?;
        assert_eq!(
            grid(&m)?,
            vec![vec![0., 0., 0., 0., F], vec![F, 0., 0., 0., 0.]]
        );
        Ok(())
    }
}
