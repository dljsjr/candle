#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};

use candle::quantized::gguf_file;
use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_gemma4::ModelWeights;
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    #[value(name = "gemma4-E2B-it")]
    Gemma4E2b,
    #[value(name = "gemma4-E4B-it")]
    Gemma4E4b,
    #[value(name = "gemma4-12B-it")]
    Gemma4_12b,
    #[value(name = "gemma4-31B-it")]
    Gemma4_31b,
    #[value(name = "gemma4-26B-A4B-it")]
    Gemma4_26bA4b,
}

impl Which {
    /// (gguf repo, gguf filename, tokenizer repo) — the official QAT q4_0 GGUFs; the GGUF repos
    /// carry no tokenizer.json, so it comes from the float repo.
    fn hub_names(&self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::Gemma4E2b => (
                "google/gemma-4-E2B-it-qat-q4_0-gguf",
                "gemma-4-E2B_q4_0-it.gguf",
                "google/gemma-4-E2B-it",
            ),
            Self::Gemma4E4b => (
                "google/gemma-4-E4B-it-qat-q4_0-gguf",
                "gemma-4-E4B_q4_0-it.gguf",
                "google/gemma-4-E4B-it",
            ),
            Self::Gemma4_12b => (
                "google/gemma-4-12B-it-qat-q4_0-gguf",
                "gemma-4-12b-it-qat-q4_0.gguf",
                "google/gemma-4-12B-it",
            ),
            Self::Gemma4_31b => (
                "google/gemma-4-31B-it-qat-q4_0-gguf",
                "gemma-4-31B_q4_0-it.gguf",
                "google/gemma-4-31B-it",
            ),
            Self::Gemma4_26bA4b => (
                "google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
                "gemma-4-26B_q4_0-it.gguf",
                "google/gemma-4-26B-A4B-it",
            ),
        }
    }
}

struct TextGeneration {
    model: ModelWeights,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelWeights,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = {
            let temperature = temp.unwrap_or(0.);
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (top_k, top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(seed, sampling)
        };

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize, dump_topk: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        // Gemma expects a leading <bos>; the tokenizer's add_special_tokens isn't prepending it
        // here, so ensure it (matches transformers' apply_chat_template — required for parity and
        // correct generation).
        if let Some(bos) = self.tokenizer.get_token("<bos>") {
            if tokens.first() != Some(&bos) {
                tokens.insert(0, bos);
            }
        }
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<eos>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <eos> token"),
        };
        let end_of_turn_token = match self.tokenizer.get_token("<turn|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <turn|> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            if dump_topk > 0 && index == 0 {
                // Parity dump: input ids + top-K of the first-position (post-softcap) logits, f32.
                println!("INPUT_IDS: {ctxt:?}");
                let v: Vec<f32> = logits.to_vec1()?;
                let mut order: Vec<usize> = (0..v.len()).collect();
                order.sort_unstable_by(|&a, &b| v[b].total_cmp(&v[a]));
                print!("TOPK:");
                for &i in order.iter().take(dump_topk) {
                    print!(" {i}:{:.4}", v[i]);
                }
                println!();
                return Ok(());
            }
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if [eos_token, end_of_turn_token].contains(&next_token) {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{size_in_bytes}B")
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// GGUF file to load; defaults to downloading the official QAT q4_0 GGUF for `--which`.
    #[arg(long)]
    model: Option<String>,

    /// Which model variant (picks the hub GGUF and tokenizer when not given explicitly).
    #[arg(long, default_value = "gemma4-E2B-it")]
    which: Which,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Parity dump: print the prompt's input ids + the top-K first-position logits (f32), then exit.
    /// 0 = off (normal generation).
    #[arg(long, default_value_t = 0)]
    dump_topk: usize,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let (gguf_repo, gguf_filename, tokenizer_repo) = args.which.hub_names();

    let start = std::time::Instant::now();
    let model_path = match &args.model {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            api.repo(hf_hub::Repo::with_revision(
                gguf_repo.to_string(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            ))
            .get(gguf_filename)?
        }
    };
    let tokenizer_path = match &args.tokenizer_file {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            api.model(tokenizer_repo.to_string())
                .get("tokenizer.json")?
        }
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    let mut file = std::fs::File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&model_path))?;
    let mut total_size_in_bytes = 0;
    for (_, tensor) in content.tensor_infos.iter() {
        let elem_count = tensor.shape.elem_count();
        total_size_in_bytes +=
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }
    println!(
        "loaded {:?} tensors ({}) in {:.2}s",
        content.tensor_infos.len(),
        &format_size(total_size_in_bytes),
        start.elapsed().as_secs_f32(),
    );
    let start = std::time::Instant::now();
    let model = ModelWeights::from_gguf(content, &mut file, &device)?;
    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len, args.dump_topk)?;
    Ok(())
}
