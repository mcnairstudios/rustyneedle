mod cross;
mod encoder;
mod frontend;
mod keywords;
mod stem;
mod stream;
mod timestamps;
mod wav;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use cact::{Archive, Layout};
use model::tensors::floats;
use model::{Model, Session};

use cross::{CrossState, CrossWeights};
use encoder::{Encoder, EncoderConfig};
use frontend::{Frontend, FrontendConfig};
use keywords::KeywordTrie;
use stem::Stem;

pub use stream::{Stream, StreamStep};
pub use timestamps::Word;
pub use wav::{decode_wav, read_wav, resample, WavError};

pub const SAMPLE_RATE: usize = 16000;
const WINDOW_SAMPLES: usize = 30 * SAMPLE_RATE;
const MINIMUM_SPREAD: f32 = 1.5;
const SILENCE_LEVEL: f32 = 1e-4;

#[derive(Clone, Debug, Default)]
pub struct Options {
    pub language: Option<String>,
    pub keywords: Vec<String>,
    pub word_timestamps: bool,
}

impl Options {
    pub fn keywords_from_text(mut self, text: &str) -> Self {
        self.keywords = text.lines().map(str::trim).filter(|line| !line.is_empty()).map(str::to_owned).collect();
        self
    }
}

pub fn samples_from_bytes(bytes: &[u8]) -> Result<Vec<f32>, WhistleError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(WhistleError(format!("{} bytes is not a whole number of 32-bit samples", bytes.len())));
    }
    Ok(bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect())
}

#[derive(Clone, Debug)]
pub struct Transcript {
    pub text: String,
    pub words: Option<Vec<Word>>,
    pub language: String,
    pub first_token_ms: f64,
    pub tokens_per_second: f64,
}

pub(crate) struct Decoded {
    pub ids: Vec<u32>,
    pub probabilities: Vec<f32>,
    pub alignment: Vec<Vec<f32>>,
    pub language: usize,
    pub frames: usize,
    pub first_token_seconds: f64,
    pub decode_seconds: f64,
}

pub struct Encoded {
    pub output: Vec<f32>,
    pub frames: usize,
}

pub struct Whistle {
    model: Arc<Model>,
    frontend: Frontend,
    stem: Stem,
    encoder: Encoder,
    cross: CrossWeights,
    languages: Vec<String>,
    text_vocab: usize,
    dim: usize,
}

#[derive(Debug)]
pub struct WhistleError(pub String);

impl std::fmt::Display for WhistleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for WhistleError {}

impl Whistle {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WhistleError> {
        let wrap = |error: &dyn std::fmt::Display| WhistleError(error.to_string());
        let archive = Archive::open(path).map_err(|e| wrap(&e))?;
        let layout = Layout::resolve(&archive).map_err(|e| wrap(&e))?;
        let audio = layout.audio.as_ref().ok_or_else(|| WhistleError("archive carries no audio encoder".into()))?;
        let geometry = archive.geometry();
        let manifest = &audio.manifest;
        let model = Model::from_archive(&archive, &layout).map_err(|e| wrap(&e))?;
        let dim = geometry.d_model;
        let heads = dim / geometry.v_head_dim;
        let frontend = Frontend::new(
            FrontendConfig { mels: manifest.mels, fft_size: manifest.fft_size, window: manifest.window, hop: manifest.hop },
            floats(&archive, audio.mel),
        );
        let stem = Stem::load(&archive, &audio.stem, manifest.stem_channels);
        let encoder = Encoder::load(
            &archive,
            EncoderConfig {
                dim,
                lanes: manifest.encoder_lanes,
                heads,
                kv_heads: geometry.num_kv_heads,
                qk_dim: geometry.qk_head_dim,
                v_dim: geometry.v_head_dim,
                conv_kernel: manifest.conv_kernel,
                rope_theta: geometry.rope_theta,
                hadamard_size: geometry.hada_n,
            },
            &audio.encoder,
            &audio.encoder_lanes,
            audio.encoder_final_norm,
            layout.hadamard_perms,
        );
        let cross = CrossWeights::load(&archive, &audio.cross, audio.position_gate, dim, heads, geometry.qk_head_dim, geometry.v_head_dim);
        let text_vocab = manifest.text_vocab;
        let vocabulary = model.tokenizer().vocabulary();
        let languages =
            (text_vocab..vocabulary.len()).map(|id| vocabulary.piece(id as u32).trim_start_matches("<|").trim_end_matches("|>").to_owned()).collect();
        Ok(Self { model, frontend, stem, encoder, cross, languages, text_vocab, dim })
    }

    pub fn with_audio_depth(self, depth: usize) -> Result<Whistle, WhistleError> {
        let model = self.model.with_depth(depth).map_err(|error| WhistleError(error.to_string()))?;
        Ok(Whistle { model, ..self })
    }

    pub fn decoder_layers(&self) -> usize {
        self.model.active_layers().len()
    }

    pub fn languages(&self) -> &[String] {
        &self.languages
    }

    pub fn encode(&self, samples: &[f32]) -> Encoded {
        let features = self.frontend.log_mel(samples);
        let stem = self.stem.forward(&features.data, features.frames, features.valid, features.mels);
        let frames = stem.valid.min(stem.frames);
        let output = self.encoder.forward(&stem.data[..frames * self.dim], frames);
        Encoded { output, frames }
    }

    fn spread(&self, samples: &[f32]) -> f32 {
        self.frontend.log_mel(samples).spread
    }

    pub fn embed(&self, samples: &[f32]) -> Vec<f32> {
        self.encode(samples).output
    }

    pub fn transcribe(&self, samples: &[f32], options: &Options) -> Result<Transcript, WhistleError> {
        if samples.len() <= WINDOW_SAMPLES {
            return self.transcribe_window(samples, options);
        }
        self.transcribe_windows(samples, options)
    }

    fn transcribe_windows(&self, samples: &[f32], options: &Options) -> Result<Transcript, WhistleError> {
        let mut offset = 0usize;
        let mut texts: Vec<String> = Vec::new();
        let mut words: Vec<Word> = Vec::new();
        let mut language = options.language.clone();
        let (mut first_token_ms, mut speed, mut windows) = (0.0, 0.0, 0usize);
        while offset < samples.len() {
            let end = (offset + WINDOW_SAMPLES).min(samples.len());
            let last = end == samples.len();
            let pass = Options { language: language.clone(), word_timestamps: true, keywords: options.keywords.clone() };
            let window = self.transcribe_window(&samples[offset..end], &pass)?;
            language.get_or_insert(window.language.clone());
            let window_words = window.words.clone().unwrap_or_default();
            let seconds = offset as f64 / SAMPLE_RATE as f64;
            let sentence_end = if last { None } else { window_words.iter().rposition(|word| word.word.ends_with(['.', '?', '!'])) };
            let kept = sentence_end.map_or(window_words.len(), |index| index + 1);
            texts.push(window_words[..kept].iter().map(|word| word.word.as_str()).collect::<Vec<_>>().join(" "));
            words.extend(window_words[..kept].iter().map(|word| Word { start: word.start + seconds, end: word.end + seconds, ..word.clone() }));
            if windows == 0 {
                first_token_ms = window.first_token_ms;
            }
            speed += window.tokens_per_second;
            windows += 1;
            offset = match sentence_end {
                Some(index) => offset + ((window_words[index].end * SAMPLE_RATE as f64) as usize).max(SAMPLE_RATE),
                None => end,
            };
        }
        Ok(Transcript {
            text: texts.into_iter().filter(|text| !text.is_empty()).collect::<Vec<_>>().join(" "),
            words: options.word_timestamps.then_some(words),
            language: language.unwrap_or_default(),
            first_token_ms,
            tokens_per_second: speed / windows.max(1) as f64,
        })
    }

    fn transcribe_window(&self, samples: &[f32], options: &Options) -> Result<Transcript, WhistleError> {
        let end = samples.iter().rposition(|value| value.abs() > SILENCE_LEVEL).map_or(0, |last| last + 1);
        let samples = &samples[..end];
        if samples.len() < SAMPLE_RATE / 100 || self.spread(samples) < MINIMUM_SPREAD {
            return Ok(Transcript {
                text: String::new(),
                words: options.word_timestamps.then(Vec::new),
                language: String::new(),
                first_token_ms: 0.0,
                tokens_per_second: 0.0,
            });
        }
        let decoded = self.decode(samples, options)?;
        let text = self.model.tokenizer().decode(&decoded.ids).trim().to_owned();
        let elapsed = decoded.decode_seconds.max(1e-9);
        let words = options.word_timestamps.then(|| timestamps::words(&decoded, self.model.tokenizer()));
        Ok(Transcript {
            text,
            words,
            language: self.languages[decoded.language].clone(),
            first_token_ms: decoded.first_token_seconds * 1000.0,
            tokens_per_second: decoded.ids.len() as f64 / elapsed,
        })
    }

    pub(crate) fn decode(&self, samples: &[f32], options: &Options) -> Result<Decoded, WhistleError> {
        let started = Instant::now();
        let encoded = self.encode(samples);
        let memory = self.cross.memory(&encoded.output, encoded.frames);
        let mut cross = CrossState::new(&self.cross, &memory);
        let tokenizer = self.model.tokenizer();
        let special = tokenizer.vocabulary();
        let (bos, eos) = (special.bos_id(), special.eos_id());
        let keywords = KeywordTrie::new(&options.keywords, tokenizer);
        let mut session = Session::new(&self.model);
        session.push_with(bos, Some(&mut cross));
        cross.take_alignment();
        let language = match &options.language {
            Some(code) => self.languages.iter().position(|known| known == code).ok_or_else(|| WhistleError(format!("unsupported language {code}")))?,
            None => self.best(&session.logits(), |id| id >= self.text_vocab).saturating_sub(self.text_vocab),
        };
        session.push_with((self.text_vocab + language) as u32, Some(&mut cross));
        let mut alignment = vec![cross.take_alignment()];
        let mut ids = Vec::new();
        let mut probabilities = Vec::new();
        let mut first_token_seconds = 0.0;
        let decode_started = Instant::now();
        let limit = self.model.config().max_seq_len - 1;
        let mut keyword_state = keywords.as_ref().map(KeywordTrie::start);
        while session.len() < limit {
            let mut logits = session.logits();
            if let (Some(trie), Some(state)) = (&keywords, &keyword_state) {
                let top = logits.iter().take(self.text_vocab).copied().fold(f32::NEG_INFINITY, f32::max);
                trie.bias(state, &mut logits, top);
            }
            let next = self.best(&logits, |id| id < self.text_vocab);
            let probability = softmax_probability(&logits, next, |id| id < self.text_vocab);
            if ids.is_empty() {
                first_token_seconds = started.elapsed().as_secs_f64();
            }
            if next as u32 == eos {
                break;
            }
            ids.push(next as u32);
            probabilities.push(probability);
            if let (Some(trie), Some(state)) = (&keywords, keyword_state.as_mut()) {
                trie.advance(state, next as u32);
            }
            session.push_with(next as u32, Some(&mut cross));
            alignment.push(cross.take_alignment());
        }
        Ok(Decoded {
            ids,
            probabilities,
            alignment,
            language,
            frames: encoded.frames,
            first_token_seconds,
            decode_seconds: decode_started.elapsed().as_secs_f64(),
        })
    }

    fn best(&self, logits: &[f32], allowed: impl Fn(usize) -> bool) -> usize {
        let mut best = usize::MAX;
        for (id, value) in logits.iter().enumerate() {
            if allowed(id) && (best == usize::MAX || *value > logits[best]) {
                best = id;
            }
        }
        best
    }
}

fn softmax_probability(logits: &[f32], token: usize, allowed: impl Fn(usize) -> bool) -> f32 {
    let peak = logits.iter().enumerate().filter(|(id, _)| allowed(*id)).map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = logits.iter().enumerate().filter(|(id, _)| allowed(*id)).map(|(_, v)| (v - peak).exp()).sum();
    (logits[token] - peak).exp() / total
}
