use std::sync::Arc;

use kernels::{fake_quantise_int8, rotary_frequencies, sigmoid, unit_rms, zero_centred_norm, zero_centred_norm_in_place};

use crate::attention::{AttentionCache, AttentionContext};
use crate::engram::{table_indices, EngramOutput, EngramSiteState};
use crate::heads::ProbeAccumulator;
use crate::model::Model;

const BATCH_THRESHOLD: usize = 4;
const PREFILL_CHUNK: usize = 128;

pub trait CrossAttention {
    fn apply(&mut self, layer: usize, x: &mut [f32]);
}

#[derive(Clone)]
pub struct Session {
    pub(crate) model: Arc<Model>,
    pub(crate) tokens: Vec<u32>,
    pub(crate) caches: Vec<AttentionCache>,
    pub(crate) engram_states: Vec<EngramSiteState>,
    pub(crate) frequencies: Vec<f32>,
    pub(crate) hidden: Vec<f32>,
    pub(crate) cells: Vec<Vec<f32>>,
    pub(crate) accumulators: Vec<ProbeAccumulator>,
}

impl Session {
    pub fn new(model: &Arc<Model>) -> Self {
        let config = model.config();
        Self {
            model: Arc::clone(model),
            tokens: Vec::new(),
            caches: (0..config.num_layers).map(|_| AttentionCache::new()).collect(),
            engram_states: (0..config.engram.sites.len()).map(|_| EngramSiteState::new()).collect(),
            frequencies: rotary_frequencies(config.qk_head_dim, config.rope_theta),
            hidden: Vec::new(),
            cells: Vec::new(),
            accumulators: model.weights().heads.iter().map(|head| head.accumulator()).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn feed(&mut self, tokens: &[u32]) {
        if tokens.len() >= BATCH_THRESHOLD {
            for chunk in tokens.chunks(PREFILL_CHUNK) {
                if chunk.len() >= BATCH_THRESHOLD {
                    self.prefill(chunk);
                } else {
                    chunk.iter().for_each(|&token| self.push(token));
                }
            }
        } else {
            for &token in tokens {
                self.push(token);
            }
        }
    }

    pub fn push(&mut self, token: u32) {
        self.push_with(token, None);
    }

    pub fn push_with(&mut self, token: u32, mut cross: Option<&mut dyn CrossAttention>) {
        let model = Arc::clone(&self.model);
        let config = model.config();
        let weights = model.weights();
        let width = config.d_model;
        let position = self.tokens.len();
        self.tokens.push(token);

        let mut row = Vec::new();
        weights.embedding.dequantise_row(token as usize, &mut row);
        let scale = (width as f32).sqrt();
        let mut stream = Vec::with_capacity(width * config.lanes);
        for _ in 0..config.lanes {
            stream.extend(row.iter().map(|value| value * scale));
        }

        let mut cells = Vec::with_capacity(config.num_layers + 1);
        cells.push(lane_mean(&stream, width, config.lanes));
        let indices = table_indices(&config.engram, &self.tokens, position);
        let site_outputs: Vec<Option<EngramOutput>> = self
            .engram_states
            .iter_mut()
            .zip(&weights.engrams)
            .zip(&config.engram.sites)
            .map(|((state, site_weights), site_layer)| {
                model.active_layers().contains(site_layer).then(|| state.contribute(site_weights, &config.engram, &indices, position))
            })
            .collect();

        let mut next_stream = vec![0.0f32; stream.len()];
        let mut normed = vec![0.0f32; width];
        let mut pre = vec![0.0f32; width];
        let mut delta = vec![0.0f32; width];
        for (slot, &layer) in model.active_layers().iter().enumerate() {
            let layer_weights = &weights.layers[layer];
            let routing = weights.lanes.route(layer, slot, &stream, width);
            let mut x = routing.input.clone();
            if let Some(site) = config.engram_site_at(layer) {
                if let Some(output) = &site_outputs[site] {
                    add_engram(&mut x, output, width);
                }
            }
            zero_centred_norm(&x, &layer_weights.norm_in, &mut normed);
            let context = AttentionContext { config, layer, position, frequencies: &self.frequencies };
            let mut attended = layer_weights.attention.attend(&mut self.caches[layer], &context, &normed);
            zero_centred_norm_in_place(&mut attended, &layer_weights.post_norm);
            let gate = sigmoid(layer_weights.attention_gate);
            for (slot, value) in x.iter_mut().zip(&attended) {
                *slot += gate * value;
            }
            if let Some(hook) = cross.as_deref_mut() {
                hook.apply(layer, &mut x);
            }
            zero_centred_norm(&x, &layer_weights.pre_hadamard_norm, &mut pre);
            let mlp = layer_weights.hadamard.apply(&pre);
            for (slot, value) in x.iter_mut().zip(&mlp) {
                *slot += value;
            }
            crate::lanes::LaneWeights::block_delta(&routing, &x, &mut delta);
            weights.lanes.combine_into(&routing, &stream, &delta, width, &mut next_stream);
            std::mem::swap(&mut stream, &mut next_stream);
            cells.push(lane_mean(&stream, width, config.lanes));
        }

        let mean = lane_mean(&stream, width, config.lanes);
        let mut hidden = vec![0.0; width];
        zero_centred_norm(&mean, &weights.final_norm, &mut hidden);
        self.hidden = hidden;
        for (head, accumulator) in weights.heads.iter().zip(self.accumulators.iter_mut()) {
            head.observe(accumulator, &cells);
        }
        self.cells = cells;
    }

    pub fn head_output(&self, code: u8) -> Option<Vec<f32>> {
        let weights = self.model.weights();
        let index = weights.heads.iter().position(|head| head.code == code)?;
        Some(weights.heads[index].project(&weights.heads[index].pooled(&self.accumulators[index])))
    }

    pub fn pooled_readout(&self, code: u8) -> Option<Vec<f32>> {
        let weights = self.model.weights();
        let index = weights.heads.iter().position(|head| head.code == code)?;
        Some(weights.heads[index].pooled(&self.accumulators[index]))
    }

    pub fn cells(&self) -> &[Vec<f32>] {
        &self.cells
    }

    pub fn logits(&self) -> Vec<f32> {
        let model = &self.model;
        let weights = model.weights();
        let config = model.config();
        let mut quantised = self.hidden.clone();
        fake_quantise_int8(&mut quantised);
        let mut rotated = Vec::new();
        weights.embedding.prepare(&quantised, &mut rotated);
        let mut logits = vec![0.0; config.output_vocab];
        weights.embedding.multiply_rows(0, &rotated, &mut logits);
        logits
    }
}

pub(crate) fn add_engram(x: &mut [f32], output: &EngramOutput, width: usize) {
    let mut unit_x = vec![0.0; width];
    let mut unit_key = vec![0.0; width];
    unit_rms(x, &mut unit_x);
    unit_rms(&output.key, &mut unit_key);
    let dot: f32 = unit_x.iter().zip(&unit_key).map(|(a, b)| a * b).sum();
    let alpha = sigmoid(dot / (width as f32).sqrt());
    for (slot, value) in x.iter_mut().zip(&output.value) {
        *slot += alpha * value;
    }
}

pub(crate) fn lane_mean(stream: &[f32], width: usize, lanes: usize) -> Vec<f32> {
    let mut mean = vec![0.0; width];
    for lane in stream.chunks_exact(width) {
        for (slot, value) in mean.iter_mut().zip(lane) {
            *slot += value / lanes as f32;
        }
    }
    mean
}
