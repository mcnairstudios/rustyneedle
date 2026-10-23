use cact::{Archive, CrossTensors};
use kernels::{fake_quantise_int8, quantise_int8, sigmoid, softmax_in_place, zero_centred_norm, zero_centred_norm_in_place, QuantisedMatrix};
use model::tensors::{floats, scalar};
use model::CrossAttention;

const POSITION_BASE: f32 = 10000.0;

pub struct CrossLayer {
    norm: Vec<f32>,
    q_proj: QuantisedMatrix,
    k_proj: QuantisedMatrix,
    v_proj: QuantisedMatrix,
    gate_proj: QuantisedMatrix,
    out_proj: QuantisedMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_norm: Vec<f32>,
    gate: f32,
}

pub struct CrossWeights {
    layers: Vec<CrossLayer>,
    position_gate: f32,
    dim: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
}

pub struct CrossMemory {
    frames: usize,
    keys: Vec<Vec<i8>>,
    key_scales: Vec<Vec<f32>>,
    values: Vec<Vec<i8>>,
    value_scales: Vec<Vec<f32>>,
}

impl CrossWeights {
    pub fn load(archive: &Archive, tensors: &[CrossTensors], position_gate: usize, dim: usize, heads: usize, qk_dim: usize, v_dim: usize) -> Self {
        let layers = tensors
            .iter()
            .map(|t| CrossLayer {
                norm: floats(archive, t.norm),
                q_proj: QuantisedMatrix::from_archive(archive, t.q_proj),
                k_proj: QuantisedMatrix::from_archive(archive, t.k_proj),
                v_proj: QuantisedMatrix::from_archive(archive, t.v_proj),
                gate_proj: QuantisedMatrix::from_archive(archive, t.gate_proj),
                out_proj: QuantisedMatrix::from_archive(archive, t.out_proj),
                q_norm: floats(archive, t.q_norm),
                k_norm: floats(archive, t.k_norm),
                post_norm: floats(archive, t.post_norm),
                gate: scalar(archive, t.gate),
            })
            .collect();
        Self { layers, position_gate: scalar(archive, position_gate), dim, heads, qk_dim, v_dim }
    }

    pub fn memory(&self, encoded: &[f32], frames: usize) -> CrossMemory {
        let dim = self.dim;
        let gate = sigmoid(self.position_gate);
        let half = dim / 2;
        let rate = -(POSITION_BASE.ln()) / (half as f32 - 1.0);
        let mut memory = vec![0.0f32; frames * dim];
        kernels::parallel_chunks_mut(&mut memory, dim, &|t, out| {
            let row = &encoded[t * dim..(t + 1) * dim];
            for j in 0..half {
                let angle = t as f32 * (rate * j as f32).exp();
                out[j] = row[j] + gate * angle.sin();
                out[half + j] = row[half + j] + gate * angle.cos();
            }
            fake_quantise_int8(out);
        });
        let (qk, vd, heads) = (self.qk_dim, self.v_dim, self.heads);
        let mut keys = Vec::new();
        let mut key_scales = Vec::new();
        let mut values = Vec::new();
        let mut value_scales = Vec::new();
        for layer in &self.layers {
            let mut rotated = Vec::new();
            layer.k_proj.prepare_batch(&memory, frames, &mut rotated);
            let mut k = vec![0.0f32; frames * layer.k_proj.rows()];
            layer.k_proj.multiply_batch(&rotated, frames, &mut k);
            let mut v = vec![0.0f32; frames * layer.v_proj.rows()];
            layer.v_proj.multiply_batch(&rotated, frames, &mut v);
            let mut key_codes = vec![0i8; k.len()];
            let mut scales_k = vec![0.0f32; frames * heads];
            for ((codes, head), scale) in key_codes.chunks_mut(qk).zip(k.chunks_mut(qk)).zip(scales_k.iter_mut()) {
                zero_centred_norm_in_place(head, &layer.k_norm);
                *scale = quantise_int8(head, codes);
            }
            let mut value_codes = vec![0i8; v.len()];
            let mut scales_v = vec![0.0f32; frames * heads];
            for ((codes, head), scale) in value_codes.chunks_mut(vd).zip(v.chunks(vd)).zip(scales_v.iter_mut()) {
                *scale = quantise_int8(head, codes);
            }
            keys.push(key_codes);
            key_scales.push(scales_k);
            values.push(value_codes);
            value_scales.push(scales_v);
        }
        CrossMemory { frames, keys, key_scales, values, value_scales }
    }
}

pub struct CrossState<'a> {
    weights: &'a CrossWeights,
    memory: &'a CrossMemory,
    alignment: Vec<f32>,
    contributions: usize,
}

impl<'a> CrossState<'a> {
    pub fn new(weights: &'a CrossWeights, memory: &'a CrossMemory) -> Self {
        Self { weights, memory, alignment: vec![0.0; memory.frames], contributions: 0 }
    }

    pub fn take_alignment(&mut self) -> Vec<f32> {
        let count = self.contributions.max(1) as f32;
        let averaged = self.alignment.iter().map(|value| value / count).collect();
        self.alignment.iter_mut().for_each(|value| *value = 0.0);
        self.contributions = 0;
        averaged
    }
}

impl CrossAttention for CrossState<'_> {
    fn apply(&mut self, layer_index: usize, x: &mut [f32]) {
        let weights = self.weights;
        let memory = self.memory;
        let layer = &weights.layers[layer_index];
        let (qk, vd, heads, frames) = (weights.qk_dim, weights.v_dim, weights.heads, memory.frames);
        let mut normed = vec![0.0f32; x.len()];
        zero_centred_norm(x, &layer.norm, &mut normed);
        fake_quantise_int8(&mut normed);
        let mut rotated = Vec::new();
        layer.q_proj.prepare(&normed, &mut rotated);
        let mut q = vec![0.0f32; layer.q_proj.rows()];
        layer.q_proj.multiply_vector(&rotated, &mut q);
        let mut gate = vec![0.0f32; layer.gate_proj.rows()];
        layer.gate_proj.multiply_vector(&rotated, &mut gate);
        let inverse_root = 1.0 / (qk as f32).sqrt();
        let mut context = vec![0.0f32; heads * vd];
        let mut scores = vec![0.0f32; frames];
        for head in 0..heads {
            let query = &mut q[head * qk..(head + 1) * qk];
            zero_centred_norm_in_place(query, &layer.q_norm);
            let mut codes = vec![0i8; qk];
            let query_scale = quantise_int8(query, &mut codes) * inverse_root;
            for (frame, slot) in scores.iter_mut().enumerate() {
                let base = (frame * heads + head) * qk;
                let keys = &memory.keys[layer_index][base..base + qk];
                let dot: i32 = codes.iter().zip(keys).map(|(a, b)| *a as i32 * *b as i32).sum();
                *slot = dot as f32 * memory.key_scales[layer_index][frame * heads + head] * query_scale;
            }
            softmax_in_place(&mut scores);
            for (slot, weight) in self.alignment.iter_mut().zip(&scores) {
                *slot += weight;
            }
            self.contributions += 1;
            let out = &mut context[head * vd..(head + 1) * vd];
            for (frame, weight) in scores.iter().enumerate() {
                let base = (frame * heads + head) * vd;
                let scale = memory.value_scales[layer_index][frame * heads + head] * weight;
                for (slot, code) in out.iter_mut().zip(&memory.values[layer_index][base..base + vd]) {
                    *slot += scale * *code as f32;
                }
            }
        }
        for (value, gate_value) in context.iter_mut().zip(&gate) {
            *value *= sigmoid(*gate_value);
        }
        fake_quantise_int8(&mut context);
        let mut rotated_out = Vec::new();
        layer.out_proj.prepare(&context, &mut rotated_out);
        let mut attended = vec![0.0f32; layer.out_proj.rows()];
        layer.out_proj.multiply_vector(&rotated_out, &mut attended);
        zero_centred_norm_in_place(&mut attended, &layer.post_norm);
        let gain = sigmoid(layer.gate);
        for (slot, value) in x.iter_mut().zip(&attended) {
            *slot += gain * value;
        }
    }
}
