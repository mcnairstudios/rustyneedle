use std::collections::VecDeque;

use cact::{Archive, LayerTensors};
use kernels::{apply_rotary, fake_quantise_int8, int8_scale, quantise_int8, sigmoid, zero_centred_norm_in_place, QuantisedMatrix};

use crate::config::ModelConfig;
use crate::tensors::floats;

const PARALLEL_POSITIONS: usize = 128;

pub struct AttentionWeights {
    q_proj: QuantisedMatrix,
    k_proj: QuantisedMatrix,
    v_proj: QuantisedMatrix,
    gate_proj: QuantisedMatrix,
    out_proj: QuantisedMatrix,
    conv_taps: Option<[Vec<f32>; 3]>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

impl AttentionWeights {
    pub fn load(archive: &Archive, tensors: &LayerTensors) -> Self {
        Self {
            q_proj: QuantisedMatrix::from_archive(archive, tensors.q_proj),
            k_proj: QuantisedMatrix::from_archive(archive, tensors.k_proj),
            v_proj: QuantisedMatrix::from_archive(archive, tensors.v_proj),
            gate_proj: QuantisedMatrix::from_archive(archive, tensors.gate_proj),
            out_proj: QuantisedMatrix::from_archive(archive, tensors.out_proj),
            conv_taps: tensors.conv_taps.map(|[q, k, v]| [floats(archive, q), floats(archive, k), floats(archive, v)]),
            q_norm: floats(archive, tensors.q_norm),
            k_norm: floats(archive, tensors.k_norm),
        }
    }
}

#[derive(Clone)]
pub struct AttentionCache {
    keys: Vec<i8>,
    key_scales: Vec<Vec<f32>>,
    values: Vec<i8>,
    value_scales: Vec<Vec<f32>>,
    value_panels: Vec<Vec<i8>>,
    raw_history: VecDeque<[Vec<f32>; 3]>,
}

impl Default for AttentionCache {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionCache {
    fn store_value(&mut self, kv: usize, position: usize, codes: &[i8]) {
        let width = codes.len();
        let positions = kernels::PANEL_POSITIONS;
        if self.value_panels.len() <= kv {
            self.value_panels.resize_with(kv + 1, Vec::new);
        }
        let panel = &mut self.value_panels[kv];
        let needed = (position / positions + 1) * width * positions;
        if panel.len() < needed {
            panel.resize(needed, 0);
        }
        let base = position / positions * width * positions + position % positions;
        for (dimension, code) in codes.iter().enumerate() {
            panel[base + dimension * positions] = *code;
        }
    }

    pub fn new() -> Self {
        Self { keys: Vec::new(), key_scales: Vec::new(), values: Vec::new(), value_scales: Vec::new(), value_panels: Vec::new(), raw_history: VecDeque::new() }
    }
}

pub struct AttentionContext<'a> {
    pub config: &'a ModelConfig,
    pub layer: usize,
    pub position: usize,
    pub frequencies: &'a [f32],
}

fn mix_taps(taps: &[Vec<f32>; 3], which: usize, history: &VecDeque<[Vec<f32>; 3]>, count: usize, position: usize) -> Vec<f32> {
    let width = history[0][which].len();
    let mut mixed = vec![0.0; width];
    for tap in 0..count.min(position + 1) {
        let gains = &taps[which][tap * width..(tap + 1) * width];
        for ((slot, gain), sample) in mixed.iter_mut().zip(gains).zip(&history[tap][which]) {
            *slot += gain * sample;
        }
    }
    mixed
}

impl AttentionWeights {
    pub fn attend(&self, cache: &mut AttentionCache, context: &AttentionContext<'_>, hidden: &[f32]) -> Vec<f32> {
        let config = context.config;
        let (qk, vd) = (config.qk_head_dim, config.v_head_dim);
        let mut quantised = hidden.to_vec();
        fake_quantise_int8(&mut quantised);
        let mut rotated = Vec::new();
        self.q_proj.prepare(&quantised, &mut rotated);
        let mut q = vec![0.0; config.query_width()];
        let mut k = vec![0.0; config.key_width()];
        let mut v = vec![0.0; config.value_width()];
        let mut gate = vec![0.0; config.attention_width()];
        QuantisedMatrix::multiply_fused(
            vec![(&self.q_proj, 0, &mut q[..]), (&self.k_proj, 0, &mut k[..]), (&self.v_proj, 0, &mut v[..]), (&self.gate_proj, 0, &mut gate[..])],
            &rotated,
        );
        if let Some(taps) = &self.conv_taps {
            cache.raw_history.push_front([q, k, v]);
            cache.raw_history.truncate(config.conv_taps);
            q = mix_taps(taps, 0, &cache.raw_history, config.conv_taps, context.position);
            k = mix_taps(taps, 1, &cache.raw_history, config.conv_taps, context.position);
            v = mix_taps(taps, 2, &cache.raw_history, config.conv_taps, context.position);
        }
        let mut query_codes = vec![0i8; q.len()];
        let mut query_scales = Vec::with_capacity(config.num_heads);
        for (head, codes) in q.chunks_exact_mut(qk).zip(query_codes.chunks_exact_mut(qk)) {
            zero_centred_norm_in_place(head, &self.q_norm);
            apply_rotary(head, context.position, context.frequencies);
            query_scales.push(quantise_int8(head, codes));
        }
        for head in k.chunks_exact_mut(qk) {
            zero_centred_norm_in_place(head, &self.k_norm);
            apply_rotary(head, context.position, context.frequencies);
        }
        cache.key_scales.resize_with(config.num_kv_heads, Vec::new);
        cache.value_scales.resize_with(config.num_kv_heads, Vec::new);
        for (kv, head) in k.chunks_exact(qk).enumerate() {
            let mut codes = vec![0i8; qk];
            let scale = quantise_int8(head, &mut codes);
            cache.keys.extend_from_slice(&codes);
            cache.key_scales[kv].push(scale);
        }
        for (kv, head) in v.chunks_exact(vd).enumerate() {
            let mut codes = vec![0i8; vd];
            let scale = quantise_int8(head, &mut codes);
            cache.values.extend_from_slice(&codes);
            cache.value_scales[kv].push(scale);
            cache.store_value(kv, context.position, &codes);
        }
        let mut attended = self.weighted_values(cache, context, &query_codes, &query_scales, true);
        for (value, gate) in attended.iter_mut().zip(&gate) {
            *value *= sigmoid(*gate);
        }
        fake_quantise_int8(&mut attended);
        let mut rotated_out = Vec::new();
        self.out_proj.prepare(&attended, &mut rotated_out);
        let mut output = vec![0.0; self.out_proj.rows()];
        self.out_proj.multiply_vector(&rotated_out, &mut output);
        output
    }

    fn weighted_values(&self, cache: &AttentionCache, context: &AttentionContext<'_>, query_codes: &[i8], query_scales: &[f32], parallel: bool) -> Vec<f32> {
        let config = context.config;
        let vd = config.v_head_dim;
        let mut output = vec![0.0; config.attention_width()];
        let total = context.position + 1;
        let count = total - self.window_start(config, context.layer, total);
        if parallel && count >= PARALLEL_POSITIONS {
            kernels::spin_pool()
                .for_each_chunk(&mut output, vd, &|head, slot| self.attend_heads(cache, context, query_codes, query_scales, head..head + 1, slot));
        } else {
            self.attend_heads(cache, context, query_codes, query_scales, 0..config.num_heads, &mut output);
        }
        output
    }

    fn window_start(&self, config: &ModelConfig, layer: usize, total: usize) -> usize {
        if config.sliding_window > 0 && !config.is_global_layer(layer) {
            total.saturating_sub(config.sliding_window)
        } else {
            0
        }
    }

    fn attend_heads(
        &self,
        cache: &AttentionCache,
        context: &AttentionContext<'_>,
        query_codes: &[i8],
        query_scales: &[f32],
        heads: std::ops::Range<usize>,
        out: &mut [f32],
    ) {
        let config = context.config;
        let (qk, vd, kv_heads) = (config.qk_head_dim, config.v_head_dim, config.num_kv_heads);
        let total = context.position + 1;
        let first = self.window_start(config, context.layer, total);
        let group = config.num_heads / kv_heads;
        let inverse_root = 1.0 / (qk as f32).sqrt();
        for (head, slot) in heads.zip(out.chunks_exact_mut(vd)) {
            let kv = head / group;
            let inputs = kernels::HeadInputs {
                query: &query_codes[head * qk..(head + 1) * qk],
                query_scale: query_scales[head] * inverse_root,
                keys: &cache.keys[first * kv_heads * qk..],
                key_stride: kv_heads * qk,
                key_offset: kv * qk,
                key_scales: &cache.key_scales[kv][first..total],
                values: &cache.values[first * kv_heads * vd..],
                value_stride: kv_heads * vd,
                value_offset: kv * vd,
                value_panels: &cache.value_panels[kv],
                first,
                value_scales: &cache.value_scales[kv][first..total],
                width: vd,
            };
            kernels::attend_head(&inputs, slot);
        }
    }
}

impl AttentionWeights {
    pub fn attend_batch(
        &self,
        cache: &mut AttentionCache,
        config: &ModelConfig,
        layer: usize,
        first_position: usize,
        hidden: &[f32],
        frequencies: &[f32],
    ) -> Vec<f32> {
        let count = hidden.len() / config.d_model;
        let (qk, vd, kv_heads) = (config.qk_head_dim, config.v_head_dim, config.num_kv_heads);
        let mut q = vec![0.0f32; count * config.query_width()];
        let mut k = vec![0.0f32; count * config.key_width()];
        let mut v = vec![0.0f32; count * config.value_width()];
        let mut gate = vec![0.0f32; count * config.attention_width()];
        let jobs = vec![(&self.q_proj, &mut q[..]), (&self.k_proj, &mut k[..]), (&self.v_proj, &mut v[..]), (&self.gate_proj, &mut gate[..])];
        match self.q_proj.natural_batch(hidden, count, &|_, row| fake_quantise_int8(row)) {
            Some(input) => QuantisedMatrix::multiply_natural_fused(jobs, &input, count),
            None => {
                let mut quantised = hidden.to_vec();
                kernels::parallel_chunks_mut(&mut quantised, config.d_model, &|_, row| fake_quantise_int8(row));
                let mut rotated = Vec::new();
                self.q_proj.prepare_batch(&quantised, count, &mut rotated);
                QuantisedMatrix::multiply_batch_fused(jobs, &rotated, count);
            }
        }
        if let Some(taps) = &self.conv_taps {
            let widths = [config.query_width(), config.key_width(), config.value_width()];
            let previous: Vec<[Vec<f32>; 3]> = cache.raw_history.iter().cloned().collect();
            let mixed: Vec<Vec<f32>> = [&q, &k, &v]
                .iter()
                .enumerate()
                .map(|(which, rows)| {
                    let width = widths[which];
                    let mut out = vec![0.0f32; count * width];
                    kernels::parallel_chunks_mut(&mut out, width, &|t, slot| {
                        let position = first_position + t;
                        for tap in 0..config.conv_taps.min(position + 1) {
                            let source: &[f32] = if tap <= t { &rows[(t - tap) * width..(t - tap + 1) * width] } else { &previous[tap - t - 1][which] };
                            let gains = &taps[which][tap * width..(tap + 1) * width];
                            for ((value, gain), sample) in slot.iter_mut().zip(gains).zip(source) {
                                *value += gain * sample;
                            }
                        }
                    });
                    out
                })
                .collect();
            let newest = count.min(config.conv_taps);
            for t in count - newest..count {
                cache.raw_history.push_front([
                    q[t * widths[0]..(t + 1) * widths[0]].to_vec(),
                    k[t * widths[1]..(t + 1) * widths[1]].to_vec(),
                    v[t * widths[2]..(t + 1) * widths[2]].to_vec(),
                ]);
            }
            cache.raw_history.truncate(config.conv_taps);
            let mut parts = mixed.into_iter();
            q = parts.next().expect("query rows");
            k = parts.next().expect("key rows");
            v = parts.next().expect("value rows");
        }
        let query_width = config.query_width();
        kernels::parallel_chunks_mut(&mut q, query_width, &|t, row| {
            for values in row.chunks_exact_mut(qk) {
                zero_centred_norm_in_place(values, &self.q_norm);
                apply_rotary(values, first_position + t, frequencies);
            }
        });
        let mut query_codes = vec![0i8; q.len()];
        kernels::parallel_chunks_mut(&mut query_codes, query_width, &|t, codes| {
            for (code_head, values) in codes.chunks_exact_mut(qk).zip(q[t * query_width..(t + 1) * query_width].chunks_exact(qk)) {
                quantise_int8(values, code_head);
            }
        });
        let query_scales: Vec<f32> = q.chunks_exact(qk).map(int8_scale).collect();
        let key_width = config.key_width();
        kernels::parallel_chunks_mut(&mut k, key_width, &|t, row| {
            for values in row.chunks_exact_mut(qk) {
                zero_centred_norm_in_place(values, &self.k_norm);
                apply_rotary(values, first_position + t, frequencies);
            }
        });
        let mut key_codes = vec![0i8; k.len()];
        kernels::parallel_chunks_mut(&mut key_codes, key_width, &|t, codes| {
            for (code_head, values) in codes.chunks_exact_mut(qk).zip(k[t * key_width..(t + 1) * key_width].chunks_exact(qk)) {
                quantise_int8(values, code_head);
            }
        });
        let key_scales: Vec<f32> = k.chunks_exact(qk).map(int8_scale).collect();
        let value_width = config.value_width();
        let mut value_codes = vec![0i8; v.len()];
        kernels::parallel_chunks_mut(&mut value_codes, value_width, &|t, codes| {
            for (code_head, values) in codes.chunks_exact_mut(vd).zip(v[t * value_width..(t + 1) * value_width].chunks_exact(vd)) {
                quantise_int8(values, code_head);
            }
        });
        let value_scales: Vec<f32> = v.chunks_exact(vd).map(int8_scale).collect();
        cache.keys.extend_from_slice(&key_codes);
        cache.key_scales.resize_with(kv_heads, Vec::new);
        cache.value_scales.resize_with(kv_heads, Vec::new);
        for per_head in key_scales.chunks_exact(kv_heads) {
            for (kv, scale) in per_head.iter().enumerate() {
                cache.key_scales[kv].push(*scale);
            }
        }
        cache.values.extend_from_slice(&value_codes);
        for t in 0..count {
            for kv in 0..kv_heads {
                let start = t * value_width + kv * vd;
                cache.store_value(kv, first_position + t, &value_codes[start..start + vd]);
            }
        }
        for per_head in value_scales.chunks_exact(kv_heads) {
            for (kv, scale) in per_head.iter().enumerate() {
                cache.value_scales[kv].push(*scale);
            }
        }
        let attention_width = config.attention_width();
        let mut attended = vec![0.0f32; count * attention_width];
        let shared: &AttentionCache = cache;
        let group = config.num_heads / kv_heads;
        let group_width = group * vd;
        let mut grouped = vec![0.0f32; kv_heads * count * group_width];
        kernels::parallel_chunks_mut(&mut grouped, group_width, &|item, out| {
            let (kv, t) = (item / count, item % count);
            let context = AttentionContext { config, layer, position: first_position + t, frequencies };
            let codes = &query_codes[t * query_width..(t + 1) * query_width];
            let scales = &query_scales[t * config.num_heads..(t + 1) * config.num_heads];
            self.attend_heads(shared, &context, codes, scales, kv * group..(kv + 1) * group, out);
        });
        for (item, block) in grouped.chunks_exact(group_width).enumerate() {
            let (kv, t) = (item / count, item % count);
            attended[t * attention_width + kv * group_width..t * attention_width + (kv + 1) * group_width].copy_from_slice(block);
        }
        let gate_and_quantise = |t: usize, row: &mut [f32]| {
            for (value, gate_value) in row.iter_mut().zip(&gate[t * attention_width..(t + 1) * attention_width]) {
                *value *= sigmoid(*gate_value);
            }
            fake_quantise_int8(row);
        };
        let mut output = vec![0.0f32; count * config.d_model];
        match self.out_proj.natural_batch(&attended, count, &gate_and_quantise) {
            Some(input) => QuantisedMatrix::multiply_natural_fused(vec![(&self.out_proj, &mut output[..])], &input, count),
            None => {
                kernels::parallel_chunks_mut(&mut attended, attention_width, &|t, row| gate_and_quantise(t, row));
                let mut rotated_out = Vec::new();
                self.out_proj.prepare_batch(&attended, count, &mut rotated_out);
                self.out_proj.multiply_batch(&rotated_out, count, &mut output);
            }
        }
        output
    }
}
