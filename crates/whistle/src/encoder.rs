use cact::{Archive, EncoderTensors, LaneTensors};
use kernels::{
    apply_rotary, fake_quantise_int8, int8_scale, quantise_int8, rotary_frequencies, sigmoid, silu, softmax_in_place, zero_centred_norm,
    zero_centred_norm_in_place, QuantisedMatrix,
};
use model::hadamard::HadamardWeights;
use model::lanes::LaneWeights;
use model::tensors::{floats, scalar};

const HALF_RESIDUAL: f32 = 0.5;

pub struct EncoderConfig {
    pub dim: usize,
    pub lanes: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub qk_dim: usize,
    pub v_dim: usize,
    pub conv_kernel: usize,
    pub rope_theta: f32,
    pub hadamard_size: usize,
}

struct EncoderLayer {
    first_norm: Vec<f32>,
    attention_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    conv_norm: Vec<f32>,
    conv_out_norm: Vec<f32>,
    last_norm: Vec<f32>,
    q_proj: QuantisedMatrix,
    k_proj: QuantisedMatrix,
    v_proj: QuantisedMatrix,
    gate_proj: QuantisedMatrix,
    out_proj: QuantisedMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    attention_gate: f32,
    pointwise_in: QuantisedMatrix,
    pointwise_out: QuantisedMatrix,
    depthwise: Vec<f32>,
    hadamard_first: HadamardWeights,
    hadamard: HadamardWeights,
}

pub struct Encoder {
    config: EncoderConfig,
    lanes: LaneWeights,
    layers: Vec<EncoderLayer>,
    final_norm: Vec<f32>,
    frequencies: Vec<f32>,
}

fn rotated_batch(matrix: &QuantisedMatrix, inputs: &[f32], count: usize) -> Vec<f32> {
    let mut rotated = Vec::new();
    matrix.prepare_batch(inputs, count, &mut rotated);
    rotated
}

fn project(matrix: &QuantisedMatrix, rotated: &[f32], count: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; count * matrix.rows()];
    matrix.multiply_batch(rotated, count, &mut output);
    output
}

impl Encoder {
    pub fn load(
        archive: &Archive,
        config: EncoderConfig,
        layer_tensors: &[EncoderTensors],
        lane_tensors: &LaneTensors,
        final_norm: usize,
        permutations: [usize; 2],
    ) -> Self {
        let layers = layer_tensors
            .iter()
            .map(|t| EncoderLayer {
                first_norm: floats(archive, t.pre_hada_norm_first),
                attention_norm: floats(archive, t.attn_norm),
                post_attention_norm: floats(archive, t.post_attn_norm),
                conv_norm: floats(archive, t.conv_norm),
                conv_out_norm: floats(archive, t.conv_out_norm),
                last_norm: floats(archive, t.pre_hada_norm),
                q_proj: QuantisedMatrix::from_archive(archive, t.q_proj),
                k_proj: QuantisedMatrix::from_archive(archive, t.k_proj),
                v_proj: QuantisedMatrix::from_archive(archive, t.v_proj),
                gate_proj: QuantisedMatrix::from_archive(archive, t.gate_proj),
                out_proj: QuantisedMatrix::from_archive(archive, t.out_proj),
                q_norm: floats(archive, t.q_norm),
                k_norm: floats(archive, t.k_norm),
                attention_gate: scalar(archive, t.attn_gate),
                pointwise_in: QuantisedMatrix::from_archive(archive, t.pw1),
                pointwise_out: QuantisedMatrix::from_archive(archive, t.pw2),
                depthwise: floats(archive, t.depthwise),
                hadamard_first: HadamardWeights::load(archive, &t.hadamard_first, permutations, config.hadamard_size),
                hadamard: HadamardWeights::load(archive, &t.hadamard, permutations, config.hadamard_size),
            })
            .collect();
        Self {
            lanes: LaneWeights::load(archive, lane_tensors, config.lanes),
            final_norm: floats(archive, final_norm),
            frequencies: rotary_frequencies(config.qk_dim, config.rope_theta),
            layers,
            config,
        }
    }

    pub fn forward(&self, input: &[f32], frames: usize) -> Vec<f32> {
        let dim = self.config.dim;
        let lanes = self.config.lanes;
        let frame_width = lanes * dim;
        let mut stream = vec![0.0f32; frames * frame_width];
        kernels::parallel_chunks_mut(&mut stream, frame_width, &|t, frame| {
            let source = &input[t * dim..(t + 1) * dim];
            for lane in 0..lanes {
                frame[lane * dim..(lane + 1) * dim].copy_from_slice(source);
            }
        });
        let mut next_stream = vec![0.0f32; frames * frame_width];
        for (index, layer) in self.layers.iter().enumerate() {
            let routings = kernels::parallel_map(frames, &|frame| self.lanes.route(index, index, &stream[frame * frame_width..(frame + 1) * frame_width], dim));
            let mut x = vec![0.0f32; frames * dim];
            kernels::parallel_chunks_mut(&mut x, dim, &|t, out| out.copy_from_slice(&routings[t].input));
            self.run_block(layer, &mut x, frames);
            kernels::parallel_chunks_mut(&mut next_stream, frame_width, &|t, out| {
                let mut delta = vec![0.0f32; dim];
                LaneWeights::block_delta(&routings[t], &x[t * dim..(t + 1) * dim], &mut delta);
                self.lanes.combine_into(&routings[t], &stream[t * frame_width..(t + 1) * frame_width], &delta, dim, out);
            });
            std::mem::swap(&mut stream, &mut next_stream);
        }
        let mut output = vec![0.0f32; frames * dim];
        kernels::parallel_chunks_mut(&mut output, dim, &|t, slot| {
            let mut mean = vec![0.0f32; dim];
            for lane in stream[t * frame_width..(t + 1) * frame_width].chunks_exact(dim) {
                for (m, value) in mean.iter_mut().zip(lane) {
                    *m += value / lanes as f32;
                }
            }
            zero_centred_norm(&mean, &self.final_norm, slot);
        });
        output
    }

    fn run_block(&self, layer: &EncoderLayer, x: &mut [f32], frames: usize) {
        let dim = self.config.dim;
        kernels::parallel_chunks_mut(x, dim, &|_, frame| {
            let mut normed = vec![0.0f32; dim];
            zero_centred_norm(frame, &layer.first_norm, &mut normed);
            let update = layer.hadamard_first.apply(&normed);
            for (slot, value) in frame.iter_mut().zip(&update) {
                *slot += HALF_RESIDUAL * value;
            }
        });
        let mut normed: Vec<f32> = vec![0.0; x.len()];
        kernels::parallel_chunks_mut(&mut normed, dim, &|t, out| zero_centred_norm(&x[t * dim..(t + 1) * dim], &layer.attention_norm, out));
        let attended = self.attend(layer, &normed, frames);
        let gate = sigmoid(layer.attention_gate);
        kernels::parallel_chunks_mut(x, dim, &|t, frame| {
            let mut attention = attended[t * dim..(t + 1) * dim].to_vec();
            zero_centred_norm_in_place(&mut attention, &layer.post_attention_norm);
            for (slot, value) in frame.iter_mut().zip(&attention) {
                *slot += gate * value;
            }
        });
        let convolved = self.convolution(layer, x, frames);
        x.iter_mut().zip(&convolved).for_each(|(slot, value)| *slot += value);
        kernels::parallel_chunks_mut(x, dim, &|_, frame| {
            let mut normed = vec![0.0f32; dim];
            zero_centred_norm(frame, &layer.last_norm, &mut normed);
            let update = layer.hadamard.apply(&normed);
            for (slot, value) in frame.iter_mut().zip(&update) {
                *slot += HALF_RESIDUAL * value;
            }
        });
    }

    fn convolution(&self, layer: &EncoderLayer, x: &[f32], frames: usize) -> Vec<f32> {
        let dim = self.config.dim;
        let mut normed = vec![0.0f32; x.len()];
        kernels::parallel_chunks_mut(&mut normed, dim, &|t, out| {
            zero_centred_norm(&x[t * dim..(t + 1) * dim], &layer.conv_norm, out);
            fake_quantise_int8(out);
        });
        let rotated = rotated_batch(&layer.pointwise_in, &normed, frames);
        let expanded = project(&layer.pointwise_in, &rotated, frames);
        let mut gated = vec![0.0f32; frames * dim];
        kernels::parallel_chunks_mut(&mut gated, dim, &|t, out| {
            let row = &expanded[t * 2 * dim..(t + 1) * 2 * dim];
            for channel in 0..dim {
                out[channel] = row[channel] * sigmoid(row[dim + channel]);
            }
        });
        let kernel = self.config.conv_kernel;
        let half = kernel / 2;
        let mut filtered = vec![0.0f32; frames * dim];
        kernels::parallel_chunks_mut(&mut filtered, dim, &|t, out| {
            for tap in 0..kernel {
                let source = t as isize + tap as isize - half as isize;
                if source < 0 || source as usize >= frames {
                    continue;
                }
                let row = &gated[source as usize * dim..(source as usize + 1) * dim];
                let weights = &layer.depthwise[tap * dim..(tap + 1) * dim];
                for channel in 0..dim {
                    out[channel] += row[channel] * weights[channel];
                }
            }
            zero_centred_norm_in_place(out, &layer.conv_out_norm);
            out.iter_mut().for_each(|value| *value = silu(*value));
            fake_quantise_int8(out);
        });
        let rotated = rotated_batch(&layer.pointwise_out, &filtered, frames);
        project(&layer.pointwise_out, &rotated, frames)
    }

    fn attend(&self, layer: &EncoderLayer, normed: &[f32], frames: usize) -> Vec<f32> {
        let config = &self.config;
        let (qk, vd) = (config.qk_dim, config.v_dim);
        let mut quantised = normed.to_vec();
        kernels::parallel_chunks_mut(&mut quantised, config.dim, &|_, row| fake_quantise_int8(row));
        let rotated = rotated_batch(&layer.q_proj, &quantised, frames);
        let mut q = project(&layer.q_proj, &rotated, frames);
        let mut k = project(&layer.k_proj, &rotated, frames);
        let v = project(&layer.v_proj, &rotated, frames);
        let gate = project(&layer.gate_proj, &rotated, frames);
        let query_width = config.heads * qk;
        let key_width = config.kv_heads * qk;
        let value_width = config.kv_heads * vd;
        kernels::parallel_chunks_mut(&mut q, query_width, &|t, row| {
            for head_values in row.chunks_exact_mut(qk) {
                zero_centred_norm_in_place(head_values, &layer.q_norm);
                apply_rotary(head_values, t, &self.frequencies);
            }
        });
        let mut query_codes = vec![0i8; frames * query_width];
        kernels::parallel_chunks_mut(&mut query_codes, query_width, &|t, codes| {
            for (code_head, head_values) in codes.chunks_exact_mut(qk).zip(q[t * query_width..(t + 1) * query_width].chunks_exact(qk)) {
                quantise_int8(head_values, code_head);
            }
        });
        let query_scales: Vec<f32> = q.chunks_exact(qk).map(int8_scale).collect();
        kernels::parallel_chunks_mut(&mut k, key_width, &|t, row| {
            for head_values in row.chunks_exact_mut(qk) {
                zero_centred_norm_in_place(head_values, &layer.k_norm);
                apply_rotary(head_values, t, &self.frequencies);
            }
        });
        let mut key_codes = vec![0i8; frames * key_width];
        kernels::parallel_chunks_mut(&mut key_codes, key_width, &|t, codes| {
            for (code_head, head_values) in codes.chunks_exact_mut(qk).zip(k[t * key_width..(t + 1) * key_width].chunks_exact(qk)) {
                quantise_int8(head_values, code_head);
            }
        });
        let key_scales: Vec<f32> = k.chunks_exact(qk).map(int8_scale).collect();
        let mut value_codes = vec![0i8; frames * value_width];
        kernels::parallel_chunks_mut(&mut value_codes, value_width, &|t, codes| {
            for (code_head, head_values) in codes.chunks_exact_mut(vd).zip(v[t * value_width..(t + 1) * value_width].chunks_exact(vd)) {
                quantise_int8(head_values, code_head);
            }
        });
        let value_scales: Vec<f32> = v.chunks_exact(vd).map(int8_scale).collect();
        let group = config.heads / config.kv_heads;
        let inverse_root = 1.0 / (qk as f32).sqrt();
        let attention_width = config.heads * vd;
        let mut context = vec![0.0f32; frames * attention_width];
        kernels::parallel_chunks_mut(&mut context, attention_width, &|t, out| {
            let mut scores = vec![0.0f32; frames];
            for head in 0..config.heads {
                let kv = head / group;
                let query = &query_codes[t * query_width + head * qk..t * query_width + (head + 1) * qk];
                let query_scale = query_scales[t * config.heads + head] * inverse_root;
                for (s, slot) in scores.iter_mut().enumerate() {
                    let keys = &key_codes[s * key_width + kv * qk..s * key_width + (kv + 1) * qk];
                    let dot: i32 = query.iter().zip(keys).map(|(a, b)| *a as i32 * *b as i32).sum();
                    *slot = dot as f32 * key_scales[s * config.kv_heads + kv] * query_scale;
                }
                softmax_in_place(&mut scores);
                let head_out = &mut out[head * vd..(head + 1) * vd];
                for (s, weight) in scores.iter().enumerate() {
                    let scale = value_scales[s * config.kv_heads + kv] * weight;
                    let codes = &value_codes[s * value_width + kv * vd..s * value_width + (kv + 1) * vd];
                    for (slot, code) in head_out.iter_mut().zip(codes) {
                        *slot += scale * *code as f32;
                    }
                }
            }
        });
        kernels::parallel_chunks_mut(&mut context, attention_width, &|t, row| {
            for (value, gate) in row.iter_mut().zip(&gate[t * attention_width..(t + 1) * attention_width]) {
                *value *= sigmoid(*gate);
            }
            fake_quantise_int8(row);
        });
        let rotated = rotated_batch(&layer.out_proj, &context, frames);
        project(&layer.out_proj, &rotated, frames)
    }
}
