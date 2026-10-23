use std::collections::VecDeque;

use cact::{Archive, EngramTensors};
use kernels::{fake_quantise_int8, QuantisedMatrix};

use crate::config::EngramConfig;
use crate::tensors::floats;

const HASH_SEED: u32 = 0x9E37_79B9;
const HASH_PRIME: u32 = 0x0100_0193;

pub struct EngramWeights {
    tables: QuantisedMatrix,
    key_proj: QuantisedMatrix,
    value_proj: QuantisedMatrix,
    taps: Vec<f32>,
}

impl EngramWeights {
    pub fn load(archive: &Archive, tensors: &EngramTensors) -> Self {
        Self {
            tables: QuantisedMatrix::from_archive(archive, tensors.tables),
            key_proj: QuantisedMatrix::from_archive(archive, tensors.key_proj),
            value_proj: QuantisedMatrix::from_archive(archive, tensors.value_proj),
            taps: floats(archive, tensors.taps),
        }
    }
}

pub fn table_indices(config: &EngramConfig, tokens: &[u32], position: usize) -> Vec<usize> {
    let stride = if config.seed_heads == 0 { config.heads_per_order } else { config.seed_heads };
    let mut indices = Vec::with_capacity(config.tables);
    for (order_index, &order) in config.orders.iter().enumerate() {
        for head in 0..config.heads_per_order {
            let mut accumulator = HASH_SEED.wrapping_mul((order_index * stride + head + 1) as u32);
            for back in 0..order {
                let token = position.checked_sub(back).map_or(0, |at| tokens[at]);
                accumulator = (accumulator ^ token).wrapping_mul(HASH_PRIME);
            }
            accumulator ^= accumulator >> 15;
            indices.push((accumulator % config.slots as u32) as usize);
        }
    }
    indices
}

#[derive(Clone)]
pub struct EngramSiteState {
    value_history: VecDeque<Vec<f32>>,
}

pub struct EngramOutput {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

impl Default for EngramSiteState {
    fn default() -> Self {
        Self::new()
    }
}

impl EngramSiteState {
    pub fn new() -> Self {
        Self { value_history: VecDeque::new() }
    }

    pub fn contribute(&mut self, weights: &EngramWeights, config: &EngramConfig, indices: &[usize], position: usize) -> EngramOutput {
        let mut gathered = Vec::with_capacity(config.tables * config.sub_dim);
        let mut row = Vec::new();
        for (table, (&index, order)) in indices.iter().zip(table_orders(config)).enumerate() {
            if position + 1 >= order {
                weights.tables.dequantise_row(table * config.slots + index, &mut row);
                gathered.extend_from_slice(&row);
            } else {
                gathered.extend(std::iter::repeat_n(0.0, config.sub_dim));
            }
        }
        fake_quantise_int8(&mut gathered);
        let mut rotated = Vec::new();
        weights.key_proj.prepare(&gathered, &mut rotated);
        let mut key = vec![0.0; weights.key_proj.rows()];
        let mut raw_value = vec![0.0; weights.value_proj.rows()];
        QuantisedMatrix::multiply_fused(vec![(&weights.key_proj, 0, &mut key[..]), (&weights.value_proj, 0, &mut raw_value[..])], &rotated);
        let width = raw_value.len();
        self.value_history.push_front(raw_value);
        self.value_history.truncate((config.conv_taps - 1) * config.dilation + 1);
        let mut value = vec![0.0; width];
        for tap in 0..config.conv_taps {
            let back = tap * config.dilation;
            if position < back {
                continue;
            }
            let past = &self.value_history[back];
            let gains = &weights.taps[tap * width..(tap + 1) * width];
            for ((slot, gain), sample) in value.iter_mut().zip(gains).zip(past) {
                *slot += gain * sample;
            }
        }
        EngramOutput { key, value }
    }
}

fn table_orders(config: &EngramConfig) -> impl Iterator<Item = usize> + '_ {
    config.orders.iter().flat_map(|&order| std::iter::repeat_n(order, config.heads_per_order))
}

impl EngramSiteState {
    pub fn contribute_batch(
        &mut self,
        weights: &EngramWeights,
        config: &EngramConfig,
        tokens: &[u32],
        first_position: usize,
        count: usize,
    ) -> Vec<EngramOutput> {
        let width = config.tables * config.sub_dim;
        let mut gathered = vec![0.0f32; count * width];
        kernels::parallel_chunks_mut(&mut gathered, width, &|t, slot| {
            let position = first_position + t;
            let indices = table_indices(config, tokens, position);
            let mut row = Vec::new();
            for (table, (&index, order)) in indices.iter().zip(table_orders(config)).enumerate() {
                if position + 1 >= order {
                    weights.tables.dequantise_row(table * config.slots + index, &mut row);
                    slot[table * config.sub_dim..(table + 1) * config.sub_dim].copy_from_slice(&row);
                }
            }
            fake_quantise_int8(slot);
        });
        let mut rotated = Vec::new();
        weights.key_proj.prepare_batch(&gathered, count, &mut rotated);
        let mut keys = vec![0.0f32; count * weights.key_proj.rows()];
        weights.key_proj.multiply_batch(&rotated, count, &mut keys);
        let mut raw = vec![0.0f32; count * weights.value_proj.rows()];
        weights.value_proj.multiply_batch(&rotated, count, &mut raw);
        let dim = weights.value_proj.rows();
        let previous: Vec<Vec<f32>> = self.value_history.iter().cloned().collect();
        let mut values = vec![0.0f32; count * dim];
        kernels::parallel_chunks_mut(&mut values, dim, &|t, slot| {
            let position = first_position + t;
            for tap in 0..config.conv_taps {
                let back = tap * config.dilation;
                if position < back {
                    continue;
                }
                let source: &[f32] = if back <= t { &raw[(t - back) * dim..(t - back + 1) * dim] } else { &previous[back - t - 1] };
                let gains = &weights.taps[tap * dim..(tap + 1) * dim];
                for ((value, gain), sample) in slot.iter_mut().zip(gains).zip(source) {
                    *value += gain * sample;
                }
            }
        });
        let keep = (config.conv_taps - 1) * config.dilation + 1;
        for t in count.saturating_sub(keep)..count {
            self.value_history.push_front(raw[t * dim..(t + 1) * dim].to_vec());
        }
        self.value_history.truncate(keep);
        keys.chunks_exact(dim).zip(values.chunks_exact(dim)).map(|(key, value)| EngramOutput { key: key.to_vec(), value: value.to_vec() }).collect()
    }
}
