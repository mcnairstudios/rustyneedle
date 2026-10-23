use cact::{Archive, HeadTensors};
use kernels::{unit_rms, QuantisedMatrix};

use crate::tensors::floats;

pub struct ProbeHead {
    pub code: u8,
    layers: usize,
    probes_per_layer: usize,
    queries: usize,
    width: usize,
    probes: Vec<f32>,
    gain: Vec<f32>,
    query: Vec<f32>,
    row_bias: Vec<f32>,
    projection: QuantisedMatrix,
    bias: Vec<f32>,
}

fn dense(matrix: &QuantisedMatrix) -> Vec<f32> {
    let mut all = Vec::with_capacity(matrix.rows() * matrix.columns());
    let mut row = Vec::new();
    for index in 0..matrix.rows() {
        matrix.dequantise_row(index, &mut row);
        all.extend_from_slice(&row);
    }
    all
}

impl ProbeHead {
    pub fn load(archive: &Archive, tensors: &HeadTensors, layers: usize) -> Self {
        let probes = QuantisedMatrix::from_archive(archive, tensors.probes);
        let query = QuantisedMatrix::from_archive(archive, tensors.query);
        let gain = floats(archive, tensors.gain);
        let width = probes.columns();
        let probes_per_layer = probes.rows() / layers;
        Self {
            code: tensors.code,
            layers,
            probes_per_layer,
            queries: query.rows(),
            width,
            probes: dense(&probes),
            gain,
            query: dense(&query),
            row_bias: floats(archive, tensors.row_bias),
            projection: QuantisedMatrix::from_archive(archive, tensors.proj),
            bias: floats(archive, tensors.bias),
        }
    }

    pub fn with_rows(&self, rows: &[usize]) -> ProbeHead {
        let (k, queries, width) = (self.probes_per_layer, self.queries, self.width);
        let old_slots = self.layers * k;
        let mut probes = Vec::with_capacity(rows.len() * k * width);
        let mut gain = Vec::with_capacity(rows.len() * k);
        for &row in rows {
            probes.extend_from_slice(&self.probes[row * k * width..(row + 1) * k * width]);
            gain.extend_from_slice(&self.gain[row * k..(row + 1) * k]);
        }
        let mut row_bias = Vec::with_capacity(queries * rows.len() * k);
        for query in 0..queries {
            for &row in rows {
                row_bias.extend_from_slice(&self.row_bias[query * old_slots + row * k..query * old_slots + (row + 1) * k]);
            }
        }
        ProbeHead {
            code: self.code,
            layers: rows.len(),
            probes_per_layer: k,
            queries,
            width,
            probes,
            gain,
            query: self.query.clone(),
            row_bias,
            projection: self.projection.clone(),
            bias: self.bias.clone(),
        }
    }

    pub fn accumulator(&self) -> ProbeAccumulator {
        let slots = self.layers * self.probes_per_layer;
        ProbeAccumulator { peak: vec![f32::NEG_INFINITY; slots], total: vec![0.0; slots], weighted: vec![0.0; slots * self.width] }
    }

    pub fn pooled(&self, accumulator: &ProbeAccumulator) -> Vec<f32> {
        let width = self.width;
        let slots = self.layers * self.probes_per_layer;
        let mut readouts = vec![0.0f32; slots * width];
        for slot in 0..slots {
            let inverse = 1.0 / accumulator.total[slot];
            let averaged: Vec<f32> = accumulator.weighted[slot * width..(slot + 1) * width].iter().map(|v| v * inverse).collect();
            unit_rms(&averaged, &mut readouts[slot * width..(slot + 1) * width]);
            readouts[slot * width..(slot + 1) * width].iter_mut().for_each(|v| *v *= self.gain[slot]);
        }
        let root = (width as f32).sqrt();
        let mut pooled = vec![0.0f32; self.queries * width];
        for query in 0..self.queries {
            let query_row = &self.query[query * width..(query + 1) * width];
            let mut weights: Vec<f32> = (0..slots)
                .map(|slot| {
                    let dot = kernels::dot_product(&readouts[slot * width..(slot + 1) * width], query_row);
                    dot / root + self.row_bias[query * slots + slot]
                })
                .collect();
            kernels::softmax_in_place(&mut weights);
            let out = &mut pooled[query * width..(query + 1) * width];
            for (slot, weight) in weights.iter().enumerate() {
                for (slot_out, value) in out.iter_mut().zip(&readouts[slot * width..(slot + 1) * width]) {
                    *slot_out += weight * value;
                }
            }
        }
        pooled
    }

    pub fn project(&self, pooled: &[f32]) -> Vec<f32> {
        let mut rotated = Vec::new();
        self.projection.prepare(pooled, &mut rotated);
        let mut output = vec![0.0; self.projection.rows()];
        self.projection.multiply_vector(&rotated, &mut output);
        for (value, bias) in output.iter_mut().zip(&self.bias) {
            *value += bias;
        }
        output
    }

    pub fn observe(&self, accumulator: &mut ProbeAccumulator, cells: &[Vec<f32>]) {
        let width = self.width;
        let root = (width as f32).sqrt();
        for (layer, cell) in cells.iter().enumerate() {
            for probe in 0..self.probes_per_layer {
                let slot = layer * self.probes_per_layer + probe;
                let direction = &self.probes[slot * width..(slot + 1) * width];
                let score = kernels::dot_product(cell, direction) / root;
                let peak = accumulator.peak[slot].max(score);
                let rescale = if accumulator.peak[slot] == f32::NEG_INFINITY { 0.0 } else { (accumulator.peak[slot] - peak).exp() };
                let weight = (score - peak).exp();
                accumulator.total[slot] = accumulator.total[slot] * rescale + weight;
                let target = &mut accumulator.weighted[slot * width..(slot + 1) * width];
                for (slot_value, value) in target.iter_mut().zip(cell) {
                    *slot_value = *slot_value * rescale + weight * value;
                }
                accumulator.peak[slot] = peak;
            }
        }
    }
}

#[derive(Clone)]
pub struct ProbeAccumulator {
    peak: Vec<f32>,
    total: Vec<f32>,
    weighted: Vec<f32>,
}
