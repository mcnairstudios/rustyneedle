use cact::{Archive, Layout};
use kernels::QuantisedMatrix;

use crate::attention::AttentionWeights;
use crate::config::ModelConfig;
use crate::engram::EngramWeights;
use crate::hadamard::HadamardWeights;
use crate::heads::ProbeHead;
use crate::lanes::LaneWeights;
use crate::tensors::{floats, scalar};

pub struct LayerWeights {
    pub norm_in: Vec<f32>,
    pub attention: AttentionWeights,
    pub post_norm: Vec<f32>,
    pub attention_gate: f32,
    pub pre_hadamard_norm: Vec<f32>,
    pub hadamard: HadamardWeights,
}

pub struct Weights {
    pub embedding: QuantisedMatrix,
    pub layers: Vec<LayerWeights>,
    pub lanes: LaneWeights,
    pub engrams: Vec<EngramWeights>,
    pub final_norm: Vec<f32>,
    pub heads: Vec<ProbeHead>,
}

impl Weights {
    pub fn load(archive: &Archive, layout: &Layout, config: &ModelConfig, rows: &[usize]) -> Self {
        let layers = layout
            .layers
            .iter()
            .map(|tensors| LayerWeights {
                norm_in: floats(archive, tensors.norm_in),
                attention: AttentionWeights::load(archive, tensors),
                post_norm: floats(archive, tensors.post_norm),
                attention_gate: scalar(archive, tensors.attn_gate),
                pre_hadamard_norm: floats(archive, tensors.pre_hada),
                hadamard: HadamardWeights::load(archive, &tensors.hadamard, layout.hadamard_perms, config.hadamard_size),
            })
            .collect();
        Self {
            embedding: QuantisedMatrix::from_archive(archive, layout.embedding),
            layers,
            lanes: LaneWeights::load(archive, &layout.lanes, config.lanes),
            engrams: layout.engrams.iter().map(|tensors| EngramWeights::load(archive, tensors)).collect(),
            final_norm: floats(archive, layout.final_norm),
            heads: layout
                .heads
                .iter()
                .map(|tensors| ProbeHead::load(archive, tensors, config.num_layers + 1))
                .map(|head| if rows.len() == config.num_layers + 1 { head } else { head.with_rows(rows) })
                .collect(),
        }
    }
}
