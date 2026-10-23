use cact::{Archive, LaneTensors};
use kernels::{fake_quantise_int8, sigmoid, sinkhorn_in_place, unit_rms, QuantisedMatrix};

use crate::tensors::floats;

const SINKHORN_ITERATIONS: usize = 20;
const PRE_LANE_BOOST: f32 = 8.0;
const PRE_LANE_FLOOR: f32 = -4.0;
const POST_LANE_FLOOR: f32 = -4.0;

pub struct LaneWeights {
    lanes: usize,
    a_pre: Vec<f32>,
    a_post: Vec<f32>,
    a_res: Vec<f32>,
    b_pre: Vec<f32>,
    b_post: Vec<f32>,
    b_res: Vec<f32>,
    phi_pre: QuantisedMatrix,
    phi_post: QuantisedMatrix,
    phi_res: QuantisedMatrix,
}

pub struct Routing {
    pub input: Vec<f32>,
    post: Vec<f32>,
    residual: Vec<f32>,
}

impl LaneWeights {
    pub fn load(archive: &Archive, tensors: &LaneTensors, lanes: usize) -> Self {
        Self {
            lanes,
            a_pre: floats(archive, tensors.a_pre),
            a_post: floats(archive, tensors.a_post),
            a_res: floats(archive, tensors.a_res),
            b_pre: floats(archive, tensors.b_pre),
            b_post: floats(archive, tensors.b_post),
            b_res: floats(archive, tensors.b_res),
            phi_pre: QuantisedMatrix::from_archive(archive, tensors.phi_pre),
            phi_post: QuantisedMatrix::from_archive(archive, tensors.phi_post),
            phi_res: QuantisedMatrix::from_archive(archive, tensors.phi_res),
        }
    }

    pub fn route(&self, layer: usize, slot: usize, stream: &[f32], width: usize) -> Routing {
        let lanes = self.lanes;
        let mut normalised = vec![0.0; stream.len()];
        unit_rms(stream, &mut normalised);
        fake_quantise_int8(&mut normalised);
        let mut rotated = Vec::new();
        self.phi_pre.prepare(&normalised, &mut rotated);
        let mut pre = vec![0.0; lanes];
        let mut post = vec![0.0; lanes];
        let mut residual = vec![0.0; lanes * lanes];
        QuantisedMatrix::multiply_fused(
            vec![
                (&self.phi_pre, layer * lanes, &mut pre[..]),
                (&self.phi_post, layer * lanes, &mut post[..]),
                (&self.phi_res, layer * lanes * lanes, &mut residual[..]),
            ],
            &rotated,
        );
        let own = slot % lanes;
        let mut input = vec![0.0; width];
        for lane in 0..lanes {
            let pre_offset = if lane == own { PRE_LANE_BOOST + PRE_LANE_FLOOR } else { PRE_LANE_FLOOR };
            let gate = sigmoid(self.a_pre[layer] * pre[lane] + self.b_pre[layer * lanes + lane] + pre_offset);
            for (slot, value) in input.iter_mut().zip(&stream[lane * width..(lane + 1) * width]) {
                *slot += gate * value;
            }
            let post_offset = if lane == own { 0.0 } else { POST_LANE_FLOOR };
            post[lane] = 2.0 * sigmoid(self.a_post[layer] * post[lane] + self.b_post[layer * lanes + lane] + post_offset);
        }
        let base = layer * lanes * lanes;
        for (index, value) in residual.iter_mut().enumerate() {
            *value = self.a_res[layer] * *value + self.b_res[base + index];
        }
        sinkhorn_in_place(&mut residual, lanes, SINKHORN_ITERATIONS);
        Routing { input, post, residual }
    }

    pub fn combine_into(&self, routing: &Routing, stream: &[f32], block_delta: &[f32], width: usize, next: &mut [f32]) {
        let lanes = self.lanes;
        for lane in 0..lanes {
            let out = &mut next[lane * width..(lane + 1) * width];
            out.iter_mut().for_each(|slot| *slot = 0.0);
            for source in 0..lanes {
                let weight = routing.residual[lane * lanes + source];
                for (slot, value) in out.iter_mut().zip(&stream[source * width..(source + 1) * width]) {
                    *slot += weight * value;
                }
            }
            for (slot, delta) in out.iter_mut().zip(block_delta) {
                *slot += routing.post[lane] * delta;
            }
        }
    }

    pub fn block_delta(routing: &Routing, output: &[f32], delta: &mut [f32]) {
        for ((slot, out), input) in delta.iter_mut().zip(output).zip(&routing.input) {
            *slot = out - input;
        }
    }
}
