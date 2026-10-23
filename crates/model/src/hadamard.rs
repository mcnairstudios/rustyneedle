use cact::{Archive, HadamardTensors};
use kernels::{kronecker_apply, silu, softmax_in_place};

use crate::tensors::floats;

const CONDITION_CHAINS_RANK: usize = 8;

pub struct HadamardWeights {
    size: usize,
    d1: Vec<f32>,
    d2: Vec<f32>,
    b2: Vec<f32>,
    d3: Vec<f32>,
    d4: Vec<f32>,
    factors: [(Vec<f32>, usize, Vec<f32>, usize); 3],
    cond_v: Vec<f32>,
    cond_u: Vec<f32>,
    rank: usize,
    first_permutation: Vec<usize>,
    second_permutation: Vec<usize>,
}

fn block_size(matrix: &[f32]) -> usize {
    (matrix.len() as f64).sqrt().round() as usize
}

fn factor(archive: &Archive, left: usize, right: usize) -> (Vec<f32>, usize, Vec<f32>, usize) {
    let (left, right) = (floats(archive, left), floats(archive, right));
    let (left_size, right_size) = (block_size(&left), block_size(&right));
    (left, left_size, right, right_size)
}

impl HadamardWeights {
    pub fn load(archive: &Archive, tensors: &HadamardTensors, permutations: [usize; 2], size: usize) -> Self {
        let permutation = |index| floats(archive, index).into_iter().map(|value| value as usize).collect();
        let cond_v = floats(archive, tensors.cond_v);
        let cond_u = floats(archive, tensors.cond_u);
        let rank = cond_u.len() / size;
        Self {
            size,
            d1: floats(archive, tensors.d1),
            d2: floats(archive, tensors.d2),
            b2: floats(archive, tensors.b2),
            d3: floats(archive, tensors.d3),
            d4: floats(archive, tensors.d4),
            factors: [factor(archive, tensors.w1a, tensors.w1b), factor(archive, tensors.w2a, tensors.w2b), factor(archive, tensors.w3a, tensors.w3b)],
            cond_v,
            cond_u,
            rank,
            first_permutation: permutation(permutations[0]),
            second_permutation: permutation(permutations[1]),
        }
    }

    pub fn apply(&self, input: &[f32]) -> Vec<f32> {
        let width = input.len();
        let mut weights = vec![0.0f32; self.rank];
        if self.rank == CONDITION_CHAINS_RANK {
            let mut chains = [[0.0f32; CONDITION_CHAINS_RANK]; 4];
            for (group, rows) in input.chunks(4).zip(self.cond_v.chunks(4 * CONDITION_CHAINS_RANK)) {
                for ((value, row), chain) in group.iter().zip(rows.chunks_exact(CONDITION_CHAINS_RANK)).zip(chains.iter_mut()) {
                    for (slot, factor) in chain.iter_mut().zip(row) {
                        *slot += value * factor;
                    }
                }
            }
            for chain in &chains {
                for (slot, partial) in weights.iter_mut().zip(chain) {
                    *slot += partial;
                }
            }
        } else {
            for (dimension, value) in input.iter().enumerate() {
                let row = &self.cond_v[dimension * self.rank..(dimension + 1) * self.rank];
                for (slot, factor) in weights.iter_mut().zip(row) {
                    *slot += value * factor;
                }
            }
        }
        softmax_in_place(&mut weights);
        let mut modulation = vec![1.0f32; self.size];
        for (rank, weight) in weights.iter().enumerate() {
            let row = &self.cond_u[rank * self.size..(rank + 1) * self.size];
            for (slot, value) in modulation.iter_mut().zip(row) {
                *slot += weight * value;
            }
        }
        let mut padded = vec![0.0f32; self.size];
        padded[..width].copy_from_slice(input);
        let mut scaled: Vec<f32> = padded.iter().zip(&self.d1).map(|(z, d)| z * d).collect();
        let first = self.apply_factor(0, &scaled);
        let gathered: Vec<f32> = self.first_permutation.iter().map(|&source| first[source]).collect();
        scaled = (0..self.size).map(|i| silu(self.d2[i] * modulation[i] * gathered[i] + self.b2[i])).collect();
        let second = self.apply_factor(1, &scaled);
        let regathered: Vec<f32> = self.second_permutation.iter().map(|&source| second[source]).collect();
        scaled = regathered.iter().zip(&self.d3).map(|(z, d)| z * d).collect();
        let third = self.apply_factor(2, &scaled);
        third.iter().zip(&self.d4).map(|(z, d)| z * d).take(width).collect()
    }

    fn apply_factor(&self, which: usize, input: &[f32]) -> Vec<f32> {
        let (left, left_size, right, right_size) = &self.factors[which];
        let mut output = vec![0.0f32; input.len()];
        kronecker_apply(input, left, *left_size, right, *right_size, &mut output);
        output
    }
}
