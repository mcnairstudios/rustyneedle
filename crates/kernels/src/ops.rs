const NORM_EPSILON: f32 = 1e-6;
const INT8_MAX: f32 = 127.0;
const INT8_MIN: f32 = -128.0;

const LANES: usize = 8;

fn sum_of_squares(values: &[f32]) -> f32 {
    let mut lanes = [0.0f32; LANES];
    let mut chunks = values.chunks_exact(LANES);
    for chunk in &mut chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            *lane += value * value;
        }
    }
    let tail: f32 = chunks.remainder().iter().map(|value| value * value).sum();
    lanes.iter().sum::<f32>() + tail
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; LANES];
    let mut left = a.chunks_exact(LANES);
    let mut right = b.chunks_exact(LANES);
    for (x, y) in (&mut left).zip(&mut right) {
        for ((lane, p), q) in lanes.iter_mut().zip(x).zip(y) {
            *lane += p * q;
        }
    }
    let tail: f32 = left.remainder().iter().zip(right.remainder()).map(|(p, q)| p * q).sum();
    lanes.iter().sum::<f32>() + tail
}

pub(crate) fn peak_magnitude(values: &[f32]) -> f32 {
    let mut lanes = [0.0f32; LANES];
    let mut chunks = values.chunks_exact(LANES);
    for chunk in &mut chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            *lane = lane.max(value.abs());
        }
    }
    let tail = chunks.remainder().iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
    lanes.iter().fold(tail, |acc, value| acc.max(*value))
}

pub fn zero_centred_norm(input: &[f32], scale: &[f32], output: &mut [f32]) {
    let mean_square = sum_of_squares(input) / input.len() as f32;
    let inverse = 1.0 / (mean_square + NORM_EPSILON).sqrt();
    for ((out, value), gain) in output.iter_mut().zip(input).zip(scale) {
        *out = (1.0 + gain) * value * inverse;
    }
}

pub fn zero_centred_norm_in_place(values: &mut [f32], scale: &[f32]) {
    let mean_square = sum_of_squares(values) / values.len() as f32;
    let inverse = 1.0 / (mean_square + NORM_EPSILON).sqrt();
    for (value, gain) in values.iter_mut().zip(scale) {
        *value = (1.0 + gain) * *value * inverse;
    }
}

pub fn unit_rms(input: &[f32], output: &mut [f32]) {
    let mean_square = sum_of_squares(input) / input.len() as f32;
    let inverse = 1.0 / (mean_square + NORM_EPSILON).sqrt();
    for (out, value) in output.iter_mut().zip(input) {
        *out = value * inverse;
    }
}

#[inline]
pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[inline]
pub fn silu(value: f32) -> f32 {
    value / (1.0 + fast_exp(-value))
}

const LOG2_E: f32 = std::f32::consts::LOG2_E;
const EXP_FLOOR: f32 = -87.0;
const EXP_CEILING: f32 = 87.0;

#[inline(always)]
pub fn fast_exp(value: f32) -> f32 {
    let scaled = value.clamp(EXP_FLOOR, EXP_CEILING) * LOG2_E;
    let whole = scaled.round();
    let fraction = scaled - whole;
    let polynomial = 1.0
        + fraction * (std::f32::consts::LN_2 + fraction * (0.240_226_5 + fraction * (0.055_504_11 + fraction * (0.009_618_129 + fraction * 0.001_333_355_8))));
    polynomial * f32::from_bits(((whole as i32 + 127) << 23) as u32)
}

pub fn softmax_in_place(values: &mut [f32]) {
    let peak = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for value in values.iter_mut() {
        *value = fast_exp(*value - peak);
        total += *value;
    }
    let inverse = 1.0 / total;
    values.iter_mut().for_each(|value| *value *= inverse);
}

pub fn int8_scale(values: &[f32]) -> f32 {
    let peak = peak_magnitude(values);
    if peak > 0.0 {
        peak / INT8_MAX
    } else {
        1.0
    }
}

pub fn quantise_int8(values: &[f32], codes: &mut [i8]) -> f32 {
    let scale = int8_scale(values);
    for (code, value) in codes.iter_mut().zip(values) {
        *code = (value / scale).round_ties_even().clamp(INT8_MIN, INT8_MAX) as i8;
    }
    scale
}

pub fn fake_quantise_int8(values: &mut [f32]) {
    let scale = int8_scale(values);
    for value in values.iter_mut() {
        *value = (*value / scale).round_ties_even().clamp(INT8_MIN, INT8_MAX) * scale;
    }
}

pub fn rotary_frequencies(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim / 2).map(|index| 1.0 / theta.powf(2.0 * index as f32 / head_dim as f32)).collect()
}

pub fn apply_rotary(head: &mut [f32], position: usize, frequencies: &[f32]) {
    let half = head.len() / 2;
    let (first, second) = head.split_at_mut(half);
    for index in 0..half {
        let angle = position as f32 * frequencies[index];
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (first[index], second[index]);
        first[index] = a * cos - b * sin;
        second[index] = b * cos + a * sin;
    }
}

fn kronecker_apply_fixed<const R: usize>(input: &[f32], left: &[f32], left_size: usize, right: &[f32], output: &mut [f32]) {
    let mut mixed = vec![0.0f32; left_size * R];
    for (row_in, row_out) in input.chunks_exact(R).zip(mixed.chunks_exact_mut(R)) {
        let mut accumulator = [0.0f32; R];
        for (value, right_row) in row_in.iter().zip(right.chunks_exact(R)) {
            for (slot, factor) in accumulator.iter_mut().zip(right_row) {
                *slot += value * factor;
            }
        }
        row_out.copy_from_slice(&accumulator);
    }
    for (k, out_row) in output.chunks_exact_mut(R).enumerate() {
        let mut accumulator = [0.0f32; R];
        for (i, mixed_row) in mixed.chunks_exact(R).enumerate() {
            let gain = left[i * left_size + k];
            for (slot, value) in accumulator.iter_mut().zip(mixed_row) {
                *slot += gain * value;
            }
        }
        out_row.copy_from_slice(&accumulator);
    }
}

pub fn kronecker_apply(input: &[f32], left: &[f32], left_size: usize, right: &[f32], right_size: usize, output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if right_size == 32 && left_size == 32 {
        return kronecker_neon::apply(input, left, right, output);
    }
    if right_size == 32 {
        return kronecker_apply_fixed::<32>(input, left, left_size, right, output);
    }
    let mut mixed = vec![0.0f32; left_size * right_size];
    for i in 0..left_size {
        let mixed_row = &mut mixed[i * right_size..(i + 1) * right_size];
        for j in 0..right_size {
            let value = input[i * right_size + j];
            let right_row = &right[j * right_size..(j + 1) * right_size];
            for (slot, factor) in mixed_row.iter_mut().zip(right_row) {
                *slot += value * factor;
            }
        }
    }
    output.iter_mut().for_each(|value| *value = 0.0);
    for i in 0..left_size {
        let mixed_row = &mixed[i * right_size..(i + 1) * right_size];
        for k in 0..left_size {
            let gain = left[i * left_size + k];
            let out_row = &mut output[k * right_size..(k + 1) * right_size];
            for (slot, value) in out_row.iter_mut().zip(mixed_row) {
                *slot += gain * value;
            }
        }
    }
}

pub fn sinkhorn_in_place(logits: &mut [f32], size: usize, iterations: usize) {
    for row in logits.chunks_exact_mut(size) {
        let peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.iter_mut().for_each(|value| *value = (*value - peak).exp());
    }
    for _ in 0..iterations {
        for row in logits.chunks_exact_mut(size) {
            let total: f32 = row.iter().sum();
            let inverse = 1.0 / total.max(f32::MIN_POSITIVE);
            row.iter_mut().for_each(|value| *value *= inverse);
        }
        for column in 0..size {
            let total: f32 = (0..size).map(|row| logits[row * size + column]).sum();
            let inverse = 1.0 / total.max(f32::MIN_POSITIVE);
            for row in 0..size {
                logits[row * size + column] *= inverse;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod kronecker_neon {
    use std::arch::aarch64::*;

    const SIZE: usize = 32;

    #[target_feature(enable = "neon")]
    unsafe fn first_stage(input: &[f32], right: &[f32], mixed: &mut [f32]) {
        for row_block in (0..SIZE).step_by(4) {
            for column_half in (0..SIZE).step_by(16) {
                let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
                for chunk in (0..SIZE).step_by(4) {
                    let coefficients = [
                        vld1q_f32(input.as_ptr().add(row_block * SIZE + chunk)),
                        vld1q_f32(input.as_ptr().add((row_block + 1) * SIZE + chunk)),
                        vld1q_f32(input.as_ptr().add((row_block + 2) * SIZE + chunk)),
                        vld1q_f32(input.as_ptr().add((row_block + 3) * SIZE + chunk)),
                    ];
                    macro_rules! step {
                        ($lane:literal) => {{
                            let base = right.as_ptr().add((chunk + $lane) * SIZE + column_half);
                            let columns = [vld1q_f32(base), vld1q_f32(base.add(4)), vld1q_f32(base.add(8)), vld1q_f32(base.add(12))];
                            for (row, slot) in acc.iter_mut().enumerate() {
                                for (column, value) in slot.iter_mut().enumerate() {
                                    *value = vfmaq_laneq_f32::<$lane>(*value, columns[column], coefficients[row]);
                                }
                            }
                        }};
                    }
                    step!(0);
                    step!(1);
                    step!(2);
                    step!(3);
                }
                for (row, slot) in acc.iter().enumerate() {
                    for (column, value) in slot.iter().enumerate() {
                        vst1q_f32(mixed.as_mut_ptr().add((row_block + row) * SIZE + column_half + column * 4), *value);
                    }
                }
            }
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn second_stage(left: &[f32], mixed: &[f32], output: &mut [f32]) {
        for row_block in (0..SIZE).step_by(4) {
            for column_half in (0..SIZE).step_by(16) {
                let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
                for source in 0..SIZE {
                    let coefficients = vld1q_f32(left.as_ptr().add(source * SIZE + row_block));
                    let base = mixed.as_ptr().add(source * SIZE + column_half);
                    let columns = [vld1q_f32(base), vld1q_f32(base.add(4)), vld1q_f32(base.add(8)), vld1q_f32(base.add(12))];
                    macro_rules! row {
                        ($row:literal) => {
                            for (column, value) in acc[$row].iter_mut().enumerate() {
                                *value = vfmaq_laneq_f32::<$row>(*value, columns[column], coefficients);
                            }
                        };
                    }
                    row!(0);
                    row!(1);
                    row!(2);
                    row!(3);
                }
                for (row, slot) in acc.iter().enumerate() {
                    for (column, value) in slot.iter().enumerate() {
                        vst1q_f32(output.as_mut_ptr().add((row_block + row) * SIZE + column_half + column * 4), *value);
                    }
                }
            }
        }
    }

    pub fn apply(input: &[f32], left: &[f32], right: &[f32], output: &mut [f32]) {
        assert!(input.len() >= SIZE * SIZE && left.len() >= SIZE * SIZE && right.len() >= SIZE * SIZE && output.len() >= SIZE * SIZE);
        let mut mixed = [0.0f32; SIZE * SIZE];
        unsafe {
            first_stage(input, right, &mut mixed);
            second_stage(left, &mixed, output);
        }
    }
}
