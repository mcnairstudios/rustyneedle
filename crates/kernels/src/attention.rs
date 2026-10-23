use std::cell::RefCell;

const COEFFICIENT_LEVELS: f32 = 2047.0;

thread_local! {
    static ENTRIES: RefCell<Vec<(u32, i16)>> = const { RefCell::new(Vec::new()) };
    static SCRATCH: RefCell<(Vec<i32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

pub struct HeadInputs<'a> {
    pub query: &'a [i8],
    pub query_scale: f32,
    pub keys: &'a [i8],
    pub key_stride: usize,
    pub key_offset: usize,
    pub key_scales: &'a [f32],
    pub values: &'a [i8],
    pub value_stride: usize,
    pub value_offset: usize,
    pub value_panels: &'a [i8],
    pub first: usize,
    pub value_scales: &'a [f32],
    pub width: usize,
}

#[cfg(not(target_arch = "aarch64"))]
fn peak_of(values: &[f32]) -> f32 {
    let mut lanes = [f32::NEG_INFINITY; 8];
    let mut chunks = values.chunks_exact(8);
    for chunk in &mut chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            *lane = lane.max(*value);
        }
    }
    let tail = chunks.remainder().iter().fold(f32::NEG_INFINITY, |acc, value| acc.max(*value));
    lanes.iter().fold(tail, |acc, value| acc.max(*value))
}

fn weighted_into(coefficients: &[f32], inputs: &HeadInputs<'_>, out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if inputs.width.is_multiple_of(16) {
        let peak = crate::ops::peak_magnitude(coefficients);
        if peak <= 0.0 {
            out.iter_mut().for_each(|value| *value = 0.0);
            return;
        }
        crate::neon::accumulate_panels(coefficients, inputs.first, peak / COEFFICIENT_LEVELS, inputs.value_panels, inputs.width, out);
        return;
    }
    weighted_values(coefficients, inputs.values, inputs.value_stride, inputs.value_offset, inputs.width, out);
}

pub fn attend_head(inputs: &HeadInputs<'_>, out: &mut [f32]) {
    let count = inputs.key_scales.len();
    SCRATCH.with(|cell| {
        let (integer_scores, coefficients) = &mut *cell.borrow_mut();
        integer_scores.clear();
        integer_scores.resize(count, 0);
        score_positions(inputs.query, inputs.keys, inputs.key_stride, inputs.key_offset, count, integer_scores);
        coefficients.clear();
        coefficients.resize(count, 0.0);
        #[cfg(target_arch = "aarch64")]
        let total = crate::neon::softmax_weights(integer_scores, inputs.key_scales, inputs.value_scales, inputs.query_scale, coefficients);
        #[cfg(not(target_arch = "aarch64"))]
        let total = {
            for ((slot, score), scale) in coefficients.iter_mut().zip(integer_scores.iter()).zip(inputs.key_scales) {
                *slot = *score as f32 * scale * inputs.query_scale;
            }
            let peak = peak_of(coefficients);
            let mut total = 0.0f32;
            for (slot, scale) in coefficients.iter_mut().zip(inputs.value_scales) {
                let exponent = crate::ops::fast_exp(*slot - peak);
                total += exponent;
                *slot = exponent * scale;
            }
            total
        };
        weighted_into(coefficients, inputs, out);
        let inverse = 1.0 / total;
        out.iter_mut().for_each(|value| *value *= inverse);
    });
}

pub fn score_positions(query: &[i8], keys: &[i8], stride: usize, offset: usize, count: usize, out: &mut [i32]) {
    #[cfg(target_arch = "aarch64")]
    if query.len().is_multiple_of(16) && query.len() <= 128 {
        crate::neon::score_positions_i8(query, keys, stride, offset, count, out);
        return;
    }
    for (position, slot) in out.iter_mut().enumerate().take(count) {
        let base = position * stride + offset;
        *slot = query.iter().zip(&keys[base..base + query.len()]).map(|(a, b)| *a as i32 * *b as i32).sum();
    }
}

pub fn weighted_values(coefficients: &[f32], values: &[i8], stride: usize, offset: usize, width: usize, out: &mut [f32]) {
    let peak = coefficients.iter().fold(0.0f32, |acc, c| acc.max(c.abs()));
    if peak <= 0.0 {
        out.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    let scale = peak / COEFFICIENT_LEVELS;
    ENTRIES.with(|cell| {
        let mut entries = cell.borrow_mut();
        entries.clear();
        #[cfg(target_arch = "aarch64")]
        crate::neon::collect_entries(coefficients, scale, &mut entries);
        #[cfg(not(target_arch = "aarch64"))]
        entries.extend(coefficients.iter().enumerate().map(|(position, c)| (position as u32, (c / scale).round() as i16)).filter(|(_, weight)| *weight != 0));
        accumulate(&entries, values, stride, offset, width, scale, out);
    });
}

fn accumulate(entries: &[(u32, i16)], values: &[i8], stride: usize, offset: usize, width: usize, scale: f32, out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if width.is_multiple_of(16) && width <= 128 {
        crate::neon::accumulate_values(entries, values, stride, offset, width, scale, out);
        return;
    }
    let mut totals = vec![0i64; width];
    for &(position, weight) in entries {
        let base = position as usize * stride + offset;
        for (slot, code) in totals.iter_mut().zip(&values[base..base + width]) {
            *slot += weight as i64 * *code as i64;
        }
    }
    for (slot, total) in out.iter_mut().zip(totals) {
        *slot = total as f32 * scale;
    }
}
