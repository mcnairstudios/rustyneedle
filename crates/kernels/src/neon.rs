use crate::dot::{has_dotprod, DotKernel, Sdot, Widening};
use std::arch::aarch64::*;

const BLOCK_BYTES: usize = 16;

pub struct PlaneInput {
    pub planes: Vec<i8>,
    pub scales: Vec<f32>,
    pub bits: u32,
    pub group: usize,
}

pub fn supports(bits: u32, group: usize) -> bool {
    match bits {
        2 => group.is_multiple_of(4 * BLOCK_BYTES),
        4 => group.is_multiple_of(2 * BLOCK_BYTES),
        _ => false,
    }
}

#[target_feature(enable = "neon")]
unsafe fn quantise_sixteen(values: *const f32, divisor: float32x4_t, out: *mut i8) {
    let limit = vdupq_n_f32(127.0);
    let negative = vdupq_n_f32(-127.0);
    let mut lanes = [vdupq_n_s32(0); 4];
    for (index, lane) in lanes.iter_mut().enumerate() {
        let scaled = vdivq_f32(vld1q_f32(values.add(index * 4)), divisor);
        *lane = vcvtq_s32_f32(vminq_f32(vmaxq_f32(vrndnq_f32(scaled), negative), limit));
    }
    let low = vcombine_s16(vqmovn_s32(lanes[0]), vqmovn_s32(lanes[1]));
    let high = vcombine_s16(vqmovn_s32(lanes[2]), vqmovn_s32(lanes[3]));
    vst1q_s8(out, vcombine_s8(vqmovn_s16(low), vqmovn_s16(high)));
}

#[target_feature(enable = "neon")]
unsafe fn quantise_natural_inner(values: &[f32], scale: f32, out: &mut [i8]) {
    let divisor = vdupq_n_f32(scale);
    for chunk in 0..values.len() / 16 {
        quantise_sixteen(values.as_ptr().add(chunk * 16), divisor, out.as_mut_ptr().add(chunk * 16));
    }
}

pub fn quantise_natural(values: &[f32], scale: f32, out: &mut [i8]) {
    assert!(values.len().is_multiple_of(16) && out.len() >= values.len());
    unsafe { quantise_natural_inner(values, scale, out) }
}

#[target_feature(enable = "neon")]
unsafe fn interleave_four_bit(natural: &[i8], planes: &mut [i8]) {
    for block in 0..natural.len() / 32 {
        let first = vld1q_s8(natural.as_ptr().add(block * 32));
        let second = vld1q_s8(natural.as_ptr().add(block * 32 + 16));
        vst1q_s8(planes.as_mut_ptr().add(block * 32), vuzp1q_s8(first, second));
        vst1q_s8(planes.as_mut_ptr().add(block * 32 + 16), vuzp2q_s8(first, second));
    }
}

impl PlaneInput {
    pub fn new(rotated: &[f32], bits: u32, group: usize) -> Self {
        if bits == 4 && group.is_multiple_of(32) {
            return Self::four_bit(rotated, group);
        }
        Self::generic(rotated, bits, group)
    }

    fn four_bit(rotated: &[f32], group: usize) -> Self {
        let groups = rotated.len() / group;
        let mut planes = vec![0i8; rotated.len()];
        let mut scales = vec![1.0f32; groups];
        let mut natural = vec![0i8; group];
        for g in 0..groups {
            let slice = &rotated[g * group..(g + 1) * group];
            let peak = crate::ops::peak_magnitude(slice);
            let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
            scales[g] = scale;
            quantise_natural(slice, scale, &mut natural);
            unsafe { interleave_four_bit(&natural, &mut planes[g * group..(g + 1) * group]) };
        }
        Self { planes, scales, bits: 4, group }
    }

    fn generic(rotated: &[f32], bits: u32, group: usize) -> Self {
        let per_byte = (8 / bits) as usize;
        let block_elements = per_byte * BLOCK_BYTES;
        let groups = rotated.len() / group;
        let mut planes = vec![0i8; rotated.len()];
        let mut scales = vec![1.0f32; groups];
        for g in 0..groups {
            let slice = &rotated[g * group..(g + 1) * group];
            let peak = slice.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
            scales[g] = scale;
            let target = &mut planes[g * group..(g + 1) * group];
            for block in 0..group / block_elements {
                for plane in 0..per_byte {
                    for byte in 0..BLOCK_BYTES {
                        let element = block * block_elements + byte * per_byte + plane;
                        let value = (slice[element] / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
                        target[(block * per_byte + plane) * BLOCK_BYTES + byte] = value;
                    }
                }
            }
        }
        Self { planes, scales, bits, group }
    }
}

#[derive(Clone)]
pub struct IntegerLevels {
    pub table: [i8; 16],
    pub scale: f32,
}

impl IntegerLevels {
    pub fn new(levels: &[f32; 16], count: usize) -> Self {
        let peak = levels[..count].iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
        let mut table = [0i8; 16];
        for (slot, value) in table.iter_mut().zip(&levels[..count]) {
            *slot = (value / scale).round() as i8;
        }
        Self { table, scale }
    }
}

pub fn row_dot(packed: &[u8], norms: &[f32], input: &PlaneInput, levels: &IntegerLevels) -> f32 {
    unsafe {
        match (input.bits, has_dotprod()) {
            (2, true) => row_dot_two_bit::<Sdot>(packed, norms, input, levels),
            (2, false) => row_dot_two_bit::<Widening>(packed, norms, input, levels),
            (_, true) => row_dot_four_bit::<Sdot>(packed, norms, input, levels),
            (_, false) => row_dot_four_bit::<Widening>(packed, norms, input, levels),
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn row_dot_two_bit<D: DotKernel>(packed: &[u8], norms: &[f32], input: &PlaneInput, levels: &IntegerLevels) -> f32 {
    let table = vld1q_u8(levels.table.as_ptr().cast());
    let mask = vdupq_n_u8(3);
    let blocks_per_group = input.group / (4 * BLOCK_BYTES);
    let mut total = 0.0f32;
    for (group, norm) in norms.iter().enumerate() {
        let mut acc = vdupq_n_s32(0);
        for block in 0..blocks_per_group {
            let bytes = vld1q_u8(packed.as_ptr().add((group * blocks_per_group + block) * BLOCK_BYTES));
            let base = group * input.group + block * 4 * BLOCK_BYTES;
            let planes = input.planes.as_ptr().add(base);
            let first = vandq_u8(bytes, mask);
            let second = vandq_u8(vshrq_n_u8::<2>(bytes), mask);
            let third = vandq_u8(vshrq_n_u8::<4>(bytes), mask);
            let fourth = vshrq_n_u8::<6>(bytes);
            acc = D::groups(acc, vreinterpretq_s8_u8(vqtbl1q_u8(table, first)), vld1q_s8(planes));
            acc = D::groups(acc, vreinterpretq_s8_u8(vqtbl1q_u8(table, second)), vld1q_s8(planes.add(BLOCK_BYTES)));
            acc = D::groups(acc, vreinterpretq_s8_u8(vqtbl1q_u8(table, third)), vld1q_s8(planes.add(2 * BLOCK_BYTES)));
            acc = D::groups(acc, vreinterpretq_s8_u8(vqtbl1q_u8(table, fourth)), vld1q_s8(planes.add(3 * BLOCK_BYTES)));
        }
        total += vaddvq_s32(acc) as f32 * norm * levels.scale * input.scales[group];
    }
    total
}

#[target_feature(enable = "neon")]
unsafe fn row_dot_four_bit<D: DotKernel>(packed: &[u8], norms: &[f32], input: &PlaneInput, levels: &IntegerLevels) -> f32 {
    let table = vld1q_u8(levels.table.as_ptr().cast());
    let mask = vdupq_n_u8(15);
    let blocks_per_group = input.group / (2 * BLOCK_BYTES);
    let mut total = 0.0f32;
    for (group, norm) in norms.iter().enumerate() {
        let mut acc = vdupq_n_s32(0);
        for block in 0..blocks_per_group {
            let bytes = vld1q_u8(packed.as_ptr().add((group * blocks_per_group + block) * BLOCK_BYTES));
            let base = group * input.group + block * 2 * BLOCK_BYTES;
            let planes = input.planes.as_ptr().add(base);
            let low = vandq_u8(bytes, mask);
            let high = vshrq_n_u8::<4>(bytes);
            acc = D::groups(acc, vreinterpretq_s8_u8(vqtbl1q_u8(table, low)), vld1q_s8(planes));
            acc = D::groups(acc, vreinterpretq_s8_u8(vqtbl1q_u8(table, high)), vld1q_s8(planes.add(BLOCK_BYTES)));
        }
        total += vaddvq_s32(acc) as f32 * norm * levels.scale * input.scales[group];
    }
    total
}

#[target_feature(enable = "neon")]
unsafe fn dot_i8_inner<D: DotKernel>(a: &[i8], b: &[i8]) -> i32 {
    let mut acc = vdupq_n_s32(0);
    let mut at = 0;
    while at < a.len() {
        acc = D::groups(acc, vld1q_s8(a.as_ptr().add(at)), vld1q_s8(b.as_ptr().add(at)));
        at += 16;
    }
    vaddvq_s32(acc)
}

#[target_feature(enable = "neon")]
unsafe fn score_inner<D: DotKernel>(query: &[i8], keys: &[i8], stride: usize, offset: usize, count: usize, out: &mut [i32]) {
    const MAX_CHUNKS: usize = 8;
    let chunks = query.len() / 16;
    let mut loaded = [vdupq_n_s8(0); MAX_CHUNKS];
    for (chunk, slot) in loaded.iter_mut().enumerate().take(chunks) {
        *slot = vld1q_s8(query.as_ptr().add(chunk * 16));
    }
    let mut position = 0;
    while position + 4 <= count {
        let mut sums = [vdupq_n_s32(0); 4];
        for (lane, sum) in sums.iter_mut().enumerate() {
            let base = keys.as_ptr().add((position + lane) * stride + offset);
            for (chunk, query_chunk) in loaded.iter().enumerate().take(chunks) {
                *sum = D::groups(*sum, *query_chunk, vld1q_s8(base.add(chunk * 16)));
            }
        }
        let paired = vpaddq_s32(vpaddq_s32(sums[0], sums[1]), vpaddq_s32(sums[2], sums[3]));
        vst1q_s32(out.as_mut_ptr().add(position), paired);
        position += 4;
    }
    while position < count {
        let base = position * stride + offset;
        out[position] = dot_i8_inner::<D>(query, &keys[base..base + query.len()]);
        position += 1;
    }
}

pub fn score_positions_i8(query: &[i8], keys: &[i8], stride: usize, offset: usize, count: usize, out: &mut [i32]) {
    unsafe {
        if has_dotprod() {
            score_inner::<Sdot>(query, keys, stride, offset, count, out)
        } else {
            score_inner::<Widening>(query, keys, stride, offset, count, out)
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn accumulate_inner(entries: &[(u32, i16)], values: &[i8], stride: usize, offset: usize, width: usize, scale: f32, out: &mut [f32]) {
    const MAX_CHUNKS: usize = 8;
    let chunks = width / 16;
    let mut acc = [vdupq_n_s32(0); MAX_CHUNKS * 4];
    for &(position, coefficient) in entries {
        let base = values.as_ptr().add(position as usize * stride + offset);
        for chunk in 0..chunks {
            let bytes = vld1q_s8(base.add(chunk * 16));
            let low = vmovl_s8(vget_low_s8(bytes));
            let high = vmovl_high_s8(bytes);
            let slot = chunk * 4;
            acc[slot] = vmlal_n_s16(acc[slot], vget_low_s16(low), coefficient);
            acc[slot + 1] = vmlal_high_n_s16(acc[slot + 1], low, coefficient);
            acc[slot + 2] = vmlal_n_s16(acc[slot + 2], vget_low_s16(high), coefficient);
            acc[slot + 3] = vmlal_high_n_s16(acc[slot + 3], high, coefficient);
        }
    }
    for (chunk, accumulator) in acc.iter().enumerate().take(chunks * 4) {
        let scaled = vmulq_n_f32(vcvtq_f32_s32(*accumulator), scale);
        vst1q_f32(out.as_mut_ptr().add(chunk * 4), scaled);
    }
}

pub fn accumulate_values(entries: &[(u32, i16)], values: &[i8], stride: usize, offset: usize, width: usize, scale: f32, out: &mut [f32]) {
    unsafe { accumulate_inner(entries, values, stride, offset, width, scale, out) }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn exp_neon(value: float32x4_t) -> float32x4_t {
    let clamped = vminq_f32(vmaxq_f32(value, vdupq_n_f32(-87.0)), vdupq_n_f32(87.0));
    let scaled = vmulq_n_f32(clamped, std::f32::consts::LOG2_E);
    let whole = vrndaq_f32(scaled);
    let fraction = vsubq_f32(scaled, whole);
    let mut polynomial = vdupq_n_f32(0.001_333_355_8);
    for coefficient in [0.009_618_129, 0.055_504_11, 0.240_226_5, std::f32::consts::LN_2, 1.0] {
        polynomial = vaddq_f32(vmulq_f32(polynomial, fraction), vdupq_n_f32(coefficient));
    }
    let exponent = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(whole), vdupq_n_s32(127)));
    vmulq_f32(polynomial, vreinterpretq_f32_s32(exponent))
}

#[target_feature(enable = "neon")]
unsafe fn softmax_inner(integer_scores: &[i32], key_scales: &[f32], value_scales: &[f32], query_scale: f32, coefficients: &mut [f32]) -> f32 {
    let count = integer_scores.len();
    let vector_count = count - count % 4;
    let mut peak_vector = vdupq_n_f32(f32::NEG_INFINITY);
    for index in (0..vector_count).step_by(4) {
        let score = vcvtq_f32_s32(vld1q_s32(integer_scores.as_ptr().add(index)));
        let scaled = vmulq_n_f32(vmulq_f32(score, vld1q_f32(key_scales.as_ptr().add(index))), query_scale);
        vst1q_f32(coefficients.as_mut_ptr().add(index), scaled);
        peak_vector = vmaxq_f32(peak_vector, scaled);
    }
    let mut peak = vmaxvq_f32(peak_vector);
    for index in vector_count..count {
        coefficients[index] = integer_scores[index] as f32 * key_scales[index] * query_scale;
        peak = peak.max(coefficients[index]);
    }
    let shift = vdupq_n_f32(peak);
    let mut total_vector = vdupq_n_f32(0.0);
    for index in (0..vector_count).step_by(4) {
        let exponent = exp_neon(vsubq_f32(vld1q_f32(coefficients.as_ptr().add(index)), shift));
        total_vector = vaddq_f32(total_vector, exponent);
        vst1q_f32(coefficients.as_mut_ptr().add(index), vmulq_f32(exponent, vld1q_f32(value_scales.as_ptr().add(index))));
    }
    let mut total = vaddvq_f32(total_vector);
    for index in vector_count..count {
        let exponent = crate::ops::fast_exp(coefficients[index] - peak);
        total += exponent;
        coefficients[index] = exponent * value_scales[index];
    }
    total
}

pub fn softmax_weights(integer_scores: &[i32], key_scales: &[f32], value_scales: &[f32], query_scale: f32, coefficients: &mut [f32]) -> f32 {
    unsafe { softmax_inner(integer_scores, key_scales, value_scales, query_scale, coefficients) }
}

#[target_feature(enable = "neon")]
unsafe fn collect_inner(coefficients: &[f32], scale: f32, entries: &mut Vec<(u32, i16)>) {
    let count = coefficients.len();
    let vector_count = count - count % 8;
    let divisor = vdupq_n_f32(scale);
    for index in (0..vector_count).step_by(8) {
        let low = vcvtaq_s32_f32(vdivq_f32(vld1q_f32(coefficients.as_ptr().add(index)), divisor));
        let high = vcvtaq_s32_f32(vdivq_f32(vld1q_f32(coefficients.as_ptr().add(index + 4)), divisor));
        let narrowed = vcombine_s16(vqmovn_s32(low), vqmovn_s32(high));
        if vmaxvq_u16(vtstq_s16(narrowed, narrowed)) == 0 {
            continue;
        }
        let mut lanes = [0i16; 8];
        vst1q_s16(lanes.as_mut_ptr(), narrowed);
        for (offset, weight) in lanes.iter().enumerate() {
            if *weight != 0 {
                entries.push(((index + offset) as u32, *weight));
            }
        }
    }
    for (index, coefficient) in coefficients.iter().enumerate().skip(vector_count) {
        let weight = (coefficient / scale).round() as i16;
        if weight != 0 {
            entries.push((index as u32, weight));
        }
    }
}

pub fn collect_entries(coefficients: &[f32], scale: f32, entries: &mut Vec<(u32, i16)>) {
    unsafe { collect_inner(coefficients, scale, entries) }
}

use crate::PANEL_POSITIONS;

#[target_feature(enable = "neon")]
unsafe fn split_coefficients(coefficients: &[f32], scale: f32, lead: usize, high: &mut [i8], low: &mut [i8]) {
    let divisor = vdupq_n_f32(scale);
    let count = coefficients.len();
    let vector_count = count - count % 8;
    let mask = vdupq_n_s16(63);
    for index in (0..vector_count).step_by(8) {
        let first = vcvtaq_s32_f32(vdivq_f32(vld1q_f32(coefficients.as_ptr().add(index)), divisor));
        let second = vcvtaq_s32_f32(vdivq_f32(vld1q_f32(coefficients.as_ptr().add(index + 4)), divisor));
        let narrowed = vcombine_s16(vqmovn_s32(first), vqmovn_s32(second));
        vst1_s8(high.as_mut_ptr().add(lead + index), vmovn_s16(vshrq_n_s16::<6>(narrowed)));
        vst1_s8(low.as_mut_ptr().add(lead + index), vmovn_s16(vandq_s16(narrowed, mask)));
    }
    for index in vector_count..count {
        let value = (coefficients[index] / scale).round() as i32;
        high[lead + index] = (value >> 6) as i8;
        low[lead + index] = (value & 63) as i8;
    }
}

#[target_feature(enable = "neon")]
unsafe fn accumulate_panel_chunks<D: DotKernel, const STEPS: usize>(
    blocks: &[u32],
    high: &[i8],
    low: &[i8],
    panels: &[i8],
    width: usize,
    scale: f32,
    out: &mut [f32],
) {
    let block_bytes = width * PANEL_POSITIONS;
    let chunk_bytes = STEPS * 16;
    for chunk in 0..width / (STEPS * 4) {
        let mut high_sums = [vdupq_n_s32(0); STEPS];
        let mut low_sums = [vdupq_n_s32(0); STEPS];
        for &block in blocks {
            let block = block as usize;
            let high_quad = vreinterpretq_s8_s32(vld1q_dup_s32(high.as_ptr().add(block * PANEL_POSITIONS).cast::<i32>()));
            let low_quad = vreinterpretq_s8_s32(vld1q_dup_s32(low.as_ptr().add(block * PANEL_POSITIONS).cast::<i32>()));
            let base = panels.as_ptr().add(block * block_bytes + chunk * chunk_bytes);
            for step in 0..STEPS {
                let weights = vld1q_s8(base.add(step * 16));
                high_sums[step] = D::lane::<0>(high_sums[step], weights, high_quad);
                low_sums[step] = D::lane::<0>(low_sums[step], weights, low_quad);
            }
        }
        for step in 0..STEPS {
            let total = vaddq_s32(vshlq_n_s32::<6>(high_sums[step]), low_sums[step]);
            vst1q_f32(out.as_mut_ptr().add(chunk * STEPS * 4 + step * 4), vmulq_n_f32(vcvtq_f32_s32(total), scale));
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn accumulate_panels_inner(blocks: &[u32], high: &[i8], low: &[i8], panels: &[i8], width: usize, scale: f32, out: &mut [f32]) {
    match (width.is_multiple_of(32), has_dotprod()) {
        (true, true) => accumulate_panel_chunks::<Sdot, 8>(blocks, high, low, panels, width, scale, out),
        (true, false) => accumulate_panel_chunks::<Widening, 8>(blocks, high, low, panels, width, scale, out),
        (false, true) => accumulate_panel_chunks::<Sdot, 4>(blocks, high, low, panels, width, scale, out),
        (false, false) => accumulate_panel_chunks::<Widening, 4>(blocks, high, low, panels, width, scale, out),
    }
}

pub fn accumulate_panels(coefficients: &[f32], first: usize, scale: f32, panels: &[i8], width: usize, out: &mut [f32]) {
    thread_local! {
        static SCRATCH: std::cell::RefCell<(Vec<i8>, Vec<i8>, Vec<u32>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
    }
    let lead = first % PANEL_POSITIONS;
    let blocks = (lead + coefficients.len()).div_ceil(PANEL_POSITIONS);
    SCRATCH.with(|cell| {
        let (high, low, active) = &mut *cell.borrow_mut();
        high.clear();
        high.resize(blocks * PANEL_POSITIONS, 0);
        low.clear();
        low.resize(blocks * PANEL_POSITIONS, 0);
        unsafe { split_coefficients(coefficients, scale, lead, high, low) };
        active.clear();
        for block in 0..blocks {
            let (high_quad, low_quad) = unsafe {
                (
                    high.as_ptr().add(block * PANEL_POSITIONS).cast::<u32>().read_unaligned(),
                    low.as_ptr().add(block * PANEL_POSITIONS).cast::<u32>().read_unaligned(),
                )
            };
            if high_quad | low_quad != 0 {
                active.push(block as u32);
            }
        }
        let panel_offset = first / PANEL_POSITIONS * width * PANEL_POSITIONS;
        unsafe { accumulate_panels_inner(active, high, low, &panels[panel_offset..], width, scale, out) };
    });
}
