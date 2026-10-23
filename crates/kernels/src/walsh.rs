#[cfg(target_arch = "aarch64")]
mod neon_group {
    use std::arch::aarch64::*;

    const VECTORS: usize = 32;

    #[target_feature(enable = "neon")]
    unsafe fn within_vector(x: float32x4_t) -> float32x4_t {
        let signs_first = vld1q_f32([1.0f32, -1.0, 1.0, -1.0].as_ptr());
        let signs_second = vld1q_f32([1.0f32, 1.0, -1.0, -1.0].as_ptr());
        let pairs = vaddq_f32(vtrn1q_f32(x, x), vmulq_f32(vtrn2q_f32(x, x), signs_first));
        let low = vcombine_f32(vget_low_f32(pairs), vget_low_f32(pairs));
        let high = vcombine_f32(vget_high_f32(pairs), vget_high_f32(pairs));
        vaddq_f32(low, vmulq_f32(high, signs_second))
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn transform(values: &mut [f32]) {
        let mut registers = [vdupq_n_f32(0.0); VECTORS];
        for (index, register) in registers.iter_mut().enumerate() {
            *register = within_vector(vld1q_f32(values.as_ptr().add(index * 4)));
        }
        let mut span = 1;
        while span < VECTORS {
            for block in (0..VECTORS).step_by(span * 2) {
                for offset in 0..span {
                    let (a, b) = (registers[block + offset], registers[block + offset + span]);
                    registers[block + offset] = vaddq_f32(a, b);
                    registers[block + offset + span] = vsubq_f32(a, b);
                }
            }
            span *= 2;
        }
        let scale = 1.0 / (128.0f32).sqrt();
        for (index, register) in registers.iter().enumerate() {
            vst1q_f32(values.as_mut_ptr().add(index * 4), vmulq_n_f32(*register, scale));
        }
    }
}

pub fn transform_in_place(values: &mut [f32]) {
    let length = values.len();
    #[cfg(target_arch = "aarch64")]
    if length == 128 {
        unsafe { neon_group::transform(values) };
        return;
    }
    debug_assert!(length.is_power_of_two());
    let mut span = 1;
    if length >= 4 {
        for quad in values.chunks_exact_mut(4) {
            let (a, b, c, d) = (quad[0], quad[1], quad[2], quad[3]);
            let (low_sum, low_difference) = (a + b, a - b);
            let (high_sum, high_difference) = (c + d, c - d);
            quad[0] = low_sum + high_sum;
            quad[1] = low_difference + high_difference;
            quad[2] = low_sum - high_sum;
            quad[3] = low_difference - high_difference;
        }
        span = 4;
    }
    while span < length {
        for block in values.chunks_exact_mut(span * 2) {
            let (low, high) = block.split_at_mut(span);
            for (a, b) in low.iter_mut().zip(high.iter_mut()) {
                let sum = *a + *b;
                let difference = *a - *b;
                *a = sum;
                *b = difference;
            }
        }
        span *= 2;
    }
    let scale = 1.0 / (length as f32).sqrt();
    values.iter_mut().for_each(|value| *value *= scale);
}

pub fn rotate_groups(values: &mut [f32], group: usize) {
    values.chunks_exact_mut(group).for_each(transform_in_place);
}
