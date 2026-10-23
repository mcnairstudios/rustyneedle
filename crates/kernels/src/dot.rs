use std::arch::aarch64::*;
use std::arch::asm;
use std::sync::OnceLock;

pub trait DotKernel {
    unsafe fn groups(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t;
    unsafe fn lane<const LANE: i32>(acc: int32x4_t, weights: int8x16_t, activations: int8x16_t) -> int32x4_t;
}

pub struct Sdot;
pub struct Widening;

impl DotKernel for Sdot {
    #[inline(always)]
    unsafe fn groups(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut result = acc;
        asm!(
            ".arch_extension dotprod",
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) result,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack)
        );
        result
    }

    #[inline(always)]
    unsafe fn lane<const LANE: i32>(acc: int32x4_t, weights: int8x16_t, activations: int8x16_t) -> int32x4_t {
        let mut result = acc;
        asm!(
            ".arch_extension dotprod",
            "sdot {acc:v}.4s, {w:v}.16b, {a:v}.4b[{lane}]",
            acc = inout(vreg) result,
            w = in(vreg) weights,
            a = in(vreg) activations,
            lane = const LANE,
            options(pure, nomem, nostack)
        );
        result
    }
}

impl DotKernel for Widening {
    #[inline(always)]
    unsafe fn groups(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let low = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        let high = vmull_high_s8(a, b);
        vaddq_s32(acc, vpaddq_s32(vpaddlq_s16(low), vpaddlq_s16(high)))
    }

    #[inline(always)]
    unsafe fn lane<const LANE: i32>(acc: int32x4_t, weights: int8x16_t, activations: int8x16_t) -> int32x4_t {
        let quad = vreinterpretq_s8_s32(vdupq_laneq_s32::<LANE>(vreinterpretq_s32_s8(activations)));
        Self::groups(acc, weights, quad)
    }
}

pub fn has_dotprod() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| std::env::var_os("RUSTYNEEDLE_NO_DOTPROD").is_none() && std::arch::is_aarch64_feature_detected!("dotprod"))
}
