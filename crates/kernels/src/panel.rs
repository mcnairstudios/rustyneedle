use crate::dot::{has_dotprod, DotKernel, Sdot, Widening};
use std::arch::aarch64::*;

use crate::neon::IntegerLevels;

const PANEL_ROWS: usize = 4;
const STEP: usize = 16;
const TOKEN_TILE: usize = 4;

pub struct PanelMatrix {
    rows: usize,
    group: usize,
    codes: Vec<u8>,
    norms: Vec<f32>,
    levels: IntegerLevels,
}

pub struct NaturalInput {
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
    pub columns: usize,
    pub groups: usize,
}

impl NaturalInput {
    pub fn build(inputs: &[f32], tokens: usize, width: usize, columns: usize, group: usize, prepare_row: &(dyn Fn(usize, &mut [f32]) + Sync)) -> Self {
        let groups = columns / group;
        let mut codes = vec![0i8; tokens * columns];
        let mut scales = vec![1.0f32; tokens * groups];
        crate::pool::parallel_chunks_pair_mut(&mut codes, columns, &mut scales, groups, &|token, token_codes, token_scales| {
            let mut row = vec![0.0f32; columns];
            row[..width].copy_from_slice(&inputs[token * width..(token + 1) * width]);
            prepare_row(token, &mut row[..width]);
            crate::walsh::rotate_groups(&mut row, group);
            for (g, scale_slot) in token_scales.iter_mut().enumerate() {
                let slice = &row[g * group..(g + 1) * group];
                let peak = crate::ops::peak_magnitude(slice);
                let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
                *scale_slot = scale;
                crate::neon::quantise_natural(slice, scale, &mut token_codes[g * group..(g + 1) * group]);
            }
        });
        Self { codes, scales, columns, groups }
    }

    pub fn new(rotated: &[f32], tokens: usize, columns: usize, group: usize) -> Self {
        let groups = columns / group;
        let mut codes = vec![0i8; tokens * columns];
        let mut scales = vec![1.0f32; tokens * groups];
        crate::pool::parallel_chunks_mut(&mut codes, columns, &|token, token_codes| {
            for g in 0..groups {
                let slice = &rotated[token * columns + g * group..token * columns + (g + 1) * group];
                let peak = crate::ops::peak_magnitude(slice);
                let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
                crate::neon::quantise_natural(slice, scale, &mut token_codes[g * group..(g + 1) * group]);
            }
        });
        for (token, token_scales) in scales.chunks_exact_mut(groups).enumerate() {
            for (g, slot) in token_scales.iter_mut().enumerate() {
                let peak = crate::ops::peak_magnitude(&rotated[token * columns + g * group..token * columns + (g + 1) * group]);
                *slot = if peak > 0.0 { peak / 127.0 } else { 1.0 };
            }
        }
        Self { codes, scales, columns, groups }
    }
}

impl PanelMatrix {
    pub fn build(
        rows: usize,
        columns: usize,
        group: usize,
        levels: IntegerLevels,
        packed: &[u8],
        row_stride: usize,
        norm_at: impl Fn(usize, usize) -> f32 + Sync,
    ) -> Self {
        let panels = rows.div_ceil(PANEL_ROWS);
        let groups = columns / group;
        let steps = group / STEP;
        let panel_bytes = groups * steps * 16;
        let mut codes = vec![0u8; panels * panel_bytes];
        let mut norms = vec![0.0f32; panels * groups * PANEL_ROWS];
        crate::pool::parallel_chunks_pair_mut(&mut codes, panel_bytes, &mut norms, groups * PANEL_ROWS, &|panel, panel_codes, panel_norms| {
            for g in 0..groups {
                for lane in 0..PANEL_ROWS {
                    let row = panel * PANEL_ROWS + lane;
                    if row < rows {
                        panel_norms[g * PANEL_ROWS + lane] = norm_at(row, g);
                    }
                }
                for step in 0..steps {
                    let base = (g * steps + step) * 16;
                    let source = (g * group + step * STEP) / 4;
                    for lane in 0..PANEL_ROWS {
                        let row = panel * PANEL_ROWS + lane;
                        if row >= rows {
                            continue;
                        }
                        let start = row * row_stride + source;
                        let word = u32::from_le_bytes([packed[start], packed[start + 1], packed[start + 2], packed[start + 3]]);
                        for kk in 0..4 {
                            let spread = (word >> (2 * kk)) & 0x0303_0303;
                            panel_codes[base + lane * 4 + kk] = (spread | spread >> 6 | spread >> 12 | spread >> 18) as u8;
                        }
                    }
                }
            }
        });
        Self { rows, group, codes, norms, levels }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub const fn panel_rows() -> usize {
        PANEL_ROWS
    }

    pub fn panels(&self) -> usize {
        self.rows.div_ceil(PANEL_ROWS)
    }

    pub fn multiply_panels(&self, input: &NaturalInput, tokens: usize, first_panel: usize, panel_count: usize, out: &mut [f32]) {
        unsafe {
            if has_dotprod() {
                self.multiply_inner::<Sdot>(input, tokens, first_panel, panel_count, out)
            } else {
                self.multiply_inner::<Widening>(input, tokens, first_panel, panel_count, out)
            }
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn multiply_inner<D: DotKernel>(&self, input: &NaturalInput, tokens: usize, first_panel: usize, panel_count: usize, out: &mut [f32]) {
        let mut done = 0;
        while done + TOKEN_TILE <= tokens {
            for panel in first_panel..first_panel + panel_count {
                self.tile::<D, TOKEN_TILE>(input, done, panel, first_panel, panel_count * PANEL_ROWS, out);
            }
            done += TOKEN_TILE;
        }
        while done < tokens {
            for panel in first_panel..first_panel + panel_count {
                self.tile::<D, 1>(input, done, panel, first_panel, panel_count * PANEL_ROWS, out);
            }
            done += 1;
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn tile<D: DotKernel, const T: usize>(
        &self,
        input: &NaturalInput,
        first_token: usize,
        panel: usize,
        first_panel: usize,
        stride: usize,
        out: &mut [f32],
    ) {
        let table = vld1q_u8(self.levels.table.as_ptr().cast());
        let mask = vdupq_n_u8(3);
        let groups = input.groups;
        let steps = self.group / STEP;
        let mut total = [vdupq_n_f32(0.0); T];
        for g in 0..groups {
            let mut acc = [vdupq_n_s32(0); T];
            let weights = self.codes.as_ptr().add((panel * groups + g) * steps * 16);
            for step in 0..steps {
                let bytes = vld1q_u8(weights.add(step * 16));
                let w0 = vreinterpretq_s8_u8(vqtbl1q_u8(table, vandq_u8(bytes, mask)));
                let w1 = vreinterpretq_s8_u8(vqtbl1q_u8(table, vandq_u8(vshrq_n_u8::<2>(bytes), mask)));
                let w2 = vreinterpretq_s8_u8(vqtbl1q_u8(table, vandq_u8(vshrq_n_u8::<4>(bytes), mask)));
                let w3 = vreinterpretq_s8_u8(vqtbl1q_u8(table, vshrq_n_u8::<6>(bytes)));
                for (t, slot) in acc.iter_mut().enumerate() {
                    let a = vld1q_s8(input.codes.as_ptr().add((first_token + t) * input.columns + g * self.group + step * STEP));
                    *slot = D::lane::<0>(*slot, w0, a);
                    *slot = D::lane::<1>(*slot, w1, a);
                    *slot = D::lane::<2>(*slot, w2, a);
                    *slot = D::lane::<3>(*slot, w3, a);
                }
            }
            let norms = vmulq_n_f32(vld1q_f32(self.norms.as_ptr().add((panel * groups + g) * PANEL_ROWS)), self.levels.scale);
            for (t, (sum, counts)) in total.iter_mut().zip(&acc).enumerate() {
                let scaled = vmulq_n_f32(norms, input.scales[(first_token + t) * groups + g]);
                *sum = vfmaq_f32(*sum, vcvtq_f32_s32(*counts), scaled);
            }
        }
        for (t, sum) in total.iter().enumerate() {
            let base = (first_token + t) * stride + (panel - first_panel) * PANEL_ROWS;
            let mut spill = [0.0f32; 4];
            vst1q_f32(spill.as_mut_ptr(), *sum);
            let lanes = (out.len() - base).min(PANEL_ROWS);
            out[base..base + lanes].copy_from_slice(&spill[..lanes]);
        }
    }
}
