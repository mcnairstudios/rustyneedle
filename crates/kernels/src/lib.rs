mod attention;
pub const PANEL_POSITIONS: usize = 4;
mod codebook;
#[cfg(target_arch = "aarch64")]
mod dot;
mod matrix;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
mod panel;
#[cfg(target_arch = "aarch64")]
pub use panel::PanelMatrix;
mod ops;
mod pool;
mod walsh;

pub use attention::{attend_head, score_positions, weighted_values, HeadInputs};
pub use codebook::Codebooks;
pub use matrix::{NaturalInput, QuantisedMatrix};
pub use ops::{
    apply_rotary, dot_product, fake_quantise_int8, fast_exp, int8_scale, kronecker_apply, quantise_int8, rotary_frequencies, sigmoid, silu, sinkhorn_in_place,
    softmax_in_place, unit_rms, zero_centred_norm, zero_centred_norm_in_place,
};
pub use pool::{parallel_chunks_mut, parallel_chunks_pair_mut, parallel_map, shared as spin_pool, SpinPool, PARALLEL_MIN_ITEMS};
pub use walsh::{rotate_groups, transform_in_place};
