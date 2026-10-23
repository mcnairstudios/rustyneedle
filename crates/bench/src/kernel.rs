use std::hint::black_box;
use std::time::Instant;

use kernels::{attend_head, fake_quantise_int8, kronecker_apply, rotate_groups, score_positions, unit_rms, weighted_values, zero_centred_norm, HeadInputs};

fn microseconds(iterations: usize, mut work: impl FnMut()) -> f64 {
    for _ in 0..iterations / 10 {
        work();
    }
    let started = Instant::now();
    for _ in 0..iterations {
        work();
    }
    started.elapsed().as_secs_f64() * 1e6 / iterations as f64
}

pub fn primitives() {
    let wide: Vec<f32> = (0..3072).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
    let narrow = wide[..768].to_vec();
    let gain = vec![0.1f32; 768];
    let mut out = vec![0.0f32; 3072];
    let mut scratch = wide.clone();
    println!("unit_rms 3072             {:6.2} us", microseconds(20000, || unit_rms(black_box(&wide), &mut out)));
    println!(
        "fake_quantise_int8 3072   {:6.2} us",
        microseconds(20000, || {
            scratch.copy_from_slice(&wide);
            fake_quantise_int8(black_box(&mut scratch));
        })
    );
    println!(
        "rotate_groups 3072        {:6.2} us",
        microseconds(20000, || {
            scratch.copy_from_slice(&wide);
            rotate_groups(black_box(&mut scratch), 128);
        })
    );
    println!("zero_centred_norm 768     {:6.2} us", microseconds(20000, || zero_centred_norm(black_box(&narrow), &gain, &mut out[..768])));
    let factor: Vec<f32> = (0..1024).map(|i| ((i * 31) % 17) as f32 / 17.0).collect();
    let input: Vec<f32> = (0..1024).map(|i| ((i * 13) % 11) as f32 / 11.0).collect();
    let mut product = vec![0.0f32; 1024];
    println!("kronecker 32x32           {:6.2} us", microseconds(20000, || kronecker_apply(black_box(&input), &factor, 32, &factor, 32, &mut product)));
}

pub fn attention(positions: usize) {
    let (kv_heads, qk, vd) = (2usize, 48usize, 64usize);
    let keys: Vec<i8> = (0..positions * kv_heads * qk).map(|i| ((i * 31 + 7) % 255) as i32 as i8).collect();
    let values: Vec<i8> = (0..positions * kv_heads * vd).map(|i| ((i * 17 + 3) % 255) as i32 as i8).collect();
    let key_scales: Vec<f32> = (0..positions).map(|i| 0.01 + (i % 7) as f32 * 0.001).collect();
    let value_scales: Vec<f32> = (0..positions).map(|i| 0.02 + (i % 5) as f32 * 0.002).collect();
    let query: Vec<i8> = (0..qk).map(|i| ((i * 13) % 200) as i32 as i8).collect();
    let mut panels = vec![0i8; positions.div_ceil(kernels::PANEL_POSITIONS) * vd * kernels::PANEL_POSITIONS];
    for position in 0..positions {
        for dimension in 0..vd {
            panels[(position / 4) * vd * 4 + dimension * 4 + position % 4] = values[position * kv_heads * vd + vd + dimension];
        }
    }
    let inputs = HeadInputs {
        query: &query,
        query_scale: 0.001,
        keys: &keys,
        key_stride: kv_heads * qk,
        key_offset: qk,
        key_scales: &key_scales,
        values: &values,
        value_stride: kv_heads * vd,
        value_offset: vd,
        value_panels: &panels,
        first: 0,
        value_scales: &value_scales,
        width: vd,
    };
    let mut out = vec![0.0f32; vd];
    println!("attention head over {positions} positions: {:.2} us", microseconds(100000, || attend_head(&inputs, &mut out)));
    let mut scores = vec![0i32; positions];
    println!(
        "  key scores only:                  {:.2} us",
        microseconds(100000, || score_positions(&query, &keys, kv_heads * qk, qk, positions, &mut scores))
    );
    let peaky: Vec<f32> = (0..positions).map(|i| (-(i as f32) / 40.0).exp()).collect();
    let flat: Vec<f32> = (0..positions).map(|i| 1.0 - (i as f32) * 1e-4).collect();
    let mut accumulated = vec![0.0f32; vd];
    println!(
        "  value sum, peaky weights:         {:.2} us",
        microseconds(100000, || weighted_values(&peaky, &values, kv_heads * vd, vd, vd, &mut accumulated))
    );
    println!("  value sum, flat weights:          {:.2} us", microseconds(100000, || weighted_values(&flat, &values, kv_heads * vd, vd, vd, &mut accumulated)));
}
