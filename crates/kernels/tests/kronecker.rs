use rustyneedle_kernels::kronecker_apply;

fn reference(input: &[f32], left: &[f32], size: usize, right: &[f32], output: &mut [f32]) {
    let mut mixed = vec![0.0f64; size * size];
    for i in 0..size {
        for j in 0..size {
            for c in 0..size {
                mixed[i * size + c] += input[i * size + j] as f64 * right[j * size + c] as f64;
            }
        }
    }
    for k in 0..size {
        for c in 0..size {
            output[k * size + c] = (0..size).map(|i| left[i * size + k] as f64 * mixed[i * size + c]).sum::<f64>() as f32;
        }
    }
}

#[test]
fn the_32_by_32_product_matches_a_double_precision_reference() {
    let make = |seed: usize| -> Vec<f32> { (0..1024).map(|i| (((i * 7919 + seed * 104729) % 2003) as f32 - 1000.0) / 500.0).collect() };
    let (input, left, right) = (make(1), make(2), make(3));
    let mut expected = vec![0.0f32; 1024];
    reference(&input, &left, 32, &right, &mut expected);
    let mut actual = vec![0.0f32; 1024];
    kronecker_apply(&input, &left, 32, &right, 32, &mut actual);
    let worst = expected.iter().zip(&actual).map(|(a, b)| (a - b).abs() / (1.0 + a.abs())).fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "worst relative error {worst}");
}
