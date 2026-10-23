use rustyneedle_kernels::transform_in_place;

fn reference(input: &[f32]) -> Vec<f64> {
    let size = input.len();
    let scale = 1.0 / (size as f64).sqrt();
    (0..size)
        .map(|row| {
            let total: f64 = (0..size).map(|column| if (row & column).count_ones() % 2 == 0 { input[column] as f64 } else { -(input[column] as f64) }).sum();
            total * scale
        })
        .collect()
}

#[test]
fn the_transform_matches_the_dense_hadamard_matrix_for_every_size_used() {
    for size in [4usize, 8, 32, 64, 128, 256] {
        let input: Vec<f32> = (0..size).map(|i| (((i * 7919 + size) % 2003) as f32 - 1000.0) / 400.0).collect();
        let mut output = input.clone();
        transform_in_place(&mut output);
        let expected = reference(&input);
        let worst = output.iter().zip(&expected).map(|(a, b)| (*a as f64 - b).abs() / (1.0 + b.abs())).fold(0.0, f64::max);
        assert!(worst < 1e-5, "size {size}: worst relative error {worst}");
    }
}

#[test]
fn applying_the_transform_twice_returns_the_input() {
    let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut output = input.clone();
    transform_in_place(&mut output);
    transform_in_place(&mut output);
    assert!(input.iter().zip(&output).all(|(a, b)| (a - b).abs() < 1e-5));
}
