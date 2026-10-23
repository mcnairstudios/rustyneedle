use rustyneedle_kernels::{fast_exp, softmax_in_place};

#[test]
fn the_fast_exponential_stays_within_a_millionth_of_the_exact_one() {
    let mut worst = 0.0f64;
    let mut x = -40.0f32;
    while x <= 5.0 {
        let relative = ((fast_exp(x) as f64 - (x as f64).exp()) / (x as f64).exp()).abs();
        worst = worst.max(relative);
        x += 0.013;
    }
    assert!(worst < 5e-6, "worst relative error {worst}");
    assert_eq!(fast_exp(0.0), 1.0);
    assert!(fast_exp(-200.0) >= 0.0 && fast_exp(-200.0) < 1e-30);
}

#[test]
fn softmax_matches_the_exact_distribution_and_sums_to_one() {
    let logits = [3.1f32, -2.0, 0.5, 7.25, 7.2, -11.0];
    let mut probabilities = logits;
    softmax_in_place(&mut probabilities);
    let peak = logits.iter().copied().fold(f32::MIN, f32::max) as f64;
    let total: f64 = logits.iter().map(|v| (*v as f64 - peak).exp()).sum();
    for (value, exact) in probabilities.iter().zip(logits.iter().map(|v| (*v as f64 - peak).exp() / total)) {
        assert!((*value as f64 - exact).abs() < 1e-6);
    }
    assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}
