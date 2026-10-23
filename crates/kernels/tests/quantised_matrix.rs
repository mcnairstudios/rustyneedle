use rustyneedle_kernels::QuantisedMatrix;
use serde::Deserialize;

#[derive(Deserialize)]
struct Sample {
    archive: String,
    tensor: usize,
    rows: Vec<usize>,
    values: Vec<Vec<f64>>,
    input: Vec<f64>,
    product: Vec<f64>,
}

fn samples() -> Vec<Sample> {
    testkit::golden("tensors")
}

fn closeness(a: f64, b: f64) -> f64 {
    (a - b).abs() / (1.0 + b.abs())
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn dequantised_rows_match_the_python_exporter() {
    for sample in samples() {
        let (archive, _) = testkit::archive(&sample.archive);
        let matrix = QuantisedMatrix::from_archive(&archive, sample.tensor);
        let mut row = Vec::new();
        for (&index, expected) in sample.rows.iter().zip(&sample.values) {
            matrix.dequantise_row(index, &mut row);
            let worst = row.iter().zip(expected).map(|(a, b)| closeness(*a as f64, *b)).fold(0.0, f64::max);
            assert!(worst < 1e-4, "{} tensor {} row {index}: worst {worst}", sample.archive, sample.tensor);
        }
    }
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn rotated_matrix_vector_product_matches_the_dense_reference() {
    for sample in samples() {
        let (archive, _) = testkit::archive(&sample.archive);
        let matrix = QuantisedMatrix::from_archive(&archive, sample.tensor);
        let input: Vec<f32> = sample.input.iter().map(|&v| v as f32).collect();
        let mut rotated = Vec::new();
        matrix.prepare(&input, &mut rotated);
        let mut output = vec![0.0f32; matrix.rows()];
        matrix.multiply_vector(&rotated, &mut output);
        let error = output.iter().zip(&sample.product).map(|(a, b)| (*a as f64 - b).powi(2)).sum::<f64>();
        let energy = sample.product.iter().map(|b| b * b).sum::<f64>();
        let relative = (error / energy).sqrt();
        assert!(relative < 0.02, "{} tensor {}: relative rms error {relative}", sample.archive, sample.tensor);
    }
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn batched_product_agrees_with_one_vector_at_a_time() {
    let sample = samples().into_iter().find(|s| s.archive == "needle3" && s.tensor == 2).expect("sample present");
    let (archive, _) = testkit::archive("needle3");
    let matrix = QuantisedMatrix::from_archive(&archive, sample.tensor);
    let base: Vec<f32> = sample.input.iter().map(|&v| v as f32).collect();
    let inputs: Vec<Vec<f32>> = (0..3).map(|t| base.iter().map(|v| v * (t as f32 + 1.0)).collect()).collect();
    let mut stacked = Vec::new();
    let mut rotated = Vec::new();
    for input in &inputs {
        matrix.prepare(input, &mut rotated);
        stacked.extend_from_slice(&rotated);
    }
    let mut batch = vec![0.0f32; 3 * matrix.rows()];
    matrix.multiply_batch(&stacked, 3, &mut batch);
    for (token, input) in inputs.iter().enumerate() {
        matrix.prepare(input, &mut rotated);
        let mut single = vec![0.0f32; matrix.rows()];
        matrix.multiply_vector(&rotated, &mut single);
        let batch_row = &batch[token * matrix.rows()..(token + 1) * matrix.rows()];
        let error = single.iter().zip(batch_row).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>();
        let energy = batch_row.iter().map(|b| (*b as f64).powi(2)).sum::<f64>();
        assert!((error / energy).sqrt() < 0.02, "token {token}: relative rms error {}", (error / energy).sqrt());
    }
}
