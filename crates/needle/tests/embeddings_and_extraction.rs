use rustyneedle_needle::{extract, ExtractOptions, Needle};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct EmbeddingCase {
    text: String,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct ExtractionCase {
    name: String,
    text: String,
    tool: Value,
    result: Value,
}

fn open_model() -> std::sync::Arc<model::Model> {
    model::Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap()
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn embeddings_point_the_same_way_as_the_engine() {
    let model = open_model();
    let agent = Needle::builder().model(&model).build();
    let cases: Vec<EmbeddingCase> = testkit::golden("embeddings");
    for case in cases {
        let got = agent.embed(&case.text).expect("confidence head present");
        let norm = case.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cosine: f32 = got.iter().zip(&case.vector).map(|(a, b)| a * b / norm).sum();
        assert!(cosine > 0.998, "{:?}: cosine {cosine}", case.text);
        assert_eq!(got.len(), case.vector.len());
    }
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn extraction_returns_the_records_the_engine_returned() {
    let model = open_model();
    let cases: Vec<ExtractionCase> = testkit::golden("extraction");
    for case in cases {
        let options = ExtractOptions { system: Some("date: 2026-10-21 Wed 23:36".into()), strict: false, ..ExtractOptions::default() };
        let got = extract(&model, case.tool.clone(), &case.text, &options).unwrap();
        assert_eq!(got.map(Value::Object), Some(case.result.clone()), "{}", case.name);
    }
}
