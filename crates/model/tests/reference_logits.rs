use rustyneedle_model::{Model, Session};
use serde::Deserialize;
use testkit::{argmax, cosine};

#[derive(Deserialize)]
struct Case {
    query: String,
    ids: Vec<u32>,
    last_logits: Vec<f32>,
    cells_last: Vec<Vec<f32>>,
    cells_first: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn last_position_logits_track_the_quantised_reference() {
    let model = Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap();
    let reference: Reference = testkit::golden("reference");
    for case in &reference.cases {
        let mut session = Session::new(&model);
        session.feed(&case.ids);
        for (layer, (ours, theirs)) in session.cells().iter().zip(&case.cells_last).enumerate() {
            let similarity = cosine(ours, theirs);
            assert!(similarity > 0.995, "{:?} layer {layer}: cosine {similarity}", case.query);
        }
        let logits = session.logits();
        let similarity = cosine(&logits, &case.last_logits);
        assert!(similarity > 0.999, "{:?}: cosine {similarity}", case.query);
        assert_eq!(argmax(&logits), argmax(&case.last_logits), "{:?}: top token differs", case.query);
    }
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn first_position_cells_track_the_quantised_reference() {
    let model = Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap();
    let reference: Reference = testkit::golden("reference");
    let case = &reference.cases[0];
    let mut session = Session::new(&model);
    session.feed(&case.ids[..1]);
    for (layer, (ours, theirs)) in session.cells().iter().zip(&case.cells_first).enumerate() {
        let similarity = cosine(ours, theirs);
        assert!(similarity > 0.9995, "layer {layer}: cosine {similarity}");
    }
}
