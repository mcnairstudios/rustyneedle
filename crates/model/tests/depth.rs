use std::time::Instant;

use rustyneedle_model::{Model, ModelError, Session};
use serde::Deserialize;
use testkit::cosine;

#[derive(Deserialize)]
struct Case {
    ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

fn ids() -> Vec<u32> {
    testkit::golden::<Reference>("reference").cases[0].ids.clone()
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn full_depth_is_the_base_model() {
    let base = Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap();
    let full = base.with_depth(20).unwrap();
    assert_eq!(full.active_layers().len(), 20);
    let (mut a, mut b) = (Session::new(&base), Session::new(&full));
    a.feed(&ids());
    b.feed(&ids());
    assert!(cosine(&a.logits(), &b.logits()) > 0.99999);
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn shallower_models_keep_the_endpoint_blocks_and_run_faster() {
    let base = Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap();
    let shallow = base.with_depth(8).unwrap();
    assert_eq!(shallow.active_layers(), &[0, 4, 6, 9, 11, 14, 16, 19]);
    let time = |model: &std::sync::Arc<Model>| {
        let mut session = Session::new(model);
        session.feed(&ids());
        let started = Instant::now();
        for step in 0..60u32 {
            session.push(10 + step);
            let _ = session.logits();
        }
        started.elapsed().as_secs_f64()
    };
    let (slow, fast) = (time(&base), time(&shallow));
    assert!(fast < slow * 0.8, "full {slow:.3}s vs depth 8 {fast:.3}s");
    let mut session = Session::new(&shallow);
    session.feed(&ids());
    assert!(session.head_output(2).is_some());
    assert!(session.logits().iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn the_two_layer_rung_still_decodes() {
    let base = Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap();
    let rung = base.with_depth(2).unwrap();
    assert_eq!(rung.active_layers(), &[0, 19]);
    let mut session = Session::new(&rung);
    session.feed(&ids());
    session.push(5);
    assert!(session.logits().iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn depths_outside_the_ladder_are_refused() {
    let base = Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap();
    assert!(matches!(base.with_depth(1), Err(ModelError::Depth { requested: 1, layers: 20 })));
    assert!(matches!(base.with_depth(21), Err(ModelError::Depth { .. })));
}
