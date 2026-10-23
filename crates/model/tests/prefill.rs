use rustyneedle_model::{Model, Session};
use serde::Deserialize;
use testkit::{argmax, cosine};

#[derive(Deserialize)]
struct Case {
    ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn feeding_a_batch_matches_pushing_one_token_at_a_time() {
    let model = Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap();
    let reference: Reference = testkit::golden("reference");
    for case in &reference.cases {
        let mut stepped = Session::new(&model);
        for &token in &case.ids {
            stepped.push(token);
        }
        let mut batched = Session::new(&model);
        batched.feed(&case.ids);
        assert_eq!(batched.len(), stepped.len());
        let (a, b) = (stepped.logits(), batched.logits());
        assert!(cosine(&a, &b) > 0.9995, "logit cosine {}", cosine(&a, &b));
        assert_eq!(argmax(&a), argmax(&b));
        let (ha, hb) = (stepped.head_output(2).unwrap()[0], batched.head_output(2).unwrap()[0]);
        assert!((ha - hb).abs() < 0.1, "confidence logits {ha} vs {hb}");
    }
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn decoding_can_continue_after_a_batched_prefill() {
    let model = Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap();
    let reference: Reference = testkit::golden("reference");
    let ids = &reference.cases[0].ids;
    let (head, tail) = ids.split_at(ids.len() - 6);
    let mut stepped = Session::new(&model);
    stepped.feed(&ids[..head.len() / 2]);
    for &token in &ids[head.len() / 2..] {
        stepped.push(token);
    }
    let mut mixed = Session::new(&model);
    mixed.feed(head);
    mixed.feed(tail);
    assert!(cosine(&stepped.logits(), &mixed.logits()) > 0.9995);
}
