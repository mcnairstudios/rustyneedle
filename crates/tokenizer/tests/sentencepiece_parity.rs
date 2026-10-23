use rustyneedle_tokenizer::Tokenizer;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
    decoded: String,
}

#[derive(Deserialize)]
struct Golden {
    pieces: usize,
    cases: Vec<Case>,
}

fn needle_tokenizer() -> Tokenizer {
    let (archive, layout) = testkit::archive("needle3");
    Tokenizer::from_blob(archive.bytes(layout.tokenizer)).expect("blob parses")
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn encoding_matches_sentencepiece_on_the_recorded_corpus() {
    let tokenizer = needle_tokenizer();
    let golden: Golden = testkit::golden("tokenizer");
    assert_eq!(tokenizer.vocabulary().len(), golden.pieces);
    let mismatches: Vec<_> =
        golden.cases.iter().filter(|case| tokenizer.encode(&case.text) != case.ids).map(|case| case.text.chars().take(60).collect::<String>()).collect();
    assert!(mismatches.is_empty(), "{} of {} cases differ, first: {:?}", mismatches.len(), golden.cases.len(), mismatches.first());
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn decoding_reproduces_sentencepiece_text() {
    let tokenizer = needle_tokenizer();
    let golden: Golden = testkit::golden("tokenizer");
    for case in &golden.cases {
        assert_eq!(tokenizer.decode(&case.ids), case.decoded, "text {:?}", case.text);
    }
}
