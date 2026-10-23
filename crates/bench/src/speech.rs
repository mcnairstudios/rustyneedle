use std::path::Path;
use std::time::Instant;

use whistle::{read_wav, Options, Whistle};

use crate::timing::measure;

pub fn transcribe(weights: &Path, clip: &Path, rounds: usize) {
    let whistle = Whistle::open(weights.join("whistle.cact")).expect("the Whistle archive opens");
    let samples = read_wav(clip).expect("the clip reads");
    let options = Options::default();
    let started = Instant::now();
    let first = whistle.transcribe(&samples, &options).expect("the clip transcribes");
    println!("first call {:.1} ms: {}", started.elapsed().as_secs_f64() * 1000.0, first.text);
    let timing = measure(rounds, || {
        let _ = whistle.transcribe(&samples, &options).expect("the clip transcribes");
    });
    println!("warm call: {timing}");
}
