use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use model::{Model, Session};
use needle::Needle;

use crate::timing::{measure, token_ids};

fn open_model(weights: &Path) -> Arc<Model> {
    Model::open(weights.join("needle3.cact")).expect("the Needle archive opens")
}

pub fn decode(weights: &Path, context: usize, steps: usize) {
    let model = open_model(weights);
    let mut session = Session::new(&model);
    session.feed(&token_ids(context, 37));
    let started = Instant::now();
    for step in 0..steps as u32 {
        session.push(10 + (step * 91) % 8000);
        let _ = session.logits();
    }
    let seconds = started.elapsed().as_secs_f64();
    println!("decode at {context} tokens of context: {:.0} tok/s ({:.2} ms per token)", steps as f64 / seconds, seconds * 1000.0 / steps as f64);
}

pub fn prefill(weights: &Path, tokens: usize) {
    let model = open_model(weights);
    let ids = token_ids(tokens, 37);
    let mut session = Session::new(&model);
    let started = Instant::now();
    session.feed(&ids);
    println!("prefill of {tokens} tokens: {:.0} tok/s", tokens as f64 / started.elapsed().as_secs_f64());
}

pub fn feed(weights: &Path, prefix: usize, turn: usize, rounds: usize) {
    let model = open_model(weights);
    let mut base = Session::new(&model);
    base.feed(&token_ids(prefix, 37));
    let new_tokens = token_ids(turn, 53);
    let timing = measure(rounds, || {
        let mut session = base.clone();
        session.feed(&new_tokens);
    });
    println!("feed of {turn} tokens after a {prefix}-token prefix: {timing}");
}

pub fn request(weights: &Path, tools: &Path, rounds: usize, auto_date: bool) {
    let model = open_model(weights);
    let schemas: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(tools).expect("the tool file reads")).expect("the tool file is JSON");
    let mut agent = Needle::builder().model(&model).tools(schemas).auto_date(auto_date).build();
    let queries = ["turn on the overhead light in the kitchen", "set the volume to 30", "pause the media player", "turn off the thermostat"];
    let mut round = 0;
    let timing = measure(rounds, || {
        let _ = agent.complete(queries[round % queries.len()]);
        agent.reset();
        round += 1;
    });
    println!("warm request: {timing}");
}

pub fn reset(weights: &Path, prefix: usize, rounds: usize) {
    let model = open_model(weights);
    let mut base = Session::new(&model);
    base.feed(&token_ids(prefix, 37));
    let timing = measure(rounds, || {
        let copy = base.clone();
        std::hint::black_box(&copy);
    });
    println!("cloning a session with a {prefix}-token prefix: {timing}");
}
