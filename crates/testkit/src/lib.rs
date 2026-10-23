use std::path::PathBuf;

use cact::{Archive, Layout};
use serde::de::DeserializeOwned;

pub use serde_json::Value;

pub fn workspace_file(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(relative)
}

pub fn golden<T: DeserializeOwned>(name: &str) -> T {
    let path = workspace_file(&format!("oracle/goldens/{name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("golden {name} unreadable: {error}"));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("golden {name} malformed: {error}"))
}

pub fn archive(name: &str) -> (Archive, Layout) {
    let archive = Archive::open(workspace_file(&format!("weights/{name}.cact"))).unwrap_or_else(|error| panic!("{name}.cact unreadable: {error}"));
    let layout = Layout::resolve(&archive).unwrap_or_else(|error| panic!("{name}.cact layout: {error}"));
    (archive, layout)
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

pub fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap()
}
