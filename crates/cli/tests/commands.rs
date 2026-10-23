use std::process::Command;

use serde_json::Value;

fn needle(arguments: &[&str]) -> (bool, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_rustyneedle")).args(arguments).env("HF_HUB_OFFLINE", "1").output().expect("binary runs");
    (output.status.success(), String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned())
}

fn weights(name: &str) -> String {
    format!("{}/../../weights/{name}.cact", env!("CARGO_MANIFEST_DIR"))
}

fn clip(name: &str) -> String {
    format!("{}/../../oracle/clips/{name}.wav", env!("CARGO_MANIFEST_DIR"))
}

const TOOLS: &str = r#"[{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn complete_prints_a_tool_call_as_json() {
    let (ok, out, err) = needle(&["complete", "--model", &weights("needle3"), "--tools", TOOLS, "what's the weather in Lagos?"]);
    assert!(ok, "{err}");
    let response: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(response["function_calls"][0]["name"], "get_weather");
    assert_eq!(response["function_calls"][0]["arguments"]["city"], "Lagos");
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn complete_without_the_date_fact_still_answers_a_request_that_needs_no_date() {
    let (ok, out, err) = needle(&["complete", "--no-date", "--model", &weights("needle3"), "--tools", TOOLS, "what's the weather in Lagos?"]);
    assert!(ok, "{err}");
    let response: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(response["function_calls"][0]["name"], "get_weather");
    assert_eq!(response["function_calls"][0]["arguments"]["city"], "Lagos");
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn transcribe_prints_text_and_word_times() {
    let (ok, out, err) = needle(&["transcribe", "--whistle", &weights("whistle"), "--word-timestamps", &clip("clip08")]);
    assert!(ok, "{err}");
    let result: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(result["text"], "Play Purple Rain by Prince");
    assert_eq!(result["language"], "en");
    assert!(result["words"].as_array().unwrap().len() >= 5);
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn audio_with_two_models_runs_speech_to_intent() {
    let (ok, out, err) = needle(&["--model", &weights("needle3"), "--model", &weights("whistle"), "--tools", TOOLS, "--audio", &clip("clip08")]);
    assert!(ok, "{err}");
    let response: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(response["audio_text"], "Play Purple Rain by Prince");
    assert_eq!(response["audio_language"], "en");
    assert!(response["function_calls"].is_array());
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn embed_prints_a_unit_vector() {
    let (ok, out, err) = needle(&["embed", "--model", &weights("needle3"), "hello world"]);
    assert!(ok, "{err}");
    let vector: Vec<f64> = serde_json::from_str(&out).unwrap();
    assert_eq!(vector.len(), 3072);
    assert!((vector.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-3);
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn extract_prints_the_record() {
    let schema = r#"{"type":"object","properties":{"restaurant":{"type":"string"},"party_size":{"type":"integer"}},"required":["restaurant","party_size"]}"#;
    let (ok, out, err) = needle(&["extract", "--model", &weights("needle3"), "--schema", schema, "--no-strict", "Book a table for four at Luigi's on Friday"]);
    assert!(ok, "{err}");
    let record: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(record["party_size"], 4);
}

#[test]
fn a_missing_action_is_an_error_not_a_panic() {
    let (ok, _, err) = needle(&[]);
    assert!(!ok);
    assert!(err.contains("nothing to do"));
}

#[test]
fn downloads_respect_offline_mode() {
    let (ok, _, err) = needle(&["download", "whistle", "--out", "/tmp/needle-cli-offline-test"]);
    assert!(!ok);
    assert!(err.contains("HF_HUB_OFFLINE"));
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn depth_flags_select_a_ladder_rung() {
    let (ok, out, err) = needle(&["complete", "--model", &weights("needle3"), "--depth", "16", "--tools", TOOLS, "what's the weather in Lagos?"]);
    assert!(ok, "{err}");
    let response: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(response["function_calls"][0]["arguments"]["city"], "Lagos");
    let (ok, _, err) = needle(&["complete", "--model", &weights("needle3"), "--depth", "1", "--tools", TOOLS, "hello"]);
    assert!(!ok);
    assert!(err.contains("depth 1"));
    let (ok, out, err) = needle(&["transcribe", "--whistle", &weights("whistle"), "--audio-depth", "6", &clip("clip08")]);
    assert!(ok, "{err}");
    assert!(out.to_lowercase().contains("purple rain"));
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn stream_prints_one_json_line_per_chunk_and_ends_with_nothing_pending() {
    let (ok, out, err) = needle(&["stream", "--whistle", &weights("whistle"), "--chunk-seconds", "0.5", &clip("clip08")]);
    assert!(ok, "{err}");
    let steps: Vec<Value> = out.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert!(steps.len() >= 4);
    let committed: Vec<&str> = steps.iter().filter_map(|step| step["text"].as_str()).filter(|text| !text.is_empty()).collect();
    assert_eq!(committed.join(" ").to_lowercase().replace('.', ""), "play purple rain by prince");
    assert_eq!(steps.last().unwrap()["pending"], "");
    assert!(steps.windows(2).all(|pair| pair[1]["received"].as_f64() >= pair[0]["received"].as_f64()));
}
