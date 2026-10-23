use std::sync::Arc;

use chrono::TimeZone;
use rustyneedle_needle::{Needle, ResponseKind};
use serde_json::{json, Value};

fn clock() -> chrono::DateTime<chrono::Local> {
    chrono::Local.with_ymd_and_hms(2026, 7, 21, 14, 30, 0).unwrap()
}

fn model() -> Arc<model::Model> {
    model::Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap()
}

fn timer() -> Value {
    json!({"name": "set_timer", "description": "Set a timer.", "parameters": {"type": "object", "properties": {
        "minutes": {"type": "integer", "minimum": 1, "maximum": 600},
        "label": {"anyOf": [{"type": "string", "maxLength": 20}, {"type": "null"}]}}, "required": ["minutes"]}})
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn an_agent_without_tools_refuses_everything_without_failing() {
    let mut agent = Needle::builder().model(&model()).clock(clock()).build();
    let response = agent.complete_within("turn on the lights", 256);
    assert!(response.function_calls.is_empty());
    assert!(response.failure.is_none());
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn empty_and_whitespace_queries_are_refused() {
    let mut agent = Needle::builder().model(&model()).clock(clock()).tool(timer()).build();
    for query in ["", "   ", "\n\n"] {
        agent.reset();
        let response = agent.complete_within(query, 256);
        assert!(response.function_calls.is_empty(), "{query:?}");
    }
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn unicode_and_punctuation_heavy_queries_are_handled() {
    let mut agent = Needle::builder().model(&model()).clock(clock()).tool(timer()).build();
    for query in ["¿puedes poner un temporizador de 10 minutos?", "⏰ 10 minutes!!! 🍝", "set a timer for 10 minutes, please — thanks"] {
        agent.reset();
        let response = agent.complete_within(query, 256);
        assert_eq!(response.kind, ResponseKind::Call, "{query:?}");
    }
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn nullable_and_bounded_parameters_come_back_valid() {
    let mut agent = Needle::builder().model(&model()).clock(clock()).tool(timer()).build();
    let response = agent.complete_within("set a timer for 25 minutes called pasta", 256);
    let call = &response.function_calls[0];
    assert_eq!(call.arguments["minutes"], 25);
    assert!(call.arguments.get("label").is_none_or(|label| label.is_null() || label.as_str().is_some_and(|text| text.len() <= 20)));
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn nested_objects_and_arrays_are_decoded_under_their_schema() {
    let order = json!({"name": "place_order", "description": "Place an order for items.", "parameters": {"type": "object", "properties": {
        "customer": {"type": "object", "properties": {"name": {"type": "string"}, "vip": {"type": "boolean"}}, "required": ["name"]},
        "items": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}}, "required": ["customer", "items"]}});
    let mut agent = Needle::builder().model(&model()).clock(clock()).tool(order).build();
    let response = agent.complete_within("order a pizza and a cola for Dana", 256);
    let call = &response.function_calls[0];
    assert_eq!(call.arguments["customer"]["name"], "Dana");
    let items = call.arguments["items"].as_array().unwrap();
    assert!((1..=3).contains(&items.len()));
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn a_prompt_that_cannot_fit_is_reported_not_decoded() {
    let mut agent = Needle::builder().model(&model()).clock(clock()).tool(timer()).build();
    let huge = "set a timer for ten minutes and ".repeat(2500);
    let response = agent.complete_within(&huge, 256);
    let failure = response.failure.expect("an oversized prompt is a failure");
    assert_eq!(failure.code, "context_overflow");
    assert!(response.function_calls.is_empty());
    agent.reset();
    assert_eq!(agent.complete_within("set a timer for 5 minutes", 256).function_calls[0].arguments["minutes"], 5);
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn a_tiny_token_budget_yields_a_truncation_error() {
    let mut agent = Needle::builder().model(&model()).clock(clock()).tool(timer()).build();
    let response = agent.complete_within("set a timer for 25 minutes called pasta", 12);
    assert_eq!(response.failure.expect("budget exhausted").code, "truncated");
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn many_turns_in_one_conversation_stay_usable() {
    let mut agent = Needle::builder().model(&model()).clock(clock()).tool(timer()).build();
    for minutes in [5, 10, 15, 20] {
        let response = agent.complete_within(&format!("set a timer for {minutes} minutes"), 256);
        assert_eq!(response.function_calls[0].arguments["minutes"], minutes);
    }
}
