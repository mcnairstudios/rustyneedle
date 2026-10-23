mod common;

use common::{arguments, dated_call, finished, Stub};
use repair::Call;
use rustyneedle_needle::{Needle, ResponseKind, RunOptions};
use serde_json::json;

fn resets(handle: &common::StubHandle) -> usize {
    handle.calls.lock().unwrap().iter().filter(|call| call.as_str() == "reset").count()
}

#[test]
fn a_stateless_agent_resets_before_every_query() {
    let (engine, handle) = Stub::new(vec![finished()]);
    let mut agent = Needle::builder().engine(engine).stateless(true).build();
    agent.complete_within("one", 64);
    agent.complete_within("two", 64);
    assert_eq!(resets(&handle), 2);
    agent.run("three", &RunOptions::default());
    assert_eq!(resets(&handle), 3);
}

#[test]
fn a_stateful_agent_keeps_the_conversation_until_reset() {
    let (engine, handle) = Stub::new(vec![finished()]);
    let mut agent = Needle::builder().engine(engine).build();
    agent.complete_within("one", 64);
    agent.complete_within("two", 64);
    assert_eq!(resets(&handle), 0);
    agent.reset();
    assert_eq!(resets(&handle), 1);
}

#[test]
fn run_feeds_tool_results_back_as_python_style_json_and_attaches_them() {
    let mut first = dated_call("2026-01-02");
    first.function_calls = vec![Call { name: "ping".into(), arguments: arguments(json!({"x": "é"})) }];
    let (engine, handle) = Stub::new(vec![first, finished()]);
    let mut agent = Needle::builder()
        .engine(engine)
        .tool_with_handler(json!({"name": "ping", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}), |args| {
            Ok(json!({"ok": true, "echo": args["x"]}))
        })
        .build();
    let response = agent.run("go", &RunOptions::default());
    assert_eq!(response.kind, ResponseKind::Respond);
    assert_eq!(response.results.unwrap(), vec![json!({"ok": true, "echo": "é"})]);
    assert_eq!(handle.calls.lock().unwrap().last().unwrap(), "[{\"ok\": true, \"echo\": \"\\u00e9\"}]");
}

#[test]
fn an_unknown_tool_and_a_failing_tool_become_error_results() {
    let mut first = dated_call("2026-01-02");
    first.function_calls =
        vec![Call { name: "missing".into(), arguments: arguments(json!({})) }, Call { name: "boom".into(), arguments: arguments(json!({})) }];
    let (engine, _) = Stub::new(vec![first, finished()]);
    let mut agent = Needle::builder()
        .engine(engine)
        .tool_with_handler(json!({"name": "boom", "parameters": {"type": "object", "properties": {}}}), |_| Err("it broke".into()))
        .build();
    let results = agent.run("go", &RunOptions::default()).results.unwrap();
    assert_eq!(results, vec![json!({"error": "unknown tool: missing"}), json!({"error": "it broke"})]);
}

#[test]
fn run_stops_after_the_step_budget() {
    let mut looping = dated_call("2026-01-02");
    looping.function_calls = vec![Call { name: "ping".into(), arguments: arguments(json!({})) }];
    let (engine, handle) = Stub::new(vec![looping]);
    let mut agent = Needle::builder()
        .engine(engine)
        .tool_with_handler(json!({"name": "ping", "parameters": {"type": "object", "properties": {}}}), |_| Ok(json!({})))
        .build();
    let options = RunOptions { max_steps: 3, ..RunOptions::default() };
    assert_eq!(agent.run("go", &options).results.unwrap().len(), 3);
    assert_eq!(handle.calls.lock().unwrap().len(), 4);
}

#[test]
fn the_system_text_carries_a_date_unless_disabled() {
    let (engine, _) = Stub::new(vec![finished()]);
    let agent = Needle::builder().engine(engine).system("device: phone").build();
    assert!(agent.system_text().starts_with("date: "));
    assert!(agent.system_text().ends_with("; device: phone"));
    let (engine, _) = Stub::new(vec![finished()]);
    assert_eq!(Needle::builder().engine(engine).auto_date(false).build().system_text(), "");
}
