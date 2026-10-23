use std::sync::Arc;

use rustyneedle_needle::{extract, ExtractOptions, Needle, ResponseKind, RunOptions};
use serde_json::{json, Value};

fn model() -> Arc<model::Model> {
    model::Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap()
}

fn tool(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}})
}

fn weather() -> Value {
    tool("get_weather", "Get the weather for a city.", json!({"city": {"type": "string"}}), &["city"])
}

fn thermostat() -> Value {
    tool("set_thermostat", "Set the thermostat temperature.", json!({"temperature": {"type": "integer"}}), &["temperature"])
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn a_tool_call_reaches_the_declared_tool() {
    let email = tool(
        "send_email",
        "Send an email to a recipient.",
        json!({"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}),
        &["to", "subject", "body"],
    );
    let mut agent = Needle::builder().model(&model()).tool(email).build();
    let response = agent.complete("email finance@cactus.dev, subject expenses, body coffee $14");
    assert_eq!(response.kind, ResponseKind::Call);
    assert_eq!(response.function_calls[0].name, "send_email");
    assert_eq!(response.function_calls[0].arguments["to"], "finance@cactus.dev");
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn structured_extraction_returns_a_record() {
    let contact = tool("Contact", "A contact.", json!({"name": {"type": "string"}, "email": {"type": "string"}}), &["name", "email"]);
    let options = ExtractOptions { system: Some("date: 2026-07-21 Tue 14:30".into()), ..ExtractOptions::default() };
    let record = extract(&model(), contact, "John Doe can be reached at john@doe.com", &options).unwrap().unwrap();
    assert_eq!(record["name"], "John Doe");
    assert_eq!(record["email"], "john@doe.com");
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn the_agentic_loop_executes_the_tool_and_finishes() {
    let mut agent = Needle::builder().model(&model()).tool_with_handler(weather(), |arguments| Ok(json!({"temp_c": 20, "city": arguments["city"]}))).build();
    let response = agent.run("what's the weather in Paris?", &RunOptions::default());
    let results = response.results.unwrap();
    assert_eq!(results[0]["temp_c"], 20);
    assert_eq!(results[0]["city"], "Paris");
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn two_agents_keep_their_own_tools() {
    let model = model();
    let mut heating = Needle::builder().model(&model).tool(thermostat()).build();
    let mut outside = Needle::builder().model(&model).tool(weather()).build();
    let first = heating.complete("set it to 21 degrees");
    assert_eq!(first.function_calls[0].name, "set_thermostat");
    assert_eq!(first.function_calls[0].arguments["temperature"], 21);
    let second = outside.complete("what's the weather in Paris?");
    assert_eq!(second.function_calls[0].name, "get_weather");
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn extracting_does_not_disturb_an_agent_that_is_mid_conversation() {
    let model = model();
    let mut agent = Needle::builder().model(&model).tool(thermostat()).build();
    let contact = tool("Contact", "A contact.", json!({"name": {"type": "string"}, "email": {"type": "string"}}), &["name", "email"]);
    extract(&model, contact, "John Doe can be reached at john@doe.com", &ExtractOptions { strict: false, ..ExtractOptions::default() }).unwrap();
    let response = agent.complete("set it to 21 degrees");
    assert_eq!(response.function_calls[0].name, "set_thermostat");
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn off_topic_requests_are_refused_with_an_empty_call() {
    let mut agent = Needle::builder().model(&model()).tool(weather()).build();
    let response = agent.complete("tell me a joke about cats");
    assert!(response.function_calls.is_empty());
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn two_calls_come_back_in_the_order_they_were_asked_for() {
    let mut agent = Needle::builder().model(&model()).tool(weather()).tool(thermostat()).build();
    let response = agent.complete("what's the weather in Lagos and set it to 22 degrees");
    let names: Vec<&str> = response.function_calls.iter().map(|call| call.name.as_str()).collect();
    assert_eq!(names, vec!["get_weather", "set_thermostat"]);
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn a_shallower_model_still_answers_an_easy_request() {
    let base = model();
    let shallow = base.with_depth(16).unwrap();
    let lights = tool(
        "set_lights",
        "Turn the lights in a room on or off.",
        json!({"room": {"type": "string", "enum": ["kitchen", "bedroom"]}, "state": {"type": "string", "enum": ["on", "off"]}}),
        &["room", "state"],
    );
    let mut agent = Needle::builder().model(&shallow).tool(lights).build();
    let response = agent.complete("turn on the kitchen lights");
    assert_eq!(response.function_calls[0].name, "set_lights");
    assert_eq!(response.function_calls[0].arguments["room"], "kitchen");
    assert_eq!(response.function_calls[0].arguments["state"], "on");
}
