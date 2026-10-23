use std::sync::Arc;

use rustyneedle::{Call, Engine, Model, Needle, Response, ResponseKind, RunOptions, ToolFile};
use serde_json::{json, Map, Value};

fn assert_send<T: Send>() {}
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn models_can_be_shared_across_threads_and_agents_moved_between_them() {
    assert_send_sync::<Arc<Model>>();
    assert_send::<Needle>();
    assert_send::<Response>();
}

struct Scripted(Vec<Call>);

impl Engine for Scripted {
    fn complete(&mut self, _text: &str, _max_new_tokens: usize) -> Response {
        let mut response = Response::empty(ResponseKind::Call);
        response.function_calls = self.0.clone();
        response
    }

    fn reset(&mut self) {}

    fn embed(&self, _text: &str) -> Option<Vec<f32>> {
        None
    }
}

#[test]
fn an_agent_runs_a_handler_for_each_call_the_engine_produces() {
    let arguments: Map<String, Value> = json!({"city": "Lagos"}).as_object().cloned().unwrap();
    let weather = json!({"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}});
    let mut agent = Needle::builder()
        .engine(Box::new(Scripted(vec![Call { name: "get_weather".into(), arguments }])))
        .tool_with_handler(weather, |arguments| Ok(json!({"seen": arguments["city"]})))
        .auto_date(false)
        .build();
    let response = agent.run("weather in Lagos", &RunOptions { max_steps: 1, ..RunOptions::default() });
    assert_eq!(response.results.unwrap(), vec![json!({"seen": "Lagos"})]);
}

fn example_files(directory: &str) -> Vec<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples").join(directory);
    let mut files: Vec<_> =
        std::fs::read_dir(root).unwrap().map(|entry| entry.unwrap().path()).filter(|path| path.extension().is_some_and(|e| e == "json")).collect();
    files.sort();
    files
}

#[test]
fn every_bundled_tool_file_is_loadable_and_well_formed() {
    let tools = example_files("tools");
    let environments = example_files("environments");
    assert!(tools.len() >= 5);
    assert_eq!(environments.len(), 6);
    let files = tools.into_iter().chain(environments);
    for path in files {
        let file = ToolFile::from_path(&path).unwrap_or_else(|error| panic!("{error}"));
        assert!(!file.tools.is_empty(), "{} has no tools", path.display());
        for tool in &file.tools {
            let name = tool["name"].as_str().unwrap_or_else(|| panic!("{} has an unnamed tool", path.display()));
            assert_eq!(tool["parameters"]["type"], "object", "{name} in {}", path.display());
            assert!(tool["description"].as_str().is_some_and(|text| !text.is_empty()), "{name} in {} has no description", path.display());
        }
    }
}

#[test]
fn a_tool_file_may_be_an_array_a_single_tool_or_an_environment() {
    let tool = json!({"name": "ping", "parameters": {"type": "object", "properties": {}}});
    assert_eq!(ToolFile::from_value(json!([tool.clone(), tool.clone()])).tools.len(), 2);
    assert_eq!(ToolFile::from_value(tool.clone()).tools.len(), 1);
    let environment = ToolFile::from_value(json!({"system": "Be brief.", "tools": [tool]}));
    assert_eq!(environment.tools.len(), 1);
    assert_eq!(environment.system.as_deref(), Some("Be brief."));
}

#[test]
fn a_tool_file_supplies_the_system_text_only_when_none_was_given() {
    let tool = json!({"name": "ping", "parameters": {"type": "object", "properties": {}}});
    let file = || ToolFile::from_value(json!({"system": "from file", "tools": [tool.clone()]}));
    let from_file = Needle::builder().engine(Box::new(Scripted(Vec::new()))).tool_file(file()).auto_date(false).build();
    assert_eq!(from_file.system_text(), "from file");
    let explicit = Needle::builder().engine(Box::new(Scripted(Vec::new()))).system("explicit").tool_file(file()).auto_date(false).build();
    assert_eq!(explicit.system_text(), "explicit");
}

#[test]
fn building_without_a_model_or_engine_is_an_error_not_a_panic() {
    assert!(Needle::builder().try_build().is_err());
}

#[test]
fn the_tool_builder_writes_choices_bounds_and_optional_parameters() {
    use rustyneedle::{tool, Parameter};
    let lights = tool("set_lights")
        .description("Turn the lights in a room on or off.")
        .parameter(Parameter::string("room").choices(["kitchen", "bedroom"]))
        .parameter(Parameter::integer("brightness").at_least(0).at_most(100).optional())
        .triggers([r"\blights?\b"])
        .build();
    assert_eq!(lights["name"], "set_lights");
    assert_eq!(lights["parameters"]["properties"]["room"]["enum"], json!(["kitchen", "bedroom"]));
    assert_eq!(lights["parameters"]["properties"]["brightness"]["minimum"], 0);
    assert_eq!(lights["parameters"]["properties"]["brightness"]["maximum"], 100);
    assert_eq!(lights["parameters"]["required"], json!(["room"]));
    assert_eq!(lights["triggers"], json!([r"\blights?\b"]));
}
