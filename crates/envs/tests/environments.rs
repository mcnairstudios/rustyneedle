use std::collections::BTreeSet;

use needle::{Engine, Response, ResponseKind, RunOptions};
use repair::Call;
use rustyneedle_envs::{all, named, Case, Environment};
use serde_json::{json, Value};

const CATEGORIES: [&str; 6] = ["invalid", "irrelevant", "missing", "negation", "parallel", "positive"];

#[test]
fn the_registry_holds_the_six_environments() {
    let names: Vec<String> = all().into_iter().map(|environment| environment.name).collect();
    assert_eq!(names, vec!["data_capture", "kitchen_appliance", "media_player", "productivity", "smart_home", "wearable"]);
    assert!(named("smart_home").is_some() && named("nothing").is_none());
    for environment in all() {
        assert!(!environment.system.is_empty() && !environment.tools.is_empty() && !environment.cases.is_empty());
    }
}

#[test]
fn every_environment_keeps_to_five_described_tools_with_typed_properties() {
    for environment in all() {
        assert!((1..=5).contains(&environment.tools.len()), "{}", environment.name);
        for schema in &environment.tools {
            assert!(!schema["description"].as_str().unwrap_or("").is_empty());
            for property in schema["parameters"]["properties"].as_object().unwrap().values() {
                assert!(property.get("type").is_some());
            }
        }
    }
}

#[test]
fn cases_cover_all_six_categories() {
    for environment in all() {
        let categories: BTreeSet<&str> = environment.cases.iter().map(|case| case.category.as_str()).collect();
        assert_eq!(categories, CATEGORIES.iter().copied().collect(), "{}", environment.name);
        assert_eq!(environment.cases.len(), 32);
    }
}

#[test]
fn cases_match_their_declared_tools() {
    for environment in all() {
        for case in &environment.cases {
            for call in &case.calls {
                let schema = environment.tools.iter().find(|tool| tool["name"] == call["name"]).expect("case names a declared tool");
                let parameters = &schema["parameters"];
                let arguments = call["arguments"].as_object().unwrap();
                let properties = parameters["properties"].as_object().unwrap();
                assert!(arguments.keys().all(|key| properties.contains_key(key)));
                for required in parameters["required"].as_array().into_iter().flatten() {
                    assert!(arguments.contains_key(required.as_str().unwrap()));
                }
                for (key, value) in arguments {
                    let property = &properties[key];
                    if let Some(options) = property.get("enum").and_then(Value::as_array) {
                        assert!(options.contains(value));
                    }
                    if let (Some(minimum), Some(number)) = (property.get("minimum").and_then(Value::as_f64), value.as_f64()) {
                        assert!(number >= minimum);
                    }
                    if let (Some(maximum), Some(number)) = (property.get("maximum").and_then(Value::as_f64), value.as_f64()) {
                        assert!(number <= maximum);
                    }
                }
            }
        }
    }
}

struct Scripted {
    calls: Vec<Call>,
}

impl Engine for Scripted {
    fn complete(&mut self, _text: &str, _max_new_tokens: usize) -> Response {
        let mut response = Response::empty(if self.calls.is_empty() { ResponseKind::Respond } else { ResponseKind::Call });
        response.function_calls = std::mem::take(&mut self.calls);
        response
    }

    fn reset(&mut self) {}

    fn embed(&self, _text: &str) -> Option<Vec<f32>> {
        None
    }
}

#[test]
fn every_declared_tool_executes_with_its_cases() {
    for environment in all() {
        for case in &environment.cases {
            let calls: Vec<Call> = case.calls.iter().map(|call| Call::from_value(call).unwrap()).collect();
            let expected = calls.len();
            let mut agent = environment.agent_with_engine(Box::new(Scripted { calls }));
            let results = agent.run(&case.query, &RunOptions::default()).results.unwrap();
            assert_eq!(results.len(), expected);
            assert!(results.iter().all(|result| result["ok"] == json!(true)));
        }
    }
}

struct Chosen;

const MATCHING: &str = r#"{"name":"set_level","arguments":{"level":5}}"#;
const OTHER: &str = r#"{"name":"set_level","arguments":{"level":9}}"#;

impl Engine for Chosen {
    fn complete(&mut self, text: &str, _max_new_tokens: usize) -> Response {
        let query = text.trim();
        let chosen = if query.starts_with("fail") { OTHER } else { MATCHING };
        let mut response = Response::empty(ResponseKind::Call);
        response.confidence = Some(if query.starts_with("low") { 0.2 } else { 0.9 });
        response.function_calls = vec![Call::from_value(&serde_json::from_str::<Value>(chosen).unwrap()).unwrap()];
        response
    }

    fn reset(&mut self) {}

    fn embed(&self, _text: &str) -> Option<Vec<f32>> {
        None
    }
}

fn synthetic(passes: usize, failures: usize, critical: usize) -> Environment {
    let matching: Value = serde_json::from_str(MATCHING).unwrap();
    let mut cases: Vec<Case> =
        (0..passes).map(|i| Case { query: format!("pass {i}"), calls: vec![matching.clone()], category: "positive".into(), critical: false }).collect();
    cases.extend((0..failures).map(|i| Case {
        query: format!("fail {i}"),
        calls: vec![matching.clone()],
        category: "synthetic".into(),
        critical: i < critical,
    }));
    Environment {
        name: "synthetic".into(),
        system: String::new(),
        tools: vec![
            json!({"name": "set_level", "description": "Set the level.", "parameters": {"type": "object", "properties": {"level": {"type": "integer"}}, "required": ["level"]}}),
        ],
        cases,
    }
}

fn outcome(environment: &Environment, min_confidence: f32) -> bool {
    environment.run_tests(&mut environment.agent_with_engine(Box::new(Chosen)), min_confidence).succeeded()
}

#[test]
fn a_suite_passes_at_29_of_32_and_fails_at_28() {
    assert!(outcome(&synthetic(29, 3, 0), 0.0));
    assert!(!outcome(&synthetic(28, 4, 0), 0.0));
}

#[test]
fn one_critical_failure_fails_a_suite_that_clears_the_rate() {
    assert!(!outcome(&synthetic(31, 1, 1), 0.0));
}

#[test]
fn a_confidence_gate_turns_an_unsure_call_into_a_refusal() {
    let refusal = Environment { cases: vec![Case { query: "low".into(), calls: vec![], category: "missing".into(), critical: false }], ..synthetic(0, 0, 0) };
    assert!(!outcome(&refusal, 0.0));
    assert!(outcome(&refusal, 0.4));
}

#[test]
fn a_call_exactly_at_the_gate_is_acted_on() {
    let accepted = Environment {
        cases: vec![Case { query: "pass".into(), calls: vec![serde_json::from_str(MATCHING).unwrap()], category: "positive".into(), critical: false }],
        ..synthetic(0, 0, 0)
    };
    assert!(outcome(&accepted, 0.9));
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn the_acceptance_suites_pass_on_the_real_model() {
    let model = model::Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap();
    let mut passed = 0;
    let mut total = 0;
    for report in rustyneedle_envs::run_all(&model, 0.0) {
        println!("{}: {}/{} passed, {} critical failures", report.environment, report.passed, report.total, report.critical_failures);
        for failure in &report.failures {
            println!("   [{}] {} -> {:?}", failure.case.category, failure.case.query, failure.got);
        }
        passed += report.passed;
        total += report.total;
    }
    println!("{passed}/{total}");
    assert!(passed >= 170, "{passed}/{total}");
}
