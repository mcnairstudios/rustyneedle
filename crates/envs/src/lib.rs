use std::sync::Arc;

use model::Model;
use needle::{Engine, Needle, Response};
use serde_json::{json, Map, Value};

const SOURCES: [&str; 6] = [
    include_str!("../../../examples/environments/data_capture.json"),
    include_str!("../../../examples/environments/kitchen_appliance.json"),
    include_str!("../../../examples/environments/media_player.json"),
    include_str!("../../../examples/environments/productivity.json"),
    include_str!("../../../examples/environments/smart_home.json"),
    include_str!("../../../examples/environments/wearable.json"),
];

const PASS_RATE: f64 = 0.9;

#[derive(Clone, Debug)]
pub struct Case {
    pub query: String,
    pub calls: Vec<Value>,
    pub category: String,
    pub critical: bool,
}

#[derive(Clone, Debug)]
pub struct Environment {
    pub name: String,
    pub system: String,
    pub tools: Vec<Value>,
    pub cases: Vec<Case>,
}

#[derive(Debug)]
pub struct Failure {
    pub case: Case,
    pub got: Vec<Value>,
}

#[derive(Debug)]
pub struct Report {
    pub environment: String,
    pub passed: usize,
    pub total: usize,
    pub critical_failures: usize,
    pub failures: Vec<Failure>,
}

impl Report {
    pub fn succeeded(&self) -> bool {
        self.passed as f64 >= (PASS_RATE * self.total as f64).round() && self.critical_failures == 0
    }
}

fn parse(source: &str) -> Environment {
    let data: Value = serde_json::from_str(source).expect("bundled environment parses");
    let cases = data["cases"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|case| Case {
            query: case["query"].as_str().unwrap_or_default().to_owned(),
            calls: case["calls"].as_array().cloned().unwrap_or_default(),
            category: case["category"].as_str().unwrap_or_default().to_owned(),
            critical: case["critical"].as_bool().unwrap_or(false),
        })
        .collect();
    Environment {
        name: data["name"].as_str().unwrap_or_default().to_owned(),
        system: data["system"].as_str().unwrap_or_default().to_owned(),
        tools: data["tools"].as_array().cloned().unwrap_or_default(),
        cases,
    }
}

pub fn all() -> Vec<Environment> {
    SOURCES.iter().map(|source| parse(source)).collect()
}

pub fn named(name: &str) -> Option<Environment> {
    all().into_iter().find(|environment| environment.name == name)
}

fn acknowledge(arguments: &Map<String, Value>) -> Result<Value, String> {
    let mut result = Map::new();
    result.insert("ok".into(), json!(true));
    result.extend(arguments.iter().map(|(key, value)| (key.clone(), value.clone())));
    Ok(Value::Object(result))
}

impl Environment {
    pub fn agent(&self, model: &Arc<Model>) -> Needle {
        self.builder().model(model).build()
    }

    pub fn agent_with_engine(&self, engine: Box<dyn Engine>) -> Needle {
        self.builder().engine(engine).build()
    }

    fn builder(&self) -> needle::NeedleBuilder {
        let mut builder = Needle::builder().system(&self.system);
        for schema in &self.tools {
            builder = builder.tool_with_handler(schema.clone(), acknowledge);
        }
        builder
    }

    pub fn run_tests(&self, agent: &mut Needle, min_confidence: f32) -> Report {
        let mut failures = Vec::new();
        let mut critical_failures = 0;
        for case in &self.cases {
            agent.reset();
            let got = calls_for(&agent.complete(&case.query), min_confidence);
            if !same_calls(&got, &case.calls) {
                critical_failures += case.critical as usize;
                failures.push(Failure { case: case.clone(), got });
            }
        }
        Report { environment: self.name.clone(), passed: self.cases.len() - failures.len(), total: self.cases.len(), critical_failures, failures }
    }
}

fn calls_for(response: &Response, min_confidence: f32) -> Vec<Value> {
    let flagged = response.validation.as_ref().is_some_and(|validation| !validation.ungrounded.is_empty() || validation.negation == Some(true));
    let too_unsure = response.confidence.is_some_and(|confidence| confidence < min_confidence);
    if response.function_calls.is_empty() || flagged || too_unsure {
        return Vec::new();
    }
    response.function_calls.iter().map(|call| call.to_value()).collect()
}

fn fold(value: &Value) -> Value {
    match value {
        Value::String(text) => Value::String(text.to_lowercase()),
        Value::Number(number) => match number.as_f64() {
            Some(float) if float.fract() == 0.0 && float.abs() < 1e15 => json!(float as i64),
            _ => value.clone(),
        },
        Value::Object(map) => Value::Object(map.iter().map(|(key, item)| (key.clone(), fold(item))).collect()),
        Value::Array(list) => Value::Array(list.iter().map(fold).collect()),
        other => other.clone(),
    }
}

fn same_calls(got: &[Value], want: &[Value]) -> bool {
    let key = |calls: &[Value]| {
        let mut keys: Vec<String> = calls.iter().map(|call| serde_json::to_string(&sorted(&fold(call))).unwrap_or_default()).collect();
        keys.sort();
        keys
    };
    key(got) == key(want)
}

fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let ordered: std::collections::BTreeMap<_, _> = map.iter().map(|(key, item)| (key.clone(), sorted(item))).collect();
            Value::Object(ordered.into_iter().collect())
        }
        Value::Array(list) => Value::Array(list.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

pub fn run_all(model: &Arc<Model>, min_confidence: f32) -> Vec<Report> {
    all().iter().map(|environment| environment.run_tests(&mut environment.agent(model), min_confidence)).collect()
}
