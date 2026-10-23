use repair::Call;
use serde_json::{json, Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseKind {
    Call,
    Respond,
}

#[derive(Clone, Debug, Default)]
pub struct Validation {
    pub ungrounded: Vec<String>,
    pub negation: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct Failure {
    pub message: String,
    pub code: String,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct Response {
    pub kind: ResponseKind,
    pub function_calls: Vec<Call>,
    pub suppressed_calls: Vec<Call>,
    pub reasoning: String,
    pub confidence: Option<f32>,
    pub validation: Option<Validation>,
    pub failure: Option<Failure>,
    pub results: Option<Vec<Value>>,
}

impl Response {
    pub fn empty(kind: ResponseKind) -> Self {
        Self {
            kind,
            function_calls: Vec::new(),
            suppressed_calls: Vec::new(),
            reasoning: String::new(),
            confidence: None,
            validation: None,
            failure: None,
            results: None,
        }
    }

    pub fn to_json(&self) -> Value {
        let mut object = Map::new();
        object.insert("type".into(), json!(if self.kind == ResponseKind::Call { "call" } else { "respond" }));
        object.insert("success".into(), json!(self.failure.is_none()));
        object.insert("error".into(), self.failure.as_ref().map_or(Value::Null, |f| json!(f.message)));
        object.insert("error_code".into(), self.failure.as_ref().map_or(Value::Null, |f| json!(f.code)));
        object.insert("reason".into(), self.failure.as_ref().map_or(Value::Null, |f| json!(f.reason)));
        object.insert("function_calls".into(), calls_value(&self.function_calls));
        object.insert("suppressed_calls".into(), calls_value(&self.suppressed_calls));
        object.insert("reasoning".into(), json!(self.reasoning));
        object.insert("confidence".into(), self.confidence.map_or(Value::Null, |value| json!((value as f64 * 10000.0).round() / 10000.0)));
        if let Some(validation) = &self.validation {
            let mut inner = Map::new();
            inner.insert("ungrounded".into(), json!(validation.ungrounded));
            if let Some(negation) = validation.negation {
                inner.insert("negation".into(), json!(negation));
            }
            object.insert("validation".into(), Value::Object(inner));
        }
        if let Some(results) = &self.results {
            object.insert("results".into(), Value::Array(results.clone()));
        }
        Value::Object(object)
    }
}

fn calls_value(calls: &[Call]) -> Value {
    Value::Array(calls.iter().map(Call::to_value).collect())
}
