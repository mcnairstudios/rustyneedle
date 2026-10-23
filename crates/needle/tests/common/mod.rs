#![allow(dead_code, clippy::new_ret_no_self)]
use std::sync::{Arc, Mutex};

use repair::Call;
use rustyneedle_needle::{Engine, Response, ResponseKind, Validation};
use serde_json::{json, Map, Value};

pub struct Stub {
    pub envelopes: Arc<Mutex<Vec<Response>>>,
    pub calls: Arc<Mutex<Vec<String>>>,
}

pub struct StubHandle {
    pub envelopes: Arc<Mutex<Vec<Response>>>,
    pub calls: Arc<Mutex<Vec<String>>>,
}

impl Stub {
    pub fn new(envelopes: Vec<Response>) -> (Box<dyn Engine>, StubHandle) {
        let envelopes = Arc::new(Mutex::new(envelopes));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let handle = StubHandle { envelopes: Arc::clone(&envelopes), calls: Arc::clone(&calls) };
        (Box::new(Stub { envelopes, calls }), handle)
    }
}

impl StubHandle {
    pub fn load(&self, envelopes: Vec<Response>) {
        *self.envelopes.lock().unwrap() = envelopes;
    }
}

impl Engine for Stub {
    fn complete(&mut self, text: &str, _max_new_tokens: usize) -> Response {
        self.calls.lock().unwrap().push(text.to_owned());
        let mut envelopes = self.envelopes.lock().unwrap();
        if envelopes.len() > 1 {
            envelopes.remove(0)
        } else {
            envelopes[0].clone()
        }
    }

    fn reset(&mut self) {
        self.calls.lock().unwrap().push("reset".to_owned());
    }

    fn embed(&self, _text: &str) -> Option<Vec<f32>> {
        None
    }
}

pub fn arguments(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

pub fn envelope(name: &str, args: Value, ungrounded: &[&str]) -> Response {
    let mut response = Response::empty(ResponseKind::Call);
    response.confidence = Some(0.9);
    response.function_calls = vec![Call { name: name.into(), arguments: arguments(args) }];
    response.validation = Some(Validation { ungrounded: ungrounded.iter().map(|s| s.to_string()).collect(), negation: Some(false) });
    response
}

pub fn dated_call(due_date: &str) -> Response {
    let mut response = Response::empty(ResponseKind::Call);
    response.confidence = Some(0.9);
    response.function_calls = vec![Call { name: "Invoice".into(), arguments: arguments(json!({"vendor": "Acme", "due_date": due_date})) }];
    response
}

pub fn finished() -> Response {
    Response::empty(ResponseKind::Respond)
}

pub fn invoice_schema() -> Value {
    json!({"name": "Invoice", "description": "An invoice.", "parameters": {"type": "object", "properties": {
        "vendor": {"type": "string"},
        "due_date": {"type": "string", "format": "date"},
        "total": {"type": "number", "default": 0.0}}, "required": ["vendor", "due_date"]}})
}

pub fn order_schema() -> Value {
    json!({"name": "Order", "description": "An order.", "parameters": {"type": "object", "properties": {
        "items": {"type": "array", "items": {"$ref": "#/$defs/LineItem"}}}, "required": ["items"],
        "$defs": {"LineItem": {"type": "object", "properties": {"price": {"type": "number"}}, "required": ["price"]}}}})
}

pub fn thermostat_schema() -> Value {
    json!({"name": "set_thermostat", "description": "Set the thermostat.", "parameters": {"type": "object", "properties": {
        "temperature": {"type": "integer", "description": "target temperature in Celsius"},
        "mode": {"type": "string", "description": "heating strategy to use"}}, "required": ["temperature"]}})
}

pub fn invoice_args(total: f64) -> Value {
    json!({"vendor": "Acme Corp", "total": total, "due_date": "2026-09-01"})
}
