use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub name: String,
    pub arguments: Map<String, Value>,
}

impl Call {
    pub fn from_value(value: &Value) -> Option<Call> {
        let name = value.get("name")?.as_str()?.to_owned();
        let arguments = value.get("arguments").and_then(Value::as_object).cloned().unwrap_or_default();
        Some(Call { name, arguments })
    }

    pub fn to_value(&self) -> Value {
        let mut object = Map::new();
        object.insert("name".into(), Value::String(self.name.clone()));
        object.insert("arguments".into(), Value::Object(self.arguments.clone()));
        Value::Object(object)
    }
}

pub fn parse_calls(text: &str) -> Vec<Call> {
    serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|value| value.as_array().map(|list| list.iter().filter_map(Call::from_value).collect()))
        .unwrap_or_default()
}
