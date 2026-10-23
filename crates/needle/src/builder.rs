use serde_json::{json, Map, Value};

#[derive(Clone, Debug)]
pub struct Parameter {
    name: String,
    schema: Map<String, Value>,
    required: bool,
}

impl Parameter {
    fn typed(name: &str, kind: &str) -> Self {
        let mut schema = Map::new();
        schema.insert("type".into(), json!(kind));
        Self { name: name.to_owned(), schema, required: true }
    }

    pub fn string(name: &str) -> Self {
        Self::typed(name, "string")
    }

    pub fn integer(name: &str) -> Self {
        Self::typed(name, "integer")
    }

    pub fn number(name: &str) -> Self {
        Self::typed(name, "number")
    }

    pub fn boolean(name: &str) -> Self {
        Self::typed(name, "boolean")
    }

    pub fn object(name: &str) -> Self {
        Self::typed(name, "object")
    }

    pub fn array_of(name: &str, items: Parameter) -> Self {
        let mut parameter = Self::typed(name, "array");
        parameter.schema.insert("items".into(), Value::Object(items.schema));
        parameter
    }

    fn with(mut self, key: &str, value: Value) -> Self {
        self.schema.insert(key.to_owned(), value);
        self
    }

    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    pub fn description(self, text: &str) -> Self {
        self.with("description", json!(text))
    }

    pub fn choices<V: Into<Value>>(mut self, values: impl IntoIterator<Item = V>) -> Self {
        let values: Vec<Value> = values.into_iter().map(Into::into).collect();
        if let Some(first) = values.first() {
            let kind = match first {
                Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
                Value::Number(_) => "number",
                Value::Bool(_) => "boolean",
                _ => "string",
            };
            self.schema.insert("type".into(), json!(kind));
        }
        self.with("enum", Value::Array(values))
    }

    pub fn constant(self, value: impl Into<Value>) -> Self {
        self.with("const", value.into())
    }

    pub fn at_least(self, bound: impl Into<Value>) -> Self {
        self.with("minimum", bound.into())
    }

    pub fn at_most(self, bound: impl Into<Value>) -> Self {
        self.with("maximum", bound.into())
    }

    pub fn above(self, bound: impl Into<Value>) -> Self {
        self.with("exclusiveMinimum", bound.into())
    }

    pub fn below(self, bound: impl Into<Value>) -> Self {
        self.with("exclusiveMaximum", bound.into())
    }

    pub fn multiple_of(self, step: impl Into<Value>) -> Self {
        self.with("multipleOf", step.into())
    }

    pub fn min_length(self, length: usize) -> Self {
        self.with("minLength", json!(length))
    }

    pub fn max_length(self, length: usize) -> Self {
        self.with("maxLength", json!(length))
    }

    pub fn pattern(self, expression: &str) -> Self {
        self.with("pattern", json!(expression))
    }

    pub fn format(self, name: &str) -> Self {
        self.with("format", json!(name))
    }

    pub fn min_items(self, count: usize) -> Self {
        self.with("minItems", json!(count))
    }

    pub fn max_items(self, count: usize) -> Self {
        self.with("maxItems", json!(count))
    }

    pub fn unique_items(self, unique: bool) -> Self {
        self.with("uniqueItems", json!(unique))
    }
}

pub struct ToolBuilder {
    name: String,
    description: Option<String>,
    parameters: Vec<Parameter>,
    triggers: Vec<String>,
}

pub fn tool(name: &str) -> ToolBuilder {
    ToolBuilder { name: name.to_owned(), description: None, parameters: Vec::new(), triggers: Vec::new() }
}

impl ToolBuilder {
    pub fn description(mut self, text: &str) -> Self {
        self.description = Some(text.trim().to_owned());
        self
    }

    pub fn parameter(mut self, parameter: Parameter) -> Self {
        self.parameters.push(parameter);
        self
    }

    pub fn triggers<S: Into<String>>(mut self, patterns: impl IntoIterator<Item = S>) -> Self {
        self.triggers.extend(patterns.into_iter().map(Into::into));
        self
    }

    pub fn build(self) -> Value {
        let required: Vec<Value> = self.parameters.iter().filter(|p| p.required).map(|p| json!(p.name)).collect();
        let properties: Map<String, Value> = self.parameters.into_iter().map(|p| (p.name, Value::Object(p.schema))).collect();
        let mut parameters = Map::new();
        parameters.insert("type".into(), json!("object"));
        parameters.insert("properties".into(), Value::Object(properties));
        if !required.is_empty() {
            parameters.insert("required".into(), Value::Array(required));
        }
        let mut out = Map::new();
        out.insert("name".into(), json!(self.name));
        if let Some(description) = self.description.filter(|text| !text.is_empty()) {
            out.insert("description".into(), json!(description));
        }
        out.insert("parameters".into(), Value::Object(parameters));
        if !self.triggers.is_empty() {
            out.insert("triggers".into(), json!(self.triggers));
        }
        Value::Object(out)
    }
}
