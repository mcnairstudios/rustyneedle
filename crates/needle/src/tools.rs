use std::collections::BTreeMap;

use fancy_regex::Regex;
use serde_json::{Map, Value};

const DROPPED_KEYS: &[&str] = &["triggers"];

pub fn to_snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    let mut previous: Option<char> = None;
    for character in name.chars() {
        if character.is_alphanumeric() {
            if character.is_uppercase() && previous.is_some_and(|p| p.is_lowercase() || p.is_ascii_digit()) {
                out.push('_');
            }
            out.extend(character.to_lowercase());
        } else if !out.ends_with('_') && !out.is_empty() {
            out.push('_');
        }
        previous = Some(character);
    }
    out.trim_end_matches('_').to_owned()
}

pub struct ToolSet {
    original: Vec<Value>,
    normalised: Vec<Value>,
    names: BTreeMap<String, String>,
    properties: BTreeMap<String, BTreeMap<String, String>>,
    triggers: Vec<Vec<Regex>>,
}

fn unwrap_function(tool: &Value) -> &Value {
    if tool.get("type").and_then(Value::as_str) == Some("function") {
        tool.get("function").unwrap_or(tool)
    } else {
        tool
    }
}

fn normalise_properties(parameters: &Value, aliases: &mut BTreeMap<String, String>) -> Value {
    let Some(object) = parameters.as_object() else { return parameters.clone() };
    let mut out = Map::new();
    for (key, value) in object {
        match key.as_str() {
            "properties" => {
                let mut properties = Map::new();
                for (name, child) in value.as_object().into_iter().flatten() {
                    let snake = to_snake_case(name);
                    aliases.insert(snake.clone(), name.clone());
                    properties.insert(snake, child.clone());
                }
                out.insert(key.clone(), Value::Object(properties));
            }
            "required" => {
                let names: Vec<Value> =
                    value.as_array().into_iter().flatten().filter_map(Value::as_str).map(|name| Value::String(to_snake_case(name))).collect();
                out.insert(key.clone(), Value::Array(names));
            }
            _ => {
                out.insert(key.clone(), value.clone());
            }
        }
    }
    Value::Object(out)
}

impl ToolSet {
    pub fn new(tools: &Value) -> Self {
        let mut original = Vec::new();
        let mut normalised = Vec::new();
        let mut names = BTreeMap::new();
        let mut properties = BTreeMap::new();
        let mut triggers = Vec::new();
        for tool in tools.as_array().into_iter().flatten() {
            let tool = unwrap_function(tool);
            let Some(name) = tool.get("name").and_then(Value::as_str) else { continue };
            let snake = to_snake_case(name);
            let mut aliases = BTreeMap::new();
            let mut entry = Map::new();
            for (key, value) in tool.as_object().into_iter().flatten() {
                if DROPPED_KEYS.contains(&key.as_str()) {
                    continue;
                }
                match key.as_str() {
                    "name" => entry.insert(key.clone(), Value::String(snake.clone())),
                    "parameters" => entry.insert(key.clone(), normalise_properties(value, &mut aliases)),
                    _ => entry.insert(key.clone(), value.clone()),
                };
            }
            triggers.push(
                tool.get("triggers")
                    .and_then(Value::as_array)
                    .map(|list| list.iter().filter_map(Value::as_str).filter_map(|pattern| Regex::new(&format!("(?i){pattern}")).ok()).collect())
                    .unwrap_or_default(),
            );
            names.insert(snake.clone(), name.to_owned());
            properties.insert(snake, aliases);
            original.push(tool.clone());
            normalised.push(Value::Object(entry));
        }
        Self { original, normalised, names, properties, triggers }
    }

    pub fn original(&self) -> &[Value] {
        &self.original
    }

    pub fn prompt_tools(&self) -> Value {
        Value::Array(self.normalised.clone())
    }

    pub fn len(&self) -> usize {
        self.normalised.len()
    }

    pub fn is_empty(&self) -> bool {
        self.normalised.is_empty()
    }

    pub fn triggered(&self, request: &str) -> Option<Vec<usize>> {
        let matched: Vec<usize> = self
            .triggers
            .iter()
            .enumerate()
            .filter(|(_, patterns)| patterns.iter().any(|pattern| pattern.is_match(request).unwrap_or(false)))
            .map(|(index, _)| index)
            .collect();
        (!matched.is_empty()).then_some(matched)
    }

    pub fn original_name(&self, snake: &str) -> String {
        self.names.get(snake).cloned().unwrap_or_else(|| snake.to_owned())
    }

    pub fn original_arguments(&self, snake_tool: &str, arguments: &Map<String, Value>) -> Map<String, Value> {
        let aliases = self.properties.get(snake_tool);
        arguments.iter().map(|(key, value)| (aliases.and_then(|map| map.get(key)).cloned().unwrap_or_else(|| key.clone()), value.clone())).collect()
    }
}
