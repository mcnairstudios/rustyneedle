use serde_json::{Number, Value};

use crate::call::Call;
use crate::dates::explicit_date;
use crate::evidence::Evidence;
use crate::negation::request_is_negated;

const CAPITALISED_ARGUMENTS: &[&str] = &["title", "message", "label"];

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Findings {
    pub ungrounded: Vec<String>,
    pub negation: bool,
    pub withhold: bool,
}

fn find_tool<'a>(tools: &'a Value, name: &str) -> Option<&'a Value> {
    tools.as_array()?.iter().map(|tool| tool.get("function").unwrap_or(tool)).find(|tool| tool.get("name").and_then(Value::as_str) == Some(name))
}

fn is_required(parameters: &Value, argument: &str) -> bool {
    parameters.get("required").and_then(Value::as_array).is_some_and(|list| list.iter().any(|name| name.as_str() == Some(argument)))
}

fn capitalise_first(text: &str) -> String {
    let mut characters = text.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
        None => String::new(),
    }
}

pub fn repair_calls(calls: Vec<Call>, tools: &Value, query: &str) -> (Vec<Call>, Findings) {
    let evidence = Evidence::new(query);
    let mut findings = Findings { ungrounded: Vec::new(), negation: request_is_negated(query), withhold: false };
    let mut repaired = Vec::with_capacity(calls.len());
    for mut call in calls {
        let parameters = find_tool(tools, &call.name).and_then(|tool| tool.get("parameters")).cloned().unwrap_or(Value::Null);
        let properties = parameters.get("properties").and_then(Value::as_object);
        let names: Vec<String> = call.arguments.keys().cloned().collect();
        for argument in names {
            let schema = properties.and_then(|map| map.get(&argument));
            let kind = schema.and_then(|schema| schema.get("type")).and_then(Value::as_str).unwrap_or("string");
            let has_choices = schema.is_some_and(|schema| schema.get("enum").is_some());
            let required = is_required(&parameters, &argument);
            let value = call.arguments[&argument].clone();
            match (&value, kind) {
                (Value::Number(number), "integer" | "number") => {
                    let amount = number.as_f64().unwrap_or(0.0);
                    if required && !evidence.explicit_quantity && schema.is_none_or(|schema| schema.get("default").is_none()) {
                        findings.withhold = true;
                    }
                    if !evidence.mentions_number(amount) {
                        if required {
                            findings.ungrounded.push(format!("{}.{}", call.name, argument));
                        } else {
                            call.arguments.shift_remove(&argument);
                            continue;
                        }
                    }
                    if kind == "number" {
                        if let Some(float) = Number::from_f64(amount) {
                            call.arguments.insert(argument.clone(), Value::Number(float));
                        }
                    }
                }
                (Value::String(text), "string") if has_choices => {
                    if !evidence.names_option(text) {
                        let options: Vec<String> = schema
                            .and_then(|schema| schema.get("enum"))
                            .and_then(Value::as_array)
                            .map(|list| list.iter().filter_map(Value::as_str).filter(|option| evidence.names_option(option)).map(str::to_owned).collect())
                            .unwrap_or_default();
                        if options.len() == 1 {
                            call.arguments.insert(argument.clone(), Value::String(options[0].clone()));
                        }
                    }
                }
                (Value::String(text), "string") => {
                    if argument.contains("date") {
                        if let Some(resolved) = explicit_date(query) {
                            call.arguments.insert(argument.clone(), Value::String(resolved));
                            continue;
                        }
                    }
                    let duplicates_other = call
                        .arguments
                        .iter()
                        .any(|(other, other_value)| other != &argument && other_value.as_str().is_some_and(|o| o.eq_ignore_ascii_case(text)));
                    if !required && !text.is_empty() && (!evidence.mentions_text(text) || duplicates_other) {
                        call.arguments.shift_remove(&argument);
                        continue;
                    }
                    if required && !text.is_empty() && !evidence.mentions_text(text) && !looks_like_resolved_date(text) {
                        findings.ungrounded.push(format!("{}.{}", call.name, argument));
                    }
                    if CAPITALISED_ARGUMENTS.contains(&argument.as_str()) {
                        call.arguments.insert(argument.clone(), Value::String(capitalise_first(text)));
                    }
                }
                _ => {}
            }
        }
        repaired.push(call);
    }
    (repaired, findings)
}

fn looks_like_resolved_date(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() >= 10 && bytes[..4].iter().all(u8::is_ascii_digit) && bytes[4] == b'-' && bytes[7] == b'-'
}
