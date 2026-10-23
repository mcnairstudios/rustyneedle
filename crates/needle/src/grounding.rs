use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use chrono::{DateTime, Local};
use fancy_regex::Regex;
use serde_json::Value;

use crate::response::{Response, Validation};

const MONTHS: &str =
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?";
const NOT_TIME: &str = r"(?![0-9A-Za-z]|:\d|\s*[ap]\.?m\b)";

fn compile(pattern: &str) -> Regex {
    Regex::new(pattern).expect("pattern compiles")
}

static YEAR_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        format!(r"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}[\s,]+(\d{{1,4}}){NOT_TIME}"),
        format!(r"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s+(\d{{1,4}}){NOT_TIME}"),
        format!(r"\b{MONTHS}[\s,]+(\d{{3,4}})(?![0-9A-Za-z])"),
        r"\byear\s+(\d{1,4})(?![0-9A-Za-z])".to_owned(),
        r"(?<![0-9])(\d{4})(?=[-/]\d{1,2}[-/]\d{1,2}(?![0-9]))".to_owned(),
    ]
    .iter()
    .map(|pattern| compile(pattern))
    .collect()
});

static RELATIVE_CUE: LazyLock<Regex> = LazyLock::new(|| {
    compile(
        r"(?i)\b(today|tonight|tomorrow|yesterday|next|this|coming|now|in \d+ (?:days?|weeks?|months?|years?)|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    )
});
static DATE_KEY: LazyLock<Regex> = LazyLock::new(|| compile(r"(?<![A-Za-z])date:"));
static DATE_STAMP: LazyLock<Regex> = LazyLock::new(|| compile(r"\d{4}-\d{2}-\d{2}"));
static DATE_FACT: LazyLock<Regex> =
    LazyLock::new(|| compile(r"date:\s*\d{4}-\d{2}-\d{2}(?:\s+[A-Za-z]{3})?(?:\s+\d{2}:\d{2})?|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?"));
static NUMBER_TOKEN: LazyLock<Regex> = LazyLock::new(|| compile(r"(?<![\w.,])(?:[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)(?!\d)"));

pub fn source_years(text: &str) -> BTreeSet<i32> {
    let lowered = text.to_lowercase();
    let mut years = BTreeSet::new();
    for pattern in YEAR_PATTERNS.iter() {
        for captures in pattern.captures_iter(&lowered).flatten() {
            if let Some(year) = captures.get(1).and_then(|m| m.as_str().parse::<i32>().ok()) {
                years.insert(year);
            }
        }
    }
    years
}

pub fn relative_cue(text: &str) -> bool {
    !text.is_empty() && RELATIVE_CUE.is_match(text).unwrap_or(false)
}

pub fn licensed_years(seen: &BTreeSet<i32>, system: Option<&str>, relative: bool) -> BTreeSet<i32> {
    let mut years = seen.clone();
    if let (false, Some(system), true) = (years.is_empty(), system, relative) {
        if !system.is_empty() {
            years.extend(source_years(system));
        }
    }
    years
}

pub fn with_date_fact(system: &str, now: DateTime<Local>) -> String {
    if DATE_KEY.is_match(system).unwrap_or(false) || DATE_STAMP.is_match(system).unwrap_or(false) {
        return system.to_owned();
    }
    let fact = now.format("date: %Y-%m-%d %a %H:%M").to_string();
    if system.trim().is_empty() {
        return fact;
    }
    if system.trim_start().starts_with('{') {
        return system.to_owned();
    }
    format!("{fact}; {system}")
}

fn resolve_ref<'a>(node: &'a Value, root: &'a Value) -> &'a Value {
    let mut current = node;
    let mut seen = BTreeSet::new();
    while let Some(reference) = current.get("$ref").and_then(Value::as_str) {
        if !seen.insert(reference.to_owned()) {
            break;
        }
        let mut target = root;
        for part in reference.trim_start_matches("#/").split('/') {
            target = target.get(part.replace("~1", "/").replace("~0", "~").as_str()).unwrap_or(&Value::Null);
        }
        if std::ptr::eq(target, current) {
            break;
        }
        current = target;
    }
    current
}

fn schema_parameters(schema: &Value) -> &Value {
    schema.get("parameters").unwrap_or(schema)
}

pub fn walk_grounding(schema: &Value, arguments: &Value, years: &BTreeSet<i32>) -> (BTreeSet<String>, BTreeSet<String>) {
    let root = schema_parameters(schema);
    let mut checked = BTreeSet::new();
    let mut failures = BTreeSet::new();
    walk(arguments, root, root, "", years, &mut checked, &mut failures);
    (checked, failures)
}

fn walk(value: &Value, node: &Value, root: &Value, path: &str, years: &BTreeSet<i32>, checked: &mut BTreeSet<String>, failures: &mut BTreeSet<String>) {
    let mut node = resolve_ref(node, root);
    let variants = node.get("anyOf").or_else(|| node.get("oneOf")).and_then(Value::as_array).cloned().unwrap_or_default();
    let concrete: Vec<&Value> = variants.iter().filter(|variant| resolve_ref(variant, root).get("type").and_then(Value::as_str) != Some("null")).collect();
    if concrete.len() == 1 {
        node = resolve_ref(concrete[0], root);
    }
    let format = node.get("format").and_then(Value::as_str);
    if matches!(format, Some("date") | Some("date-time")) {
        if let Some(text) = value.as_str() {
            let leading = text.get(..5).filter(|head| head.len() == 5 && head.ends_with('-') && head[..4].chars().all(|c| c.is_ascii_digit()));
            if let (Some(head), false) = (leading, years.is_empty()) {
                checked.insert(path.to_owned());
                let year: i32 = head[..4].parse().unwrap_or(0);
                if !years.contains(&year) {
                    failures.insert(path.to_owned());
                }
            }
            return;
        }
    }
    match value {
        Value::Object(map) => {
            let properties = node.get("properties").and_then(Value::as_object);
            for (key, item) in map {
                if let Some(child) = properties.and_then(|props| props.get(key)) {
                    let next = if path.is_empty() { key.clone() } else { format!("{path}.{key}") };
                    walk(item, child, root, &next, years, checked, failures);
                }
            }
        }
        Value::Array(list) => {
            if let Some(items) = node.get("items") {
                for (index, item) in list.iter().enumerate() {
                    walk(item, items, root, &format!("{path}[{index}]"), years, checked, failures);
                }
            }
        }
        _ => {}
    }
}

pub fn temporal_grounding(text: &str, schema: &Value, arguments: &Value, system: Option<&str>) -> (BTreeSet<String>, BTreeSet<String>) {
    let years = licensed_years(&source_years(text), system, relative_cue(text));
    walk_grounding(schema, arguments, &years)
}

fn without_date_facts(text: &str) -> String {
    DATE_FACT.replace_all(text, " ").into_owned()
}

fn source_numbers(sources: &[&str]) -> Vec<f64> {
    let joined = sources.iter().filter(|source| !source.is_empty()).map(|source| without_date_facts(source)).collect::<Vec<_>>().join("\n");
    NUMBER_TOKEN.find_iter(&joined).flatten().filter_map(|found| found.as_str().replace(',', "").parse::<f64>().ok()).collect()
}

fn numeric_paths(value: &Value, path: &str, out: &mut Vec<(String, f64)>) {
    match value {
        Value::Number(number) => {
            if let Some(amount) = number.as_f64() {
                out.push((path.to_owned(), amount));
            }
        }
        Value::Object(map) => {
            for (key, item) in map {
                numeric_paths(item, &if path.is_empty() { key.clone() } else { format!("{path}.{key}") }, out);
            }
        }
        Value::Array(list) => {
            for (index, item) in list.iter().enumerate() {
                numeric_paths(item, &format!("{path}[{index}]"), out);
            }
        }
        _ => {}
    }
}

pub fn grounded_number_paths(arguments: &Value, sources: &[&str]) -> BTreeSet<String> {
    let numbers = source_numbers(sources);
    if numbers.is_empty() {
        return BTreeSet::new();
    }
    let mut leaves = Vec::new();
    numeric_paths(arguments, "", &mut leaves);
    leaves.into_iter().filter(|(_, value)| numbers.iter().any(|seen| seen == value)).map(|(path, _)| path).collect()
}

pub fn ungrounded_paths(response: &Response) -> BTreeMap<String, BTreeSet<String>> {
    let mut grouped: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    if let Some(validation) = &response.validation {
        for name in &validation.ungrounded {
            let (tool, path) = name.split_once('.').map_or((name.as_str(), name.as_str()), |(tool, path)| (tool, path));
            grouped.entry(tool.to_owned()).or_default().insert(if path.is_empty() { tool.to_owned() } else { path.to_owned() });
        }
    }
    grouped
}

pub fn annotate_ungrounded(response: &mut Response, tool_schemas: &[Value], seen_years: &BTreeSet<i32>, system: Option<&str>, text: Option<&str>) {
    let years = licensed_years(seen_years, system, text.is_none_or(relative_cue));
    if response.function_calls.is_empty() || years.is_empty() {
        return;
    }
    let mut found = Vec::new();
    for call in &response.function_calls {
        let Some(schema) = tool_schemas.iter().find(|entry| entry.get("name").and_then(Value::as_str) == Some(call.name.as_str())) else { continue };
        let arguments = Value::Object(call.arguments.clone());
        let (_, failures) = walk_grounding(schema, &arguments, &years);
        found.extend(failures.into_iter().map(|path| format!("{}.{}", call.name, path)));
    }
    if found.is_empty() {
        return;
    }
    let validation = response.validation.get_or_insert_with(Validation::default);
    for name in found {
        if !validation.ungrounded.contains(&name) {
            validation.ungrounded.push(name);
        }
    }
}
