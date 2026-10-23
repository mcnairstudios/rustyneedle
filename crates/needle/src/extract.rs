use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use model::Model;
use serde_json::{Map, Value};

use crate::agent::Needle;
use crate::engine::Engine;
use crate::grounding::{grounded_number_paths, temporal_grounding};
use crate::response::Response;

#[derive(Debug)]
pub struct ExtractionValidationError(pub String);

impl fmt::Display for ExtractionValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "extraction returned values not grounded in the input: {}", self.0)
    }
}

impl std::error::Error for ExtractionValidationError {}

pub struct ExtractOptions {
    pub system: Option<String>,
    pub max_new_tokens: usize,
    pub strict: bool,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self { system: None, max_new_tokens: crate::agent::DEFAULT_MAX_NEW_TOKENS, strict: true }
    }
}

pub fn extract(model: &Arc<Model>, record: Value, text: &str, options: &ExtractOptions) -> Result<Option<Map<String, Value>>, ExtractionValidationError> {
    let agent = Needle::builder().model(model).tool(record).system(options.system.as_deref().unwrap_or("")).build();
    extract_with(agent, text, options)
}

pub fn extract_with_engine(
    engine: Box<dyn Engine>,
    record: Value,
    text: &str,
    options: &ExtractOptions,
) -> Result<Option<Map<String, Value>>, ExtractionValidationError> {
    let agent = Needle::builder().engine(engine).tool(record).system(options.system.as_deref().unwrap_or("")).build();
    extract_with(agent, text, options)
}

fn extract_with(mut agent: Needle, text: &str, options: &ExtractOptions) -> Result<Option<Map<String, Value>>, ExtractionValidationError> {
    let facts = agent.system_text().to_owned();
    let schema = agent.tool_schemas()[0].clone();
    let response = agent.complete_within(text, options.max_new_tokens);
    let Some(call) = response.function_calls.first().or_else(|| response.suppressed_calls.first()) else { return Ok(None) };
    let arguments = call.arguments.clone();
    if options.strict {
        validate_extraction(text, &schema, &arguments, &response, &facts)?;
    }
    Ok(Some(arguments))
}

pub fn validate_extraction(
    text: &str,
    schema: &Value,
    arguments: &Map<String, Value>,
    response: &Response,
    system: &str,
) -> Result<(), ExtractionValidationError> {
    let arguments_value = Value::Object(arguments.clone());
    let (checked, mut failures) = temporal_grounding(text, schema, &arguments_value, Some(system));
    let flagged: Vec<String> = response.validation.as_ref().map(|validation| validation.ungrounded.clone()).unwrap_or_default();
    let grounded = if flagged.is_empty() { BTreeSet::new() } else { grounded_number_paths(&arguments_value, &[text, system]) };
    for name in &flagged {
        let path = name.split_once('.').map_or(name.as_str(), |(_, path)| path).to_owned();
        if grounded.contains(&path) {
            continue;
        }
        if !checked.contains(&path) || failures.contains(&path) {
            failures.insert(path);
        }
    }
    if response.validation.as_ref().and_then(|validation| validation.negation).unwrap_or(false) {
        failures.insert("negated request".to_owned());
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(ExtractionValidationError(failures.into_iter().collect::<Vec<_>>().join(", ")))
    }
}
