use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use chrono::{DateTime, Local};
use model::Model;
use serde_json::{json, Map, Value};

use crate::engine::{Engine, ModelEngine};
use crate::grounding::{annotate_ungrounded, grounded_number_paths, source_years, ungrounded_paths, with_date_fact};
use crate::pyjson;
use crate::response::{Response, ResponseKind};
use crate::tool_file::ToolFile;

pub const DEFAULT_MAX_NEW_TOKENS: usize = 512;
pub const DEFAULT_MAX_STEPS: usize = 8;

type Handler = Box<dyn FnMut(&Map<String, Value>) -> Result<Value, String> + Send>;

#[derive(Debug)]
pub struct MissingEngine;

impl std::fmt::Display for MissingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "a Needle needs a model or an engine")
    }
}

impl std::error::Error for MissingEngine {}

pub struct RunOptions {
    pub max_steps: usize,
    pub max_new_tokens: usize,
    pub strict: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self { max_steps: DEFAULT_MAX_STEPS, max_new_tokens: DEFAULT_MAX_NEW_TOKENS, strict: true }
    }
}

pub struct NeedleBuilder {
    tools: Vec<Value>,
    handlers: BTreeMap<String, Handler>,
    system: String,
    auto_date: bool,
    stateless: bool,
    model: Option<Arc<Model>>,
    engine: Option<Box<dyn Engine>>,
    now: Option<DateTime<Local>>,
}

impl NeedleBuilder {
    pub fn model(mut self, model: &Arc<Model>) -> Self {
        self.model = Some(Arc::clone(model));
        self
    }

    pub fn engine(mut self, engine: Box<dyn Engine>) -> Self {
        self.engine = Some(engine);
        self
    }

    pub fn tool(mut self, schema: Value) -> Self {
        self.tools.push(schema);
        self
    }

    pub fn tool_with_handler(mut self, schema: Value, handler: impl FnMut(&Map<String, Value>) -> Result<Value, String> + Send + 'static) -> Self {
        if let Some(name) = schema.get("name").and_then(Value::as_str) {
            self.handlers.insert(name.to_owned(), Box::new(handler));
        }
        self.tools.push(schema);
        self
    }

    pub fn tools(mut self, schemas: impl IntoIterator<Item = Value>) -> Self {
        self.tools.extend(schemas);
        self
    }

    pub fn tool_file(mut self, file: ToolFile) -> Self {
        self.tools.extend(file.tools);
        if self.system.is_empty() {
            self.system = file.system.unwrap_or_default();
        }
        self
    }

    pub fn system(mut self, system: &str) -> Self {
        self.system = system.to_owned();
        self
    }

    pub fn auto_date(mut self, enabled: bool) -> Self {
        self.auto_date = enabled;
        self
    }

    pub fn stateless(mut self, enabled: bool) -> Self {
        self.stateless = enabled;
        self
    }

    pub fn clock(mut self, now: DateTime<Local>) -> Self {
        self.now = Some(now);
        self
    }

    pub fn build(self) -> Needle {
        self.try_build().expect("a Needle needs a model or an engine")
    }

    pub fn try_build(self) -> Result<Needle, MissingEngine> {
        let now = self.now.unwrap_or_else(Local::now);
        let system_text = if self.auto_date { with_date_fact(&self.system, now) } else { self.system.clone() };
        let schemas = Value::Array(self.tools.clone());
        let engine: Box<dyn Engine> = match (self.engine, &self.model) {
            (Some(engine), _) => engine,
            (None, Some(model)) => Box::new(ModelEngine::new(model, &schemas, &system_text)),
            (None, None) => return Err(MissingEngine),
        };
        Ok(Needle { engine, tool_schemas: self.tools, handlers: self.handlers, system_text, seen_years: BTreeSet::new(), stateless: self.stateless })
    }
}

pub struct Needle {
    engine: Box<dyn Engine>,
    tool_schemas: Vec<Value>,
    handlers: BTreeMap<String, Handler>,
    system_text: String,
    seen_years: BTreeSet<i32>,
    stateless: bool,
}

impl Needle {
    pub fn builder() -> NeedleBuilder {
        NeedleBuilder {
            tools: Vec::new(),
            handlers: BTreeMap::new(),
            system: String::new(),
            auto_date: true,
            stateless: false,
            model: None,
            engine: None,
            now: None,
        }
    }

    pub fn system_text(&self) -> &str {
        &self.system_text
    }

    pub fn tool_schemas(&self) -> &[Value] {
        &self.tool_schemas
    }

    pub fn complete(&mut self, text: &str) -> Response {
        self.complete_within(text, DEFAULT_MAX_NEW_TOKENS)
    }

    pub fn complete_within(&mut self, text: &str, max_new_tokens: usize) -> Response {
        if self.stateless {
            self.reset();
        }
        self.turn(text, max_new_tokens, true)
    }

    pub(crate) fn turn(&mut self, text: &str, max_new_tokens: usize, ground: bool) -> Response {
        self.seen_years.extend(source_years(text));
        let mut response = self.engine.complete(text, max_new_tokens);
        if ground {
            annotate_ungrounded(&mut response, &self.tool_schemas, &self.seen_years, Some(&self.system_text), Some(text));
        }
        response
    }

    pub fn embed(&self, text: &str) -> Option<Vec<f32>> {
        self.engine.embed(text)
    }

    pub fn reset(&mut self) {
        self.engine.reset();
        self.seen_years.clear();
    }

    pub fn run(&mut self, query: &str, options: &RunOptions) -> Response {
        if self.stateless {
            self.reset();
        }
        let mut response = self.turn(query, options.max_new_tokens, true);
        let mut executed: Vec<Value> = Vec::new();
        for _ in 0..options.max_steps {
            if response.kind != ResponseKind::Call || response.function_calls.is_empty() {
                break;
            }
            let ungrounded = ungrounded_paths(&response);
            let calls = response.function_calls.clone();
            let mut results = Vec::with_capacity(calls.len());
            for call in &calls {
                let mut fabricated: Vec<String> = ungrounded.get(&call.name).map(|paths| paths.iter().cloned().collect()).unwrap_or_default();
                if options.strict && !fabricated.is_empty() {
                    let grounded = grounded_number_paths(&Value::Object(call.arguments.clone()), &[query, &self.system_text]);
                    fabricated.retain(|path| !grounded.contains(path));
                }
                if options.strict && !fabricated.is_empty() {
                    results.push(json!({"error": format!("ungrounded {}", fabricated.join(", "))}));
                    continue;
                }
                let outcome = match self.handlers.get_mut(&call.name) {
                    Some(handler) => handler(&call.arguments).unwrap_or_else(|error| json!({"error": error})),
                    None => json!({"error": format!("unknown tool: {}", call.name)}),
                };
                results.push(outcome);
            }
            executed.extend(results.iter().cloned());
            response = self.turn(&pyjson::dumps(&Value::Array(results)), options.max_new_tokens, false);
        }
        response.results = Some(executed);
        response
    }
}
