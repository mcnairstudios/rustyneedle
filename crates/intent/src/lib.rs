use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use needle::{Needle, Response};
use serde_json::{json, Value};
use whistle::{read_wav, Options, Transcript, Whistle, WhistleError};

pub struct IntentResponse {
    pub transcript: Transcript,
    pub response: Response,
}

impl IntentResponse {
    pub fn to_json(&self) -> Value {
        let mut object = self.response.to_json();
        if let Value::Object(map) = &mut object {
            map.insert("audio_text".into(), json!(self.transcript.text));
            map.insert("audio_language".into(), json!(self.transcript.language));
            map.insert("audio_ttft_ms".into(), json!(round1(self.transcript.first_token_ms)));
            map.insert("audio_decode_tps".into(), json!(round1(self.transcript.tokens_per_second)));
        }
        object
    }
}

fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

pub struct Intent {
    whistle: Arc<Whistle>,
    agent: Needle,
    options: Options,
}

pub fn schema_keywords(tools: &[Value]) -> Vec<String> {
    let mut found = BTreeSet::new();
    for tool in tools {
        collect_options(tool, &mut found);
    }
    found.into_iter().collect()
}

fn collect_options(value: &Value, found: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            if let Some(options) = map.get("enum").and_then(Value::as_array) {
                found.extend(options.iter().filter_map(Value::as_str).map(|option| option.replace('_', " ")));
            }
            map.values().for_each(|child| collect_options(child, found));
        }
        Value::Array(list) => list.iter().for_each(|child| collect_options(child, found)),
        _ => {}
    }
}

impl Intent {
    pub fn new(whistle: Arc<Whistle>, agent: Needle) -> Self {
        let keywords = schema_keywords(agent.tool_schemas());
        Self { whistle, agent, options: Options { keywords, ..Options::default() } }
    }

    pub fn with_keywords(mut self, extra: impl IntoIterator<Item = String>) -> Self {
        self.options.keywords.extend(extra);
        self
    }

    pub fn with_language(mut self, language: Option<String>) -> Self {
        self.options.language = language;
        self
    }

    pub fn agent(&mut self) -> &mut Needle {
        &mut self.agent
    }

    pub fn reset(&mut self) {
        self.agent.reset();
    }

    pub fn process(&mut self, samples: &[f32]) -> Result<IntentResponse, WhistleError> {
        let transcript = self.whistle.transcribe(samples, &self.options)?;
        let response = self.agent.complete(&transcript.text);
        Ok(IntentResponse { transcript, response })
    }

    pub fn process_file(&mut self, path: impl AsRef<Path>) -> Result<IntentResponse, WhistleError> {
        let samples = read_wav(path).map_err(|error| WhistleError(error.to_string()))?;
        self.process(&samples)
    }
}
