use std::path::Path;
use std::sync::Arc;

use intent::Intent;
use model::Model;
use needle::{extract, ExtractOptions, Needle, ToolFile};
use serde_json::{json, Value};
use whistle::{read_wav, Options, Stream, Whistle};

use crate::args::SpeechOptions;
use crate::resources::{keywords, open_model, open_whistle, read_json_argument, CliError};

pub fn render(value: &Value, pretty: bool) -> String {
    if pretty {
        serde_json::to_string_pretty(value).unwrap_or_default()
    } else {
        value.to_string()
    }
}

pub struct AgentSettings<'a> {
    pub system: Option<&'a str>,
    pub auto_date: bool,
}

fn tool_file(argument: &str) -> Result<ToolFile, CliError> {
    Ok(ToolFile::from_value(read_json_argument(argument)?))
}

fn build_agent(model: &Arc<Model>, tools: &str, settings: &AgentSettings<'_>) -> Result<Needle, CliError> {
    let builder = Needle::builder().model(model).system(settings.system.unwrap_or_default());
    Ok(builder.tool_file(tool_file(tools)?).auto_date(settings.auto_date).build())
}

pub fn complete(
    model: Option<&Path>,
    depth: Option<usize>,
    tools: &str,
    settings: AgentSettings<'_>,
    max_new_tokens: usize,
    query: &str,
    pretty: bool,
) -> Result<String, CliError> {
    let model = open_model(model, depth)?;
    let mut agent = build_agent(&model, tools, &settings)?;
    Ok(render(&agent.complete_within(query, max_new_tokens).to_json(), pretty))
}

pub fn extraction(
    model: Option<&Path>,
    depth: Option<usize>,
    schema: &str,
    system: Option<&str>,
    strict: bool,
    text: &str,
    pretty: bool,
) -> Result<String, CliError> {
    let model = open_model(model, depth)?;
    let mut record = read_json_argument(schema)?;
    if record.get("parameters").is_none() {
        record = json!({"name": "record", "description": "Record the extracted fields.", "parameters": record});
    }
    let options = ExtractOptions { system: system.map(str::to_owned), strict, ..ExtractOptions::default() };
    let result = extract(&model, record, text, &options)?;
    Ok(render(&result.map_or(Value::Null, Value::Object), pretty))
}

pub fn embedding(model: Option<&Path>, depth: Option<usize>, text: &str) -> Result<String, CliError> {
    let model = open_model(model, depth)?;
    let agent = Needle::builder().model(&model).auto_date(false).build();
    let vector = agent.embed(text).ok_or_else(|| CliError("this archive carries no confidence head to embed with".into()))?;
    Ok(Value::from(vector.iter().map(|&v| json!(v)).collect::<Vec<_>>()).to_string())
}

fn speech_options(speech: &SpeechOptions, word_timestamps: bool) -> Result<Options, CliError> {
    Ok(Options { language: speech.language.clone(), keywords: keywords(&speech.keywords, speech.keywords_file.as_deref())?, word_timestamps })
}

pub fn transcript_json(transcript: &whistle::Transcript) -> Value {
    let mut object = json!({
        "text": transcript.text,
        "language": transcript.language,
        "ttft_ms": (transcript.first_token_ms * 10.0).round() / 10.0,
        "decode_tps": (transcript.tokens_per_second * 10.0).round() / 10.0,
    });
    if let Some(words) = &transcript.words {
        object["words"] = Value::Array(
            words
                .iter()
                .map(|w| json!({"word": w.word, "start": w.start, "end": w.end, "probability": (w.probability as f64 * 1000.0).round() / 1000.0}))
                .collect(),
        );
    }
    object
}

pub fn transcription(speech: &SpeechOptions, word_timestamps: bool, audio: &Path, pretty: bool) -> Result<String, CliError> {
    let whistle = open_whistle(speech.whistle.as_deref(), speech.audio_depth)?;
    let samples = read_wav(audio)?;
    let transcript = whistle.transcribe(&samples, &speech_options(speech, word_timestamps)?)?;
    Ok(render(&transcript_json(&transcript), pretty))
}

pub fn streaming(speech: &SpeechOptions, chunk_seconds: f64, audio: &Path) -> Result<Vec<String>, CliError> {
    let whistle: Whistle = open_whistle(speech.whistle.as_deref(), speech.audio_depth)?;
    let samples = read_wav(audio)?;
    let mut stream = Stream::new(&whistle, speech_options(speech, true)?);
    let size = ((chunk_seconds * whistle::SAMPLE_RATE as f64) as usize).max(1);
    let mut lines = Vec::new();
    let mut steps: Vec<whistle::StreamStep> = samples.chunks(size).map(|chunk| stream.process(chunk)).collect::<Result<_, _>>()?;
    steps.push(stream.stop()?);
    for step in steps {
        lines.push(json!({"text": step.text, "pending": step.pending, "language": step.language, "received": step.received, "pass_ms": step.pass_ms, "words": step.words.iter().map(|w| json!({"word": w.word, "start": w.start, "end": w.end})).collect::<Vec<_>>()}).to_string());
    }
    Ok(lines)
}

pub fn intent(
    model: Option<&Path>,
    depth: Option<usize>,
    speech: &SpeechOptions,
    tools: &str,
    settings: AgentSettings<'_>,
    audio: &Path,
    pretty: bool,
) -> Result<String, CliError> {
    let model = open_model(model, depth)?;
    let whistle = Arc::new(open_whistle(speech.whistle.as_deref(), speech.audio_depth)?);
    let agent = build_agent(&model, tools, &settings)?;
    let mut pipeline =
        Intent::new(whistle, agent).with_language(speech.language.clone()).with_keywords(keywords(&speech.keywords, speech.keywords_file.as_deref())?);
    Ok(render(&pipeline.process_file(audio)?.to_json(), pretty))
}

pub fn download(spec: &str, directory: &Path) -> Result<String, CliError> {
    Ok(fetch::fetch_spec(spec, directory)?.display().to_string())
}

pub fn playground(model: Option<&Path>, depth: Option<usize>, host: &str, port: u16) -> Result<(), CliError> {
    let weights = crate::resources::needle_weights(model)?;
    let playground = playground::Playground::bind(&playground::Config { host: host.to_owned(), port, model: weights, depth })?;
    let address = playground.address().map(|a| a.to_string()).unwrap_or_default();
    println!("rustyneedle playground ready: http://{address}  ({})", playground.name());
    playground.run();
    Ok(())
}
