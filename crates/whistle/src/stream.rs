use std::time::Instant;

use crate::{Options, Whistle, WhistleError, Word, SAMPLE_RATE};

const TRIM_AFTER_SECONDS: f64 = 15.0;
const MAX_BUFFER_SECONDS: f64 = 28.0;

#[derive(Clone, Debug)]
pub struct StreamStep {
    pub text: String,
    pub words: Vec<Word>,
    pub pending: String,
    pub language: String,
    pub received: f64,
    pub pass_ms: f64,
}

pub struct Stream<'a> {
    whistle: &'a Whistle,
    options: Options,
    buffer: Vec<f32>,
    base_seconds: f64,
    previous: Vec<Word>,
    committed: usize,
    received_samples: usize,
    language: Option<String>,
}

fn normalised(word: &str) -> String {
    word.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

impl<'a> Stream<'a> {
    pub fn new(whistle: &'a Whistle, options: Options) -> Self {
        let language = options.language.clone();
        Self {
            whistle,
            options: Options { word_timestamps: true, ..options },
            buffer: Vec::new(),
            base_seconds: 0.0,
            previous: Vec::new(),
            committed: 0,
            received_samples: 0,
            language,
        }
    }

    pub fn process(&mut self, chunk: &[f32]) -> Result<StreamStep, WhistleError> {
        self.buffer.extend_from_slice(chunk);
        self.received_samples += chunk.len();
        self.pass(false)
    }

    pub fn stop(&mut self) -> Result<StreamStep, WhistleError> {
        self.pass(true)
    }

    fn pass(&mut self, finish: bool) -> Result<StreamStep, WhistleError> {
        let started = Instant::now();
        let options = Options { language: self.language.clone(), ..self.options.clone() };
        let transcript = if self.buffer.is_empty() { None } else { Some(self.whistle.transcribe(&self.buffer, &options)?) };
        let (words, language) =
            transcript.map(|t| (t.words.unwrap_or_default(), t.language)).unwrap_or((Vec::new(), self.language.clone().unwrap_or_default()));
        if !language.is_empty() {
            self.language.get_or_insert(language.clone());
        }
        let buffer_seconds = self.buffer.len() as f64 / SAMPLE_RATE as f64;
        let agreed = if finish || buffer_seconds > MAX_BUFFER_SECONDS {
            words.len()
        } else {
            self.previous.iter().zip(&words).take_while(|(before, now)| normalised(&before.word) == normalised(&now.word)).count()
        };
        let agreed = agreed.max(self.committed);
        let committed: Vec<Word> = words[self.committed..agreed]
            .iter()
            .map(|word| Word { start: word.start + self.base_seconds, end: word.end + self.base_seconds, ..word.clone() })
            .collect();
        let text = committed.iter().map(|word| word.word.as_str()).collect::<Vec<_>>().join(" ");
        let pending = words[agreed..].iter().map(|word| word.word.as_str()).collect::<Vec<_>>().join(" ");
        self.committed = agreed;
        if buffer_seconds > TRIM_AFTER_SECONDS && agreed > 0 {
            let cut_seconds = words[agreed - 1].end;
            let cut_samples = ((cut_seconds * SAMPLE_RATE as f64) as usize).min(self.buffer.len());
            self.buffer.drain(..cut_samples);
            self.base_seconds += cut_samples as f64 / SAMPLE_RATE as f64;
            self.previous = words[agreed..]
                .iter()
                .map(|word| Word { start: (word.start - cut_seconds).max(0.0), end: (word.end - cut_seconds).max(0.0), ..word.clone() })
                .collect();
            self.committed = 0;
        } else {
            self.previous = words;
        }
        Ok(StreamStep {
            text,
            words: committed,
            pending,
            language,
            received: self.received_samples as f64 / SAMPLE_RATE as f64,
            pass_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }
}
