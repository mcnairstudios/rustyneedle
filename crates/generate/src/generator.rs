use std::sync::Arc;

use grammar::{CallGrammar, TokenBytes};
use model::{Model, Session};

use crate::prompt;
use crate::special::SpecialTokens;

const CONTEXT_RESERVE: usize = 16;

pub struct RawTurn {
    pub reasoning: String,
    pub call_probability: f32,
    pub minimum_allowed_mass: f32,
    pub head_probability: Option<f32>,
    pub calls_json: Option<String>,
    pub tokens: Vec<u32>,
    pub finished: bool,
    pub overflow: bool,
}

pub struct Generator {
    model: Arc<Model>,
    special: SpecialTokens,
    grammar: CallGrammar,
    token_bytes: TokenBytes,
    base: Session,
    session: Session,
    turns: usize,
}

impl Generator {
    pub fn new(model: &Arc<Model>, system: &str, tools_json: &str, grammar_tools: &serde_json::Value) -> Self {
        let tokenizer = model.tokenizer();
        let special = SpecialTokens::resolve(tokenizer);
        let mut base = Session::new(model);
        let model = Arc::clone(model);
        base.push(special.bos);
        base.feed(&tokenizer.encode(&prompt::prefix(system, tools_json)));
        let grammar = CallGrammar::from_tools_json(grammar_tools);
        Self { model, special, grammar, token_bytes: TokenBytes::new(tokenizer), session: base.clone(), base, turns: 0 }
    }

    pub fn reset(&mut self) {
        self.session = self.base.clone();
        self.turns = 0;
    }

    pub fn special(&self) -> &SpecialTokens {
        &self.special
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn user_turn(&mut self, text: &str, max_new_tokens: usize, required: Option<&[usize]>) -> RawTurn {
        let rendered = if self.turns == 0 { prompt::first_user_turn(text) } else { prompt::later_user_turn(text) };
        self.run_turn(&rendered, max_new_tokens, required)
    }

    pub fn tool_result_turn(&mut self, result_json: &str, max_new_tokens: usize) -> RawTurn {
        self.run_turn(&prompt::tool_result_turn(result_json), max_new_tokens, None)
    }

    pub fn turn_count(&self) -> usize {
        self.turns
    }

    fn run_turn(&mut self, rendered: &str, max_new_tokens: usize, required: Option<&[usize]>) -> RawTurn {
        let grammar = match required {
            Some(permitted) => self.grammar.clone().require_call(Some(permitted.to_vec())),
            None => self.grammar.clone(),
        };
        let tokenizer = self.model.tokenizer();
        let incoming = tokenizer.encode_continuation(rendered);
        let window = self.model.config().max_seq_len;
        if self.session.len() + incoming.len() + CONTEXT_RESERVE > window {
            return RawTurn {
                reasoning: String::new(),
                call_probability: 0.0,
                minimum_allowed_mass: 1.0,
                head_probability: None,
                calls_json: None,
                tokens: Vec::new(),
                finished: false,
                overflow: true,
            };
        }
        let mut prompt_tokens = incoming;
        prompt_tokens.push(self.special.think_start);
        self.session.feed(&prompt_tokens);
        self.turns += 1;
        let mut generated = vec![self.special.think_start];
        let mut finished = false;
        let mut machine: Option<grammar::CallMachine> = None;
        let mut call_log_probability = 0.0f32;
        let mut minimum_allowed_mass = 1.0f32;
        while generated.len() < max_new_tokens && self.session.len() + 1 < window {
            let logits = self.session.logits();
            let mut constraint: Option<Vec<bool>> = None;
            let next = match &machine {
                Some(active) if active.is_complete() => self.special.tool_call_end,
                Some(active) => {
                    let allowed = self.token_bytes.allowed(active);
                    let chosen = argmax_allowed(&logits, &allowed).unwrap_or_else(|| argmax(&logits)) as u32;
                    constraint = Some(allowed);
                    chosen
                }
                None => argmax(&logits) as u32,
            };
            if let Some(allowed) = constraint.filter(|_| next != self.special.tool_call_end) {
                let statistics = allowed_statistics(&logits, next as usize, &allowed);
                call_log_probability += statistics.probability.max(1e-30).ln();
                minimum_allowed_mass = minimum_allowed_mass.min(statistics.mass);
            }
            generated.push(next);
            self.session.push(next);
            if next == self.special.tool_call_start && machine.is_none() {
                machine = Some(grammar.start());
            } else if next == self.special.tool_call_end {
                machine = None;
            } else if let Some(active) = machine.as_mut() {
                self.token_bytes.advance(active, next);
            }
            if next == self.special.im_end || next == self.special.eos {
                finished = true;
                break;
            }
        }
        let text = tokenizer.decode(&generated);
        let (reasoning, calls_json) = split_output(&text);
        let head_probability = self.session.head_output(2).map(|output| 1.0 / (1.0 + (-output[0]).exp()));
        RawTurn {
            reasoning,
            call_probability: call_log_probability.exp(),
            minimum_allowed_mass,
            head_probability,
            calls_json,
            tokens: generated,
            finished,
            overflow: false,
        }
    }
}

fn argmax_allowed(values: &[f32], allowed: &[bool]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (index, value) in values.iter().enumerate() {
        if allowed[index] && best.is_none_or(|current| *value > values[current]) {
            best = Some(index);
        }
    }
    best
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

fn split_output(text: &str) -> (String, Option<String>) {
    let reasoning = text.split_once("<think>").and_then(|(_, rest)| rest.split_once("</think>")).map(|(inner, _)| inner.trim().to_owned()).unwrap_or_default();
    let calls = text.split_once("<tool_call>").map(|(_, rest)| rest.split_once("</tool_call>").map_or(rest, |(inner, _)| inner).trim().to_owned());
    (reasoning, calls)
}

struct AllowedStatistics {
    probability: f32,
    mass: f32,
}

fn peak_of(values: impl Iterator<Item = f32>) -> f32 {
    values.fold(f32::NEG_INFINITY, f32::max)
}

fn allowed_statistics(logits: &[f32], token: usize, allowed: &[bool]) -> AllowedStatistics {
    let peak_all = peak_of(logits.iter().copied());
    let peak_allowed = peak_of(logits.iter().zip(allowed).map(|(value, ok)| if *ok { *value } else { f32::NEG_INFINITY }));
    let mut total_all = 0.0f32;
    let mut total_allowed = 0.0f32;
    for (value, ok) in logits.iter().zip(allowed) {
        total_all += kernels::fast_exp(value - peak_all);
        total_allowed += kernels::fast_exp(value - peak_allowed) * (*ok as u8 as f32);
    }
    let probability = kernels::fast_exp(logits[token] - peak_allowed) / total_allowed;
    let mass = total_allowed * kernels::fast_exp(peak_allowed - peak_all) / total_all;
    AllowedStatistics { probability, mass }
}
