use std::sync::Arc;

use generate::{Generator, RawTurn};
use model::{Model, Session};
use repair::{parse_calls, repair_calls, Call, Findings};
use serde_json::Value;

use crate::response::{Failure, Response, ResponseKind, Validation};
use crate::tools::ToolSet;

const CONFIDENCE_FLOOR: f32 = 0.1;
const CONFIDENCE_HEAD: u8 = 2;

pub trait Engine: Send {
    fn complete(&mut self, text: &str, max_new_tokens: usize) -> Response;
    fn reset(&mut self);
    fn embed(&self, text: &str) -> Option<Vec<f32>>;
}

pub struct ModelEngine {
    model: Arc<Model>,
    generator: Generator,
    tools: ToolSet,
}

impl ModelEngine {
    pub fn new(model: &Arc<Model>, tools: &Value, system: &str) -> Self {
        let tools = ToolSet::new(tools);
        let prompt_tools = tools.prompt_tools();
        let rendered = serde_json::to_string(&prompt_tools).unwrap_or_else(|_| "[]".into());
        Self { model: Arc::clone(model), generator: Generator::new(model, system, &rendered, &prompt_tools), tools }
    }

    fn assemble(&self, turn: RawTurn, query: &str, triggered: bool, tool_result: bool) -> Response {
        let confidence = turn.head_probability.map(|head| head.min(turn.call_probability));
        let kind = ResponseKind::Call;
        if turn.overflow {
            let mut response = Response::empty(kind);
            response.failure = Some(Failure {
                message: "prompt does not fit in the context window".into(),
                code: "context_overflow".into(),
                reason: "runtime_failure".into(),
            });
            return response;
        }
        let completed = turn.calls_json.is_some() && turn.tokens.contains(&self.generator.special().tool_call_end);
        if !turn.finished || (turn.calls_json.is_some() && !completed) {
            let mut response = Response::empty(kind);
            response.reasoning = turn.reasoning;
            response.confidence = confidence;
            response.failure =
                Some(Failure { message: "tool call truncated: token budget exhausted".into(), code: "truncated".into(), reason: "runtime_failure".into() });
            return response;
        }
        let calls = turn.calls_json.as_deref().map(parse_calls).unwrap_or_default();
        let (calls, findings) = repair_calls(calls, &self.tools.prompt_tools(), query);
        let withheld = Self::withhold(&calls, &findings, confidence, triggered);
        let validation = (!calls.is_empty()).then(|| Validation { ungrounded: self.restore_paths(&findings.ungrounded), negation: Some(findings.negation) });
        let calls: Vec<Call> = calls.into_iter().map(|call| self.restore(call)).collect();
        let (function_calls, suppressed_calls) = if withheld { (Vec::new(), calls) } else { (calls, Vec::new()) };
        let kind = if tool_result && function_calls.is_empty() && suppressed_calls.is_empty() { ResponseKind::Respond } else { kind };
        let mut response = Response::empty(kind);
        response.function_calls = function_calls;
        response.suppressed_calls = suppressed_calls;
        response.reasoning = turn.reasoning;
        response.confidence = confidence;
        response.validation = validation;
        response
    }

    fn restore(&self, call: Call) -> Call {
        Call { arguments: self.tools.original_arguments(&call.name, &call.arguments), name: self.tools.original_name(&call.name) }
    }

    fn restore_paths(&self, paths: &[String]) -> Vec<String> {
        paths
            .iter()
            .map(|path| match path.split_once('.') {
                Some((tool, rest)) => format!("{}.{}", self.tools.original_name(tool), rest),
                None => path.clone(),
            })
            .collect()
    }

    fn withhold(calls: &[Call], findings: &Findings, confidence: Option<f32>, triggered: bool) -> bool {
        if calls.is_empty() {
            return false;
        }
        let below_floor = confidence.is_some_and(|value| value < CONFIDENCE_FLOOR);
        findings.negation || (!triggered && (findings.withhold || below_floor))
    }
}

fn looks_like_tool_result(text: &str) -> bool {
    let trimmed = text.trim();
    (trimmed.starts_with('{') || trimmed.starts_with('[')) && serde_json::from_str::<Value>(trimmed).is_ok()
}

impl Engine for ModelEngine {
    fn complete(&mut self, text: &str, max_new_tokens: usize) -> Response {
        if self.generator.turn_count() > 0 && looks_like_tool_result(text) {
            let turn = self.generator.tool_result_turn(text.trim(), max_new_tokens);
            return self.assemble(turn, text, false, true);
        }
        let required = self.tools.triggered(text);
        let turn = self.generator.user_turn(text, max_new_tokens, required.as_deref());
        self.assemble(turn, text, required.is_some(), false)
    }

    fn reset(&mut self) {
        self.generator.reset();
    }

    fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let tokenizer = self.model.tokenizer();
        let mut session = Session::new(&self.model);
        session.push(self.generator.special().bos);
        session.feed(&tokenizer.encode(text));
        let mut readout = session.pooled_readout(CONFIDENCE_HEAD)?;
        let norm = readout.iter().map(|value| value * value).sum::<f32>().sqrt();
        readout.iter_mut().for_each(|value| *value /= norm);
        Some(readout)
    }
}
