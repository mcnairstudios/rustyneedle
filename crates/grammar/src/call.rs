use std::sync::Arc;

use serde_json::Value;

use crate::machine::{LiteralState, Machine};
use crate::schema::{compile, Node};

const NAME_PREFIX: &[u8] = b"{\"name\":\"";
const ARGUMENTS_PREFIX: &[u8] = b",\"arguments\":";

#[derive(Clone)]
pub struct ToolSpec {
    pub name: String,
    pub arguments: Node,
}

pub fn tool_specs(tools: &Value) -> Vec<ToolSpec> {
    tools
        .as_array()
        .map(|list| {
            list.iter()
                .filter_map(|tool| {
                    let tool = tool.get("function").unwrap_or(tool);
                    let name = tool.get("name")?.as_str()?.to_owned();
                    let parameters = tool.get("parameters").cloned().unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
                    Some(ToolSpec { name, arguments: compile(&parameters, &parameters) })
                })
                .collect()
        })
        .unwrap_or_default()
}

#[derive(Clone)]
pub struct CallGrammar {
    tools: Arc<[ToolSpec]>,
    names: Arc<[String]>,
    allow_empty: bool,
    permitted: Option<Vec<usize>>,
}

impl CallGrammar {
    pub fn new(tools: Vec<ToolSpec>) -> Self {
        let names: Vec<String> = tools.iter().map(|tool| format!("{}\"", tool.name)).collect();
        Self { tools: Arc::from(tools), names: Arc::from(names), allow_empty: true, permitted: None }
    }

    pub fn from_tools_json(tools: &Value) -> Self {
        Self::new(tool_specs(tools))
    }

    pub fn require_call(mut self, permitted: Option<Vec<usize>>) -> Self {
        self.allow_empty = false;
        self.permitted = permitted;
        self
    }

    pub fn start(&self) -> CallMachine {
        CallMachine { grammar: self.clone(), stage: CallStage::Open, completed: 0 }
    }
}

#[derive(Clone)]
enum CallStage {
    Open,
    AfterOpen,
    Prefix { position: usize },
    Name(LiteralState),
    Middle { tool: usize, position: usize },
    Arguments { machine: Machine },
    Close,
    AfterCall,
    Done,
}

#[derive(Clone)]
pub struct CallMachine {
    grammar: CallGrammar,
    stage: CallStage,
    completed: usize,
}

impl CallMachine {
    pub fn is_complete(&self) -> bool {
        matches!(self.stage, CallStage::Done)
    }

    fn name_literal(&self) -> LiteralState {
        let allowed = self.grammar.permitted.clone().unwrap_or_else(|| (0..self.grammar.tools.len()).collect());
        LiteralState::restricted(Arc::clone(&self.grammar.names), allowed)
    }

    pub fn step(&mut self, byte: u8) -> bool {
        match &mut self.stage {
            CallStage::Open => {
                if byte == b'[' {
                    self.stage = CallStage::AfterOpen;
                    true
                } else {
                    false
                }
            }
            CallStage::AfterOpen => {
                if byte == b']' && self.grammar.allow_empty && self.completed == 0 {
                    self.stage = CallStage::Done;
                    true
                } else if byte == NAME_PREFIX[0] {
                    self.stage = CallStage::Prefix { position: 1 };
                    true
                } else {
                    false
                }
            }
            CallStage::Prefix { position } => {
                if NAME_PREFIX.get(*position) != Some(&byte) {
                    return false;
                }
                *position += 1;
                if *position == NAME_PREFIX.len() {
                    self.stage = CallStage::Name(self.name_literal());
                }
                true
            }
            CallStage::Name(literal) => {
                if !literal.step(byte) {
                    return false;
                }
                if let Some(tool) = literal.chosen() {
                    self.stage = CallStage::Middle { tool, position: 0 };
                }
                true
            }
            CallStage::Middle { tool, position } => {
                if ARGUMENTS_PREFIX.get(*position) != Some(&byte) {
                    return false;
                }
                *position += 1;
                if *position == ARGUMENTS_PREFIX.len() {
                    let tool = *tool;
                    self.stage = CallStage::Arguments { machine: Machine::new(&self.grammar.tools[tool].arguments) };
                }
                true
            }
            CallStage::Arguments { machine, .. } => {
                if machine.step(byte) {
                    if machine.can_finish() {
                        self.stage = CallStage::Close;
                    }
                    return true;
                }
                false
            }
            CallStage::Close => {
                if byte == b'}' {
                    self.completed += 1;
                    self.stage = CallStage::AfterCall;
                    true
                } else {
                    false
                }
            }
            CallStage::AfterCall => match byte {
                b',' => {
                    self.stage = CallStage::Prefix { position: 0 };
                    true
                }
                b']' => {
                    self.stage = CallStage::Done;
                    true
                }
                _ => false,
            },
            CallStage::Done => false,
        }
    }
}
