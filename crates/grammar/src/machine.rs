use std::sync::Arc;

use crate::schema::{Node, Property};

const MAX_NUMBER_CHARS: usize = 20;

#[derive(Clone)]
pub enum Machine {
    Text(TextState),
    Integer(NumberState),
    Number(NumberState),
    Literals(LiteralState),
    Array(Box<ArrayState>),
    Object(Box<ObjectState>),
    Choice(Vec<Machine>),
}

impl Machine {
    pub fn new(node: &Node) -> Machine {
        match node {
            Node::Text { min_chars, max_chars } => Machine::Text(TextState::new(*min_chars, *max_chars)),
            Node::Integer { min, max } => Machine::Integer(NumberState::new(min.map(|v| v as f64), max.map(|v| v as f64), false)),
            Node::Number { min, max } => Machine::Number(NumberState::new(*min, *max, true)),
            Node::Boolean => Machine::Literals(LiteralState::new(Arc::from(vec!["true".to_owned(), "false".to_owned()]))),
            Node::Null => Machine::Literals(LiteralState::new(Arc::from(vec!["null".to_owned()]))),
            Node::Literals(options) => Machine::Literals(LiteralState::new(Arc::clone(options))),
            Node::Array { items, min_items, max_items } => Machine::Array(Box::new(ArrayState::new(Arc::clone(items), *min_items, *max_items))),
            Node::Object { properties } => Machine::Object(Box::new(ObjectState::new(Arc::clone(properties)))),
            Node::Choice(nodes) => Machine::Choice(nodes.iter().map(Machine::new).collect()),
        }
    }

    pub fn step(&mut self, byte: u8) -> bool {
        match self {
            Machine::Text(state) => state.step(byte),
            Machine::Integer(state) | Machine::Number(state) => state.step(byte),
            Machine::Literals(state) => state.step(byte),
            Machine::Array(state) => state.step(byte),
            Machine::Object(state) => state.step(byte),
            Machine::Choice(alternatives) => {
                let mut survivors = Vec::new();
                for alternative in alternatives.iter() {
                    let mut candidate = alternative.clone();
                    if candidate.step(byte) {
                        survivors.push(candidate);
                    }
                }
                if survivors.is_empty() {
                    return false;
                }
                *alternatives = survivors;
                true
            }
        }
    }

    pub fn can_finish(&self) -> bool {
        match self {
            Machine::Text(state) => state.closed,
            Machine::Integer(state) | Machine::Number(state) => state.can_finish(),
            Machine::Literals(state) => state.can_finish(),
            Machine::Array(state) => state.done,
            Machine::Object(state) => state.done,
            Machine::Choice(alternatives) => alternatives.iter().any(Machine::can_finish),
        }
    }
}

#[derive(Clone)]
pub struct TextState {
    min_chars: usize,
    max_chars: Option<usize>,
    count: usize,
    opened: bool,
    pub closed: bool,
    escape: Escape,
    continuation: u8,
}

#[derive(Clone, Copy, PartialEq)]
enum Escape {
    None,
    Slash,
    Unicode(u8),
}

impl TextState {
    fn new(min_chars: usize, max_chars: Option<usize>) -> Self {
        Self { min_chars, max_chars, count: 0, opened: false, closed: false, escape: Escape::None, continuation: 0 }
    }

    fn has_room(&self) -> bool {
        self.max_chars.is_none_or(|max| self.count < max)
    }

    fn step(&mut self, byte: u8) -> bool {
        if self.closed {
            return false;
        }
        if !self.opened {
            self.opened = byte == b'"';
            return self.opened;
        }
        if self.continuation > 0 {
            if byte & 0xC0 != 0x80 {
                return false;
            }
            self.continuation -= 1;
            return true;
        }
        match self.escape {
            Escape::Slash => {
                return match byte {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                        self.escape = Escape::None;
                        self.count += 1;
                        true
                    }
                    b'u' => {
                        self.escape = Escape::Unicode(4);
                        true
                    }
                    _ => false,
                };
            }
            Escape::Unicode(left) => {
                if !byte.is_ascii_hexdigit() {
                    return false;
                }
                if left == 1 {
                    self.escape = Escape::None;
                    self.count += 1;
                } else {
                    self.escape = Escape::Unicode(left - 1);
                }
                return true;
            }
            Escape::None => {}
        }
        match byte {
            b'"' => {
                if self.count < self.min_chars {
                    return false;
                }
                self.closed = true;
                true
            }
            b'\\' => {
                if !self.has_room() {
                    return false;
                }
                self.escape = Escape::Slash;
                true
            }
            0x00..=0x1F => false,
            0x20..=0x7F => {
                if !self.has_room() {
                    return false;
                }
                self.count += 1;
                true
            }
            0xC2..=0xDF => self.begin_multibyte(1),
            0xE0..=0xEF => self.begin_multibyte(2),
            0xF0..=0xF4 => self.begin_multibyte(3),
            _ => false,
        }
    }

    fn begin_multibyte(&mut self, continuation: u8) -> bool {
        if !self.has_room() {
            return false;
        }
        self.count += 1;
        self.continuation = continuation;
        true
    }
}

#[derive(Clone)]
pub struct NumberState {
    text: Vec<u8>,
    min: Option<f64>,
    max: Option<f64>,
    fractional: bool,
}

impl NumberState {
    fn new(min: Option<f64>, max: Option<f64>, fractional: bool) -> Self {
        Self { text: Vec::new(), min, max, fractional }
    }

    fn step(&mut self, byte: u8) -> bool {
        if self.text.len() >= MAX_NUMBER_CHARS {
            return false;
        }
        self.text.push(byte);
        if self.prefix_is_viable() {
            true
        } else {
            self.text.pop();
            false
        }
    }

    fn split(&self) -> (bool, &[u8], Option<&[u8]>) {
        let negative = self.text.first() == Some(&b'-');
        let body = if negative { &self.text[1..] } else { &self.text[..] };
        match body.iter().position(|&b| b == b'.') {
            Some(dot) => (negative, &body[..dot], Some(&body[dot + 1..])),
            None => (negative, body, None),
        }
    }

    fn prefix_is_viable(&self) -> bool {
        let (negative, integer, fraction) = self.split();
        let valid_chars = self.text.iter().enumerate().all(|(index, &b)| b.is_ascii_digit() || (b == b'-' && index == 0) || (b == b'.' && self.fractional));
        if !valid_chars || self.text.iter().filter(|&&b| b == b'.').count() > 1 {
            return false;
        }
        if integer.len() > 1 && integer[0] == b'0' {
            return false;
        }
        if let Some(fraction) = fraction {
            if integer.is_empty() || !fraction.iter().all(u8::is_ascii_digit) {
                return false;
            }
        }
        if negative && self.min.is_some_and(|min| min >= 0.0) {
            return false;
        }
        if integer.is_empty() {
            return true;
        }
        self.integer_range_viable(negative, integer, fraction.is_some())
    }

    fn integer_range_viable(&self, negative: bool, digits: &[u8], has_fraction: bool) -> bool {
        let value: f64 = std::str::from_utf8(digits).unwrap_or("0").parse().unwrap_or(0.0);
        let lower = self.min.unwrap_or(f64::NEG_INFINITY);
        let upper = self.max.unwrap_or(f64::INFINITY);
        if has_fraction {
            let (low, high) = if negative { (-value - 1.0, -value) } else { (value, value + 1.0) };
            return high >= lower && low <= upper;
        }
        let remaining = MAX_NUMBER_CHARS.saturating_sub(self.text.len());
        for extra in 0..=remaining.min(18) {
            let scale = 10f64.powi(extra as i32);
            let low = value * scale;
            let high = low + scale - 1.0;
            let (span_low, span_high) = if negative { (-high, -low) } else { (low, high) };
            if span_high >= lower && span_low <= upper {
                return true;
            }
            if self.fractional && extra == 0 {
                let (frac_low, frac_high) = if negative { (-value - 1.0, -value) } else { (value, value + 1.0) };
                if frac_high >= lower && frac_low <= upper {
                    return true;
                }
            }
        }
        false
    }

    fn can_finish(&self) -> bool {
        let (negative, integer, fraction) = self.split();
        if integer.is_empty() || fraction.is_some_and(<[u8]>::is_empty) {
            return false;
        }
        let text = std::str::from_utf8(&self.text).unwrap_or("0");
        let value: f64 = text.parse().unwrap_or(f64::NAN);
        let _ = negative;
        self.min.is_none_or(|min| value >= min) && self.max.is_none_or(|max| value <= max)
    }
}

#[derive(Clone)]
pub struct LiteralState {
    options: Arc<[String]>,
    alive: Vec<usize>,
    position: usize,
}

impl LiteralState {
    pub fn new(options: Arc<[String]>) -> Self {
        let alive = (0..options.len()).collect();
        Self { options, alive, position: 0 }
    }

    pub fn restricted(options: Arc<[String]>, allowed: Vec<usize>) -> Self {
        Self { options, alive: allowed, position: 0 }
    }

    pub fn step(&mut self, byte: u8) -> bool {
        let position = self.position;
        let survivors: Vec<usize> = self.alive.iter().copied().filter(|&index| self.options[index].as_bytes().get(position) == Some(&byte)).collect();
        if survivors.is_empty() {
            return false;
        }
        self.alive = survivors;
        self.position += 1;
        true
    }

    pub fn can_finish(&self) -> bool {
        self.alive.iter().any(|&index| self.options[index].len() == self.position)
    }

    pub fn chosen(&self) -> Option<usize> {
        self.alive.iter().copied().find(|&index| self.options[index].len() == self.position)
    }
}

#[derive(Clone)]
enum ArrayStage {
    Start,
    AfterOpen,
    Item(Machine),
    AfterItem,
    AfterComma,
}

#[derive(Clone)]
pub struct ArrayState {
    items: Arc<Node>,
    min_items: usize,
    max_items: usize,
    count: usize,
    stage: ArrayStage,
    done: bool,
}

impl ArrayState {
    fn new(items: Arc<Node>, min_items: usize, max_items: usize) -> Self {
        Self { items, min_items, max_items, count: 0, stage: ArrayStage::Start, done: false }
    }

    fn begin_item(&mut self, byte: u8) -> bool {
        if self.count >= self.max_items {
            return false;
        }
        let mut machine = Machine::new(&self.items);
        if !machine.step(byte) {
            return false;
        }
        self.count += 1;
        self.stage = ArrayStage::Item(machine);
        true
    }

    fn step(&mut self, byte: u8) -> bool {
        if self.done {
            return false;
        }
        match &mut self.stage {
            ArrayStage::Start => {
                if byte == b'[' {
                    self.stage = ArrayStage::AfterOpen;
                    true
                } else {
                    false
                }
            }
            ArrayStage::AfterOpen => {
                if byte == b']' && self.min_items == 0 {
                    self.done = true;
                    true
                } else {
                    self.begin_item(byte)
                }
            }
            ArrayStage::Item(machine) => {
                if machine.step(byte) {
                    return true;
                }
                if !machine.can_finish() {
                    return false;
                }
                self.stage = ArrayStage::AfterItem;
                self.step(byte)
            }
            ArrayStage::AfterItem => match byte {
                b',' if self.count < self.max_items => {
                    self.stage = ArrayStage::AfterComma;
                    true
                }
                b']' if self.count >= self.min_items => {
                    self.done = true;
                    true
                }
                _ => false,
            },
            ArrayStage::AfterComma => self.begin_item(byte),
        }
    }
}

#[derive(Clone)]
enum ObjectStage {
    Start,
    BeforeKey { first: bool },
    Key { machine: LiteralState },
    Colon { property: usize },
    Value { property: usize, machine: Option<Machine> },
    AfterValue,
}

#[derive(Clone)]
pub struct ObjectState {
    properties: Arc<[Property]>,
    keys: Arc<[String]>,
    next: usize,
    stage: ObjectStage,
    done: bool,
}

impl ObjectState {
    fn new(properties: Arc<[Property]>) -> Self {
        let keys: Vec<String> = properties.iter().map(|property| format!("\"{}\"", property.name)).collect();
        Self { properties, keys: Arc::from(keys), next: 0, stage: ObjectStage::Start, done: false }
    }

    fn candidate_properties(&self) -> Vec<usize> {
        let mut candidates = Vec::new();
        for index in self.next..self.properties.len() {
            candidates.push(index);
            if self.properties[index].required {
                break;
            }
        }
        candidates
    }

    fn can_close(&self) -> bool {
        self.properties[self.next..].iter().all(|property| !property.required)
    }

    fn begin_key(&mut self, byte: u8) -> bool {
        let candidates = self.candidate_properties();
        if candidates.is_empty() {
            return false;
        }
        let mut machine = LiteralState::restricted(Arc::clone(&self.keys), candidates);
        if !machine.step(byte) {
            return false;
        }
        self.stage = ObjectStage::Key { machine };
        true
    }

    fn step(&mut self, byte: u8) -> bool {
        if self.done {
            return false;
        }
        match &mut self.stage {
            ObjectStage::Start => {
                if byte == b'{' {
                    self.stage = ObjectStage::BeforeKey { first: true };
                    true
                } else {
                    false
                }
            }
            ObjectStage::BeforeKey { first } => {
                let first = *first;
                if byte == b'}' && first && self.can_close() {
                    self.done = true;
                    return true;
                }
                self.begin_key(byte)
            }
            ObjectStage::Key { machine } => {
                if machine.step(byte) {
                    return true;
                }
                match machine.chosen() {
                    Some(property) => {
                        self.stage = ObjectStage::Colon { property };
                        self.step(byte)
                    }
                    None => false,
                }
            }
            ObjectStage::Colon { property } => {
                if byte == b':' {
                    let property = *property;
                    self.stage = ObjectStage::Value { property, machine: None };
                    true
                } else {
                    false
                }
            }
            ObjectStage::Value { property, machine } => {
                let property = *property;
                match machine {
                    None => {
                        let mut fresh = Machine::new(&self.properties[property].node);
                        if fresh.step(byte) {
                            self.stage = ObjectStage::Value { property, machine: Some(fresh) };
                            true
                        } else {
                            false
                        }
                    }
                    Some(active) => {
                        if active.step(byte) {
                            return true;
                        }
                        if !active.can_finish() {
                            return false;
                        }
                        self.next = property + 1;
                        self.stage = ObjectStage::AfterValue;
                        self.step(byte)
                    }
                }
            }
            ObjectStage::AfterValue => match byte {
                b',' if self.next < self.properties.len() => {
                    self.stage = ObjectStage::BeforeKey { first: false };
                    true
                }
                b'}' if self.can_close() => {
                    self.done = true;
                    true
                }
                _ => false,
            },
        }
    }
}
