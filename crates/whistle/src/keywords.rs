use std::collections::HashMap;

use tokenizer::Tokenizer;

const MARGIN: f32 = 6.0;
const START_BOOST: f32 = 5.0;
const CONTINUE_BOOST: f32 = 10.0;

#[derive(Default)]
struct Node {
    children: HashMap<u32, usize>,
}

pub struct KeywordTrie {
    nodes: Vec<Node>,
    start_boost: f32,
    continue_boost: f32,
}

#[derive(Clone)]
pub struct KeywordState {
    active: Vec<usize>,
}

fn casings(keyword: &str) -> Vec<String> {
    let lower = keyword.to_lowercase();
    let mut capitalised = lower.clone();
    if let Some(first) = capitalised.get_mut(0..1) {
        first.make_ascii_uppercase();
    }
    let mut all = vec![keyword.to_owned(), lower, capitalised];
    all.dedup();
    all
}

const MID_SENTENCE: usize = 0;
const SENTENCE_START: usize = 1;

impl KeywordTrie {
    fn insert(&mut self, root: usize, tokens: &[u32]) {
        let mut node = root;
        for &token in tokens {
            node = match self.nodes[node].children.get(&token) {
                Some(&next) => next,
                None => {
                    self.nodes.push(Node::default());
                    let next = self.nodes.len() - 1;
                    self.nodes[node].children.insert(token, next);
                    next
                }
            };
        }
    }

    pub fn new(keywords: &[String], tokenizer: &Tokenizer) -> Option<Self> {
        let mut trie = Self { nodes: vec![Node::default(), Node::default()], start_boost: START_BOOST, continue_boost: CONTINUE_BOOST };
        let mut any = false;
        for keyword in keywords.iter().map(|k| k.trim()).filter(|k| !k.is_empty()) {
            for variant in casings(keyword) {
                trie.insert(SENTENCE_START, &tokenizer.encode_continuation(&variant));
                let spaced = tokenizer.encode_continuation(&format!(" {variant}"));
                trie.insert(SENTENCE_START, &spaced);
                trie.insert(MID_SENTENCE, &spaced);
                any = true;
            }
        }
        any.then_some(trie)
    }

    pub fn start(&self) -> KeywordState {
        KeywordState { active: vec![SENTENCE_START, MID_SENTENCE] }
    }

    pub fn bias(&self, state: &KeywordState, logits: &mut [f32], top: f32) {
        let floor = top - MARGIN;
        for &node in &state.active {
            let boost = if node <= SENTENCE_START { self.start_boost } else { self.continue_boost };
            for &token in self.nodes[node].children.keys() {
                if logits[token as usize] >= floor {
                    logits[token as usize] += boost;
                }
            }
        }
    }

    pub fn advance(&self, state: &mut KeywordState, token: u32) {
        let mut next = vec![MID_SENTENCE];
        for &node in &state.active {
            if let Some(&child) = self.nodes[node].children.get(&token) {
                if !next.contains(&child) {
                    next.push(child);
                }
            }
        }
        state.active = next;
    }
}
