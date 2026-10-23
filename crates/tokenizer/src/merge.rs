use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::vocabulary::Vocabulary;

struct Candidate {
    score: f32,
    left: usize,
    right: usize,
    left_len: usize,
    right_len: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score).then_with(|| other.left.cmp(&self.left))
    }
}

struct Symbol {
    start: usize,
    len: usize,
    previous: Option<usize>,
    next: Option<usize>,
}

pub(crate) fn merge_segment(vocabulary: &Vocabulary, text: &str, ids: &mut Vec<u32>) {
    if text.is_empty() {
        return;
    }
    let boundaries: Vec<usize> = text.char_indices().map(|(at, _)| at).chain(std::iter::once(text.len())).collect();
    let count = boundaries.len() - 1;
    let mut symbols: Vec<Symbol> = (0..count)
        .map(|index| Symbol {
            start: boundaries[index],
            len: boundaries[index + 1] - boundaries[index],
            previous: index.checked_sub(1),
            next: (index + 1 < count).then_some(index + 1),
        })
        .collect();
    let mut heap = BinaryHeap::new();
    for index in 1..count {
        push_candidate(vocabulary, text, &symbols, index - 1, index, &mut heap);
    }
    while let Some(candidate) = heap.pop() {
        let left_alive = symbols[candidate.left].len == candidate.left_len;
        let right_alive = symbols[candidate.right].len == candidate.right_len;
        if !left_alive || !right_alive || symbols[candidate.left].len == 0 || symbols[candidate.right].len == 0 {
            continue;
        }
        symbols[candidate.left].len += symbols[candidate.right].len;
        symbols[candidate.right].len = 0;
        let after = symbols[candidate.right].next;
        symbols[candidate.left].next = after;
        if let Some(after) = after {
            symbols[after].previous = Some(candidate.left);
        }
        if let Some(before) = symbols[candidate.left].previous {
            push_candidate(vocabulary, text, &symbols, before, candidate.left, &mut heap);
        }
        if let Some(after) = after {
            push_candidate(vocabulary, text, &symbols, candidate.left, after, &mut heap);
        }
    }
    let mut cursor = Some(0);
    while let Some(index) = cursor {
        let symbol = &symbols[index];
        emit_symbol(vocabulary, &text[symbol.start..symbol.start + symbol.len], ids);
        cursor = symbol.next;
    }
}

fn push_candidate(vocabulary: &Vocabulary, text: &str, symbols: &[Symbol], left: usize, right: usize, heap: &mut BinaryHeap<Candidate>) {
    let (a, b) = (&symbols[left], &symbols[right]);
    let joined = &text[a.start..b.start + b.len];
    if let Some(id) = vocabulary.piece_id(joined) {
        heap.push(Candidate { score: vocabulary.scores[id as usize], left, right, left_len: a.len, right_len: b.len });
    }
}

fn emit_symbol(vocabulary: &Vocabulary, symbol: &str, ids: &mut Vec<u32>) {
    if let Some(id) = vocabulary.piece_id(symbol) {
        ids.push(id);
    } else if vocabulary.byte_fallback {
        ids.extend(symbol.bytes().filter_map(|byte| vocabulary.byte_ids[byte as usize]));
    } else {
        ids.push(vocabulary.unknown_id);
    }
}
