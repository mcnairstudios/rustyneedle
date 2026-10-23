use tokenizer::{PieceKind, Tokenizer};

use crate::call::CallMachine;

const SPACE_MARK: &str = "\u{2581}";

pub struct TokenBytes {
    pieces: Vec<Option<Vec<u8>>>,
}

impl TokenBytes {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let vocabulary = tokenizer.vocabulary();
        let pieces = (0..vocabulary.len() as u32)
            .map(|id| match vocabulary.kind(id) {
                PieceKind::Normal => Some(vocabulary.piece(id).replace(SPACE_MARK, " ").into_bytes()),
                PieceKind::Byte => vocabulary.byte_value(id).map(|value| vec![value]),
                _ => None,
            })
            .collect();
        Self { pieces }
    }

    pub fn bytes(&self, id: u32) -> Option<&[u8]> {
        self.pieces.get(id as usize).and_then(|piece| piece.as_deref())
    }

    pub fn allowed(&self, machine: &CallMachine) -> Vec<bool> {
        let mut first_ok = [None::<bool>; 256];
        self.pieces
            .iter()
            .map(|piece| {
                let Some(bytes) = piece.as_deref().filter(|bytes| !bytes.is_empty()) else { return false };
                let ok = *first_ok[bytes[0] as usize].get_or_insert_with(|| machine.clone().step(bytes[0]));
                if !ok {
                    return false;
                }
                let mut trial = machine.clone();
                bytes.iter().all(|&byte| trial.step(byte))
            })
            .collect()
    }

    pub fn advance(&self, machine: &mut CallMachine, id: u32) -> bool {
        match self.bytes(id) {
            Some(bytes) => bytes.iter().all(|&byte| machine.step(byte)),
            None => false,
        }
    }
}
