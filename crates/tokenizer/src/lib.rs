mod merge;
mod vocabulary;

pub use vocabulary::{PieceKind, Vocabulary, VocabularyError};

pub struct Tokenizer {
    vocabulary: Vocabulary,
}

impl Tokenizer {
    pub fn from_blob(blob: &[u8]) -> Result<Self, VocabularyError> {
        Ok(Self { vocabulary: Vocabulary::parse(blob)? })
    }

    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let escaped = self.escape_spaces(text);
        let prepared = if self.vocabulary.add_dummy_prefix { format!("{}{escaped}", Vocabulary::meta_space()) } else { escaped };
        self.encode_prepared(&prepared)
    }

    pub fn encode_continuation(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        self.encode_prepared(&self.escape_spaces(text))
    }

    fn escape_spaces(&self, text: &str) -> String {
        text.replace(' ', &Vocabulary::meta_space().to_string())
    }

    fn encode_prepared(&self, prepared: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(prepared.len() / 3 + 1);
        let mut segment_start = 0;
        let mut at = 0;
        while at < prepared.len() {
            let rest = &prepared[at..];
            if let Some((marker, id)) = self.vocabulary.markers.iter().find(|(marker, _)| rest.starts_with(marker.as_str())) {
                merge::merge_segment(&self.vocabulary, &prepared[segment_start..at], &mut ids);
                ids.push(*id);
                at += marker.len();
                segment_start = at;
            } else {
                at += rest.chars().next().map_or(1, char::len_utf8);
            }
        }
        merge::merge_segment(&self.vocabulary, &prepared[segment_start..], &mut ids);
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            let index = id as usize;
            match self.vocabulary.kinds.get(index) {
                Some(PieceKind::Byte) => {
                    if let Some(value) = self.vocabulary.pieces.get(index).and_then(|piece| piece_byte(piece)) {
                        bytes.push(value);
                    }
                }
                Some(PieceKind::Control) | Some(PieceKind::Unknown) | None => {}
                Some(_) => bytes.extend_from_slice(self.vocabulary.pieces[index].as_bytes()),
            }
        }
        let text = String::from_utf8_lossy(&bytes).replace(Vocabulary::meta_space(), " ");
        match text.strip_prefix(' ') {
            Some(trimmed) if self.vocabulary.add_dummy_prefix => trimmed.to_owned(),
            _ => text,
        }
    }
}

fn piece_byte(piece: &str) -> Option<u8> {
    let digits = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(digits, 16).ok()
}
