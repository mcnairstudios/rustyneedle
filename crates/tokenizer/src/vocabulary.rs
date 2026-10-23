use std::collections::HashMap;
use std::fmt;

const HEADER_BYTES: usize = 24;
const PIECE_HEADER_BYTES: usize = 7;
const META_SPACE: char = '\u{2581}';

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte,
}

#[derive(Debug)]
pub struct VocabularyError(pub &'static str);

impl fmt::Display for VocabularyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tokenizer blob is malformed: {}", self.0)
    }
}

impl std::error::Error for VocabularyError {}

pub struct Vocabulary {
    pub(crate) pieces: Vec<String>,
    pub(crate) scores: Vec<f32>,
    pub(crate) kinds: Vec<PieceKind>,
    pub(crate) ids: HashMap<String, u32>,
    pub(crate) byte_ids: [Option<u32>; 256],
    pub(crate) markers: Vec<(String, u32)>,
    pub(crate) pad_id: u32,
    pub(crate) eos_id: u32,
    pub(crate) bos_id: u32,
    pub(crate) unknown_id: u32,
    pub(crate) add_dummy_prefix: bool,
    pub(crate) byte_fallback: bool,
}

fn read_u32(blob: &[u8], at: usize) -> Result<u32, VocabularyError> {
    let bytes = blob.get(at..at + 4).ok_or(VocabularyError("truncated header"))?;
    Ok(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
}

fn kind_from_code(code: u8) -> Result<PieceKind, VocabularyError> {
    match code {
        0 => Ok(PieceKind::Normal),
        1 => Ok(PieceKind::Unknown),
        2 => Ok(PieceKind::Control),
        3 => Ok(PieceKind::UserDefined),
        4 => Ok(PieceKind::Byte),
        _ => Err(VocabularyError("unknown piece type")),
    }
}

fn byte_value(piece: &str) -> Option<u8> {
    let digits = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(digits, 16).ok()
}

impl Vocabulary {
    pub fn parse(blob: &[u8]) -> Result<Self, VocabularyError> {
        let count = read_u32(blob, 0)? as usize;
        let pad_id = read_u32(blob, 4)?;
        let eos_id = read_u32(blob, 8)?;
        let bos_id = read_u32(blob, 12)?;
        let unknown_id = read_u32(blob, 16)?;
        let add_dummy_prefix = *blob.get(20).ok_or(VocabularyError("truncated header"))? != 0;
        let byte_fallback = *blob.get(21).ok_or(VocabularyError("truncated header"))? != 0;
        let mut pieces = Vec::with_capacity(count);
        let mut scores = Vec::with_capacity(count);
        let mut kinds = Vec::with_capacity(count);
        let mut at = HEADER_BYTES;
        for _ in 0..count {
            let head = blob.get(at..at + PIECE_HEADER_BYTES).ok_or(VocabularyError("truncated piece"))?;
            scores.push(f32::from_le_bytes(head[0..4].try_into().expect("four bytes")));
            kinds.push(kind_from_code(head[4])?);
            let length = u16::from_le_bytes([head[5], head[6]]) as usize;
            let surface = blob.get(at + PIECE_HEADER_BYTES..at + PIECE_HEADER_BYTES + length).ok_or(VocabularyError("truncated piece text"))?;
            pieces.push(String::from_utf8_lossy(surface).into_owned());
            at += PIECE_HEADER_BYTES + length;
        }
        let ids: HashMap<String, u32> = pieces.iter().enumerate().map(|(id, piece)| (piece.clone(), id as u32)).collect();
        let mut byte_ids = [None; 256];
        let mut markers = Vec::new();
        for (id, kind) in kinds.iter().enumerate() {
            match kind {
                PieceKind::Byte => {
                    if let Some(value) = byte_value(&pieces[id]) {
                        byte_ids[value as usize] = Some(id as u32);
                    }
                }
                PieceKind::UserDefined => markers.push((pieces[id].clone(), id as u32)),
                _ => {}
            }
        }
        markers.sort_by_key(|(piece, _)| std::cmp::Reverse(piece.len()));
        Ok(Self { pieces, scores, kinds, ids, byte_ids, markers, pad_id, eos_id, bos_id, unknown_id, add_dummy_prefix, byte_fallback })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    pub fn unknown_id(&self) -> u32 {
        self.unknown_id
    }

    pub fn piece_id(&self, piece: &str) -> Option<u32> {
        self.ids.get(piece).copied()
    }

    pub fn kind(&self, id: u32) -> PieceKind {
        self.kinds[id as usize]
    }

    pub fn byte_value(&self, id: u32) -> Option<u8> {
        byte_value(&self.pieces[id as usize])
    }

    pub fn piece(&self, id: u32) -> &str {
        &self.pieces[id as usize]
    }

    pub(crate) fn meta_space() -> char {
        META_SPACE
    }
}
