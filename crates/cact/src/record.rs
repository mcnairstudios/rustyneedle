use crate::cursor::Cursor;
use crate::error::{CactError, Result};

pub const RECORD_BYTES: usize = 44;
pub const TERNARY_RECORD_BITS: u32 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementKind {
    Half,
    Single,
    CodebookQuantised,
    Raw,
}

impl ElementKind {
    fn from_code(code: u8) -> Result<Self> {
        match code {
            1 => Ok(ElementKind::Half),
            2 => Ok(ElementKind::Single),
            3 => Ok(ElementKind::CodebookQuantised),
            4 => Ok(ElementKind::Raw),
            code => Err(CactError::UnknownElementKind { code }),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorRecord {
    pub kind: ElementKind,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub byte_len: usize,
    pub group_size: usize,
    pub bits: u32,
}

impl TensorRecord {
    pub fn parse(cursor: &mut Cursor<'_>) -> Result<Self> {
        let kind = ElementKind::from_code(cursor.u8()?)?;
        let rank = cursor.u8()? as usize;
        cursor.u16()?;
        let mut dims = [0usize; 4];
        for dim in dims.iter_mut() {
            *dim = cursor.u32()? as usize;
        }
        let offset = cursor.u64()? as usize;
        let byte_len = cursor.u64()? as usize;
        let group_size = cursor.u32()? as usize;
        let bits = cursor.u32()?;
        Ok(Self { kind, shape: dims[..rank.min(4)].to_vec(), offset, byte_len, group_size, bits })
    }

    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    pub fn columns(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }

    pub fn padded_columns(&self) -> usize {
        let group = self.group_size.max(1);
        self.columns().div_ceil(group) * group
    }

    pub fn index_bits(&self) -> u32 {
        if self.bits == TERNARY_RECORD_BITS {
            2
        } else {
            self.bits
        }
    }

    pub fn packed_row_bytes(&self) -> usize {
        self.padded_columns() * self.index_bits() as usize / 8
    }

    pub fn packed_bytes(&self) -> usize {
        self.rows() * self.packed_row_bytes()
    }

    pub fn groups_per_row(&self) -> usize {
        self.padded_columns() / self.group_size.max(1)
    }
}
