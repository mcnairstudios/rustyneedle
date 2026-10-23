use std::fs::File;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::cursor::Cursor;
use crate::error::{CactError, Result};
use crate::geometry::{Geometry, HEADER_BYTES};
use crate::record::{ElementKind, TensorRecord, RECORD_BYTES};

#[derive(Clone)]
pub struct TensorBytes {
    map: Arc<Mmap>,
    offset: usize,
    length: usize,
}

impl Deref for TensorBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.map[self.offset..self.offset + self.length]
    }
}

#[derive(Clone)]
pub struct Archive {
    map: Arc<Mmap>,
    geometry: Geometry,
    codebook: Vec<f32>,
    records: Vec<TensorRecord>,
}

impl Archive {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        Self::from_map(Arc::new(map))
    }

    fn from_map(map: Arc<Mmap>) -> Result<Self> {
        let geometry = Geometry::parse(&map)?;
        let mut cursor = Cursor::new(&map, HEADER_BYTES, "codebook");
        let mut codebook = Vec::with_capacity(geometry.codebook_len);
        for _ in 0..geometry.codebook_len {
            codebook.push(cursor.f32()?);
        }
        let mut directory = Cursor::new(&map, cursor.position(), "tensor directory");
        let mut records = Vec::with_capacity(geometry.tensor_count);
        for _ in 0..geometry.tensor_count {
            records.push(TensorRecord::parse(&mut directory)?);
        }
        for (index, record) in records.iter().enumerate() {
            if record.offset.checked_add(record.byte_len).is_none_or(|end| end > map.len()) {
                return Err(CactError::OutOfBounds { index });
            }
            let width = match record.kind {
                ElementKind::Half => 2,
                ElementKind::Single => 4,
                _ => 1,
            };
            if record.offset % width != 0 || record.byte_len % width != 0 {
                return Err(CactError::BadLayout { reason: format!("tensor {index} is not aligned to its element width") });
            }
        }
        debug_assert_eq!(directory.position(), cursor.position() + geometry.tensor_count * RECORD_BYTES);
        Ok(Self { map, geometry, codebook, records })
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    pub fn codebook(&self) -> &[f32] {
        &self.codebook
    }

    pub fn record(&self, index: usize) -> &TensorRecord {
        &self.records[index]
    }

    pub fn records(&self) -> &[TensorRecord] {
        &self.records
    }

    pub fn bytes(&self, index: usize) -> &[u8] {
        let record = &self.records[index];
        &self.map[record.offset..record.offset + record.byte_len]
    }

    pub fn tensor_bytes(&self, index: usize) -> TensorBytes {
        let record = &self.records[index];
        TensorBytes { map: Arc::clone(&self.map), offset: record.offset, length: record.byte_len }
    }

    pub fn halves(&self, index: usize) -> &[u16] {
        reinterpret(self.bytes(index))
    }

    pub fn singles(&self, index: usize) -> &[f32] {
        reinterpret(self.bytes(index))
    }
}

fn reinterpret<T: Copy>(bytes: &[u8]) -> &[T] {
    let width = std::mem::size_of::<T>();
    assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<T>(), 0, "tensor data is misaligned");
    assert_eq!(bytes.len() % width, 0, "tensor length is not a whole number of elements");
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / width) }
}
