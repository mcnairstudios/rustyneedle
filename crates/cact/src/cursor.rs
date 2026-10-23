use crate::error::{CactError, Result};

pub struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
    what: &'static str,
}

impl<'a> Cursor<'a> {
    pub fn new(bytes: &'a [u8], position: usize, what: &'static str) -> Self {
        Self { bytes, position, what }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self.position + N;
        let slice = self.bytes.get(self.position..end).ok_or(CactError::Truncated { what: self.what })?;
        self.position = end;
        Ok(slice.try_into().expect("slice length matches"))
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take::<1>()?[0])
    }

    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take()?))
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take()?))
    }
}
