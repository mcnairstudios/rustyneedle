use std::fmt;
use std::io;

#[derive(Debug)]
pub enum CactError {
    Io(io::Error),
    Truncated { what: &'static str },
    WrongTag { found: u32 },
    UnknownElementKind { code: u8 },
    BadLayout { reason: String },
    OutOfBounds { index: usize },
}

impl fmt::Display for CactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CactError::Io(error) => write!(f, "{error}"),
            CactError::Truncated { what } => write!(f, "archive is truncated inside the {what}"),
            CactError::WrongTag { found } => write!(f, "not a Needle 3 archive (format tag 0x{found:08x})"),
            CactError::UnknownElementKind { code } => write!(f, "unknown tensor element kind {code}"),
            CactError::BadLayout { reason } => write!(f, "unexpected tensor layout: {reason}"),
            CactError::OutOfBounds { index } => write!(f, "tensor {index} lies outside the archive"),
        }
    }
}

impl std::error::Error for CactError {}

impl From<io::Error> for CactError {
    fn from(error: io::Error) -> Self {
        CactError::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, CactError>;
