pub use model::{Model, ModelError};
pub use needle::{
    extract, extract_with_engine, tool, Engine, ExtractOptions, ExtractionValidationError, Failure, MissingEngine, Needle, NeedleBuilder, Parameter, Response,
    ResponseKind, RunOptions, ToolBuilder, ToolFile, ToolFileError, Validation, DEFAULT_MAX_NEW_TOKENS, DEFAULT_MAX_STEPS,
};
pub use repair::Call;

#[cfg(feature = "speech")]
pub mod speech {
    pub use intent::{Intent, IntentResponse};
    pub use whistle::{read_wav, Options, Stream, StreamStep, Transcript, Whistle, WhistleError, Word};
}

#[cfg(feature = "download")]
pub mod download {
    pub use fetch::{base_weights, cache_directory, download, fetch_spec, Base, FetchError};
}
