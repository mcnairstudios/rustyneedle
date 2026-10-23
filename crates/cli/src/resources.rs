use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use fetch::Base;
use model::Model;
use serde_json::Value;
use whistle::Whistle;

#[derive(Debug)]
pub struct CliError(pub String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CliError {}

macro_rules! convert {
    ($($error:ty),*) => {$(impl From<$error> for CliError { fn from(error: $error) -> Self { CliError(error.to_string()) } })*};
}

convert!(
    std::io::Error,
    serde_json::Error,
    fetch::FetchError,
    model::ModelError,
    whistle::WhistleError,
    whistle::WavError,
    playground::PlaygroundError,
    needle::ExtractionValidationError
);

pub fn needle_weights(path: Option<&Path>) -> Result<PathBuf, CliError> {
    match path {
        Some(path) => Ok(path.to_path_buf()),
        None => Ok(fetch::base_weights(Base::Needle)?),
    }
}

pub fn whistle_weights(path: Option<&Path>) -> Result<PathBuf, CliError> {
    match path {
        Some(path) => Ok(path.to_path_buf()),
        None => Ok(fetch::base_weights(Base::Whistle)?),
    }
}

pub fn open_model(path: Option<&Path>, depth: Option<usize>) -> Result<Arc<Model>, CliError> {
    let model = Model::open(needle_weights(path)?)?;
    Ok(match depth {
        Some(depth) => model.with_depth(depth)?,
        None => model,
    })
}

pub fn open_whistle(path: Option<&Path>, depth: Option<usize>) -> Result<Whistle, CliError> {
    let whistle = Whistle::open(whistle_weights(path)?)?;
    Ok(match depth {
        Some(depth) => whistle.with_audio_depth(depth)?,
        None => whistle,
    })
}

pub fn carries_audio(path: &Path) -> Result<bool, CliError> {
    let archive = cact::Archive::open(path).map_err(|error| CliError(error.to_string()))?;
    let layout = cact::Layout::resolve(&archive).map_err(|error| CliError(error.to_string()))?;
    Ok(layout.audio.is_some())
}

pub fn read_json_argument(argument: &str) -> Result<Value, CliError> {
    let trimmed = argument.trim_start();
    if trimmed.starts_with('[') || trimmed.starts_with('{') {
        return Ok(serde_json::from_str(argument)?);
    }
    Ok(serde_json::from_str(&fs::read_to_string(argument)?)?)
}

pub fn keywords(inline: &[String], file: Option<&Path>) -> Result<Vec<String>, CliError> {
    let mut all = inline.to_vec();
    if let Some(path) = file {
        all.extend(fs::read_to_string(path)?.lines().map(str::trim).filter(|line| !line.is_empty()).map(str::to_owned));
    }
    Ok(all)
}
