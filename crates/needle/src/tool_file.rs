use std::fmt;
use std::fs;
use std::path::Path;

use serde_json::Value;

#[derive(Clone, Debug, Default)]
pub struct ToolFile {
    pub tools: Vec<Value>,
    pub system: Option<String>,
}

#[derive(Debug)]
pub struct ToolFileError(pub String);

impl fmt::Display for ToolFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ToolFileError {}

impl ToolFile {
    pub fn from_value(value: Value) -> Self {
        if let Some(tools) = value.get("tools").and_then(Value::as_array) {
            let system = value.get("system").and_then(Value::as_str).map(str::to_owned);
            return Self { tools: tools.clone(), system };
        }
        match value {
            Value::Array(tools) => Self { tools, system: None },
            single => Self { tools: vec![single], system: None },
        }
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ToolFileError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|error| ToolFileError(format!("{}: {error}", path.display())))?;
        let value = serde_json::from_str(&text).map_err(|error| ToolFileError(format!("{}: {error}", path.display())))?;
        Ok(Self::from_value(value))
    }
}
