use std::path::Path;
use std::sync::Arc;

use cact::{Archive, CactError, Layout};
use tokenizer::{Tokenizer, VocabularyError};

use crate::config::ModelConfig;
use crate::ladder;
use crate::weights::Weights;

#[derive(Debug)]
pub enum ModelError {
    Archive(CactError),
    Tokenizer(VocabularyError),
    Depth { requested: usize, layers: usize },
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Archive(error) => write!(f, "{error}"),
            ModelError::Tokenizer(error) => write!(f, "{error}"),
            ModelError::Depth { requested, layers } => write!(f, "depth {requested} is outside the 2 to {layers} layers this model offers"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<CactError> for ModelError {
    fn from(error: CactError) -> Self {
        ModelError::Archive(error)
    }
}

impl From<VocabularyError> for ModelError {
    fn from(error: VocabularyError) -> Self {
        ModelError::Tokenizer(error)
    }
}

pub struct Model {
    config: ModelConfig,
    weights: Weights,
    tokenizer: Tokenizer,
    active: Vec<usize>,
    archive: Archive,
    layout: Layout,
}

impl Model {
    pub fn open(path: impl AsRef<Path>) -> Result<Arc<Self>, ModelError> {
        let archive = Archive::open(path)?;
        let layout = Layout::resolve(&archive)?;
        Self::from_archive(&archive, &layout)
    }

    pub fn from_archive(archive: &Archive, layout: &Layout) -> Result<Arc<Self>, ModelError> {
        let layers = archive.geometry().num_layers;
        Self::build(archive, layout, (0..layers).collect())
    }

    pub fn with_depth(&self, depth: usize) -> Result<Arc<Self>, ModelError> {
        let active = ladder::select(self.config.num_layers, depth).ok_or(ModelError::Depth { requested: depth, layers: self.config.num_layers })?;
        Self::build(&self.archive, &self.layout, active)
    }

    fn build(archive: &Archive, layout: &Layout, active: Vec<usize>) -> Result<Arc<Self>, ModelError> {
        let config = ModelConfig::from_geometry(archive.geometry());
        let rows: Vec<usize> = std::iter::once(0).chain(active.iter().map(|layer| layer + 1)).collect();
        let weights = Weights::load(archive, layout, &config, &rows);
        let tokenizer = Tokenizer::from_blob(archive.bytes(layout.tokenizer))?;
        Ok(Arc::new(Self { config, weights, tokenizer, active, archive: archive.clone(), layout: layout.clone() }))
    }

    pub fn active_layers(&self) -> &[usize] {
        &self.active
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub(crate) fn weights(&self) -> &Weights {
        &self.weights
    }
}
