//! Experimental Rust HTTP app state.

use gpu_search_core::{DependencyGraph, DiscoveredFile, IndexOptions, discover_files};
use std::path::{Path, PathBuf};
/// Shared state for the experimental Rust HTTP server.
#[derive(Debug, Clone, Default)]
pub struct AppState {
    pub(crate) indexed_root: Option<PathBuf>,
    pub(crate) files: Vec<DiscoveredFile>,
    pub(crate) dependency_graph: Option<DependencyGraph>,
}

impl AppState {
    /// Empty server state. Used by default so routes remain cheap/not-ready.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build server state by discovering files under a repository root.
    pub fn from_directory(root: impl AsRef<Path>) -> Result<Self, gpu_search_core::DiscoveryError> {
        let root = root
            .as_ref()
            .canonicalize()
            .unwrap_or_else(|_| root.as_ref().to_path_buf());
        let files = discover_files(&root, &IndexOptions::default())?;
        let dependency_graph = DependencyGraph::from_files(&files).ok();
        Ok(Self {
            indexed_root: Some(root),
            files,
            dependency_graph,
        })
    }

    pub fn indexed_files(&self) -> usize {
        self.files.len()
    }

    pub fn indexed_roots(&self) -> usize {
        usize::from(self.indexed_root.is_some())
    }

    pub fn dependency_ready(&self) -> bool {
        self.dependency_graph.is_some()
    }
}
