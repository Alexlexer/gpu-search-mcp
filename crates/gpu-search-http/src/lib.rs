//! Experimental Rust HTTP server scaffold.
//!
//! This crate is intentionally additive and does not replace the current
//! Python HTTP/MCP runtime. It starts with a small `/health` route so the Rust
//! API surface can grow in focused PRs while preserving existing behavior.

pub mod dependency;
pub mod models;
pub mod read;
pub mod search;
pub mod signals;
pub mod state;
pub mod system;

use axum::{
    Router,
    routing::{get, post},
};
pub use dependency::dependency_impact;
pub use models::*;
pub use read::{read_block, read_skeleton};
pub use search::{search_code, search_hybrid, search_semantic};
pub use signals::scan_signals;
pub use state::AppState;
use std::path::{Path, PathBuf};
pub use system::{diagnostics, health, semantic_model_status, stats};

/// Experimental Rust HTTP crate version.
pub const RUST_HTTP_VERSION: &str = "0.1.0-prototype";

/// Default sentence-transformers model used by the Python runtime.
pub const DEFAULT_SEMANTIC_MODEL_ID: &str = "BAAI/bge-small-en-v1.5";

/// Build the experimental Rust HTTP router.
pub fn app() -> Router {
    app_with_state(AppState::empty())
}

/// Build the experimental Rust HTTP router with explicit state.
pub fn app_with_state(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/diagnostics", get(diagnostics))
        .route("/semantic/model/status", get(semantic_model_status))
        .route("/search/code", post(search_code))
        .route("/search/hybrid", post(search_hybrid))
        .route("/search/semantic", post(search_semantic))
        .route("/read/block", post(read_block))
        .route("/read/skeleton", post(read_skeleton))
        .route("/dependency/impact", post(dependency_impact))
        .route("/scan/signals", post(scan_signals))
        .with_state(state)
}

pub(crate) fn resolve_under_indexed_root(state: &AppState, filepath: &str) -> Option<PathBuf> {
    let root = state.indexed_root.as_ref()?;
    let candidate = PathBuf::from(filepath);
    let candidate = if candidate.is_absolute() {
        candidate
    } else {
        root.join(candidate)
    };
    let resolved = candidate.canonicalize().ok()?;
    let resolved_root = root.canonicalize().ok()?;
    resolved.starts_with(resolved_root).then_some(resolved)
}

pub(crate) fn display_relative(state: &AppState, filepath: &Path) -> String {
    state
        .indexed_root
        .as_ref()
        .and_then(|root| filepath.strip_prefix(root).ok())
        .unwrap_or(filepath)
        .display()
        .to_string()
}

#[cfg(test)]
mod tests;
