//! Experimental Rust HTTP system/status handlers.

use crate::{
    AppState, CapabilityResponse, DEFAULT_SEMANTIC_MODEL_ID, DiagnosticsDevice, DiagnosticsIndexes,
    DiagnosticsResponse, HealthResponse, RUST_HTTP_VERSION, SemanticModelStatusResponse,
    StatsResponse,
};
use axum::{Json, extract::State};
use gpu_search_core::{DEFAULT_INDEXED_EXTS, DEFAULT_SKIP_DIRS, RUST_CORE_VERSION};
use std::env;
/// Return semantic model status without loading or downloading a model.
pub async fn semantic_model_status() -> Json<SemanticModelStatusResponse> {
    Json(semantic_model_status_snapshot())
}

/// Return lightweight health information.
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        implementation: "rust-http-experimental",
        rust_core_version: RUST_CORE_VERSION,
        rust_http_version: RUST_HTTP_VERSION,
    })
}

/// Return cheap, in-memory stats for the experimental Rust HTTP server.
pub async fn stats(State(state): State<AppState>) -> Json<StatsResponse> {
    Json(StatsResponse {
        status: "ok",
        implementation: "rust-http-experimental",
        rust_core_version: RUST_CORE_VERSION,
        rust_http_version: RUST_HTTP_VERSION,
        indexed_roots: state.indexed_roots(),
        indexed_files: state.indexed_files(),
        default_indexed_extensions: DEFAULT_INDEXED_EXTS.len(),
        default_skip_directories: DEFAULT_SKIP_DIRS.len(),
        capabilities: CapabilityResponse {
            health: true,
            stats: true,
            diagnostics: true,
            search_code: true,
            dependency_impact: true,
            signal_scan: true,
            semantic_search: false,
        },
        limitations: vec![
            "Experimental Rust HTTP scaffold only.",
            "Python HTTP/MCP runtime remains authoritative.",
            "Semantic search is not implemented in Rust HTTP yet.",
        ],
    })
}

/// Return cheap setup diagnostics without indexing or probing external systems.
pub async fn diagnostics(State(state): State<AppState>) -> Json<DiagnosticsResponse> {
    let has_index = !state.files.is_empty();
    Json(DiagnosticsResponse {
        status: if has_index { "ok" } else { "not_ready" },
        implementation: "rust-http-experimental",
        rust_core_version: RUST_CORE_VERSION,
        rust_http_version: RUST_HTTP_VERSION,
        device: DiagnosticsDevice {
            backend: "cpu",
            reason: "Rust HTTP scaffold has no GPU device selection yet.",
            warnings: Vec::new(),
        },
        indexes: DiagnosticsIndexes {
            pattern_ready: has_index,
            semantic_ready: false,
            dependency_ready: state.dependency_ready(),
            indexed_files: state.indexed_files(),
        },
        capabilities: CapabilityResponse {
            health: true,
            stats: true,
            diagnostics: true,
            search_code: true,
            dependency_impact: true,
            signal_scan: true,
            semantic_search: false,
        },
        warnings: if has_index {
            Vec::new()
        } else {
            vec!["No repository is indexed by the Rust HTTP server yet."]
        },
        limitations: vec![
            "Experimental Rust HTTP scaffold only.",
            "Diagnostics are static and do not perform scans.",
            "Python HTTP/MCP runtime remains authoritative.",
        ],
    })
}

fn semantic_model_status_snapshot() -> SemanticModelStatusResponse {
    let model_id = env::var("GPU_SEARCH_SEMANTIC_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_SEMANTIC_MODEL_ID.to_string());

    SemanticModelStatusResponse {
        model_id: model_id.clone(),
        provider: "sentence-transformers",
        available: false,
        cached: false,
        requires_download: true,
        device: "unavailable",
        message: format!(
            "Rust HTTP does not load sentence-transformers models yet. Use the Python runtime or run: gpu-search-mcp --semantic-model {model_id} --download-semantic-model"
        ),
        limitations: vec![
            "Rust HTTP semantic model status is advisory only.",
            "Rust semantic search is not implemented yet.",
            "Python HTTP/MCP runtime remains authoritative for sentence-transformers embeddings.",
        ],
    }
}
