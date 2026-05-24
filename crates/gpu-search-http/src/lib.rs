//! Experimental Rust HTTP server scaffold.
//!
//! This crate is intentionally additive and does not replace the current
//! Python HTTP/MCP runtime. It starts with a small `/health` route so the Rust
//! API surface can grow in focused PRs while preserving existing behavior.

use axum::{Json, Router, routing::get};
use gpu_search_core::{DEFAULT_INDEXED_EXTS, DEFAULT_SKIP_DIRS, RUST_CORE_VERSION};
use serde::Serialize;

/// Experimental Rust HTTP crate version.
pub const RUST_HTTP_VERSION: &str = "0.1.0-prototype";

/// Health response for the experimental Rust HTTP server.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct HealthResponse {
    pub status: &'static str,
    pub implementation: &'static str,
    pub rust_core_version: &'static str,
    pub rust_http_version: &'static str,
}

/// Current experimental Rust HTTP capability flags.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CapabilityResponse {
    pub health: bool,
    pub stats: bool,
    pub diagnostics: bool,
    pub search_code: bool,
    pub dependency_impact: bool,
    pub semantic_search: bool,
}

/// Lightweight stats response for the experimental Rust HTTP server.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct StatsResponse {
    pub status: &'static str,
    pub implementation: &'static str,
    pub rust_core_version: &'static str,
    pub rust_http_version: &'static str,
    pub indexed_roots: usize,
    pub indexed_files: usize,
    pub default_indexed_extensions: usize,
    pub default_skip_directories: usize,
    pub capabilities: CapabilityResponse,
    pub limitations: Vec<&'static str>,
}

/// Build the experimental Rust HTTP router.
pub fn app() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
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
pub async fn stats() -> Json<StatsResponse> {
    Json(StatsResponse {
        status: "ok",
        implementation: "rust-http-experimental",
        rust_core_version: RUST_CORE_VERSION,
        rust_http_version: RUST_HTTP_VERSION,
        indexed_roots: 0,
        indexed_files: 0,
        default_indexed_extensions: DEFAULT_INDEXED_EXTS.len(),
        default_skip_directories: DEFAULT_SKIP_DIRS.len(),
        capabilities: CapabilityResponse {
            health: true,
            stats: true,
            diagnostics: false,
            search_code: false,
            dependency_impact: false,
            semantic_search: false,
        },
        limitations: vec![
            "Experimental Rust HTTP scaffold only.",
            "Python HTTP/MCP runtime remains authoritative.",
            "No repository is indexed by the Rust HTTP server yet.",
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_endpoint_returns_ok() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn stats_endpoint_returns_ok() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/stats")
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_handler_reports_experimental_rust_versions() {
        let Json(response) = health().await;

        assert_eq!(response.status, "ok");
        assert_eq!(response.implementation, "rust-http-experimental");
        assert_eq!(response.rust_core_version, RUST_CORE_VERSION);
        assert_eq!(response.rust_http_version, RUST_HTTP_VERSION);
    }

    #[tokio::test]
    async fn stats_handler_reports_capabilities_without_indexing() {
        let Json(response) = stats().await;

        assert_eq!(response.status, "ok");
        assert_eq!(response.implementation, "rust-http-experimental");
        assert_eq!(response.indexed_roots, 0);
        assert_eq!(response.indexed_files, 0);
        assert_eq!(
            response.default_indexed_extensions,
            DEFAULT_INDEXED_EXTS.len()
        );
        assert_eq!(response.default_skip_directories, DEFAULT_SKIP_DIRS.len());
        assert!(response.capabilities.health);
        assert!(response.capabilities.stats);
        assert!(!response.capabilities.search_code);
        assert!(
            response
                .limitations
                .contains(&"Python HTTP/MCP runtime remains authoritative.")
        );
    }
}
