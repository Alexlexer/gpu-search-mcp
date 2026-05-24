//! Experimental Rust HTTP server scaffold.
//!
//! This crate is intentionally additive and does not replace the current
//! Python HTTP/MCP runtime. It starts with a small `/health` route so the Rust
//! API surface can grow in focused PRs while preserving existing behavior.

use axum::{Json, Router, routing::get};
use gpu_search_core::RUST_CORE_VERSION;
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

/// Build the experimental Rust HTTP router.
pub fn app() -> Router {
    Router::new().route("/health", get(health))
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
    async fn health_handler_reports_experimental_rust_versions() {
        let Json(response) = health().await;

        assert_eq!(response.status, "ok");
        assert_eq!(response.implementation, "rust-http-experimental");
        assert_eq!(response.rust_core_version, RUST_CORE_VERSION);
        assert_eq!(response.rust_http_version, RUST_HTTP_VERSION);
    }
}
