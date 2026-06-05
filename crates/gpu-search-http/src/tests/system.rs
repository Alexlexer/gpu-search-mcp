use super::*;
use axum::body::Body;
use axum::extract::{Json, State};
use axum::http::{Request, StatusCode};
use gpu_search_core::{DEFAULT_INDEXED_EXTS, DEFAULT_SKIP_DIRS, RUST_CORE_VERSION};
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
async fn diagnostics_endpoint_returns_ok() {
    let response = app()
        .oneshot(
            Request::builder()
                .uri("/diagnostics")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn semantic_model_status_endpoint_returns_ok() {
    let response = app()
        .oneshot(
            Request::builder()
                .uri("/semantic/model/status")
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
    let Json(response) = stats(State(AppState::empty())).await;

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
    assert!(response.capabilities.search_code);
    assert!(response.capabilities.dependency_impact);
    assert!(response.capabilities.signal_scan);
    assert!(
        response
            .limitations
            .contains(&"Python HTTP/MCP runtime remains authoritative.")
    );
}

#[tokio::test]
async fn diagnostics_handler_reports_not_ready_without_side_effects() {
    let Json(response) = diagnostics(State(AppState::empty())).await;

    assert_eq!(response.status, "not_ready");
    assert_eq!(response.implementation, "rust-http-experimental");
    assert_eq!(response.device.backend, "cpu");
    assert!(!response.indexes.pattern_ready);
    assert!(!response.indexes.semantic_ready);
    assert!(!response.indexes.dependency_ready);
    assert_eq!(response.indexes.indexed_files, 0);
    assert!(response.capabilities.diagnostics);
    assert!(response.capabilities.search_code);
    assert!(response.capabilities.dependency_impact);
    assert!(response.capabilities.signal_scan);
    assert!(
        response
            .warnings
            .contains(&"No repository is indexed by the Rust HTTP server yet.")
    );
}

#[tokio::test]
async fn semantic_model_status_reports_python_sidecar_guidance() {
    let Json(response) = semantic_model_status().await;

    assert_eq!(response.model_id, DEFAULT_SEMANTIC_MODEL_ID);
    assert_eq!(response.provider, "sentence-transformers");
    assert!(!response.available);
    assert!(!response.cached);
    assert!(response.requires_download);
    assert!(response.message.contains("--download-semantic-model"));
    assert!(
        response
            .limitations
            .contains(&"Rust semantic search is not implemented yet.")
    );
}

#[tokio::test]
async fn stats_and_diagnostics_reflect_indexed_state() {
    let root = temp_root("status");
    fs::write(root.join("lib.rs"), "pub struct UserService;\n")
        .expect("sample file should be written");
    let state = AppState::from_directory(&root).expect("state should index temp root");

    let Json(stats_response) = stats(State(state.clone())).await;
    let Json(diagnostics_response) = diagnostics(State(state)).await;

    assert_eq!(stats_response.indexed_roots, 1);
    assert_eq!(stats_response.indexed_files, 1);
    assert_eq!(diagnostics_response.status, "ok");
    assert!(diagnostics_response.indexes.pattern_ready);
    assert!(diagnostics_response.indexes.dependency_ready);
    assert_eq!(diagnostics_response.indexes.indexed_files, 1);
    fs::remove_dir_all(root).ok();
}
