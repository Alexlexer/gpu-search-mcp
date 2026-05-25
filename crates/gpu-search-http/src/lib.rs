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
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::{Json, State};
    use axum::http::{Request, StatusCode};
    use gpu_search_core::{
        DEFAULT_INDEXED_EXTS, DEFAULT_SKIP_DIRS, DEPENDENCY_ANALYSIS_MODE, RUST_CORE_VERSION,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tower::ServiceExt;

    fn temp_root(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gpu_search_http_{name}_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root).expect("temp root should be created");
        root
    }

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
    async fn search_code_endpoint_returns_not_ready_shape() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/search/code")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"UserService","topK":3}"#))
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn search_hybrid_endpoint_returns_ok() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/search/hybrid")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"UserService","topK":3}"#))
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn search_semantic_endpoint_returns_ok() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/search/semantic")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"authentication middleware"}"#))
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn dependency_impact_endpoint_returns_not_ready_shape() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/dependency/impact")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"filepath":"service.py"}"#))
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn read_block_endpoint_rejects_when_no_root() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/read/block")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"filepath":"src/lib.rs"}"#))
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn read_skeleton_endpoint_rejects_when_no_root() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/read/skeleton")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"filepath":"src/lib.rs"}"#))
                    .expect("request should build"),
            )
            .await
            .expect("router should respond");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn scan_signals_endpoint_returns_not_ready_shape() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/scan/signals")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{}"#))
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
    async fn search_code_handler_preserves_request_shape_and_empty_results() {
        let Json(response) = search_code(
            State(AppState::empty()),
            Json(SearchCodeRequest {
                query: "UserService".to_string(),
                top_k: Some(3),
                mode: Some(SearchMode::Pattern),
                context_mode: Some(SearchContextMode::Compact),
            }),
        )
        .await;

        assert_eq!(response.status, "not_ready");
        assert_eq!(response.query, "UserService");
        assert_eq!(response.top_k, 3);
        assert_eq!(response.mode, SearchMode::Pattern);
        assert_eq!(response.context_mode, SearchContextMode::Compact);
        assert!(response.results.is_empty());
        assert!(
            response
                .result
                .contains("Start with --directory to enable pattern search")
        );
    }

    #[tokio::test]
    async fn search_code_returns_pattern_results_when_state_has_files() {
        let root = temp_root("pattern_search");
        fs::write(
            root.join("app.rs"),
            "fn main() {\n    let user_service = UserService::new();\n}\n",
        )
        .expect("sample file should be written");
        let state = AppState::from_directory(&root).expect("state should index temp root");

        let Json(response) = search_code(
            State(state),
            Json(SearchCodeRequest {
                query: "UserService".to_string(),
                top_k: Some(5),
                mode: Some(SearchMode::Pattern),
                context_mode: Some(SearchContextMode::Compact),
            }),
        )
        .await;

        assert_eq!(response.status, "ok");
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].line_start, 2);
        assert_eq!(response.results[0].reason, "exact token match");
        assert!(response.results[0].snippet.contains("UserService"));
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn search_hybrid_forces_hybrid_mode_and_returns_pattern_results() {
        let root = temp_root("hybrid_search");
        fs::write(
            root.join("app.rs"),
            "let user_service = UserService::new();\n",
        )
        .expect("sample file should be written");
        let state = AppState::from_directory(&root).expect("state should index temp root");

        let Json(response) = search_hybrid(
            State(state),
            Json(SearchCodeRequest {
                query: "UserService".to_string(),
                top_k: Some(5),
                mode: Some(SearchMode::Pattern),
                context_mode: Some(SearchContextMode::Compact),
            }),
        )
        .await;

        assert_eq!(response.status, "ok");
        assert_eq!(response.mode, SearchMode::Hybrid);
        assert_eq!(response.results.len(), 1);
        assert!(
            response
                .warnings
                .contains(&"Hybrid mode currently returns Rust pattern results only.")
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn search_semantic_forces_semantic_not_ready_response() {
        let root = temp_root("semantic_search");
        fs::write(
            root.join("app.rs"),
            "let user_service = UserService::new();\n",
        )
        .expect("sample file should be written");
        let state = AppState::from_directory(&root).expect("state should index temp root");

        let Json(response) = search_semantic(
            State(state),
            Json(SearchCodeRequest {
                query: "authentication middleware".to_string(),
                top_k: Some(5),
                mode: Some(SearchMode::Pattern),
                context_mode: Some(SearchContextMode::Compact),
            }),
        )
        .await;

        assert_eq!(response.status, "not_ready");
        assert_eq!(response.mode, SearchMode::Semantic);
        assert!(response.results.is_empty());
        assert!(
            response
                .warnings
                .contains(&"Semantic search remains available through the Python runtime.")
        );
        fs::remove_dir_all(root).ok();
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

    #[tokio::test]
    async fn dependency_impact_returns_reverse_dependencies_with_reason() {
        let root = temp_root("dependency_impact");
        fs::write(root.join("service.py"), "class UserService:\n    pass\n")
            .expect("service file should be written");
        fs::write(root.join("app.py"), "from service import UserService\n")
            .expect("app file should be written");
        let state = AppState::from_directory(&root).expect("state should index temp root");

        let Json(response) = dependency_impact(
            State(state),
            Json(DependencyImpactRequest {
                filepath: "service.py".to_string(),
            }),
        )
        .await;

        assert_eq!(response.status, "ok");
        assert_eq!(response.confidence, "medium");
        assert_eq!(response.analysis_mode, DEPENDENCY_ANALYSIS_MODE);
        assert_eq!(response.impacted_files.len(), 1);
        assert!(response.impacted_files[0].file.ends_with("app.py"));
        assert_eq!(response.impacted_files[0].hops, 1);
        assert_eq!(
            response.impacted_files[0].reason.as_deref(),
            Some("imports module service")
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn dependency_impact_rejects_outside_root() {
        let root = temp_root("dependency_outside");
        fs::write(root.join("service.py"), "class UserService:\n    pass\n")
            .expect("service file should be written");
        let outside = temp_root("dependency_outside_target").join("outside.py");
        fs::write(&outside, "print('outside')\n").expect("outside file should be written");
        let state = AppState::from_directory(&root).expect("state should index temp root");

        let Json(response) = dependency_impact(
            State(state),
            Json(DependencyImpactRequest {
                filepath: outside.display().to_string(),
            }),
        )
        .await;

        assert_eq!(response.status, "error");
        assert!(response.impacted_files.is_empty());
        assert!(
            response
                .warnings
                .contains(&"Requested file is outside the indexed root or does not exist.")
        );
        fs::remove_dir_all(root).ok();
        fs::remove_dir_all(outside.parent().expect("outside file has parent")).ok();
    }

    #[tokio::test]
    async fn read_block_returns_requested_line_range() {
        let root = temp_root("read_block");
        fs::write(
            root.join("app.rs"),
            "line one\nline two\nline three\nline four\n",
        )
        .expect("sample file should be written");
        let state = AppState::from_directory(&root).expect("state should index temp root");

        let Json(response) = read_block(
            State(state),
            Json(ReadBlockRequest {
                filepath: "app.rs".to_string(),
                line_start: Some(2),
                line_end: Some(3),
            }),
        )
        .await;

        assert_eq!(response.status, "ok");
        assert_eq!(response.line_start, 2);
        assert_eq!(response.line_end, 3);
        assert_eq!(response.content, "line two\nline three");
        assert!(response.file.ends_with("app.rs"));
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn read_block_rejects_outside_root() {
        let root = temp_root("read_outside");
        fs::write(root.join("app.rs"), "inside\n").expect("inside file should be written");
        let outside_root = temp_root("read_outside_target");
        let outside = outside_root.join("outside.rs");
        fs::write(&outside, "outside\n").expect("outside file should be written");
        let state = AppState::from_directory(&root).expect("state should index temp root");

        let Json(response) = read_block(
            State(state),
            Json(ReadBlockRequest {
                filepath: outside.display().to_string(),
                line_start: None,
                line_end: None,
            }),
        )
        .await;

        assert_eq!(response.status, "error");
        assert!(response.content.is_empty());
        assert!(
            response
                .warnings
                .contains(&"Requested file is outside the indexed root or does not exist.")
        );
        fs::remove_dir_all(root).ok();
        fs::remove_dir_all(outside_root).ok();
    }

    #[tokio::test]
    async fn read_skeleton_returns_csharp_symbols() {
        let root = temp_root("read_skeleton_cs");
        fs::write(
            root.join("UserController.cs"),
            "using MyApp.Services;\nnamespace MyApp.Controllers;\npublic interface IUserController { }\npublic class UserController : ControllerBase, IUserController {\n    public string GetUser() => \"ok\";\n}\n",
        )
        .expect("sample file should be written");
        let state = AppState::from_directory(&root).expect("state should index temp root");

        let Json(response) = read_skeleton(
            State(state),
            Json(ReadSkeletonRequest {
                filepath: "UserController.cs".to_string(),
            }),
        )
        .await;

        assert_eq!(response.status, "ok");
        assert!(response.symbols.iter().any(|symbol| {
            symbol.kind == "class_declaration" && symbol.name.as_deref() == Some("UserController")
        }));
        assert!(
            response
                .symbols
                .iter()
                .any(|symbol| symbol.kind == "method_declaration"
                    && symbol.name.as_deref() == Some("GetUser"))
        );
        assert!(
            response
                .symbols
                .iter()
                .any(|symbol| symbol.kind == "controller_action"
                    && symbol.name.as_deref() == Some("GetUser"))
        );
        assert!(
            response
                .symbols
                .iter()
                .any(|symbol| symbol.kind == "inherits_from"
                    && symbol.name.as_deref() == Some("ControllerBase"))
        );
        assert!(
            response
                .symbols
                .iter()
                .any(|symbol| symbol.kind == "implements_interface"
                    && symbol.name.as_deref() == Some("IUserController"))
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn read_skeleton_rejects_outside_root() {
        let root = temp_root("skeleton_outside");
        fs::write(root.join("app.rs"), "fn main() {}\n").expect("inside file should be written");
        let outside_root = temp_root("skeleton_outside_target");
        let outside = outside_root.join("outside.rs");
        fs::write(&outside, "class Outside {}\n").expect("outside file should be written");
        let state = AppState::from_directory(&root).expect("state should index temp root");

        let Json(response) = read_skeleton(
            State(state),
            Json(ReadSkeletonRequest {
                filepath: outside.display().to_string(),
            }),
        )
        .await;

        assert_eq!(response.status, "error");
        assert!(response.symbols.is_empty());
        assert!(
            response
                .warnings
                .contains(&"Requested file is outside the indexed root or does not exist.")
        );
        fs::remove_dir_all(root).ok();
        fs::remove_dir_all(outside_root).ok();
    }

    #[tokio::test]
    async fn scan_signals_returns_builtin_signal_matches() {
        let root = temp_root("signal_scan");
        fs::write(
            root.join("Repository.cs"),
            "using System.Data.SqlClient;\nusing var conn = new SqlConnection(value);\n",
        )
        .expect("sample file should be written");
        let state = AppState::from_directory(&root).expect("state should index temp root");

        let Json(response) = scan_signals(
            State(state),
            Json(SignalScanRequest {
                categories: vec!["data".to_string()],
                top_k_per_signal: Some(3),
                include_snippets: Some(true),
            }),
        )
        .await;

        assert_eq!(response.status, "ok");
        assert_eq!(response.total_matches, 1);
        assert_eq!(response.signals.len(), 1);
        assert_eq!(response.signals[0].id, "sql_connection");
        assert_eq!(response.signals[0].matches[0].line_start, 2);
        assert!(
            response.signals[0].matches[0]
                .snippet
                .as_deref()
                .unwrap_or_default()
                .contains("SqlConnection")
        );
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn scan_signals_can_omit_snippets() {
        let root = temp_root("signal_scan_no_snippets");
        fs::write(
            root.join("Repository.cs"),
            "catch (Exception ex) { throw; }\n",
        )
        .expect("sample file should be written");
        let state = AppState::from_directory(&root).expect("state should index temp root");

        let Json(response) = scan_signals(
            State(state),
            Json(SignalScanRequest {
                categories: vec!["reliability".to_string()],
                top_k_per_signal: Some(3),
                include_snippets: Some(false),
            }),
        )
        .await;

        assert_eq!(response.status, "ok");
        assert_eq!(response.total_matches, 1);
        assert!(response.signals[0].matches[0].snippet.is_none());
        fs::remove_dir_all(root).ok();
    }
}
