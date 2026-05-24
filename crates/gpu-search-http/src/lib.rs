//! Experimental Rust HTTP server scaffold.
//!
//! This crate is intentionally additive and does not replace the current
//! Python HTTP/MCP runtime. It starts with a small `/health` route so the Rust
//! API surface can grow in focused PRs while preserving existing behavior.

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use gpu_search_core::{
    ContextMode, DEFAULT_INDEXED_EXTS, DEFAULT_SKIP_DIRS, DEPENDENCY_ANALYSIS_MODE,
    DependencyGraph, DiscoveredFile, IndexOptions, PatternSearchOptions, RUST_CORE_VERSION,
    discover_files, search_files,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Experimental Rust HTTP crate version.
pub const RUST_HTTP_VERSION: &str = "0.1.0-prototype";

/// Shared state for the experimental Rust HTTP server.
#[derive(Debug, Clone, Default)]
pub struct AppState {
    indexed_root: Option<PathBuf>,
    files: Vec<DiscoveredFile>,
    dependency_graph: Option<DependencyGraph>,
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

/// Device metadata for the experimental Rust HTTP diagnostics endpoint.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticsDevice {
    pub backend: &'static str,
    pub reason: &'static str,
    pub warnings: Vec<&'static str>,
}

/// Index readiness metadata for the experimental Rust HTTP diagnostics endpoint.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticsIndexes {
    pub pattern_ready: bool,
    pub semantic_ready: bool,
    pub dependency_ready: bool,
    pub indexed_files: usize,
}

/// Lightweight diagnostics response for the experimental Rust HTTP server.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticsResponse {
    pub status: &'static str,
    pub implementation: &'static str,
    pub rust_core_version: &'static str,
    pub rust_http_version: &'static str,
    pub device: DiagnosticsDevice,
    pub indexes: DiagnosticsIndexes,
    pub capabilities: CapabilityResponse,
    pub warnings: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}

/// Search mode requested by clients. Kept stringly-compatible with Python HTTP.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    Auto,
    Pattern,
    Semantic,
    Hybrid,
}

/// Context mode requested by clients. Kept stringly-compatible with Python HTTP.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchContextMode {
    Compact,
    Normal,
    Full,
}

/// Experimental Rust `/search/code` request.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SearchCodeRequest {
    pub query: String,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub mode: Option<SearchMode>,
    #[serde(default)]
    pub context_mode: Option<SearchContextMode>,
}

/// Structured search result placeholder matching current DTO concepts.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SearchResultItem {
    pub file: String,
    pub line_start: usize,
    pub line_end: usize,
    pub score: Option<String>,
    pub reason: String,
    pub snippet: String,
}

/// Experimental Rust `/search/code` response.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SearchCodeResponse {
    pub result: String,
    pub results: Vec<SearchResultItem>,
    pub query: String,
    pub mode: SearchMode,
    pub context_mode: SearchContextMode,
    pub top_k: usize,
    pub status: &'static str,
    pub warnings: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}

/// Experimental Rust `/dependency/impact` request.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DependencyImpactRequest {
    pub filepath: String,
}

/// Structured impacted-file item for Rust dependency impact.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ImpactedFileResponse {
    pub file: String,
    pub absolute_file: String,
    pub hops: usize,
    pub reason: Option<String>,
}

/// Experimental Rust `/dependency/impact` response.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DependencyImpactResponse {
    pub status: &'static str,
    pub file: String,
    pub impacted_files: Vec<ImpactedFileResponse>,
    pub confidence: &'static str,
    pub analysis_mode: &'static str,
    pub warnings: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}

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
        .route("/search/code", post(search_code))
        .route("/dependency/impact", post(dependency_impact))
        .with_state(state)
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

/// Return heuristic dependency impact results from the experimental Rust core.
pub async fn dependency_impact(
    State(state): State<AppState>,
    Json(request): Json<DependencyImpactRequest>,
) -> Json<DependencyImpactResponse> {
    let Some(graph) = &state.dependency_graph else {
        return Json(DependencyImpactResponse {
            status: "not_ready",
            file: request.filepath,
            impacted_files: Vec::new(),
            confidence: "low",
            analysis_mode: DEPENDENCY_ANALYSIS_MODE,
            warnings: vec!["Dependency graph is not ready in the Rust HTTP server."],
            limitations: dependency_limitations(),
        });
    };

    let Some(changed_file) = resolve_under_indexed_root(&state, &request.filepath) else {
        return Json(DependencyImpactResponse {
            status: "error",
            file: request.filepath,
            impacted_files: Vec::new(),
            confidence: "low",
            analysis_mode: DEPENDENCY_ANALYSIS_MODE,
            warnings: vec!["Requested file is outside the indexed root or does not exist."],
            limitations: dependency_limitations(),
        });
    };

    let impacted_files = graph
        .impact(&changed_file)
        .into_iter()
        .map(|impacted| ImpactedFileResponse {
            file: display_relative(&state, &impacted.file),
            absolute_file: impacted.file.display().to_string(),
            hops: impacted.hops,
            reason: impacted.reason,
        })
        .collect::<Vec<_>>();

    Json(DependencyImpactResponse {
        status: "ok",
        file: display_relative(&state, &changed_file),
        impacted_files,
        confidence: "medium",
        analysis_mode: DEPENDENCY_ANALYSIS_MODE,
        warnings: Vec::new(),
        limitations: dependency_limitations(),
    })
}

/// Return structured pattern-search results from the experimental Rust core.
pub async fn search_code(
    State(state): State<AppState>,
    Json(request): Json<SearchCodeRequest>,
) -> Json<SearchCodeResponse> {
    let mode = request.mode.unwrap_or(SearchMode::Auto);
    let context_mode = request.context_mode.unwrap_or(SearchContextMode::Normal);
    let top_k = request.top_k.unwrap_or(5);
    let query = request.query;

    if state.files.is_empty() {
        return Json(SearchCodeResponse {
            result:
                "Rust HTTP search is not ready yet. Start with --directory to enable pattern search."
                    .to_string(),
            results: Vec::new(),
            query,
            mode,
            context_mode,
            top_k,
            status: "not_ready",
            warnings: vec!["No repository is indexed by the Rust HTTP server yet."],
            limitations: vec![
                "Experimental Rust HTTP scaffold only.",
                "Python HTTP/MCP runtime remains authoritative.",
            ],
        });
    }

    if matches!(mode, SearchMode::Semantic) {
        return Json(SearchCodeResponse {
            result: "Rust HTTP semantic search is not implemented yet.".to_string(),
            results: Vec::new(),
            query,
            mode,
            context_mode,
            top_k,
            status: "not_ready",
            warnings: vec!["Semantic search remains available through the Python runtime."],
            limitations: vec![
                "Rust HTTP currently supports pattern search only.",
                "Python HTTP/MCP runtime remains authoritative.",
            ],
        });
    }

    let options = PatternSearchOptions {
        max_results: top_k,
        context_mode: context_mode.to_core(),
        ..PatternSearchOptions::default()
    };

    let matches = match search_files(&state.files, &query, &options) {
        Ok(matches) => matches,
        Err(_err) => {
            return Json(SearchCodeResponse {
                result: "Rust HTTP pattern search failed while reading an indexed file."
                    .to_string(),
                results: Vec::new(),
                query,
                mode,
                context_mode,
                top_k,
                status: "error",
                warnings: vec!["Search failed while reading an indexed file."],
                limitations: vec!["Experimental Rust HTTP scaffold only."],
            });
        }
    };

    let results: Vec<SearchResultItem> = matches
        .into_iter()
        .map(|matched| SearchResultItem {
            file: matched.file.display().to_string(),
            line_start: matched.line,
            line_end: matched.line,
            score: Some("1.0".to_string()),
            reason: matched.reason,
            snippet: matched.snippet,
        })
        .collect();
    let is_hybrid = matches!(mode, SearchMode::Hybrid);

    Json(SearchCodeResponse {
        result: if results.is_empty() {
            format!("No matches for '{query}'")
        } else {
            format!("{} Rust pattern match(es) for '{query}'", results.len())
        },
        results,
        query,
        mode,
        context_mode,
        top_k,
        status: "ok",
        warnings: if is_hybrid {
            vec!["Hybrid mode currently returns Rust pattern results only."]
        } else {
            Vec::new()
        },
        limitations: vec![
            "Experimental Rust HTTP scaffold only.",
            "Rust HTTP currently supports pattern search only.",
            "Python HTTP/MCP runtime remains authoritative.",
        ],
    })
}

impl SearchContextMode {
    fn to_core(&self) -> ContextMode {
        match self {
            Self::Compact => ContextMode::Compact,
            Self::Normal => ContextMode::Normal,
            Self::Full => ContextMode::Full,
        }
    }
}

fn dependency_limitations() -> Vec<&'static str> {
    vec![
        "Dependency impact is heuristic, not compiler-accurate.",
        "Rust HTTP dependency impact is experimental.",
        "Python HTTP/MCP runtime remains authoritative.",
    ]
}

fn resolve_under_indexed_root(state: &AppState, filepath: &str) -> Option<PathBuf> {
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

fn display_relative(state: &AppState, filepath: &Path) -> String {
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
    use axum::extract::State;
    use axum::http::{Request, StatusCode};
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
        assert!(
            response
                .warnings
                .contains(&"No repository is indexed by the Rust HTTP server yet.")
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
}
