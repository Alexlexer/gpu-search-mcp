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
    discover_files, file_ext, parse_csharp_ast_summary, search_files,
};
use serde::{Deserialize, Serialize};
use std::fs;
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
    pub signal_scan: bool,
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

/// Experimental Rust `/read/block` request.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ReadBlockRequest {
    pub filepath: String,
    #[serde(default)]
    pub line_start: Option<usize>,
    #[serde(default)]
    pub line_end: Option<usize>,
}

/// Experimental Rust `/read/block` response.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ReadBlockResponse {
    pub status: &'static str,
    pub file: String,
    pub absolute_file: Option<String>,
    pub line_start: usize,
    pub line_end: usize,
    pub content: String,
    pub warnings: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}

/// Experimental Rust `/read/skeleton` request.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ReadSkeletonRequest {
    pub filepath: String,
}

/// A single symbol-like skeleton item.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SkeletonItem {
    pub kind: String,
    pub name: Option<String>,
    pub line: usize,
}

/// Experimental Rust `/read/skeleton` response.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ReadSkeletonResponse {
    pub status: &'static str,
    pub file: String,
    pub absolute_file: Option<String>,
    pub symbols: Vec<SkeletonItem>,
    pub warnings: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}

/// Experimental Rust `/scan/signals` request.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SignalScanRequest {
    #[serde(default)]
    pub categories: Vec<String>,
    #[serde(default)]
    pub top_k_per_signal: Option<usize>,
    #[serde(default)]
    pub include_snippets: Option<bool>,
}

/// A matched signal occurrence.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SignalMatchResponse {
    pub file: String,
    pub line_start: usize,
    pub line_end: usize,
    pub snippet: Option<String>,
}

/// A signal and its matches.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SignalResultResponse {
    pub id: &'static str,
    pub category: &'static str,
    pub label: &'static str,
    pub query: &'static str,
    pub matches: Vec<SignalMatchResponse>,
}

/// Experimental Rust `/scan/signals` response.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SignalScanResponse {
    pub status: &'static str,
    pub signals: Vec<SignalResultResponse>,
    pub total_matches: usize,
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
        .route("/search/hybrid", post(search_hybrid))
        .route("/search/semantic", post(search_semantic))
        .route("/read/block", post(read_block))
        .route("/read/skeleton", post(read_skeleton))
        .route("/dependency/impact", post(dependency_impact))
        .route("/scan/signals", post(scan_signals))
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

/// Scan indexed files for a small built-in set of advisory code signals.
pub async fn scan_signals(
    State(state): State<AppState>,
    Json(request): Json<SignalScanRequest>,
) -> Json<SignalScanResponse> {
    if state.files.is_empty() {
        return Json(SignalScanResponse {
            status: "not_ready",
            signals: Vec::new(),
            total_matches: 0,
            warnings: vec!["No repository is indexed by the Rust HTTP server yet."],
            limitations: signal_limitations(),
        });
    }

    let top_k = request.top_k_per_signal.unwrap_or(5).clamp(1, 20);
    let include_snippets = request.include_snippets.unwrap_or(true);
    let requested_categories: Vec<String> = request
        .categories
        .iter()
        .map(|category| category.to_ascii_lowercase())
        .collect();
    let mut signals = Vec::new();

    for signal in builtin_signals() {
        if !requested_categories.is_empty()
            && !requested_categories
                .iter()
                .any(|category| category == signal.category)
        {
            continue;
        }

        let options = PatternSearchOptions {
            max_results: top_k,
            context_mode: ContextMode::Compact,
            case_sensitive: false,
            ..PatternSearchOptions::default()
        };
        let matches = search_files(&state.files, signal.query, &options)
            .unwrap_or_default()
            .into_iter()
            .map(|matched| SignalMatchResponse {
                file: display_relative(&state, &matched.file),
                line_start: matched.line,
                line_end: matched.line,
                snippet: include_snippets.then_some(matched.snippet),
            })
            .collect::<Vec<_>>();

        if !matches.is_empty() {
            signals.push(SignalResultResponse {
                id: signal.id,
                category: signal.category,
                label: signal.label,
                query: signal.query,
                matches,
            });
        }
    }

    let total_matches = signals
        .iter()
        .map(|signal| signal.matches.len())
        .sum::<usize>();

    Json(SignalScanResponse {
        status: "ok",
        signals,
        total_matches,
        warnings: Vec::new(),
        limitations: signal_limitations(),
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

/// Read a bounded source block from inside the indexed root.
pub async fn read_block(
    State(state): State<AppState>,
    Json(request): Json<ReadBlockRequest>,
) -> Json<ReadBlockResponse> {
    let Some(path) = resolve_under_indexed_root(&state, &request.filepath) else {
        return Json(ReadBlockResponse {
            status: "error",
            file: request.filepath,
            absolute_file: None,
            line_start: 0,
            line_end: 0,
            content: String::new(),
            warnings: vec!["Requested file is outside the indexed root or does not exist."],
            limitations: read_limitations(),
        });
    };

    let Ok(text) = fs::read_to_string(&path) else {
        return Json(ReadBlockResponse {
            status: "error",
            file: display_relative(&state, &path),
            absolute_file: Some(path.display().to_string()),
            line_start: 0,
            line_end: 0,
            content: String::new(),
            warnings: vec!["Requested file could not be read as UTF-8 text."],
            limitations: read_limitations(),
        });
    };

    let lines: Vec<&str> = text.lines().collect();
    let total_lines = lines.len().max(1);
    let line_start = request.line_start.unwrap_or(1).clamp(1, total_lines);
    let line_end = request
        .line_end
        .unwrap_or(line_start.saturating_add(40))
        .clamp(line_start, total_lines);
    let content = lines[(line_start - 1)..line_end].join("\n");

    Json(ReadBlockResponse {
        status: "ok",
        file: display_relative(&state, &path),
        absolute_file: Some(path.display().to_string()),
        line_start,
        line_end,
        content,
        warnings: Vec::new(),
        limitations: read_limitations(),
    })
}

/// Read a lightweight symbol skeleton from inside the indexed root.
pub async fn read_skeleton(
    State(state): State<AppState>,
    Json(request): Json<ReadSkeletonRequest>,
) -> Json<ReadSkeletonResponse> {
    let Some(path) = resolve_under_indexed_root(&state, &request.filepath) else {
        return Json(ReadSkeletonResponse {
            status: "error",
            file: request.filepath,
            absolute_file: None,
            symbols: Vec::new(),
            warnings: vec!["Requested file is outside the indexed root or does not exist."],
            limitations: skeleton_limitations(),
        });
    };

    let Ok(text) = fs::read_to_string(&path) else {
        return Json(ReadSkeletonResponse {
            status: "error",
            file: display_relative(&state, &path),
            absolute_file: Some(path.display().to_string()),
            symbols: Vec::new(),
            warnings: vec!["Requested file could not be read as UTF-8 text."],
            limitations: skeleton_limitations(),
        });
    };

    let mut warnings = Vec::new();
    let symbols = if file_ext(&path).as_deref() == Some(".cs") {
        match parse_csharp_ast_summary(&text) {
            Ok(items) => items
                .into_iter()
                .map(|item| SkeletonItem {
                    kind: item.kind,
                    name: item.name,
                    line: item.line,
                })
                .collect(),
            Err(_) => {
                warnings
                    .push("C# Tree-sitter skeleton parsing failed; returned fallback skeleton.");
                fallback_skeleton(&text)
            }
        }
    } else {
        warnings
            .push("Tree-sitter skeleton support is currently C#-only; returned fallback skeleton.");
        fallback_skeleton(&text)
    };

    Json(ReadSkeletonResponse {
        status: "ok",
        file: display_relative(&state, &path),
        absolute_file: Some(path.display().to_string()),
        symbols,
        warnings,
        limitations: skeleton_limitations(),
    })
}

/// Return structured pattern-search results from the experimental Rust core.
pub async fn search_code(
    State(state): State<AppState>,
    Json(request): Json<SearchCodeRequest>,
) -> Json<SearchCodeResponse> {
    run_search(state, request, None)
}

/// Return structured hybrid-search results from the experimental Rust HTTP API.
pub async fn search_hybrid(
    State(state): State<AppState>,
    Json(request): Json<SearchCodeRequest>,
) -> Json<SearchCodeResponse> {
    run_search(state, request, Some(SearchMode::Hybrid))
}

/// Return a structured not-ready semantic-search response from the experimental Rust HTTP API.
pub async fn search_semantic(
    State(state): State<AppState>,
    Json(request): Json<SearchCodeRequest>,
) -> Json<SearchCodeResponse> {
    run_search(state, request, Some(SearchMode::Semantic))
}

fn run_search(
    state: AppState,
    request: SearchCodeRequest,
    forced_mode: Option<SearchMode>,
) -> Json<SearchCodeResponse> {
    let mode = request.mode.unwrap_or(SearchMode::Auto);
    let mode = forced_mode.unwrap_or(mode);
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

fn read_limitations() -> Vec<&'static str> {
    vec![
        "Rust HTTP read/block is experimental.",
        "Python HTTP/MCP runtime remains authoritative.",
    ]
}

fn skeleton_limitations() -> Vec<&'static str> {
    vec![
        "Rust HTTP read/skeleton is experimental.",
        "C# skeleton uses Tree-sitter and remains best-effort.",
        "Python HTTP/MCP runtime remains authoritative.",
    ]
}

fn signal_limitations() -> Vec<&'static str> {
    vec![
        "Rust HTTP signal scan is experimental.",
        "Signal definitions are a small built-in subset.",
        "Python HTTP/MCP runtime remains authoritative.",
    ]
}

#[derive(Debug, Clone, Copy)]
struct BuiltinSignal {
    id: &'static str,
    category: &'static str,
    label: &'static str,
    query: &'static str,
}

fn builtin_signals() -> Vec<BuiltinSignal> {
    vec![
        BuiltinSignal {
            id: "web_config",
            category: "configuration",
            label: "Web configuration file",
            query: "web.config",
        },
        BuiltinSignal {
            id: "package_config",
            category: "configuration",
            label: "Package configuration",
            query: "packages.config",
        },
        BuiltinSignal {
            id: "sql_connection",
            category: "data",
            label: "SQL connection usage",
            query: "SqlConnection",
        },
        BuiltinSignal {
            id: "catch_exception",
            category: "reliability",
            label: "Generic exception catch",
            query: "catch (Exception",
        },
    ]
}

fn fallback_skeleton(text: &str) -> Vec<SkeletonItem> {
    text.lines()
        .enumerate()
        .filter_map(|(idx, line)| {
            let trimmed = line.trim_start();
            let (kind, rest) = trimmed
                .strip_prefix("class ")
                .map(|rest| ("class", rest))
                .or_else(|| trimmed.strip_prefix("def ").map(|rest| ("function", rest)))
                .or_else(|| {
                    trimmed
                        .strip_prefix("function ")
                        .map(|rest| ("function", rest))
                })?;
            let name = rest
                .split(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
                .next()
                .filter(|value| !value.is_empty())
                .map(str::to_string);
            Some(SkeletonItem {
                kind: kind.to_string(),
                name,
                line: idx + 1,
            })
        })
        .collect()
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
            "using MyApp.Services;\nnamespace MyApp.Controllers;\npublic class UserController {\n    public string GetUser() => \"ok\";\n}\n",
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
