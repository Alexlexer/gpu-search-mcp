use axum::{Json, extract::State};
use gpu_search_core::{ContextMode, PatternSearchOptions, search_files};

use crate::{
    AppState, SearchCodeRequest, SearchCodeResponse, SearchContextMode, SearchMode,
    SearchResultItem,
};

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
