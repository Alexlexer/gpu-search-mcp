use axum::{Json, extract::State};
use gpu_search_core::{ContextMode, PatternSearchOptions, search_files};

use crate::{
    AppState, SignalMatchResponse, SignalResultResponse, SignalScanRequest, SignalScanResponse,
    display_relative,
};

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

fn signal_limitations() -> Vec<&'static str> {
    vec![
        "Rust HTTP signal scan is experimental.",
        "Signal definitions are a small built-in subset.",
        "Python HTTP/MCP runtime remains authoritative.",
    ]
}
