//! Experimental Rust MCP signal-scan tool handler.

use gpu_search_core::{
    ContextMode, IndexOptions, PatternSearchOptions, discover_files, search_files,
};
use serde_json::{Map, Value, json};
use std::path::Path;

use super::common::{display_relative, tool_error};

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

pub(crate) fn rust_scan_signals_tool_result(arguments: &Value) -> Value {
    let directory = arguments
        .get("directory")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let categories = arguments
        .get("categories")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_ascii_lowercase)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let top_k_per_signal = arguments
        .get("topKPerSignal")
        .or_else(|| arguments.get("top_k_per_signal"))
        .and_then(Value::as_u64)
        .unwrap_or(5)
        .clamp(1, 20) as usize;
    let include_snippets = arguments
        .get("includeSnippets")
        .or_else(|| arguments.get("include_snippets"))
        .and_then(Value::as_bool)
        .unwrap_or(true);

    if directory.trim().is_empty() {
        return tool_error("rust_scan_signals requires a non-empty directory argument.");
    }

    let root = match Path::new(directory).canonicalize() {
        Ok(root) => root,
        Err(error) => return tool_error(&format!("Failed to resolve directory: {error}")),
    };
    let files = match discover_files(&root, &IndexOptions::default()) {
        Ok(files) => files,
        Err(error) => return tool_error(&format!("Failed to discover files: {error}")),
    };
    let options = PatternSearchOptions {
        case_sensitive: false,
        max_results: top_k_per_signal,
        context_mode: ContextMode::Compact,
        ..PatternSearchOptions::default()
    };

    let mut total_matches = 0usize;
    let signals: Vec<Value> = builtin_signals()
        .into_iter()
        .filter(|signal| {
            categories.is_empty()
                || categories
                    .iter()
                    .any(|category| category == signal.category)
        })
        .map(|signal| {
            let matches = search_files(&files, signal.query, &options).unwrap_or_default();
            total_matches += matches.len();
            let match_values = matches
                .iter()
                .map(|matched| {
                    let mut object = Map::new();
                    object.insert(
                        "file".to_string(),
                        json!(display_relative(directory, &matched.file)),
                    );
                    object.insert("lineStart".to_string(), json!(matched.line));
                    object.insert("lineEnd".to_string(), json!(matched.line));
                    if include_snippets {
                        object.insert("snippet".to_string(), json!(matched.snippet));
                    }
                    Value::Object(object)
                })
                .collect::<Vec<_>>();

            json!({
                "id": signal.id,
                "category": signal.category,
                "label": signal.label,
                "query": signal.query,
                "matches": match_values
            })
        })
        .collect();

    json!({
        "content": [
            {
                "type": "text",
                "text": format!(
                    "Rust signal scan found {total_matches} match(es) across {} signal(s).",
                    signals.len()
                )
            }
        ],
        "structuredContent": {
            "root": root.display().to_string(),
            "signals": signals,
            "totalMatches": total_matches,
            "limitations": [
                "Experimental Rust MCP signal scan only.",
                "Signal scan is heuristic and search-based.",
                "Python MCP runtime remains authoritative."
            ]
        }
    })
}
