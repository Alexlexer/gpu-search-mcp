//! Experimental Rust MCP search tool handler.

use gpu_search_core::{
    ContextMode, IndexOptions, PatternSearchOptions, discover_files, search_files,
};
use serde_json::{Value, json};

pub(crate) fn rust_search_code_tool_result(arguments: &Value) -> Value {
    let directory = arguments
        .get("directory")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let query = arguments
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let top_k = arguments
        .get("topK")
        .or_else(|| arguments.get("top_k"))
        .and_then(Value::as_u64)
        .unwrap_or(5)
        .clamp(1, 50) as usize;

    if directory.trim().is_empty() || query.trim().is_empty() {
        return json!({
            "isError": true,
            "content": [
                {
                    "type": "text",
                    "text": "rust_search_code requires non-empty directory and query arguments."
                }
            ]
        });
    }

    let files = match discover_files(directory, &IndexOptions::default()) {
        Ok(files) => files,
        Err(error) => {
            return json!({
                "isError": true,
                "content": [
                    {
                        "type": "text",
                        "text": format!("Failed to discover files: {error}")
                    }
                ]
            });
        }
    };

    let options = PatternSearchOptions {
        max_results: top_k,
        context_mode: ContextMode::Compact,
        ..PatternSearchOptions::default()
    };
    let matches = match search_files(&files, query, &options) {
        Ok(matches) => matches,
        Err(error) => {
            return json!({
                "isError": true,
                "content": [
                    {
                        "type": "text",
                        "text": format!("Rust pattern search failed: {error}")
                    }
                ]
            });
        }
    };

    let structured_results: Vec<Value> = matches
        .iter()
        .map(|matched| {
            json!({
                "file": matched.file.display().to_string(),
                "lineStart": matched.line,
                "lineEnd": matched.line,
                "reason": "exact token match",
                "snippet": matched.snippet
            })
        })
        .collect();
    let text = if structured_results.is_empty() {
        format!("No Rust pattern matches for '{query}'.")
    } else {
        format!(
            "Rust pattern search found {} match(es) for '{}'.",
            structured_results.len(),
            query
        )
    };

    json!({
    "content": [
        {
            "type": "text",
            "text": text
        }
    ],
    "structuredContent": {
        "query": query,
        "topK": top_k,
        "results": structured_results,
        "limitations": [
            "Experimental Rust MCP pattern search only.",
            "Python MCP runtime remains authoritative.",
            "Semantic search is not available through the Rust MCP scaffold."
        ]
    }
    })
}
