//! Experimental Rust MCP tool handlers.

pub(crate) mod dependency;
pub(crate) mod read;
pub(crate) mod search;
pub(crate) mod signals;

use crate::{DEFAULT_SEMANTIC_MODEL_ID, RUST_MCP_VERSION};
use gpu_search_core::{DEPENDENCY_ANALYSIS_MODE, RUST_CORE_VERSION, parse_csharp_regex_summary};
use serde_json::{Value, json};
use std::env;
use std::path::{Path, PathBuf};

pub(crate) use dependency::rust_dependency_impact_tool_result;
pub(crate) use read::{rust_read_block_tool_result, rust_read_skeleton_tool_result};
pub(crate) use search::rust_search_code_tool_result;
pub(crate) use signals::rust_scan_signals_tool_result;

pub(crate) fn rust_semantic_model_status_tool_result(arguments: &Value) -> Value {
    let model_id = arguments
        .get("modelId")
        .or_else(|| arguments.get("model_id"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(resolve_semantic_model_id);
    let status = semantic_model_status_snapshot(&model_id);

    json!({
        "content": [
            {
                "type": "text",
                "text": status["message"].as_str().unwrap_or("Rust semantic model status is advisory.").to_string()
            }
        ],
        "structuredContent": status
    })
}

fn resolve_semantic_model_id() -> String {
    env::var("GPU_SEARCH_SEMANTIC_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_SEMANTIC_MODEL_ID.to_string())
}

pub(crate) fn semantic_model_status_snapshot(model_id: &str) -> Value {
    json!({
        "modelId": model_id,
        "provider": "sentence-transformers",
        "available": false,
        "cached": false,
        "requiresDownload": true,
        "device": "unavailable",
        "message": format!(
            "Rust MCP does not load sentence-transformers models yet. Use the Python runtime or run: gpu-search-mcp --semantic-model {model_id} --download-semantic-model"
        ),
        "limitations": [
            "Rust MCP semantic model status is advisory only.",
            "Rust semantic search is not implemented yet.",
            "Python MCP runtime remains authoritative for sentence-transformers embeddings."
        ]
    })
}

pub(crate) fn rust_get_diagnostics_tool_result() -> Value {
    let model_id = resolve_semantic_model_id();
    let semantic_model = semantic_model_status_snapshot(&model_id);
    let diagnostics = json!({
        "version": RUST_MCP_VERSION,
        "status": "degraded",
        "implementation": "rust-mcp-scaffold",
        "rustCoreVersion": RUST_CORE_VERSION,
        "rustMcpVersion": RUST_MCP_VERSION,
        "device": {
            "backend": "unavailable",
            "torchDevice": Value::Null,
            "reason": "Rust MCP scaffold does not select CUDA/MPS/CPU devices yet.",
            "warnings": [
                "Python MCP runtime remains authoritative for device selection."
            ]
        },
        "indexedRoots": [],
        "indexes": {
            "pattern": {
                "ready": false,
                "fileCount": 0,
                "cacheStatus": "not_loaded"
            },
            "semantic": {
                "ready": false,
                "chunkCount": 0,
                "modelId": semantic_model["modelId"],
                "modelAvailable": semantic_model["available"],
                "message": semantic_model["message"]
            },
            "dependency": {
                "ready": false,
                "analysisMode": DEPENDENCY_ANALYSIS_MODE,
                "confidence": "low"
            }
        },
        "cache": {
            "directory": Value::Null,
            "schemaVersion": Value::Null,
            "entries": []
        },
        "semanticModel": semantic_model,
        "capabilities": {
            "patternSearch": true,
            "semanticSearch": false,
            "hybridSearch": false,
            "dependencyImpact": true,
            "signalScan": true,
            "mcpTools": true
        },
        "warnings": [
            "Rust MCP diagnostics are scaffold-only and do not inspect runtime Python state.",
            "No indexed root is retained between Rust MCP tool calls."
        ],
        "limitations": [
            "Rust MCP diagnostics do not trigger indexing, scans, model loads, or downloads.",
            "Python MCP runtime remains authoritative.",
            "Dependency impact is heuristic, not compiler-accurate."
        ]
    });

    json!({
        "content": [
            {
                "type": "text",
                "text": "Rust MCP diagnostics: scaffold is experimental; Python MCP remains authoritative."
            }
        ],
        "structuredContent": diagnostics
    })
}

fn csharp_regex_skeleton(text: &str) -> Vec<Value> {
    parse_csharp_regex_summary(text)
        .into_iter()
        .map(|item| {
            json!({
                "kind": item.kind,
                "name": item.name,
                "line": item.line
            })
        })
        .collect()
}

fn fallback_skeleton(text: &str) -> Vec<Value> {
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
                .filter(|value| !value.is_empty());
            Some(json!({
                "kind": kind,
                "name": name,
                "line": idx + 1
            }))
        })
        .collect()
}

fn resolve_under_root(directory: &str, filepath: &str) -> Option<PathBuf> {
    let root = Path::new(directory).canonicalize().ok()?;
    let candidate = Path::new(filepath);
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root.join(candidate)
    };
    let resolved = joined.canonicalize().ok()?;
    resolved.starts_with(root).then_some(resolved)
}

fn display_relative(directory: &str, filepath: &Path) -> String {
    Path::new(directory)
        .canonicalize()
        .ok()
        .and_then(|root| filepath.strip_prefix(root).ok().map(Path::to_path_buf))
        .unwrap_or_else(|| filepath.to_path_buf())
        .display()
        .to_string()
}

fn tool_error(message: &str) -> Value {
    json!({
        "isError": true,
        "content": [
            {
                "type": "text",
                "text": message
            }
        ]
    })
}
