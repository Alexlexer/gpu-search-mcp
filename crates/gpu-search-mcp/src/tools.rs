//! Experimental Rust MCP tool handlers.

pub(crate) mod search;

use crate::{DEFAULT_SEMANTIC_MODEL_ID, RUST_MCP_VERSION};
use gpu_search_core::{
    ContextMode, DEPENDENCY_ANALYSIS_MODE, DependencyGraph, IndexOptions, LineIndex,
    PatternSearchOptions, RUST_CORE_VERSION, discover_files, file_ext, parse_csharp_ast_summary,
    parse_csharp_regex_summary, search_files,
};
use serde_json::{Map, Value, json};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) use search::rust_search_code_tool_result;

pub(crate) fn rust_read_block_tool_result(arguments: &Value) -> Value {
    let directory = arguments
        .get("directory")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let filepath = arguments
        .get("filepath")
        .or_else(|| arguments.get("file"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let line_start = arguments
        .get("lineStart")
        .or_else(|| arguments.get("line_start"))
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1) as usize;
    let line_end = arguments
        .get("lineEnd")
        .or_else(|| arguments.get("line_end"))
        .and_then(Value::as_u64)
        .unwrap_or(line_start as u64)
        .max(line_start as u64) as usize;

    if directory.trim().is_empty() || filepath.trim().is_empty() {
        return tool_error("rust_read_block requires non-empty directory and filepath arguments.");
    }

    let Some(path) = resolve_under_root(directory, filepath) else {
        return tool_error("Requested file is outside the supplied directory or does not exist.");
    };

    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => return tool_error(&format!("Failed to read file: {error}")),
    };
    if String::from_utf8(bytes.clone()).is_err() {
        return tool_error("Requested file could not be read as UTF-8 text.");
    }

    let line_index = LineIndex::new(&bytes);
    let bounded_start = line_start.min(line_index.line_count().max(1));
    let bounded_end = line_end.min(line_index.line_count().max(bounded_start));
    let Some((content_start, _)) = line_index.line_range(&bytes, bounded_start) else {
        return tool_error("Requested start line is outside the file.");
    };
    let Some((_, content_end)) = line_index.line_range(&bytes, bounded_end) else {
        return tool_error("Requested end line is outside the file.");
    };
    let content = String::from_utf8_lossy(&bytes[content_start..content_end])
        .replace("\r\n", "\n")
        .to_string();
    let display_file = path.display().to_string();
    let returned_lines = bounded_end.saturating_sub(bounded_start) + 1;
    let text_summary = format!(
        "Rust read block returned {} line(s) from {}.",
        returned_lines, display_file
    );

    json!({
        "content": [
            {
                "type": "text",
                "text": text_summary
            }
        ],
        "structuredContent": {
            "file": display_file,
            "lineStart": bounded_start,
            "lineEnd": bounded_end,
            "content": content,
            "limitations": [
                "Experimental Rust MCP read block only.",
                "Python MCP runtime remains authoritative.",
                "Reads are bounded to UTF-8 files under the explicit directory argument."
            ]
        }
    })
}

pub(crate) fn rust_dependency_impact_tool_result(arguments: &Value) -> Value {
    let directory = arguments
        .get("directory")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let filepath = arguments
        .get("filepath")
        .or_else(|| arguments.get("file"))
        .and_then(Value::as_str)
        .unwrap_or_default();

    if directory.trim().is_empty() || filepath.trim().is_empty() {
        return tool_error(
            "rust_dependency_impact requires non-empty directory and filepath arguments.",
        );
    }

    let Some(changed_file) = resolve_under_root(directory, filepath) else {
        return tool_error("Requested file is outside the supplied directory or does not exist.");
    };

    let root = match Path::new(directory).canonicalize() {
        Ok(root) => root,
        Err(error) => return tool_error(&format!("Failed to resolve directory: {error}")),
    };
    let files = match discover_files(&root, &IndexOptions::default()) {
        Ok(files) => files,
        Err(error) => return tool_error(&format!("Failed to discover files: {error}")),
    };
    let graph = match DependencyGraph::from_files(&files) {
        Ok(graph) => graph,
        Err(error) => return tool_error(&format!("Failed to build dependency graph: {error}")),
    };

    let impacted_files: Vec<Value> = graph
        .impact(&changed_file)
        .into_iter()
        .map(|impacted| {
            json!({
                "file": display_relative(directory, &impacted.file),
                "absoluteFile": impacted.file.display().to_string(),
                "hops": impacted.hops,
                "reason": impacted.reason
            })
        })
        .collect();
    let changed_display = display_relative(directory, &changed_file);
    let text = format!(
        "Rust dependency impact found {} impacted file(s) for {}.",
        impacted_files.len(),
        changed_display
    );

    json!({
        "content": [
            {
                "type": "text",
                "text": text
            }
        ],
        "structuredContent": {
            "file": changed_display,
            "absoluteFile": changed_file.display().to_string(),
            "impactedFiles": impacted_files,
            "confidence": "medium",
            "analysisMode": DEPENDENCY_ANALYSIS_MODE,
            "limitations": [
                "Experimental Rust MCP dependency impact only.",
                "Dependency impact is heuristic, not compiler-accurate.",
                "Python MCP runtime remains authoritative."
            ]
        }
    })
}

pub(crate) fn rust_read_skeleton_tool_result(arguments: &Value) -> Value {
    let directory = arguments
        .get("directory")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let filepath = arguments
        .get("filepath")
        .or_else(|| arguments.get("file"))
        .and_then(Value::as_str)
        .unwrap_or_default();

    if directory.trim().is_empty() || filepath.trim().is_empty() {
        return tool_error(
            "rust_read_skeleton requires non-empty directory and filepath arguments.",
        );
    }

    let Some(path) = resolve_under_root(directory, filepath) else {
        return tool_error("Requested file is outside the supplied directory or does not exist.");
    };

    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => return tool_error(&format!("Failed to read UTF-8 source file: {error}")),
    };

    let mut warnings = Vec::new();
    let symbols = if file_ext(&path).as_deref() == Some(".cs") {
        match parse_csharp_ast_summary(&text) {
            Ok(items) => items
                .into_iter()
                .map(|item| {
                    json!({
                        "kind": item.kind,
                        "name": item.name,
                        "line": item.line
                    })
                })
                .collect::<Vec<_>>(),
            Err(_) => {
                warnings.push(
                    "C# Tree-sitter skeleton parsing failed; returned regex fallback skeleton.",
                );
                csharp_regex_skeleton(&text)
            }
        }
    } else {
        warnings
            .push("Tree-sitter skeleton support is currently C#-only; returned fallback skeleton.");
        fallback_skeleton(&text)
    };
    let file = display_relative(directory, &path);
    let text_summary = format!(
        "Rust skeleton returned {} symbol(s) from {}.",
        symbols.len(),
        file
    );

    json!({
        "content": [
            {
                "type": "text",
                "text": text_summary
            }
        ],
        "structuredContent": {
            "file": file,
            "absoluteFile": path.display().to_string(),
            "symbols": symbols,
            "warnings": warnings,
            "limitations": [
                "Experimental Rust MCP skeleton only.",
                "C# skeleton uses Tree-sitter; non-C# files use a small fallback.",
                "Python MCP runtime remains authoritative."
            ]
        }
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
