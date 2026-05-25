//! Experimental Rust MCP scaffold.
//!
//! This crate is additive only. The Python MCP server remains authoritative
//! while Rust MCP compatibility is developed in small, testable milestones.

use gpu_search_core::{
    ContextMode, DEPENDENCY_ANALYSIS_MODE, DependencyGraph, IndexOptions, LineIndex,
    PatternSearchOptions, RUST_CORE_VERSION, discover_files, file_ext, parse_csharp_ast_summary,
    search_files,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Experimental Rust MCP crate version.
pub const RUST_MCP_VERSION: &str = "0.1.0-prototype";

/// Default sentence-transformers model used by the Python semantic runtime.
pub const DEFAULT_SEMANTIC_MODEL_ID: &str = "BAAI/bge-small-en-v1.5";

/// Minimal capability metadata for the experimental Rust MCP scaffold.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct McpScaffoldInfo {
    pub status: &'static str,
    pub implementation: &'static str,
    pub rust_core_version: &'static str,
    pub rust_mcp_version: &'static str,
    pub tools: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}

/// Return static scaffold metadata without indexing or starting an MCP loop.
pub fn scaffold_info() -> McpScaffoldInfo {
    McpScaffoldInfo {
        status: "experimental",
        implementation: "rust-mcp-scaffold",
        rust_core_version: RUST_CORE_VERSION,
        rust_mcp_version: RUST_MCP_VERSION,
        tools: vec![
            "get_scaffold_info",
            "rust_search_code",
            "rust_read_block",
            "rust_dependency_impact",
            "rust_read_skeleton",
            "rust_scan_signals",
            "rust_semantic_model_status",
        ],
        limitations: vec![
            "Python MCP runtime remains authoritative.",
            "Rust MCP stdio protocol handling is not implemented yet.",
            "No repository indexing, search, or file reads are performed by this scaffold.",
        ],
    }
}

/// Return a minimal MCP initialize result for smoke tests.
pub fn initialize_result() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "gpu-search-mcp-rust-experimental",
            "version": RUST_MCP_VERSION
        },
        "instructions": "Experimental Rust MCP scaffold. Python MCP runtime remains authoritative."
    })
}

/// Return the experimental Rust MCP tool list.
pub fn tools_list_result() -> Value {
    json!({
        "tools": [
            {
                "name": "get_scaffold_info",
                "description": "Return static metadata for the experimental Rust MCP scaffold.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            },
                {
                    "name": "rust_search_code",
                    "description": "Experimental Rust exact pattern search over an explicit local directory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Repository root to discover and search."
                        },
                        "query": {
                            "type": "string",
                            "description": "Exact text query."
                        },
                        "topK": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Maximum number of matches to return."
                        }
                    },
                    "required": ["directory", "query"],
                    "additionalProperties": false
                }
            },
                {
                    "name": "rust_read_block",
                    "description": "Experimental Rust bounded line-range read under an explicit local directory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Repository root that the file must live under."
                        },
                        "filepath": {
                            "type": "string",
                            "description": "File path relative to directory, or absolute path under directory."
                        },
                        "lineStart": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "First 1-based line to read."
                        },
                        "lineEnd": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Last 1-based line to read."
                        }
                    },
                    "required": ["directory", "filepath"],
                    "additionalProperties": false
                }
            },
            {
                "name": "rust_dependency_impact",
                "description": "Experimental Rust heuristic dependency impact over an explicit local directory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Repository root to discover and analyze."
                        },
                        "filepath": {
                            "type": "string",
                            "description": "Changed file path relative to directory, or absolute path under directory."
                        }
                    },
                    "required": ["directory", "filepath"],
                    "additionalProperties": false
                }
            },
            {
                "name": "rust_read_skeleton",
                "description": "Experimental Rust skeleton summary for a UTF-8 file under an explicit local directory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Repository root that the file must live under."
                        },
                        "filepath": {
                            "type": "string",
                            "description": "File path relative to directory, or absolute path under directory."
                        }
                    },
                    "required": ["directory", "filepath"],
                    "additionalProperties": false
                }
            },
            {
                "name": "rust_scan_signals",
                "description": "Experimental Rust signal scan over an explicit local directory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Repository root to discover and scan."
                        },
                        "categories": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional signal categories to include."
                        },
                        "topKPerSignal": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "description": "Maximum matches per signal."
                        },
                        "includeSnippets": {
                            "type": "boolean",
                            "description": "Whether to include one-line snippets."
                        }
                    },
                    "required": ["directory"],
                    "additionalProperties": false
                }
            },
            {
                "name": "rust_semantic_model_status",
                "description": "Return advisory Rust MCP semantic model status without loading or downloading models.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "modelId": {
                            "type": "string",
                            "description": "Optional sentence-transformers model id to report."
                        }
                    },
                    "additionalProperties": false
                }
            }
        ]
    })
}

/// Handle a minimal MCP tools/call request for experimental scaffold tools.
pub fn tools_call_result(name: &str, arguments: Option<&Value>) -> Value {
    match name {
        "get_scaffold_info" => json!({
        "content": [
            {
                "type": "text",
                "text": format!("Rust MCP scaffold: {}", RUST_MCP_VERSION)
            }
        ],
        "structuredContent": scaffold_info()
            }),
        "rust_search_code" => rust_search_code_tool_result(arguments.unwrap_or(&Value::Null)),
        "rust_read_block" => rust_read_block_tool_result(arguments.unwrap_or(&Value::Null)),
        "rust_dependency_impact" => {
            rust_dependency_impact_tool_result(arguments.unwrap_or(&Value::Null))
        }
        "rust_read_skeleton" => rust_read_skeleton_tool_result(arguments.unwrap_or(&Value::Null)),
        "rust_scan_signals" => rust_scan_signals_tool_result(arguments.unwrap_or(&Value::Null)),
        "rust_semantic_model_status" => {
            rust_semantic_model_status_tool_result(arguments.unwrap_or(&Value::Null))
        }
        _ => json!({
            "isError": true,
            "content": [
                {
                    "type": "text",
                    "text": format!("Unknown Rust MCP scaffold tool: {name}")
                }
            ]
        }),
    }
}

/// Handle one minimal JSON-RPC request for scaffold smoke tests.
///
/// This is intentionally tiny and does not implement full MCP protocol support.
pub fn handle_scaffold_json_rpc(request: &Value) -> Value {
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    let method = request
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();

    let result = match method {
        "initialize" => Some(initialize_result()),
        "tools/list" => Some(tools_list_result()),
        "tools/call" => {
            let params = request.get("params").unwrap_or(&Value::Null);
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let arguments = params.get("arguments");
            Some(tools_call_result(name, arguments))
        }
        "get_scaffold_info" => Some(json!(scaffold_info())),
        _ => None,
    };

    if let Some(result) = result {
        return json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        });
    }

    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": -32601,
            "message": "Method not found in Rust MCP scaffold"
        }
    })
}

/// Handle one newline-delimited JSON-RPC message for the experimental stdio loop.
///
/// Notifications (messages without an `id`) intentionally return `None`.
pub fn handle_scaffold_json_rpc_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    let request: Value = match serde_json::from_str(trimmed) {
        Ok(request) => request,
        Err(error) => {
            return Some(
                json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {error}")
                    }
                })
                .to_string(),
            );
        }
    };

    if request.get("id").is_none() {
        return None;
    }

    Some(handle_scaffold_json_rpc(&request).to_string())
}

fn rust_search_code_tool_result(arguments: &Value) -> Value {
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

fn rust_read_block_tool_result(arguments: &Value) -> Value {
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

fn rust_dependency_impact_tool_result(arguments: &Value) -> Value {
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

fn rust_read_skeleton_tool_result(arguments: &Value) -> Value {
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

fn rust_scan_signals_tool_result(arguments: &Value) -> Value {
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

fn rust_semantic_model_status_tool_result(arguments: &Value) -> Value {
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

fn semantic_model_status_snapshot(model_id: &str) -> Value {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gpu_search_mcp_{name}_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root).expect("temp root should be created");
        root
    }

    #[test]
    fn scaffold_info_reports_versions_and_limitations() {
        let info = scaffold_info();

        assert_eq!(info.status, "experimental");
        assert_eq!(info.implementation, "rust-mcp-scaffold");
        assert_eq!(info.rust_core_version, RUST_CORE_VERSION);
        assert_eq!(info.rust_mcp_version, RUST_MCP_VERSION);
        assert_eq!(
            info.tools,
            vec![
                "get_scaffold_info",
                "rust_search_code",
                "rust_read_block",
                "rust_dependency_impact",
                "rust_read_skeleton",
                "rust_scan_signals",
                "rust_semantic_model_status"
            ]
        );
        assert!(
            info.limitations
                .contains(&"Python MCP runtime remains authoritative.")
        );
    }

    #[test]
    fn json_rpc_scaffold_info_returns_result() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "get_scaffold_info"
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 7);
        assert_eq!(response["result"]["implementation"], "rust-mcp-scaffold");
    }

    #[test]
    fn initialize_result_reports_minimal_mcp_capabilities() {
        let result = initialize_result();

        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(
            result["serverInfo"]["name"],
            "gpu-search-mcp-rust-experimental"
        );
        assert_eq!(result["serverInfo"]["version"], RUST_MCP_VERSION);
        assert!(result["capabilities"].get("tools").is_some());
    }

    #[test]
    fn tools_list_result_includes_scaffold_tool_schema() {
        let result = tools_list_result();
        let tools = result["tools"]
            .as_array()
            .expect("tools should be an array");

        assert_eq!(tools.len(), 7);
        assert_eq!(tools[0]["name"], "get_scaffold_info");
        assert_eq!(tools[0]["inputSchema"]["type"], "object");
        assert_eq!(tools[0]["inputSchema"]["additionalProperties"], false);
        assert_eq!(tools[1]["name"], "rust_search_code");
        assert_eq!(tools[1]["inputSchema"]["required"][0], "directory");
        assert_eq!(tools[1]["inputSchema"]["required"][1], "query");
        assert_eq!(tools[2]["name"], "rust_read_block");
        assert_eq!(tools[2]["inputSchema"]["required"][0], "directory");
        assert_eq!(tools[2]["inputSchema"]["required"][1], "filepath");
        assert_eq!(tools[3]["name"], "rust_dependency_impact");
        assert_eq!(tools[3]["inputSchema"]["required"][0], "directory");
        assert_eq!(tools[3]["inputSchema"]["required"][1], "filepath");
        assert_eq!(tools[4]["name"], "rust_read_skeleton");
        assert_eq!(tools[4]["inputSchema"]["required"][0], "directory");
        assert_eq!(tools[4]["inputSchema"]["required"][1], "filepath");
        assert_eq!(tools[5]["name"], "rust_scan_signals");
        assert_eq!(tools[5]["inputSchema"]["required"][0], "directory");
        assert_eq!(tools[6]["name"], "rust_semantic_model_status");
        assert_eq!(tools[6]["inputSchema"]["additionalProperties"], false);
    }

    #[test]
    fn json_rpc_initialize_returns_result() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 1);
        assert_eq!(response["result"]["protocolVersion"], "2024-11-05");
    }

    #[test]
    fn json_rpc_tools_list_returns_result() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 2);
        assert_eq!(response["result"]["tools"][0]["name"], "get_scaffold_info");
        assert_eq!(response["result"]["tools"][1]["name"], "rust_search_code");
        assert_eq!(response["result"]["tools"][2]["name"], "rust_read_block");
        assert_eq!(
            response["result"]["tools"][3]["name"],
            "rust_dependency_impact"
        );
        assert_eq!(response["result"]["tools"][4]["name"], "rust_read_skeleton");
        assert_eq!(response["result"]["tools"][5]["name"], "rust_scan_signals");
        assert_eq!(
            response["result"]["tools"][6]["name"],
            "rust_semantic_model_status"
        );
    }

    #[test]
    fn tools_call_get_scaffold_info_returns_structured_content() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_scaffold_info",
                "arguments": {}
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 3);
        assert_eq!(
            response["result"]["structuredContent"]["implementation"],
            "rust-mcp-scaffold"
        );
    }

    #[test]
    fn tools_call_rust_search_code_returns_pattern_results() {
        let root = temp_root("pattern_tool");
        fs::write(
            root.join("app.rs"),
            "fn main() {\n    let service = UserService::new();\n}\n",
        )
        .expect("sample file should be written");

        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "rust_search_code",
                "arguments": {
                    "directory": root.display().to_string(),
                    "query": "UserService",
                    "topK": 5
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 4);
        assert_eq!(
            response["result"]["structuredContent"]["results"][0]["lineStart"],
            2
        );
        assert!(
            response["result"]["structuredContent"]["results"][0]["snippet"]
                .as_str()
                .unwrap_or_default()
                .contains("UserService")
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn tools_call_rust_search_code_validates_required_arguments() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "rust_search_code",
                "arguments": {
                    "directory": "",
                    "query": ""
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 5);
        assert_eq!(response["result"]["isError"], true);
    }

    #[test]
    fn tools_call_rust_read_block_returns_line_range() {
        let root = temp_root("read_block_tool");
        fs::write(root.join("app.rs"), "line one\nline two\nline three\n")
            .expect("sample file should be written");

        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "rust_read_block",
                "arguments": {
                    "directory": root.display().to_string(),
                    "filepath": "app.rs",
                    "lineStart": 2,
                    "lineEnd": 3
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 6);
        assert_eq!(response["result"]["structuredContent"]["lineStart"], 2);
        assert_eq!(response["result"]["structuredContent"]["lineEnd"], 3);
        assert_eq!(
            response["result"]["structuredContent"]["content"],
            "line two\nline three"
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn tools_call_rust_read_block_rejects_outside_root() {
        let root = temp_root("read_block_root");
        let outside_root = temp_root("read_block_outside");
        let outside = outside_root.join("outside.rs");
        fs::write(&outside, "outside\n").expect("outside file should be written");

        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "rust_read_block",
                "arguments": {
                    "directory": root.display().to_string(),
                    "filepath": outside.display().to_string()
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 7);
        assert_eq!(response["result"]["isError"], true);
        fs::remove_dir_all(root).ok();
        fs::remove_dir_all(outside_root).ok();
    }

    #[test]
    fn tools_call_rust_read_block_validates_required_arguments() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "rust_read_block",
                "arguments": {
                    "directory": "",
                    "filepath": ""
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 8);
        assert_eq!(response["result"]["isError"], true);
    }

    #[test]
    fn tools_call_rust_dependency_impact_returns_impacted_files() {
        let root = temp_root("dependency_tool");
        fs::write(root.join("service.py"), "class Service:\n    pass\n")
            .expect("service file should be written");
        fs::write(root.join("controller.py"), "import service\n")
            .expect("controller file should be written");

        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "rust_dependency_impact",
                "arguments": {
                    "directory": root.display().to_string(),
                    "filepath": "service.py"
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 9);
        assert_eq!(
            response["result"]["structuredContent"]["impactedFiles"][0]["file"],
            "controller.py"
        );
        assert_eq!(
            response["result"]["structuredContent"]["impactedFiles"][0]["hops"],
            1
        );
        assert_eq!(
            response["result"]["structuredContent"]["impactedFiles"][0]["reason"],
            "imports module service"
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn tools_call_rust_dependency_impact_rejects_outside_root() {
        let root = temp_root("dependency_root");
        let outside_root = temp_root("dependency_outside");
        let outside = outside_root.join("outside.py");
        fs::write(&outside, "print('outside')\n").expect("outside file should be written");

        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "rust_dependency_impact",
                "arguments": {
                    "directory": root.display().to_string(),
                    "filepath": outside.display().to_string()
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 10);
        assert_eq!(response["result"]["isError"], true);
        fs::remove_dir_all(root).ok();
        fs::remove_dir_all(outside_root).ok();
    }

    #[test]
    fn tools_call_rust_dependency_impact_validates_required_arguments() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "rust_dependency_impact",
                "arguments": {
                    "directory": "",
                    "filepath": ""
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 11);
        assert_eq!(response["result"]["isError"], true);
    }

    #[test]
    fn tools_call_rust_read_skeleton_returns_csharp_symbols() {
        let root = temp_root("skeleton_tool");
        fs::write(
            root.join("UserController.cs"),
            "using MyApp.Services;\nnamespace MyApp.Controllers;\npublic class UserController {\n    public string GetUser() => \"ok\";\n}\n",
        )
        .expect("sample file should be written");

        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "rust_read_skeleton",
                "arguments": {
                    "directory": root.display().to_string(),
                    "filepath": "UserController.cs"
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 12);
        let symbols = response["result"]["structuredContent"]["symbols"]
            .as_array()
            .expect("symbols should be an array");
        assert!(symbols.iter().any(|symbol| {
            symbol["kind"] == "class_declaration" && symbol["name"] == "UserController"
        }));
        assert!(symbols.iter().any(|symbol| {
            symbol["kind"] == "method_declaration" && symbol["name"] == "GetUser"
        }));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn tools_call_rust_read_skeleton_rejects_outside_root() {
        let root = temp_root("skeleton_root");
        let outside_root = temp_root("skeleton_outside");
        let outside = outside_root.join("outside.cs");
        fs::write(&outside, "public class Outside {}\n").expect("outside file should be written");

        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "rust_read_skeleton",
                "arguments": {
                    "directory": root.display().to_string(),
                    "filepath": outside.display().to_string()
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 13);
        assert_eq!(response["result"]["isError"], true);
        fs::remove_dir_all(root).ok();
        fs::remove_dir_all(outside_root).ok();
    }

    #[test]
    fn tools_call_rust_read_skeleton_validates_required_arguments() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 14,
            "method": "tools/call",
            "params": {
                "name": "rust_read_skeleton",
                "arguments": {
                    "directory": "",
                    "filepath": ""
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 14);
        assert_eq!(response["result"]["isError"], true);
    }

    #[test]
    fn tools_call_rust_scan_signals_returns_matches() {
        let root = temp_root("signal_tool");
        fs::write(
            root.join("Repository.cs"),
            "using System.Data.SqlClient;\nclass Repository {\n    void Open() { var conn = new SqlConnection(value); }\n}\n",
        )
        .expect("sample file should be written");

        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 15,
            "method": "tools/call",
            "params": {
                "name": "rust_scan_signals",
                "arguments": {
                    "directory": root.display().to_string(),
                    "categories": ["data"],
                    "topKPerSignal": 3,
                    "includeSnippets": true
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 15);
        assert_eq!(
            response["result"]["structuredContent"]["signals"][0]["id"],
            "sql_connection"
        );
        assert_eq!(response["result"]["structuredContent"]["totalMatches"], 1);
        assert!(
            response["result"]["structuredContent"]["signals"][0]["matches"][0]["snippet"]
                .as_str()
                .unwrap_or_default()
                .contains("SqlConnection")
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn tools_call_rust_scan_signals_can_omit_snippets() {
        let root = temp_root("signal_no_snippets_tool");
        fs::write(
            root.join("Repository.cs"),
            "var conn = new SqlConnection(value);\n",
        )
        .expect("sample file should be written");

        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 16,
            "method": "tools/call",
            "params": {
                "name": "rust_scan_signals",
                "arguments": {
                    "directory": root.display().to_string(),
                    "categories": ["data"],
                    "topKPerSignal": 1,
                    "includeSnippets": false
                }
            }
        }));

        let first_match = response["result"]["structuredContent"]["signals"][0]["matches"][0]
            .as_object()
            .expect("match should be an object");
        assert!(!first_match.contains_key("snippet"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn tools_call_rust_scan_signals_validates_required_arguments() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 17,
            "method": "tools/call",
            "params": {
                "name": "rust_scan_signals",
                "arguments": {
                    "directory": ""
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 17);
        assert_eq!(response["result"]["isError"], true);
    }

    #[test]
    fn tools_call_rust_semantic_model_status_returns_advisory_status() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 18,
            "method": "tools/call",
            "params": {
                "name": "rust_semantic_model_status",
                "arguments": {
                    "modelId": "custom/model"
                }
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 18);
        let status = &response["result"]["structuredContent"];
        assert_eq!(status["modelId"], "custom/model");
        assert_eq!(status["provider"], "sentence-transformers");
        assert_eq!(status["available"], false);
        assert_eq!(status["cached"], false);
        assert_eq!(status["requiresDownload"], true);
        assert!(
            status["message"]
                .as_str()
                .unwrap_or_default()
                .contains("--download-semantic-model")
        );
    }

    #[test]
    fn semantic_model_status_snapshot_uses_default_model_id() {
        let status = semantic_model_status_snapshot(DEFAULT_SEMANTIC_MODEL_ID);

        assert_eq!(status["modelId"], DEFAULT_SEMANTIC_MODEL_ID);
        assert_eq!(status["available"], false);
        assert_eq!(status["requiresDownload"], true);
    }

    #[test]
    fn tools_call_unknown_tool_returns_tool_error_result() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {
                "name": "missing_tool",
                "arguments": {}
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 20);
        assert_eq!(response["result"]["isError"], true);
    }

    #[test]
    fn json_rpc_unknown_method_returns_method_not_found() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": "abc",
            "method": "gpu_search"
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], "abc");
        assert_eq!(response["error"]["code"], -32601);
    }

    #[test]
    fn json_rpc_line_returns_response_for_requests() {
        let response =
            handle_scaffold_json_rpc_line(r#"{"jsonrpc":"2.0","id":13,"method":"tools/list"}"#)
                .expect("request should return a response");
        let parsed: Value = serde_json::from_str(&response).expect("response should be json");

        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["id"], 13);
        assert_eq!(parsed["result"]["tools"][0]["name"], "get_scaffold_info");
    }

    #[test]
    fn json_rpc_line_ignores_notifications() {
        let response = handle_scaffold_json_rpc_line(
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        );

        assert!(response.is_none());
    }

    #[test]
    fn json_rpc_line_returns_parse_error_for_invalid_json() {
        let response = handle_scaffold_json_rpc_line("{not json")
            .expect("invalid request should return an error response");
        let parsed: Value = serde_json::from_str(&response).expect("response should be json");

        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["id"], Value::Null);
        assert_eq!(parsed["error"]["code"], -32700);
    }
}
