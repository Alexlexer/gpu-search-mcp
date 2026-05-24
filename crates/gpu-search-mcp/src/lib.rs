//! Experimental Rust MCP scaffold.
//!
//! This crate is additive only. The Python MCP server remains authoritative
//! while Rust MCP compatibility is developed in small, testable milestones.

use gpu_search_core::{
    ContextMode, DEPENDENCY_ANALYSIS_MODE, DependencyGraph, IndexOptions, LineIndex,
    PatternSearchOptions, RUST_CORE_VERSION, discover_files, search_files,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};

/// Experimental Rust MCP crate version.
pub const RUST_MCP_VERSION: &str = "0.1.0-prototype";

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
                "rust_dependency_impact"
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

        assert_eq!(tools.len(), 4);
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
    fn tools_call_unknown_tool_returns_tool_error_result() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "missing_tool",
                "arguments": {}
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 12);
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
