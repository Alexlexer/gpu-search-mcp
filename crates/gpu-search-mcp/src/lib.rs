//! Experimental Rust MCP scaffold.
//!
//! This crate is additive only. The Python MCP server remains authoritative
//! while Rust MCP compatibility is developed in small, testable milestones.

mod tools;

use gpu_search_core::RUST_CORE_VERSION;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

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
            "rust_get_diagnostics",
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
            },
            {
                "name": "rust_get_diagnostics",
                "description": "Return lightweight Rust MCP scaffold diagnostics without indexing, scanning, loading, or downloading.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
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
        "rust_search_code" => {
            tools::rust_search_code_tool_result(arguments.unwrap_or(&Value::Null))
        }
        "rust_read_block" => tools::rust_read_block_tool_result(arguments.unwrap_or(&Value::Null)),
        "rust_dependency_impact" => {
            tools::rust_dependency_impact_tool_result(arguments.unwrap_or(&Value::Null))
        }
        "rust_read_skeleton" => {
            tools::rust_read_skeleton_tool_result(arguments.unwrap_or(&Value::Null))
        }
        "rust_scan_signals" => {
            tools::rust_scan_signals_tool_result(arguments.unwrap_or(&Value::Null))
        }
        "rust_semantic_model_status" => {
            tools::rust_semantic_model_status_tool_result(arguments.unwrap_or(&Value::Null))
        }
        "rust_get_diagnostics" => tools::rust_get_diagnostics_tool_result(),
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

#[cfg(test)]
mod tests;
