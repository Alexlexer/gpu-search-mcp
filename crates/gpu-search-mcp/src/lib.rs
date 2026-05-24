//! Experimental Rust MCP scaffold.
//!
//! This crate is additive only. The Python MCP server remains authoritative
//! while Rust MCP compatibility is developed in small, testable milestones.

use gpu_search_core::{
    ContextMode, IndexOptions, PatternSearchOptions, RUST_CORE_VERSION, discover_files,
    search_files,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

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
        tools: vec!["get_scaffold_info", "rust_search_code"],
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
        assert_eq!(info.tools, vec!["get_scaffold_info", "rust_search_code"]);
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

        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["name"], "get_scaffold_info");
        assert_eq!(tools[0]["inputSchema"]["type"], "object");
        assert_eq!(tools[0]["inputSchema"]["additionalProperties"], false);
        assert_eq!(tools[1]["name"], "rust_search_code");
        assert_eq!(tools[1]["inputSchema"]["required"][0], "directory");
        assert_eq!(tools[1]["inputSchema"]["required"][1], "query");
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
    fn tools_call_unknown_tool_returns_tool_error_result() {
        let response = handle_scaffold_json_rpc(&json!({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "missing_tool",
                "arguments": {}
            }
        }));

        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 6);
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
}
