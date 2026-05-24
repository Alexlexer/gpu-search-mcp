//! Experimental Rust MCP scaffold.
//!
//! This crate is additive only. The Python MCP server remains authoritative
//! while Rust MCP compatibility is developed in small, testable milestones.

use gpu_search_core::RUST_CORE_VERSION;
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
        tools: vec!["get_scaffold_info"],
        limitations: vec![
            "Python MCP runtime remains authoritative.",
            "Rust MCP stdio protocol handling is not implemented yet.",
            "No repository indexing, search, or file reads are performed by this scaffold.",
        ],
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

    if method == "get_scaffold_info" {
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": scaffold_info(),
        })
    } else {
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": -32601,
                "message": "Method not found in Rust MCP scaffold"
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_info_reports_versions_and_limitations() {
        let info = scaffold_info();

        assert_eq!(info.status, "experimental");
        assert_eq!(info.implementation, "rust-mcp-scaffold");
        assert_eq!(info.rust_core_version, RUST_CORE_VERSION);
        assert_eq!(info.rust_mcp_version, RUST_MCP_VERSION);
        assert_eq!(info.tools, vec!["get_scaffold_info"]);
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
