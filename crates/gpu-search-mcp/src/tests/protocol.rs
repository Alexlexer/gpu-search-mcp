use super::*;

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
            "rust_semantic_model_status",
            "rust_get_diagnostics"
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

    assert_eq!(tools.len(), 8);
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
    assert_eq!(tools[7]["name"], "rust_get_diagnostics");
    assert_eq!(tools[7]["inputSchema"]["additionalProperties"], false);
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
    assert_eq!(
        response["result"]["tools"][7]["name"],
        "rust_get_diagnostics"
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
