use super::*;
use gpu_search_core::DEPENDENCY_ANALYSIS_MODE;
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
        "using MyApp.Services;\nnamespace MyApp.Controllers;\npublic interface IUserController { }\npublic class UserController : ControllerBase, IUserController {\n    public string GetUser() => \"ok\";\n}\n",
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
    assert!(
        symbols.iter().any(|symbol| {
            symbol["kind"] == "method_declaration" && symbol["name"] == "GetUser"
        })
    );
    assert!(
        symbols
            .iter()
            .any(|symbol| { symbol["kind"] == "controller_action" && symbol["name"] == "GetUser" })
    );
    assert!(
        symbols.iter().any(|symbol| {
            symbol["kind"] == "inherits_from" && symbol["name"] == "ControllerBase"
        })
    );
    assert!(symbols.iter().any(|symbol| {
        symbol["kind"] == "implements_interface" && symbol["name"] == "IUserController"
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
    let status = tools::status::semantic_model_status_snapshot(DEFAULT_SEMANTIC_MODEL_ID);

    assert_eq!(status["modelId"], DEFAULT_SEMANTIC_MODEL_ID);
    assert_eq!(status["available"], false);
    assert_eq!(status["requiresDownload"], true);
}

#[test]
fn tools_call_rust_get_diagnostics_returns_scaffold_status() {
    let response = handle_scaffold_json_rpc(&json!({
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/call",
        "params": {
            "name": "rust_get_diagnostics",
            "arguments": {}
        }
    }));

    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], 20);
    let diagnostics = &response["result"]["structuredContent"];
    assert_eq!(diagnostics["implementation"], "rust-mcp-scaffold");
    assert_eq!(diagnostics["rustCoreVersion"], RUST_CORE_VERSION);
    assert_eq!(diagnostics["capabilities"]["mcpTools"], true);
    assert_eq!(diagnostics["capabilities"]["semanticSearch"], false);
    assert_eq!(
        diagnostics["indexes"]["dependency"]["analysisMode"],
        DEPENDENCY_ANALYSIS_MODE
    );
    assert_eq!(
        diagnostics["semanticModel"]["provider"],
        "sentence-transformers"
    );
}

#[test]
fn tools_call_unknown_tool_returns_tool_error_result() {
    let response = handle_scaffold_json_rpc(&json!({
        "jsonrpc": "2.0",
        "id": 21,
        "method": "tools/call",
        "params": {
            "name": "missing_tool",
            "arguments": {}
        }
    }));

    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], 21);
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
    let response =
        handle_scaffold_json_rpc_line(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);

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
