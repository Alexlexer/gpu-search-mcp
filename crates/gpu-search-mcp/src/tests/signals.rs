use super::*;

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
