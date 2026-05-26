use super::*;

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
