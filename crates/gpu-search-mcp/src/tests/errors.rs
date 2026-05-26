use super::*;

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
