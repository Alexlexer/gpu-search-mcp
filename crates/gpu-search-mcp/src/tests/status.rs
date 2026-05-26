use super::*;

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
