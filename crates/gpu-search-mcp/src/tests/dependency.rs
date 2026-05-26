use super::*;

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
