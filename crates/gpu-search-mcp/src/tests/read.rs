use super::*;

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
