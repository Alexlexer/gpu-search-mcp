use super::*;
use axum::body::Body;
use axum::extract::{Json, State};
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

#[tokio::test]
async fn read_block_endpoint_rejects_when_no_root() {
    let response = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/read/block")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filepath":"src/lib.rs"}"#))
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn read_skeleton_endpoint_rejects_when_no_root() {
    let response = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/read/skeleton")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filepath":"src/lib.rs"}"#))
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn read_block_returns_requested_line_range() {
    let root = temp_root("read_block");
    fs::write(
        root.join("app.rs"),
        "line one\nline two\nline three\nline four\n",
    )
    .expect("sample file should be written");
    let state = AppState::from_directory(&root).expect("state should index temp root");

    let Json(response) = read_block(
        State(state),
        Json(ReadBlockRequest {
            filepath: "app.rs".to_string(),
            line_start: Some(2),
            line_end: Some(3),
        }),
    )
    .await;

    assert_eq!(response.status, "ok");
    assert_eq!(response.line_start, 2);
    assert_eq!(response.line_end, 3);
    assert_eq!(response.content, "line two\nline three");
    assert!(response.file.ends_with("app.rs"));
    fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn read_block_rejects_outside_root() {
    let root = temp_root("read_outside");
    fs::write(root.join("app.rs"), "inside\n").expect("inside file should be written");
    let outside_root = temp_root("read_outside_target");
    let outside = outside_root.join("outside.rs");
    fs::write(&outside, "outside\n").expect("outside file should be written");
    let state = AppState::from_directory(&root).expect("state should index temp root");

    let Json(response) = read_block(
        State(state),
        Json(ReadBlockRequest {
            filepath: outside.display().to_string(),
            line_start: None,
            line_end: None,
        }),
    )
    .await;

    assert_eq!(response.status, "error");
    assert!(response.content.is_empty());
    assert!(
        response
            .warnings
            .contains(&"Requested file is outside the indexed root or does not exist.")
    );
    fs::remove_dir_all(root).ok();
    fs::remove_dir_all(outside_root).ok();
}

#[tokio::test]
async fn read_skeleton_returns_csharp_symbols() {
    let root = temp_root("read_skeleton_cs");
    fs::write(
            root.join("UserController.cs"),
            "using MyApp.Services;\nnamespace MyApp.Controllers;\npublic interface IUserController { }\npublic class UserController : ControllerBase, IUserController {\n    public string GetUser() => \"ok\";\n}\n",
        )
        .expect("sample file should be written");
    let state = AppState::from_directory(&root).expect("state should index temp root");

    let Json(response) = read_skeleton(
        State(state),
        Json(ReadSkeletonRequest {
            filepath: "UserController.cs".to_string(),
        }),
    )
    .await;

    assert_eq!(response.status, "ok");
    assert!(response.symbols.iter().any(|symbol| {
        symbol.kind == "class_declaration" && symbol.name.as_deref() == Some("UserController")
    }));
    assert!(
        response
            .symbols
            .iter()
            .any(|symbol| symbol.kind == "method_declaration"
                && symbol.name.as_deref() == Some("GetUser"))
    );
    assert!(
        response
            .symbols
            .iter()
            .any(|symbol| symbol.kind == "controller_action"
                && symbol.name.as_deref() == Some("GetUser"))
    );
    assert!(
        response
            .symbols
            .iter()
            .any(|symbol| symbol.kind == "inherits_from"
                && symbol.name.as_deref() == Some("ControllerBase"))
    );
    assert!(
        response
            .symbols
            .iter()
            .any(|symbol| symbol.kind == "implements_interface"
                && symbol.name.as_deref() == Some("IUserController"))
    );
    fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn read_skeleton_rejects_outside_root() {
    let root = temp_root("skeleton_outside");
    fs::write(root.join("app.rs"), "fn main() {}\n").expect("inside file should be written");
    let outside_root = temp_root("skeleton_outside_target");
    let outside = outside_root.join("outside.rs");
    fs::write(&outside, "class Outside {}\n").expect("outside file should be written");
    let state = AppState::from_directory(&root).expect("state should index temp root");

    let Json(response) = read_skeleton(
        State(state),
        Json(ReadSkeletonRequest {
            filepath: outside.display().to_string(),
        }),
    )
    .await;

    assert_eq!(response.status, "error");
    assert!(response.symbols.is_empty());
    assert!(
        response
            .warnings
            .contains(&"Requested file is outside the indexed root or does not exist.")
    );
    fs::remove_dir_all(root).ok();
    fs::remove_dir_all(outside_root).ok();
}
