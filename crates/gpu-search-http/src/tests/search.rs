use super::*;
use axum::body::Body;
use axum::extract::{Json, State};
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

#[tokio::test]
async fn search_code_endpoint_returns_not_ready_shape() {
    let response = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/search/code")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"query":"UserService","topK":3}"#))
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn search_hybrid_endpoint_returns_ok() {
    let response = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/search/hybrid")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"query":"UserService","topK":3}"#))
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn search_semantic_endpoint_returns_ok() {
    let response = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/search/semantic")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"query":"authentication middleware"}"#))
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn search_code_handler_preserves_request_shape_and_empty_results() {
    let Json(response) = search_code(
        State(AppState::empty()),
        Json(SearchCodeRequest {
            query: "UserService".to_string(),
            top_k: Some(3),
            mode: Some(SearchMode::Pattern),
            context_mode: Some(SearchContextMode::Compact),
        }),
    )
    .await;

    assert_eq!(response.status, "not_ready");
    assert_eq!(response.query, "UserService");
    assert_eq!(response.top_k, 3);
    assert_eq!(response.mode, SearchMode::Pattern);
    assert_eq!(response.context_mode, SearchContextMode::Compact);
    assert!(response.results.is_empty());
    assert!(
        response
            .result
            .contains("Start with --directory to enable pattern search")
    );
}

#[tokio::test]
async fn search_code_returns_pattern_results_when_state_has_files() {
    let root = temp_root("pattern_search");
    fs::write(
        root.join("app.rs"),
        "fn main() {\n    let user_service = UserService::new();\n}\n",
    )
    .expect("sample file should be written");
    let state = AppState::from_directory(&root).expect("state should index temp root");

    let Json(response) = search_code(
        State(state),
        Json(SearchCodeRequest {
            query: "UserService".to_string(),
            top_k: Some(5),
            mode: Some(SearchMode::Pattern),
            context_mode: Some(SearchContextMode::Compact),
        }),
    )
    .await;

    assert_eq!(response.status, "ok");
    assert_eq!(response.results.len(), 1);
    assert_eq!(response.results[0].line_start, 2);
    assert_eq!(response.results[0].reason, "exact token match");
    assert!(response.results[0].snippet.contains("UserService"));
    fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn search_hybrid_forces_hybrid_mode_and_returns_pattern_results() {
    let root = temp_root("hybrid_search");
    fs::write(
        root.join("app.rs"),
        "let user_service = UserService::new();\n",
    )
    .expect("sample file should be written");
    let state = AppState::from_directory(&root).expect("state should index temp root");

    let Json(response) = search_hybrid(
        State(state),
        Json(SearchCodeRequest {
            query: "UserService".to_string(),
            top_k: Some(5),
            mode: Some(SearchMode::Pattern),
            context_mode: Some(SearchContextMode::Compact),
        }),
    )
    .await;

    assert_eq!(response.status, "ok");
    assert_eq!(response.mode, SearchMode::Hybrid);
    assert_eq!(response.results.len(), 1);
    assert!(
        response
            .warnings
            .contains(&"Hybrid mode currently returns Rust pattern results only.")
    );
    fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn search_semantic_forces_semantic_not_ready_response() {
    let root = temp_root("semantic_search");
    fs::write(
        root.join("app.rs"),
        "let user_service = UserService::new();\n",
    )
    .expect("sample file should be written");
    let state = AppState::from_directory(&root).expect("state should index temp root");

    let Json(response) = search_semantic(
        State(state),
        Json(SearchCodeRequest {
            query: "authentication middleware".to_string(),
            top_k: Some(5),
            mode: Some(SearchMode::Pattern),
            context_mode: Some(SearchContextMode::Compact),
        }),
    )
    .await;

    assert_eq!(response.status, "not_ready");
    assert_eq!(response.mode, SearchMode::Semantic);
    assert!(response.results.is_empty());
    assert!(
        response
            .warnings
            .contains(&"Semantic search remains available through the Python runtime.")
    );
    fs::remove_dir_all(root).ok();
}
