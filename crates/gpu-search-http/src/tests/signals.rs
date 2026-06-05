use super::*;
use axum::body::Body;
use axum::extract::{Json, State};
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

#[tokio::test]
async fn scan_signals_endpoint_returns_not_ready_shape() {
    let response = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/scan/signals")
                .header("content-type", "application/json")
                .body(Body::from(r#"{}"#))
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn scan_signals_returns_builtin_signal_matches() {
    let root = temp_root("signal_scan");
    fs::write(
        root.join("Repository.cs"),
        "using System.Data.SqlClient;\nusing var conn = new SqlConnection(value);\n",
    )
    .expect("sample file should be written");
    let state = AppState::from_directory(&root).expect("state should index temp root");

    let Json(response) = scan_signals(
        State(state),
        Json(SignalScanRequest {
            categories: vec!["data".to_string()],
            top_k_per_signal: Some(3),
            include_snippets: Some(true),
        }),
    )
    .await;

    assert_eq!(response.status, "ok");
    assert_eq!(response.total_matches, 1);
    assert_eq!(response.signals.len(), 1);
    assert_eq!(response.signals[0].id, "sql_connection");
    assert_eq!(response.signals[0].matches[0].line_start, 2);
    assert!(
        response.signals[0].matches[0]
            .snippet
            .as_deref()
            .unwrap_or_default()
            .contains("SqlConnection")
    );
    fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn scan_signals_can_omit_snippets() {
    let root = temp_root("signal_scan_no_snippets");
    fs::write(
        root.join("Repository.cs"),
        "catch (Exception ex) { throw; }\n",
    )
    .expect("sample file should be written");
    let state = AppState::from_directory(&root).expect("state should index temp root");

    let Json(response) = scan_signals(
        State(state),
        Json(SignalScanRequest {
            categories: vec!["reliability".to_string()],
            top_k_per_signal: Some(3),
            include_snippets: Some(false),
        }),
    )
    .await;

    assert_eq!(response.status, "ok");
    assert_eq!(response.total_matches, 1);
    assert!(response.signals[0].matches[0].snippet.is_none());
    fs::remove_dir_all(root).ok();
}
