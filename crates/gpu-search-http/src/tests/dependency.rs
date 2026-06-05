use super::*;
use axum::body::Body;
use axum::extract::{Json, State};
use axum::http::{Request, StatusCode};
use gpu_search_core::DEPENDENCY_ANALYSIS_MODE;
use tower::ServiceExt;

#[tokio::test]
async fn dependency_impact_endpoint_returns_not_ready_shape() {
    let response = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/dependency/impact")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filepath":"service.py"}"#))
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn dependency_impact_returns_reverse_dependencies_with_reason() {
    let root = temp_root("dependency_impact");
    fs::write(root.join("service.py"), "class UserService:\n    pass\n")
        .expect("service file should be written");
    fs::write(root.join("app.py"), "from service import UserService\n")
        .expect("app file should be written");
    let state = AppState::from_directory(&root).expect("state should index temp root");

    let Json(response) = dependency_impact(
        State(state),
        Json(DependencyImpactRequest {
            filepath: "service.py".to_string(),
        }),
    )
    .await;

    assert_eq!(response.status, "ok");
    assert_eq!(response.confidence, "medium");
    assert_eq!(response.analysis_mode, DEPENDENCY_ANALYSIS_MODE);
    assert_eq!(response.impacted_files.len(), 1);
    assert!(response.impacted_files[0].file.ends_with("app.py"));
    assert_eq!(response.impacted_files[0].hops, 1);
    assert_eq!(
        response.impacted_files[0].reason.as_deref(),
        Some("imports module service")
    );
    fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn dependency_impact_rejects_outside_root() {
    let root = temp_root("dependency_outside");
    fs::write(root.join("service.py"), "class UserService:\n    pass\n")
        .expect("service file should be written");
    let outside = temp_root("dependency_outside_target").join("outside.py");
    fs::write(&outside, "print('outside')\n").expect("outside file should be written");
    let state = AppState::from_directory(&root).expect("state should index temp root");

    let Json(response) = dependency_impact(
        State(state),
        Json(DependencyImpactRequest {
            filepath: outside.display().to_string(),
        }),
    )
    .await;

    assert_eq!(response.status, "error");
    assert!(response.impacted_files.is_empty());
    assert!(
        response
            .warnings
            .contains(&"Requested file is outside the indexed root or does not exist.")
    );
    fs::remove_dir_all(root).ok();
    fs::remove_dir_all(outside.parent().expect("outside file has parent")).ok();
}
