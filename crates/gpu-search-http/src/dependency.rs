use axum::{Json, extract::State};
use gpu_search_core::DEPENDENCY_ANALYSIS_MODE;

use crate::{
    AppState, DependencyImpactRequest, DependencyImpactResponse, ImpactedFileResponse,
    display_relative, resolve_under_indexed_root,
};

/// Return heuristic dependency impact results from the experimental Rust core.
pub async fn dependency_impact(
    State(state): State<AppState>,
    Json(request): Json<DependencyImpactRequest>,
) -> Json<DependencyImpactResponse> {
    let Some(graph) = &state.dependency_graph else {
        return Json(DependencyImpactResponse {
            status: "not_ready",
            file: request.filepath,
            impacted_files: Vec::new(),
            confidence: "low",
            analysis_mode: DEPENDENCY_ANALYSIS_MODE,
            warnings: vec!["Dependency graph is not ready in the Rust HTTP server."],
            limitations: dependency_limitations(),
        });
    };

    let Some(changed_file) = resolve_under_indexed_root(&state, &request.filepath) else {
        return Json(DependencyImpactResponse {
            status: "error",
            file: request.filepath,
            impacted_files: Vec::new(),
            confidence: "low",
            analysis_mode: DEPENDENCY_ANALYSIS_MODE,
            warnings: vec!["Requested file is outside the indexed root or does not exist."],
            limitations: dependency_limitations(),
        });
    };

    let impacted_files = graph
        .impact(&changed_file)
        .into_iter()
        .map(|impacted| ImpactedFileResponse {
            file: display_relative(&state, &impacted.file),
            absolute_file: impacted.file.display().to_string(),
            hops: impacted.hops,
            reason: impacted.reason,
        })
        .collect::<Vec<_>>();

    Json(DependencyImpactResponse {
        status: "ok",
        file: display_relative(&state, &changed_file),
        impacted_files,
        confidence: "medium",
        analysis_mode: DEPENDENCY_ANALYSIS_MODE,
        warnings: Vec::new(),
        limitations: dependency_limitations(),
    })
}

fn dependency_limitations() -> Vec<&'static str> {
    vec![
        "Dependency impact is heuristic, not compiler-accurate.",
        "Rust HTTP dependency impact is experimental.",
        "Python HTTP/MCP runtime remains authoritative.",
    ]
}
