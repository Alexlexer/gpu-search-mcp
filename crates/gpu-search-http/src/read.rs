use axum::{Json, extract::State};
use gpu_search_core::{file_ext, parse_csharp_ast_summary};
use std::fs;

use crate::{
    AppState, ReadBlockRequest, ReadBlockResponse, ReadSkeletonRequest, ReadSkeletonResponse,
    SkeletonItem, display_relative, resolve_under_indexed_root,
};

/// Read a bounded source block from inside the indexed root.
pub async fn read_block(
    State(state): State<AppState>,
    Json(request): Json<ReadBlockRequest>,
) -> Json<ReadBlockResponse> {
    let Some(path) = resolve_under_indexed_root(&state, &request.filepath) else {
        return Json(ReadBlockResponse {
            status: "error",
            file: request.filepath,
            absolute_file: None,
            line_start: 0,
            line_end: 0,
            content: String::new(),
            warnings: vec!["Requested file is outside the indexed root or does not exist."],
            limitations: read_limitations(),
        });
    };

    let Ok(text) = fs::read_to_string(&path) else {
        return Json(ReadBlockResponse {
            status: "error",
            file: display_relative(&state, &path),
            absolute_file: Some(path.display().to_string()),
            line_start: 0,
            line_end: 0,
            content: String::new(),
            warnings: vec!["Requested file could not be read as UTF-8 text."],
            limitations: read_limitations(),
        });
    };

    let lines: Vec<&str> = text.lines().collect();
    let total_lines = lines.len().max(1);
    let line_start = request.line_start.unwrap_or(1).clamp(1, total_lines);
    let line_end = request
        .line_end
        .unwrap_or(line_start.saturating_add(40))
        .clamp(line_start, total_lines);
    let content = lines[(line_start - 1)..line_end].join("\n");

    Json(ReadBlockResponse {
        status: "ok",
        file: display_relative(&state, &path),
        absolute_file: Some(path.display().to_string()),
        line_start,
        line_end,
        content,
        warnings: Vec::new(),
        limitations: read_limitations(),
    })
}

/// Read a lightweight symbol skeleton from inside the indexed root.
pub async fn read_skeleton(
    State(state): State<AppState>,
    Json(request): Json<ReadSkeletonRequest>,
) -> Json<ReadSkeletonResponse> {
    let Some(path) = resolve_under_indexed_root(&state, &request.filepath) else {
        return Json(ReadSkeletonResponse {
            status: "error",
            file: request.filepath,
            absolute_file: None,
            symbols: Vec::new(),
            warnings: vec!["Requested file is outside the indexed root or does not exist."],
            limitations: skeleton_limitations(),
        });
    };

    let Ok(text) = fs::read_to_string(&path) else {
        return Json(ReadSkeletonResponse {
            status: "error",
            file: display_relative(&state, &path),
            absolute_file: Some(path.display().to_string()),
            symbols: Vec::new(),
            warnings: vec!["Requested file could not be read as UTF-8 text."],
            limitations: skeleton_limitations(),
        });
    };

    let mut warnings = Vec::new();
    let symbols = if file_ext(&path).as_deref() == Some(".cs") {
        match parse_csharp_ast_summary(&text) {
            Ok(items) => items
                .into_iter()
                .map(|item| SkeletonItem {
                    kind: item.kind,
                    name: item.name,
                    line: item.line,
                })
                .collect(),
            Err(_) => {
                warnings
                    .push("C# Tree-sitter skeleton parsing failed; returned fallback skeleton.");
                fallback_skeleton(&text)
            }
        }
    } else {
        warnings
            .push("Tree-sitter skeleton support is currently C#-only; returned fallback skeleton.");
        fallback_skeleton(&text)
    };

    Json(ReadSkeletonResponse {
        status: "ok",
        file: display_relative(&state, &path),
        absolute_file: Some(path.display().to_string()),
        symbols,
        warnings,
        limitations: skeleton_limitations(),
    })
}

fn read_limitations() -> Vec<&'static str> {
    vec![
        "Rust HTTP read/block is experimental.",
        "Python HTTP/MCP runtime remains authoritative.",
    ]
}

fn skeleton_limitations() -> Vec<&'static str> {
    vec![
        "Rust HTTP read/skeleton is experimental.",
        "C# skeleton uses Tree-sitter and remains best-effort.",
        "Python HTTP/MCP runtime remains authoritative.",
    ]
}

fn fallback_skeleton(text: &str) -> Vec<SkeletonItem> {
    text.lines()
        .enumerate()
        .filter_map(|(idx, line)| {
            let trimmed = line.trim_start();
            let (kind, rest) = trimmed
                .strip_prefix("class ")
                .map(|rest| ("class", rest))
                .or_else(|| trimmed.strip_prefix("def ").map(|rest| ("function", rest)))
                .or_else(|| {
                    trimmed
                        .strip_prefix("function ")
                        .map(|rest| ("function", rest))
                })?;
            let name = rest
                .split(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
                .next()
                .filter(|value| !value.is_empty())
                .map(str::to_string);
            Some(SkeletonItem {
                kind: kind.to_string(),
                name,
                line: idx + 1,
            })
        })
        .collect()
}
