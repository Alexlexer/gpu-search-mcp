//! Experimental Rust MCP dependency-impact tool handler.

use gpu_search_core::{DEPENDENCY_ANALYSIS_MODE, DependencyGraph, IndexOptions, discover_files};
use serde_json::{Value, json};
use std::path::Path;

use super::{display_relative, resolve_under_root, tool_error};

pub(crate) fn rust_dependency_impact_tool_result(arguments: &Value) -> Value {
    let directory = arguments
        .get("directory")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let filepath = arguments
        .get("filepath")
        .or_else(|| arguments.get("file"))
        .and_then(Value::as_str)
        .unwrap_or_default();

    if directory.trim().is_empty() || filepath.trim().is_empty() {
        return tool_error(
            "rust_dependency_impact requires non-empty directory and filepath arguments.",
        );
    }

    let Some(changed_file) = resolve_under_root(directory, filepath) else {
        return tool_error("Requested file is outside the supplied directory or does not exist.");
    };

    let root = match Path::new(directory).canonicalize() {
        Ok(root) => root,
        Err(error) => return tool_error(&format!("Failed to resolve directory: {error}")),
    };
    let files = match discover_files(&root, &IndexOptions::default()) {
        Ok(files) => files,
        Err(error) => return tool_error(&format!("Failed to discover files: {error}")),
    };
    let graph = match DependencyGraph::from_files(&files) {
        Ok(graph) => graph,
        Err(error) => return tool_error(&format!("Failed to build dependency graph: {error}")),
    };

    let impacted_files: Vec<Value> = graph
        .impact(&changed_file)
        .into_iter()
        .map(|impacted| {
            json!({
                "file": display_relative(directory, &impacted.file),
                "absoluteFile": impacted.file.display().to_string(),
                "hops": impacted.hops,
                "reason": impacted.reason
            })
        })
        .collect();
    let changed_display = display_relative(directory, &changed_file);
    let text = format!(
        "Rust dependency impact found {} impacted file(s) for {}.",
        impacted_files.len(),
        changed_display
    );

    json!({
        "content": [
            {
                "type": "text",
                "text": text
            }
        ],
        "structuredContent": {
            "file": changed_display,
            "absoluteFile": changed_file.display().to_string(),
            "impactedFiles": impacted_files,
            "confidence": "medium",
            "analysisMode": DEPENDENCY_ANALYSIS_MODE,
            "limitations": [
                "Experimental Rust MCP dependency impact only.",
                "Dependency impact is heuristic, not compiler-accurate.",
                "Python MCP runtime remains authoritative."
            ]
        }
    })
}
