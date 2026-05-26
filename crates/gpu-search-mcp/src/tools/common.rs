//! Shared helpers for experimental Rust MCP tool handlers.

use gpu_search_core::parse_csharp_regex_summary;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

pub(crate) fn csharp_regex_skeleton(text: &str) -> Vec<Value> {
    parse_csharp_regex_summary(text)
        .into_iter()
        .map(|item| {
            json!({
                "kind": item.kind,
                "name": item.name,
                "line": item.line
            })
        })
        .collect()
}

pub(crate) fn fallback_skeleton(text: &str) -> Vec<Value> {
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
                .filter(|value| !value.is_empty());
            Some(json!({
                "kind": kind,
                "name": name,
                "line": idx + 1
            }))
        })
        .collect()
}

pub(crate) fn resolve_under_root(directory: &str, filepath: &str) -> Option<PathBuf> {
    let root = Path::new(directory).canonicalize().ok()?;
    let candidate = Path::new(filepath);
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root.join(candidate)
    };
    let resolved = joined.canonicalize().ok()?;
    resolved.starts_with(root).then_some(resolved)
}

pub(crate) fn display_relative(directory: &str, filepath: &Path) -> String {
    Path::new(directory)
        .canonicalize()
        .ok()
        .and_then(|root| filepath.strip_prefix(root).ok().map(Path::to_path_buf))
        .unwrap_or_else(|| filepath.to_path_buf())
        .display()
        .to_string()
}

pub(crate) fn tool_error(message: &str) -> Value {
    json!({
        "isError": true,
        "content": [
            {
                "type": "text",
                "text": message
            }
        ]
    })
}
