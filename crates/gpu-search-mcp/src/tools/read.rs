//! Experimental Rust MCP read tool handlers.

use gpu_search_core::{LineIndex, file_ext, parse_csharp_ast_summary};
use serde_json::{Value, json};
use std::fs;

use super::common::{
    csharp_regex_skeleton, display_relative, fallback_skeleton, resolve_under_root, tool_error,
};

pub(crate) fn rust_read_block_tool_result(arguments: &Value) -> Value {
    let directory = arguments
        .get("directory")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let filepath = arguments
        .get("filepath")
        .or_else(|| arguments.get("file"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let line_start = arguments
        .get("lineStart")
        .or_else(|| arguments.get("line_start"))
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1) as usize;
    let line_end = arguments
        .get("lineEnd")
        .or_else(|| arguments.get("line_end"))
        .and_then(Value::as_u64)
        .unwrap_or(line_start as u64)
        .max(line_start as u64) as usize;

    if directory.trim().is_empty() || filepath.trim().is_empty() {
        return tool_error("rust_read_block requires non-empty directory and filepath arguments.");
    }

    let Some(path) = resolve_under_root(directory, filepath) else {
        return tool_error("Requested file is outside the supplied directory or does not exist.");
    };

    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => return tool_error(&format!("Failed to read file: {error}")),
    };
    if String::from_utf8(bytes.clone()).is_err() {
        return tool_error("Requested file could not be read as UTF-8 text.");
    }

    let line_index = LineIndex::new(&bytes);
    let bounded_start = line_start.min(line_index.line_count().max(1));
    let bounded_end = line_end.min(line_index.line_count().max(bounded_start));
    let Some((content_start, _)) = line_index.line_range(&bytes, bounded_start) else {
        return tool_error("Requested start line is outside the file.");
    };
    let Some((_, content_end)) = line_index.line_range(&bytes, bounded_end) else {
        return tool_error("Requested end line is outside the file.");
    };
    let content = String::from_utf8_lossy(&bytes[content_start..content_end])
        .replace("\r\n", "\n")
        .to_string();
    let display_file = path.display().to_string();
    let returned_lines = bounded_end.saturating_sub(bounded_start) + 1;
    let text_summary = format!(
        "Rust read block returned {} line(s) from {}.",
        returned_lines, display_file
    );

    json!({
        "content": [
            {
                "type": "text",
                "text": text_summary
            }
        ],
        "structuredContent": {
            "file": display_file,
            "lineStart": bounded_start,
            "lineEnd": bounded_end,
            "content": content,
            "limitations": [
                "Experimental Rust MCP read block only.",
                "Python MCP runtime remains authoritative.",
                "Reads are bounded to UTF-8 files under the explicit directory argument."
            ]
        }
    })
}

pub(crate) fn rust_read_skeleton_tool_result(arguments: &Value) -> Value {
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
            "rust_read_skeleton requires non-empty directory and filepath arguments.",
        );
    }

    let Some(path) = resolve_under_root(directory, filepath) else {
        return tool_error("Requested file is outside the supplied directory or does not exist.");
    };

    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => return tool_error(&format!("Failed to read UTF-8 source file: {error}")),
    };

    let mut warnings = Vec::new();
    let symbols = if file_ext(&path).as_deref() == Some(".cs") {
        match parse_csharp_ast_summary(&text) {
            Ok(items) => items
                .into_iter()
                .map(|item| {
                    json!({
                        "kind": item.kind,
                        "name": item.name,
                        "line": item.line
                    })
                })
                .collect::<Vec<_>>(),
            Err(_) => {
                warnings.push(
                    "C# Tree-sitter skeleton parsing failed; returned regex fallback skeleton.",
                );
                csharp_regex_skeleton(&text)
            }
        }
    } else {
        warnings
            .push("Tree-sitter skeleton support is currently C#-only; returned fallback skeleton.");
        fallback_skeleton(&text)
    };
    let file = display_relative(directory, &path);
    let text_summary = format!(
        "Rust skeleton returned {} symbol(s) from {}.",
        symbols.len(),
        file
    );

    json!({
        "content": [
            {
                "type": "text",
                "text": text_summary
            }
        ],
        "structuredContent": {
            "file": file,
            "absoluteFile": path.display().to_string(),
            "symbols": symbols,
            "warnings": warnings,
            "limitations": [
                "Experimental Rust MCP skeleton only.",
                "C# skeleton uses Tree-sitter; non-C# files use a small fallback.",
                "Python MCP runtime remains authoritative."
            ]
        }
    })
}
