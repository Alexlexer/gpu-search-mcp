//! CPU exact pattern search primitives for the experimental Rust core.
//!
//! This is intentionally a simple, dependency-free prototype. It searches file
//! bytes on CPU, returns line numbers and one-line snippets, and keeps result
//! structs close to the compact HTTP DTO concepts used by the Python runtime.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::{file_discovery::DiscoveredFile, line_index::LineIndex};

/// Controls how much source context exact-search matches retain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextMode {
    /// Lowest-token result: one trimmed line and metadata only.
    Compact,
    /// Default result: one-line snippet, matching the current prototype behavior.
    Normal,
    /// Expanded result: includes nearby lines around the match.
    Full,
}

impl Default for ContextMode {
    fn default() -> Self {
        Self::Normal
    }
}

/// Options for exact CPU pattern search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternSearchOptions {
    /// Match ASCII case exactly when true.
    pub case_sensitive: bool,
    /// Maximum matches returned across a search operation.
    pub max_results: usize,
    /// Maximum snippet characters retained per match.
    pub max_snippet_chars: usize,
    /// Amount of source context retained per match.
    pub context_mode: ContextMode,
}

impl Default for PatternSearchOptions {
    fn default() -> Self {
        Self {
            case_sensitive: true,
            max_results: 100,
            max_snippet_chars: 240,
            context_mode: ContextMode::default(),
        }
    }
}

/// A compact exact-search match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternMatch {
    /// Matched file path.
    pub file: PathBuf,
    /// 1-based line number.
    pub line: usize,
    /// 0-based byte offset in the file.
    pub byte_offset: usize,
    /// One-line snippet containing the match.
    pub snippet: String,
    /// Context mode used to shape this result.
    pub context_mode: ContextMode,
    /// Advisory explanation for why this result was returned.
    pub reason: String,
}

/// Error returned by file-backed pattern search.
#[derive(Debug)]
pub struct PatternSearchError {
    path: PathBuf,
    source: io::Error,
}

impl PatternSearchError {
    fn new(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self {
            path: path.into(),
            source,
        }
    }

    /// File path that failed during search.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Original IO error.
    pub fn source(&self) -> &io::Error {
        &self.source
    }
}

impl fmt::Display for PatternSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to search {}: {}",
            self.path.display(),
            self.source
        )
    }
}

impl std::error::Error for PatternSearchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Search a byte slice and return compact matches for a synthetic file path.
pub fn search_bytes(
    file: impl Into<PathBuf>,
    bytes: &[u8],
    query: &str,
    options: &PatternSearchOptions,
) -> Vec<PatternMatch> {
    let query_bytes = query.as_bytes();
    if query_bytes.is_empty() || options.max_results == 0 || bytes.len() < query_bytes.len() {
        return Vec::new();
    }

    let file = file.into();
    let line_index = LineIndex::new(bytes);
    let mut matches = Vec::new();
    let mut start = 0;

    while start + query_bytes.len() <= bytes.len() && matches.len() < options.max_results {
        let Some(relative_offset) = find_from(&bytes[start..], query_bytes, options.case_sensitive)
        else {
            break;
        };
        let byte_offset = start + relative_offset;
        matches.push(PatternMatch {
            file: file.clone(),
            line: line_index.line_number(byte_offset),
            byte_offset,
            snippet: build_snippet(bytes, &line_index, byte_offset, options),
            context_mode: options.context_mode,
            reason: match_reason(options),
        });
        start = byte_offset + query_bytes.len().max(1);
    }

    matches
}

/// Search a single file from disk.
pub fn search_file(
    file: impl AsRef<Path>,
    query: &str,
    options: &PatternSearchOptions,
) -> Result<Vec<PatternMatch>, PatternSearchError> {
    let file = file.as_ref();
    let bytes = fs::read(file).map_err(|err| PatternSearchError::new(file, err))?;
    Ok(search_bytes(file.to_path_buf(), &bytes, query, options))
}

/// Search discovered files from disk in deterministic input order.
pub fn search_files(
    files: &[DiscoveredFile],
    query: &str,
    options: &PatternSearchOptions,
) -> Result<Vec<PatternMatch>, PatternSearchError> {
    let mut results = Vec::new();
    for file in files {
        if results.len() >= options.max_results {
            break;
        }

        let remaining = options.max_results - results.len();
        let per_file_options = PatternSearchOptions {
            max_results: remaining,
            ..options.clone()
        };
        results.extend(search_file(&file.path, query, &per_file_options)?);
    }

    Ok(results)
}

fn find_from(haystack: &[u8], needle: &[u8], case_sensitive: bool) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| bytes_eq(window, needle, case_sensitive))
}

fn bytes_eq(left: &[u8], right: &[u8], case_sensitive: bool) -> bool {
    if case_sensitive {
        left == right
    } else {
        left.eq_ignore_ascii_case(right)
    }
}

fn build_snippet(
    bytes: &[u8],
    line_index: &LineIndex,
    byte_offset: usize,
    options: &PatternSearchOptions,
) -> String {
    match options.context_mode {
        ContextMode::Compact | ContextMode::Normal => {
            line_index.snippet_at(bytes, byte_offset, options.max_snippet_chars)
        }
        ContextMode::Full => expanded_snippet(bytes, byte_offset, options.max_snippet_chars),
    }
}

fn match_reason(options: &PatternSearchOptions) -> String {
    if options.case_sensitive {
        "exact token match".to_string()
    } else {
        "case-insensitive exact token match".to_string()
    }
}

fn expanded_snippet(bytes: &[u8], byte_offset: usize, max_chars: usize) -> String {
    let mut line_start = byte_offset.min(bytes.len());
    while line_start > 0 && bytes[line_start - 1] != b'\n' {
        line_start -= 1;
    }
    if line_start > 0 {
        line_start -= 1;
        while line_start > 0 && bytes[line_start - 1] != b'\n' {
            line_start -= 1;
        }
    }

    let mut line_end = byte_offset.min(bytes.len());
    let mut newlines_seen = 0;
    while line_end < bytes.len() {
        if bytes[line_end] == b'\n' {
            newlines_seen += 1;
            if newlines_seen == 2 {
                break;
            }
        }
        line_end += 1;
    }

    let snippet = String::from_utf8_lossy(&bytes[line_start..line_end])
        .replace("\r\n", "\n")
        .trim()
        .to_string();

    if snippet.chars().count() <= max_chars {
        snippet
    } else {
        snippet.chars().take(max_chars).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file(name: &str, content: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gpu_search_core_pattern_{name}_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root).expect("temp root should be created");
        let path = root.join("sample.rs");
        fs::write(&path, content).expect("file should be written");
        path
    }

    #[test]
    fn search_bytes_returns_line_and_snippet() {
        let options = PatternSearchOptions::default();
        let matches = search_bytes(
            "src/lib.rs",
            b"first line\nlet user_service = UserService::new();\nlast line",
            "UserService",
            &options,
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].line, 2);
        assert_eq!(matches[0].byte_offset, 30);
        assert_eq!(matches[0].snippet, "let user_service = UserService::new();");
        assert_eq!(matches[0].context_mode, ContextMode::Normal);
        assert_eq!(matches[0].reason, "exact token match");
    }

    #[test]
    fn search_bytes_supports_ascii_case_insensitive() {
        let options = PatternSearchOptions {
            case_sensitive: false,
            ..PatternSearchOptions::default()
        };
        let matches = search_bytes("app.cs", b"class UserService {}", "userservice", &options);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].line, 1);
        assert_eq!(matches[0].reason, "case-insensitive exact token match");
    }

    #[test]
    fn compact_context_keeps_one_line_snippet() {
        let options = PatternSearchOptions {
            context_mode: ContextMode::Compact,
            ..PatternSearchOptions::default()
        };
        let matches = search_bytes(
            "app.rs",
            b"before\nlet needle = true;\nafter",
            "needle",
            &options,
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].context_mode, ContextMode::Compact);
        assert_eq!(matches[0].snippet, "let needle = true;");
    }

    #[test]
    fn full_context_includes_neighboring_lines() {
        let options = PatternSearchOptions {
            context_mode: ContextMode::Full,
            max_snippet_chars: 200,
            ..PatternSearchOptions::default()
        };
        let matches = search_bytes(
            "app.rs",
            b"line one\nline two needle\nline three\nline four",
            "needle",
            &options,
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].context_mode, ContextMode::Full);
        assert_eq!(matches[0].snippet, "line one\nline two needle\nline three");
    }

    #[test]
    fn search_bytes_respects_max_results() {
        let options = PatternSearchOptions {
            max_results: 2,
            ..PatternSearchOptions::default()
        };
        let matches = search_bytes("app.py", b"TODO\nTODO\nTODO", "TODO", &options);

        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].line, 1);
        assert_eq!(matches[1].line, 2);
    }

    #[test]
    fn search_file_reads_from_disk() {
        let path = temp_file("disk", "fn main() {\n    println!(\"hello\");\n}\n");
        let matches = search_file(&path, "println", &PatternSearchOptions::default())
            .expect("search should read file");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].line, 2);
        fs::remove_dir_all(path.parent().expect("temp file has parent")).ok();
    }

    #[test]
    fn search_files_preserves_discovered_file_order_and_global_limit() {
        let first = temp_file("first", "needle one\nneedle two");
        let second = temp_file("second", "needle three");
        let files = vec![
            DiscoveredFile {
                path: first.clone(),
                size: 21,
                modified_ns: 1,
            },
            DiscoveredFile {
                path: second.clone(),
                size: 12,
                modified_ns: 1,
            },
        ];
        let options = PatternSearchOptions {
            max_results: 2,
            ..PatternSearchOptions::default()
        };

        let matches = search_files(&files, "needle", &options).expect("search should work");

        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].file, first);
        assert_eq!(matches[1].file, files[0].path);
        fs::remove_dir_all(files[0].path.parent().expect("first has parent")).ok();
        fs::remove_dir_all(files[1].path.parent().expect("second has parent")).ok();
    }
}
