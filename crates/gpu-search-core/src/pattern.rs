//! CPU exact pattern search primitives for the experimental Rust core.
//!
//! This is intentionally a simple, dependency-free prototype. It searches file
//! bytes on CPU, returns line numbers and one-line snippets, and keeps result
//! structs close to the compact HTTP DTO concepts used by the Python runtime.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::file_discovery::DiscoveredFile;

/// Options for exact CPU pattern search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternSearchOptions {
    /// Match ASCII case exactly when true.
    pub case_sensitive: bool,
    /// Maximum matches returned across a search operation.
    pub max_results: usize,
    /// Maximum snippet characters retained per match.
    pub max_snippet_chars: usize,
}

impl Default for PatternSearchOptions {
    fn default() -> Self {
        Self {
            case_sensitive: true,
            max_results: 100,
            max_snippet_chars: 240,
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
            line: line_number(bytes, byte_offset),
            byte_offset,
            snippet: line_snippet(bytes, byte_offset, options.max_snippet_chars),
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

fn line_number(bytes: &[u8], byte_offset: usize) -> usize {
    bytes[..byte_offset.min(bytes.len())]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        + 1
}

fn line_snippet(bytes: &[u8], byte_offset: usize, max_chars: usize) -> String {
    let offset = byte_offset.min(bytes.len());
    let line_start = bytes[..offset]
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map(|pos| pos + 1)
        .unwrap_or(0);
    let line_end = bytes[offset..]
        .iter()
        .position(|byte| *byte == b'\n' || *byte == b'\r')
        .map(|pos| offset + pos)
        .unwrap_or(bytes.len());

    let snippet = String::from_utf8_lossy(&bytes[line_start..line_end]);
    let trimmed = snippet.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }

    trimmed.chars().take(max_chars).collect()
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
