//! Experimental Rust core for `gpu-search-mcp`.
//!
//! This crate is intentionally small for the first Rust rewrite milestone. It
//! does not replace the Python runtime yet; it only establishes a tested Rust
//! workspace where indexing/search primitives can be ported incrementally.

pub mod file_discovery;
pub mod pattern;

use std::path::Path;

pub use file_discovery::{DiscoveredFile, DiscoveryError, discover_files};
pub use pattern::{
    PatternMatch, PatternSearchError, PatternSearchOptions, search_bytes, search_file, search_files,
};

/// Current experimental Rust core API version.
pub const RUST_CORE_VERSION: &str = "0.1.0-prototype";

/// File extensions indexed by the current Python implementation.
pub const DEFAULT_INDEXED_EXTS: &[&str] = &[
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".java", ".cs",
    ".rb", ".php", ".swift", ".kt", ".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".html",
    ".css", ".scss", ".sql", ".sh", ".bat", ".ps1", ".cfg", ".ini", ".xml",
];

/// Directories skipped by default while indexing.
pub const DEFAULT_SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "target",
    "bin",
    "obj",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".gpu-search-cache",
];

/// Minimal indexing options shared by future discovery/search code.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexOptions {
    /// Include `.env` dotfiles when true. Defaults to false for safety.
    pub allow_env_files: bool,
    /// Maximum file size in MiB.
    pub max_file_mb: f64,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            allow_env_files: false,
            max_file_mb: 5.0,
        }
    }
}

/// Return the normalized indexable extension for a file name.
///
/// Mirrors the Python implementation's handling of dotfiles such as `.env`,
/// whose extension is treated as the whole file name.
pub fn file_ext(path: impl AsRef<Path>) -> Option<String> {
    let path = path.as_ref();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        return Some(format!(".{ext}").to_ascii_lowercase());
    }

    let name = path.file_name()?.to_str()?;
    if name.starts_with('.') && name.matches('.').count() == 1 {
        return Some(name.to_ascii_lowercase());
    }

    None
}

/// True when a file should be considered indexable under the given options.
pub fn is_indexable_file(path: impl AsRef<Path>, options: &IndexOptions) -> bool {
    match file_ext(path) {
        Some(ext) if ext == ".env" => options.allow_env_files,
        Some(ext) => DEFAULT_INDEXED_EXTS.contains(&ext.as_str()),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_exclude_env_files() {
        let options = IndexOptions::default();

        assert!(is_indexable_file("src/main.rs", &options));
        assert!(is_indexable_file("README.md", &options));
        assert!(!is_indexable_file(".env", &options));
    }

    #[test]
    fn allow_env_files_opts_in() {
        let options = IndexOptions {
            allow_env_files: true,
            ..IndexOptions::default()
        };

        assert!(is_indexable_file(".env", &options));
    }

    #[test]
    fn file_ext_handles_dotfiles_like_python() {
        assert_eq!(file_ext(".env"), Some(".env".to_string()));
        assert_eq!(file_ext("Service.CS"), Some(".cs".to_string()));
        assert_eq!(file_ext("Dockerfile"), None);
    }
}
