//! Experimental Rust core for `gpu-search-mcp`.
//!
//! This crate is intentionally small for the first Rust rewrite milestone. It
//! does not replace the Python runtime yet; it only establishes a tested Rust
//! workspace where indexing/search primitives can be ported incrementally.

pub mod cache;
pub mod csharp_ast;
pub mod deps;
pub mod file_discovery;
pub mod index_config;
pub mod line_index;
pub mod pattern;

pub use cache::{
    CACHE_METADATA_FILE, CACHE_SCHEMA_VERSION, CacheEntry, CacheMetadata, CacheMetadataError,
    DEPENDENCY_CACHE_SCHEMA_VERSION, PATTERN_CACHE_SCHEMA_VERSION, SEMANTIC_CACHE_SCHEMA_VERSION,
    SourceFingerprint, compute_source_fingerprint, invalidate_cache_entry, is_cache_entry_valid,
    load_cache_metadata, new_cache_metadata, save_cache_metadata,
};
pub use csharp_ast::{
    CSharpAstItem, CSharpAstParseError, parse_csharp_ast_summary, parse_csharp_regex_summary,
};
pub use deps::{
    DEPENDENCY_ANALYSIS_MODE, DependencyEdge, DependencyGraph, DependencyGraphError, ImpactedFile,
};

pub use file_discovery::{DiscoveredFile, DiscoveryError, discover_files};
pub use index_config::{
    DEFAULT_INDEXED_EXTS, DEFAULT_SKIP_DIRS, IndexOptions, file_ext, is_indexable_file,
};
pub use line_index::LineIndex;
pub use pattern::{
    ContextMode, PatternMatch, PatternSearchError, PatternSearchOptions, search_bytes, search_file,
    search_files,
};

/// Current experimental Rust core API version.
pub const RUST_CORE_VERSION: &str = "0.1.0-prototype";
