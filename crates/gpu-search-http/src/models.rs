//! Experimental Rust HTTP request and response models.

use serde::{Deserialize, Serialize};
/// Health response for the experimental Rust HTTP server.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct HealthResponse {
    pub status: &'static str,
    pub implementation: &'static str,
    pub rust_core_version: &'static str,
    pub rust_http_version: &'static str,
}

/// Current experimental Rust HTTP capability flags.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CapabilityResponse {
    pub health: bool,
    pub stats: bool,
    pub diagnostics: bool,
    pub search_code: bool,
    pub dependency_impact: bool,
    pub signal_scan: bool,
    pub semantic_search: bool,
}

/// Lightweight stats response for the experimental Rust HTTP server.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct StatsResponse {
    pub status: &'static str,
    pub implementation: &'static str,
    pub rust_core_version: &'static str,
    pub rust_http_version: &'static str,
    pub indexed_roots: usize,
    pub indexed_files: usize,
    pub default_indexed_extensions: usize,
    pub default_skip_directories: usize,
    pub capabilities: CapabilityResponse,
    pub limitations: Vec<&'static str>,
}

/// Device metadata for the experimental Rust HTTP diagnostics endpoint.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticsDevice {
    pub backend: &'static str,
    pub reason: &'static str,
    pub warnings: Vec<&'static str>,
}

/// Index readiness metadata for the experimental Rust HTTP diagnostics endpoint.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticsIndexes {
    pub pattern_ready: bool,
    pub semantic_ready: bool,
    pub dependency_ready: bool,
    pub indexed_files: usize,
}

/// Lightweight diagnostics response for the experimental Rust HTTP server.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticsResponse {
    pub status: &'static str,
    pub implementation: &'static str,
    pub rust_core_version: &'static str,
    pub rust_http_version: &'static str,
    pub device: DiagnosticsDevice,
    pub indexes: DiagnosticsIndexes,
    pub capabilities: CapabilityResponse,
    pub warnings: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}

/// Semantic embedding model status for the experimental Rust HTTP server.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SemanticModelStatusResponse {
    pub model_id: String,
    pub provider: &'static str,
    pub available: bool,
    pub cached: bool,
    pub requires_download: bool,
    pub device: &'static str,
    pub message: String,
    pub limitations: Vec<&'static str>,
}

/// Search mode requested by clients. Kept stringly-compatible with Python HTTP.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    Auto,
    Pattern,
    Semantic,
    Hybrid,
}

/// Context mode requested by clients. Kept stringly-compatible with Python HTTP.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchContextMode {
    Compact,
    Normal,
    Full,
}

/// Experimental Rust `/search/code` request.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SearchCodeRequest {
    pub query: String,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub mode: Option<SearchMode>,
    #[serde(default)]
    pub context_mode: Option<SearchContextMode>,
}

/// Structured search result placeholder matching current DTO concepts.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SearchResultItem {
    pub file: String,
    pub line_start: usize,
    pub line_end: usize,
    pub score: Option<String>,
    pub reason: String,
    pub snippet: String,
}

/// Experimental Rust `/search/code` response.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SearchCodeResponse {
    pub result: String,
    pub results: Vec<SearchResultItem>,
    pub query: String,
    pub mode: SearchMode,
    pub context_mode: SearchContextMode,
    pub top_k: usize,
    pub status: &'static str,
    pub warnings: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}

/// Experimental Rust `/dependency/impact` request.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DependencyImpactRequest {
    pub filepath: String,
}

/// Structured impacted-file item for Rust dependency impact.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ImpactedFileResponse {
    pub file: String,
    pub absolute_file: String,
    pub hops: usize,
    pub reason: Option<String>,
}

/// Experimental Rust `/dependency/impact` response.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DependencyImpactResponse {
    pub status: &'static str,
    pub file: String,
    pub impacted_files: Vec<ImpactedFileResponse>,
    pub confidence: &'static str,
    pub analysis_mode: &'static str,
    pub warnings: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}

/// Experimental Rust `/read/block` request.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ReadBlockRequest {
    pub filepath: String,
    #[serde(default)]
    pub line_start: Option<usize>,
    #[serde(default)]
    pub line_end: Option<usize>,
}

/// Experimental Rust `/read/block` response.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ReadBlockResponse {
    pub status: &'static str,
    pub file: String,
    pub absolute_file: Option<String>,
    pub line_start: usize,
    pub line_end: usize,
    pub content: String,
    pub warnings: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}

/// Experimental Rust `/read/skeleton` request.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ReadSkeletonRequest {
    pub filepath: String,
}

/// A single symbol-like skeleton item.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SkeletonItem {
    pub kind: String,
    pub name: Option<String>,
    pub line: usize,
}

/// Experimental Rust `/read/skeleton` response.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ReadSkeletonResponse {
    pub status: &'static str,
    pub file: String,
    pub absolute_file: Option<String>,
    pub symbols: Vec<SkeletonItem>,
    pub warnings: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}

/// Experimental Rust `/scan/signals` request.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SignalScanRequest {
    #[serde(default)]
    pub categories: Vec<String>,
    #[serde(default)]
    pub top_k_per_signal: Option<usize>,
    #[serde(default)]
    pub include_snippets: Option<bool>,
}

/// A matched signal occurrence.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SignalMatchResponse {
    pub file: String,
    pub line_start: usize,
    pub line_end: usize,
    pub snippet: Option<String>,
}

/// A signal and its matches.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SignalResultResponse {
    pub id: &'static str,
    pub category: &'static str,
    pub label: &'static str,
    pub query: &'static str,
    pub matches: Vec<SignalMatchResponse>,
}

/// Experimental Rust `/scan/signals` response.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SignalScanResponse {
    pub status: &'static str,
    pub signals: Vec<SignalResultResponse>,
    pub total_matches: usize,
    pub warnings: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}
