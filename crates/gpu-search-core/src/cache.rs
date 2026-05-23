//! Cache metadata helpers for the experimental Rust core.
//!
//! This module intentionally avoids external JSON dependencies for now so the
//! early Rust workspace stays dependency-free. It writes a stable
//! `.gpu-search-cache/cache-meta.json` shape compatible in spirit with the
//! Python cache metadata work and safely treats unreadable/legacy metadata as
//! absent.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{IndexOptions, RUST_CORE_VERSION, file_discovery::DiscoveredFile};

/// Current Rust cache metadata schema version.
pub const CACHE_SCHEMA_VERSION: u32 = 1;
/// Pattern cache schema version.
pub const PATTERN_CACHE_SCHEMA_VERSION: u32 = 1;
/// Dependency cache schema version.
pub const DEPENDENCY_CACHE_SCHEMA_VERSION: u32 = 1;
/// Semantic cache schema version placeholder for compatibility tracking.
pub const SEMANTIC_CACHE_SCHEMA_VERSION: u32 = 1;
/// Cache metadata file name under `.gpu-search-cache/`.
pub const CACHE_METADATA_FILE: &str = "cache-meta.json";

/// Lightweight source fingerprint used to detect stale cache metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFingerprint {
    /// Repository root associated with this fingerprint.
    pub repo_root: PathBuf,
    /// Number of indexed source files.
    pub indexed_file_count: usize,
    /// Maximum indexed file modified timestamp in nanoseconds.
    pub max_modified_ns: u128,
    /// Whether `.env` files were included.
    pub allow_env_files: bool,
    /// Max indexed file size setting in MiB, preserved as text to avoid float equality issues.
    pub max_file_mb: String,
}

/// A single cache entry tracked by metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    /// Cache entry name, e.g. `pattern`, `dependency`, or `semantic`.
    pub name: String,
    /// Entry-specific schema version.
    pub schema_version: u32,
    /// Relative or absolute cache file path.
    pub file_path: String,
    /// Creation timestamp as seconds since Unix epoch.
    pub created_at_utc: String,
    /// Update timestamp as seconds since Unix epoch.
    pub updated_at_utc: String,
    /// Source fingerprint associated with the entry.
    pub source_fingerprint: SourceFingerprint,
    /// Entry status, e.g. `built`, `loaded`, or `invalidated`.
    pub status: String,
}

/// Top-level cache metadata document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheMetadata {
    /// Metadata schema version.
    pub schema_version: u32,
    /// Creation timestamp as seconds since Unix epoch.
    pub created_at_utc: String,
    /// Update timestamp as seconds since Unix epoch.
    pub updated_at_utc: String,
    /// Rust core version that wrote this metadata.
    pub gpu_search_version: String,
    /// Repository root associated with this cache directory.
    pub repo_root: PathBuf,
    /// Python version field kept for shape compatibility; Rust writes `rust-core`.
    pub python_version: String,
    /// Platform string.
    pub platform: String,
    /// Cache entries.
    pub cache_entries: Vec<CacheEntry>,
}

/// Error returned by cache metadata helpers.
#[derive(Debug)]
pub struct CacheMetadataError {
    path: PathBuf,
    source: io::Error,
}

impl CacheMetadataError {
    fn new(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self {
            path: path.into(),
            source,
        }
    }
}

impl fmt::Display for CacheMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cache metadata error at {}: {}",
            self.path.display(),
            self.source
        )
    }
}

impl std::error::Error for CacheMetadataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Compute a cheap source fingerprint from already-discovered files.
pub fn compute_source_fingerprint(
    repo_root: impl AsRef<Path>,
    files: &[DiscoveredFile],
    options: &IndexOptions,
) -> SourceFingerprint {
    SourceFingerprint {
        repo_root: repo_root.as_ref().to_path_buf(),
        indexed_file_count: files.len(),
        max_modified_ns: files
            .iter()
            .map(|file| file.modified_ns)
            .max()
            .unwrap_or_default(),
        allow_env_files: options.allow_env_files,
        max_file_mb: format!("{:.6}", options.max_file_mb),
    }
}

/// Create new metadata with no entries.
pub fn new_cache_metadata(repo_root: impl AsRef<Path>) -> CacheMetadata {
    let now = now_utc_string();
    CacheMetadata {
        schema_version: CACHE_SCHEMA_VERSION,
        created_at_utc: now.clone(),
        updated_at_utc: now,
        gpu_search_version: RUST_CORE_VERSION.to_string(),
        repo_root: repo_root.as_ref().to_path_buf(),
        python_version: "rust-core".to_string(),
        platform: std::env::consts::OS.to_string(),
        cache_entries: Vec::new(),
    }
}

/// Return true when a cache entry matches schema, fingerprint, and active status.
pub fn is_cache_entry_valid(
    entry: &CacheEntry,
    expected_schema_version: u32,
    expected_fingerprint: &SourceFingerprint,
) -> bool {
    entry.schema_version == expected_schema_version
        && entry.source_fingerprint == *expected_fingerprint
        && !matches!(entry.status.as_str(), "invalid" | "invalidated" | "stale")
}

/// Mark a cache entry invalidated in-place.
pub fn invalidate_cache_entry(entry: &mut CacheEntry) {
    entry.status = "invalidated".to_string();
    entry.updated_at_utc = now_utc_string();
}

/// Load cache metadata if present and compatible. Missing, legacy, corrupt, or
/// mismatched top-level schema metadata returns `Ok(None)` instead of crashing.
pub fn load_cache_metadata(
    cache_dir: impl AsRef<Path>,
) -> Result<Option<CacheMetadata>, CacheMetadataError> {
    let path = cache_dir.as_ref().join(CACHE_METADATA_FILE);
    if !path.exists() {
        return Ok(None);
    }

    let text = fs::read_to_string(&path).map_err(|err| CacheMetadataError::new(&path, err))?;
    let Some(metadata) = parse_metadata(&text) else {
        return Ok(None);
    };
    if metadata.schema_version != CACHE_SCHEMA_VERSION {
        return Ok(None);
    }

    Ok(Some(metadata))
}

/// Save cache metadata, creating the cache directory if needed.
pub fn save_cache_metadata(
    cache_dir: impl AsRef<Path>,
    metadata: &CacheMetadata,
) -> Result<(), CacheMetadataError> {
    let cache_dir = cache_dir.as_ref();
    fs::create_dir_all(cache_dir).map_err(|err| CacheMetadataError::new(cache_dir, err))?;
    let path = cache_dir.join(CACHE_METADATA_FILE);
    fs::write(&path, metadata_to_json(metadata)).map_err(|err| CacheMetadataError::new(path, err))
}

fn metadata_to_json(metadata: &CacheMetadata) -> String {
    let entries = metadata
        .cache_entries
        .iter()
        .map(entry_to_json)
        .collect::<Vec<_>>()
        .join(",\n    ");
    format!(
        concat!(
            "{{\n",
            "  \"schemaVersion\": {},\n",
            "  \"createdAtUtc\": \"{}\",\n",
            "  \"updatedAtUtc\": \"{}\",\n",
            "  \"gpuSearchVersion\": \"{}\",\n",
            "  \"repoRoot\": \"{}\",\n",
            "  \"pythonVersion\": \"{}\",\n",
            "  \"platform\": \"{}\",\n",
            "  \"cacheEntries\": [\n    {}\n  ]\n",
            "}}\n"
        ),
        metadata.schema_version,
        escape_json(&metadata.created_at_utc),
        escape_json(&metadata.updated_at_utc),
        escape_json(&metadata.gpu_search_version),
        escape_json(&metadata.repo_root.to_string_lossy()),
        escape_json(&metadata.python_version),
        escape_json(&metadata.platform),
        entries
    )
}

fn entry_to_json(entry: &CacheEntry) -> String {
    format!(
        concat!(
            "{{\n",
            "      \"name\": \"{}\",\n",
            "      \"schemaVersion\": {},\n",
            "      \"filePath\": \"{}\",\n",
            "      \"createdAtUtc\": \"{}\",\n",
            "      \"updatedAtUtc\": \"{}\",\n",
            "      \"sourceFingerprint\": {},\n",
            "      \"status\": \"{}\"\n",
            "    }}"
        ),
        escape_json(&entry.name),
        entry.schema_version,
        escape_json(&entry.file_path),
        escape_json(&entry.created_at_utc),
        escape_json(&entry.updated_at_utc),
        fingerprint_to_json(&entry.source_fingerprint),
        escape_json(&entry.status)
    )
}

fn fingerprint_to_json(fingerprint: &SourceFingerprint) -> String {
    format!(
        concat!(
            "{{",
            "\"repoRoot\": \"{}\", ",
            "\"indexedFileCount\": {}, ",
            "\"maxModifiedNs\": {}, ",
            "\"allowEnvFiles\": {}, ",
            "\"maxFileMb\": \"{}\"",
            "}}"
        ),
        escape_json(&fingerprint.repo_root.to_string_lossy()),
        fingerprint.indexed_file_count,
        fingerprint.max_modified_ns,
        fingerprint.allow_env_files,
        escape_json(&fingerprint.max_file_mb)
    )
}

fn parse_metadata(text: &str) -> Option<CacheMetadata> {
    let schema_version = extract_u32(text, "schemaVersion")?;
    let mut metadata = CacheMetadata {
        schema_version,
        created_at_utc: extract_string(text, "createdAtUtc")?,
        updated_at_utc: extract_string(text, "updatedAtUtc")?,
        gpu_search_version: extract_string(text, "gpuSearchVersion")?,
        repo_root: PathBuf::from(extract_string(text, "repoRoot")?),
        python_version: extract_string(text, "pythonVersion").unwrap_or_default(),
        platform: extract_string(text, "platform").unwrap_or_default(),
        cache_entries: Vec::new(),
    };

    for object in entry_objects(text) {
        let fingerprint_block = object_between(object, "sourceFingerprint")?;
        metadata.cache_entries.push(CacheEntry {
            name: extract_string(object, "name")?,
            schema_version: extract_u32(object, "schemaVersion")?,
            file_path: extract_string(object, "filePath")?,
            created_at_utc: extract_string(object, "createdAtUtc")?,
            updated_at_utc: extract_string(object, "updatedAtUtc")?,
            source_fingerprint: SourceFingerprint {
                repo_root: PathBuf::from(extract_string(fingerprint_block, "repoRoot")?),
                indexed_file_count: extract_usize(fingerprint_block, "indexedFileCount")?,
                max_modified_ns: extract_u128(fingerprint_block, "maxModifiedNs")?,
                allow_env_files: extract_bool(fingerprint_block, "allowEnvFiles")?,
                max_file_mb: extract_string(fingerprint_block, "maxFileMb")?,
            },
            status: extract_string(object, "status")?,
        });
    }

    Some(metadata)
}

fn entry_objects(text: &str) -> Vec<&str> {
    let Some(entries_start) = text.find("\"cacheEntries\"") else {
        return Vec::new();
    };
    let entries = &text[entries_start..];
    let mut objects = Vec::new();
    let mut cursor = 0;
    while let Some(relative_start) = entries[cursor..].find("{\n      \"name\"") {
        let start = cursor + relative_start;
        let Some(relative_end) = entries[start..].find("\n    }") else {
            break;
        };
        let end = start + relative_end + "\n    }".len();
        objects.push(&entries[start..end]);
        cursor = end;
    }
    objects
}

fn object_between<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("\"{key}\"");
    let start = text.find(&marker)?;
    let object_start = text[start..].find('{')? + start;
    let object_end = text[object_start..].find('}')? + object_start + 1;
    Some(&text[object_start..object_end])
}

fn extract_string(text: &str, key: &str) -> Option<String> {
    let marker = format!("\"{key}\"");
    let start = text.find(&marker)?;
    let after_key = &text[start + marker.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    let value = after_colon.strip_prefix('"')?;
    let end = value.find('"')?;
    Some(unescape_json(&value[..end]))
}

fn extract_u32(text: &str, key: &str) -> Option<u32> {
    extract_number(text, key)?.parse().ok()
}

fn extract_usize(text: &str, key: &str) -> Option<usize> {
    extract_number(text, key)?.parse().ok()
}

fn extract_u128(text: &str, key: &str) -> Option<u128> {
    extract_number(text, key)?.parse().ok()
}

fn extract_bool(text: &str, key: &str) -> Option<bool> {
    let marker = format!("\"{key}\"");
    let start = text.find(&marker)?;
    let after_key = &text[start + marker.len()..];
    let colon = after_key.find(':')?;
    let value = after_key[colon + 1..].trim_start();
    if value.starts_with("true") {
        Some(true)
    } else if value.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn extract_number<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("\"{key}\"");
    let start = text.find(&marker)?;
    let after_key = &text[start + marker.len()..];
    let colon = after_key.find(':')?;
    let value = after_key[colon + 1..].trim_start();
    let end = value
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(value.len());
    if end == 0 { None } else { Some(&value[..end]) }
}

fn escape_json(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn unescape_json(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(next) = chars.next() {
                output.push(next);
            }
        } else {
            output.push(ch);
        }
    }
    output
}

fn now_utc_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_cache_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "gpu_search_core_cache_{name}_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&dir).expect("temp cache dir should be created");
        dir
    }

    fn sample_fingerprint(root: &Path) -> SourceFingerprint {
        let files = vec![
            DiscoveredFile {
                path: root.join("a.rs"),
                size: 10,
                modified_ns: 100,
            },
            DiscoveredFile {
                path: root.join("b.rs"),
                size: 20,
                modified_ns: 250,
            },
        ];
        compute_source_fingerprint(root, &files, &IndexOptions::default())
    }

    #[test]
    fn source_fingerprint_uses_count_max_mtime_and_options() {
        let root = PathBuf::from("repo");
        let fingerprint = sample_fingerprint(&root);

        assert_eq!(fingerprint.repo_root, root);
        assert_eq!(fingerprint.indexed_file_count, 2);
        assert_eq!(fingerprint.max_modified_ns, 250);
        assert!(!fingerprint.allow_env_files);
        assert_eq!(fingerprint.max_file_mb, "5.000000");
    }

    #[test]
    fn cache_metadata_round_trips_to_disk() {
        let dir = temp_cache_dir("roundtrip");
        let repo = dir.join("repo root");
        let fingerprint = sample_fingerprint(&repo);
        let mut metadata = new_cache_metadata(&repo);
        metadata.cache_entries.push(CacheEntry {
            name: "pattern".to_string(),
            schema_version: PATTERN_CACHE_SCHEMA_VERSION,
            file_path: "pattern-index-v1.bin".to_string(),
            created_at_utc: "1".to_string(),
            updated_at_utc: "2".to_string(),
            source_fingerprint: fingerprint.clone(),
            status: "loaded".to_string(),
        });

        save_cache_metadata(&dir, &metadata).expect("metadata should save");
        let loaded = load_cache_metadata(&dir)
            .expect("metadata should load")
            .expect("metadata should exist");

        assert_eq!(loaded.schema_version, CACHE_SCHEMA_VERSION);
        assert_eq!(loaded.repo_root, repo);
        assert_eq!(loaded.cache_entries.len(), 1);
        assert_eq!(loaded.cache_entries[0].source_fingerprint, fingerprint);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn missing_or_legacy_metadata_returns_none() {
        let dir = temp_cache_dir("missing_legacy");
        assert!(
            load_cache_metadata(&dir)
                .expect("missing should not error")
                .is_none()
        );

        fs::write(dir.join(CACHE_METADATA_FILE), "{\"old\":true}")
            .expect("legacy metadata written");
        assert!(
            load_cache_metadata(&dir)
                .expect("legacy should not error")
                .is_none()
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn mismatched_top_level_schema_returns_none() {
        let dir = temp_cache_dir("schema");
        fs::write(
            dir.join(CACHE_METADATA_FILE),
            "{\"schemaVersion\":999,\"createdAtUtc\":\"1\",\"updatedAtUtc\":\"1\",\"gpuSearchVersion\":\"x\",\"repoRoot\":\"repo\",\"cacheEntries\":[]}",
        )
        .expect("metadata written");

        assert!(
            load_cache_metadata(&dir)
                .expect("schema mismatch should not error")
                .is_none()
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn cache_entry_validates_schema_fingerprint_and_status() {
        let fingerprint = sample_fingerprint(Path::new("repo"));
        let mut entry = CacheEntry {
            name: "pattern".to_string(),
            schema_version: PATTERN_CACHE_SCHEMA_VERSION,
            file_path: "pattern-index-v1.bin".to_string(),
            created_at_utc: "1".to_string(),
            updated_at_utc: "1".to_string(),
            source_fingerprint: fingerprint.clone(),
            status: "loaded".to_string(),
        };

        assert!(is_cache_entry_valid(
            &entry,
            PATTERN_CACHE_SCHEMA_VERSION,
            &fingerprint
        ));
        assert!(!is_cache_entry_valid(&entry, 999, &fingerprint));

        invalidate_cache_entry(&mut entry);
        assert_eq!(entry.status, "invalidated");
        assert!(!is_cache_entry_valid(
            &entry,
            PATTERN_CACHE_SCHEMA_VERSION,
            &fingerprint
        ));
    }
}
