//! File discovery primitives for the experimental Rust core.
//!
//! This module mirrors the current Python index boundary rules at a small scale:
//! skip known build/cache directories, exclude `.env` by default, reject large
//! files, skip binary-looking files, and return deterministic sorted output.

use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::{DEFAULT_SKIP_DIRS, IndexOptions, is_indexable_file};

const BINARY_SNIFF_BYTES: usize = 8192;

/// A source file discovered for indexing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredFile {
    /// File path as discovered under the requested root.
    pub path: PathBuf,
    /// File size in bytes.
    pub size: u64,
    /// Last modification timestamp in nanoseconds since the Unix epoch.
    pub modified_ns: u128,
}

/// Error returned by file discovery.
#[derive(Debug)]
pub struct DiscoveryError {
    path: PathBuf,
    source: io::Error,
}

impl DiscoveryError {
    fn new(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self {
            path: path.into(),
            source,
        }
    }

    /// Path that failed during discovery.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Original IO error.
    pub fn source(&self) -> &io::Error {
        &self.source
    }
}

impl fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to discover files under {}: {}",
            self.path.display(),
            self.source
        )
    }
}

impl std::error::Error for DiscoveryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Discover indexable files under `root` using deterministic ordering.
pub fn discover_files(
    root: impl AsRef<Path>,
    options: &IndexOptions,
) -> Result<Vec<DiscoveredFile>, DiscoveryError> {
    let root = root.as_ref();
    let metadata = fs::metadata(root).map_err(|err| DiscoveryError::new(root, err))?;
    if !metadata.is_dir() {
        return Err(DiscoveryError::new(
            root,
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "discovery root is not a directory",
            ),
        ));
    }

    let mut files = Vec::new();
    visit_dir(root, options, &mut files)?;
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(files)
}

fn visit_dir(
    directory: &Path,
    options: &IndexOptions,
    files: &mut Vec<DiscoveredFile>,
) -> Result<(), DiscoveryError> {
    let entries = fs::read_dir(directory).map_err(|err| DiscoveryError::new(directory, err))?;

    for entry in entries {
        let entry = entry.map_err(|err| DiscoveryError::new(directory, err))?;
        let path = entry.path();
        let metadata = entry
            .metadata()
            .map_err(|err| DiscoveryError::new(&path, err))?;

        if metadata.is_dir() {
            if !is_skipped_dir(&path) {
                visit_dir(&path, options, files)?;
            }
            continue;
        }

        if !metadata.is_file() || !is_indexable_file(&path, options) {
            continue;
        }

        if metadata.len() > max_file_bytes(options) || is_binary_file(&path)? {
            continue;
        }

        files.push(DiscoveredFile {
            path,
            size: metadata.len(),
            modified_ns: modified_ns(&metadata),
        });
    }

    Ok(())
}

fn is_skipped_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| DEFAULT_SKIP_DIRS.contains(&name))
        .unwrap_or(false)
}

fn max_file_bytes(options: &IndexOptions) -> u64 {
    if !options.max_file_mb.is_finite() || options.max_file_mb <= 0.0 {
        return 0;
    }

    (options.max_file_mb * 1024.0 * 1024.0).floor() as u64
}

fn modified_ns(metadata: &fs::Metadata) -> u128 {
    metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

fn is_binary_file(path: &Path) -> Result<bool, DiscoveryError> {
    let mut file = File::open(path).map_err(|err| DiscoveryError::new(path, err))?;
    let mut buffer = [0_u8; BINARY_SNIFF_BYTES];
    let read = file
        .read(&mut buffer)
        .map_err(|err| DiscoveryError::new(path, err))?;

    Ok(buffer[..read].contains(&0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gpu_search_core_{name}_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root).expect("temp root should be created");
        root
    }

    fn write(path: &Path, content: impl AsRef<[u8]>) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("parent should be created");
        }
        fs::write(path, content).expect("file should be written");
    }

    fn relative_files(root: &Path, files: &[DiscoveredFile]) -> Vec<String> {
        files
            .iter()
            .map(|file| {
                file.path
                    .strip_prefix(root)
                    .expect("file should live under root")
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect()
    }

    #[test]
    fn discovery_returns_stable_sorted_indexable_files() {
        let root = temp_root("sorted");
        write(&root.join("z.cs"), "class Z {}");
        write(&root.join("a.py"), "print('a')");
        write(&root.join("notes.tmp"), "skip me");

        let files = discover_files(&root, &IndexOptions::default()).expect("discovery should work");

        assert_eq!(relative_files(&root, &files), vec!["a.py", "z.cs"]);
        assert!(files.iter().all(|file| file.size > 0));
        assert!(files.iter().all(|file| file.modified_ns > 0));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn discovery_skips_default_directories_and_env_files() {
        let root = temp_root("skip_dirs");
        write(&root.join("src/app.rs"), "fn main() {}");
        write(&root.join("target/generated.rs"), "fn generated() {}");
        write(
            &root.join("node_modules/pkg/index.js"),
            "console.log('skip')",
        );
        write(&root.join(".env"), "SECRET=skip");

        let files = discover_files(&root, &IndexOptions::default()).expect("discovery should work");

        assert_eq!(relative_files(&root, &files), vec!["src/app.rs"]);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn discovery_allows_env_files_when_opted_in() {
        let root = temp_root("env_opt_in");
        write(&root.join(".env"), "SAFE_FOR_TESTS=yes");

        let options = IndexOptions {
            allow_env_files: true,
            ..IndexOptions::default()
        };
        let files = discover_files(&root, &options).expect("discovery should work");

        assert_eq!(relative_files(&root, &files), vec![".env"]);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn discovery_skips_large_and_binary_files() {
        let root = temp_root("large_binary");
        write(&root.join("small.txt"), "ok");
        write(&root.join("large.txt"), vec![b'a'; 2048]);
        write(&root.join("binary.txt"), b"hello\0world");

        let options = IndexOptions {
            max_file_mb: 0.001,
            ..IndexOptions::default()
        };
        let files = discover_files(&root, &options).expect("discovery should work");

        assert_eq!(relative_files(&root, &files), vec!["small.txt"]);
        fs::remove_dir_all(root).ok();
    }
}
