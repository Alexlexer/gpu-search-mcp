//! Heuristic dependency graph primitives for the experimental Rust core.
//!
//! This module starts Phase 6 with Python import parsing only. It intentionally
//! mirrors the project stance: dependency impact is best-effort and not compiler
//! accurate. Later PRs can add JS/TS, C#, reasons parity, and richer resolvers.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::{file_discovery::DiscoveredFile, file_ext};

/// Dependency analysis mode reported by Rust graph results.
pub const DEPENDENCY_ANALYSIS_MODE: &str = "heuristic";

/// Directed dependency edge from one source file to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyEdge {
    /// File containing the import/reference.
    pub from: PathBuf,
    /// File being imported/referenced.
    pub to: PathBuf,
    /// Human-readable heuristic reason.
    pub reason: String,
}

/// Impact result for a changed file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactedFile {
    /// Impacted file path.
    pub file: PathBuf,
    /// Graph hop distance from the changed file.
    pub hops: usize,
    /// Human-readable heuristic reason for the first discovered impact path.
    pub reason: Option<String>,
}

/// Heuristic dependency graph.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DependencyGraph {
    edges: Vec<DependencyEdge>,
    reverse_edges: BTreeMap<PathBuf, Vec<DependencyEdge>>,
}

impl DependencyGraph {
    /// Build a heuristic dependency graph from discovered files.
    pub fn from_files(files: &[DiscoveredFile]) -> Result<Self, DependencyGraphError> {
        let mut graph = Self::default();
        let python_modules = python_module_map(files);

        for file in files {
            if file_ext(&file.path).as_deref() != Some(".py") {
                continue;
            }

            let text = fs::read_to_string(&file.path)
                .map_err(|source| DependencyGraphError::new(&file.path, source))?;
            for import in parse_python_imports(&text) {
                if let Some(target) = resolve_python_import(&import.module, &python_modules) {
                    graph.add_edge(DependencyEdge {
                        from: file.path.clone(),
                        to: target,
                        reason: format!("imports module {}", import.module),
                    });
                }
            }
        }

        Ok(graph)
    }

    /// Return all graph edges.
    pub fn edges(&self) -> &[DependencyEdge] {
        &self.edges
    }

    /// Return impacted files that transitively depend on `changed_file`.
    pub fn impact(&self, changed_file: impl AsRef<Path>) -> Vec<ImpactedFile> {
        let changed_file = changed_file.as_ref();
        let mut impacted = Vec::new();
        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((changed_file.to_path_buf(), 0_usize));
        visited.insert(changed_file.to_path_buf());

        while let Some((current, hops)) = queue.pop_front() {
            let Some(edges) = self.reverse_edges.get(&current) else {
                continue;
            };

            for edge in edges {
                if visited.insert(edge.from.clone()) {
                    let next_hops = hops + 1;
                    impacted.push(ImpactedFile {
                        file: edge.from.clone(),
                        hops: next_hops,
                        reason: Some(edge.reason.clone()),
                    });
                    queue.push_back((edge.from.clone(), next_hops));
                }
            }
        }

        impacted.sort_by(|left, right| left.hops.cmp(&right.hops).then(left.file.cmp(&right.file)));
        impacted
    }

    fn add_edge(&mut self, edge: DependencyEdge) {
        self.reverse_edges
            .entry(edge.to.clone())
            .or_default()
            .push(edge.clone());
        self.edges.push(edge);
    }
}

/// Error returned by dependency graph construction.
#[derive(Debug)]
pub struct DependencyGraphError {
    path: PathBuf,
    source: io::Error,
}

impl DependencyGraphError {
    fn new(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self {
            path: path.into(),
            source,
        }
    }

    /// Path that failed during graph construction.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Original IO error.
    pub fn source(&self) -> &io::Error {
        &self.source
    }
}

impl std::fmt::Display for DependencyGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "failed to build dependency graph for {}: {}",
            self.path.display(),
            self.source
        )
    }
}

impl std::error::Error for DependencyGraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PythonImport {
    module: String,
}

fn parse_python_imports(text: &str) -> Vec<PythonImport> {
    let mut imports = Vec::new();

    for line in text.lines() {
        let line = strip_python_comment(line).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix("import ") {
            for part in rest.split(',') {
                let Some(module) = normalize_import_module(part) else {
                    continue;
                };
                imports.push(PythonImport { module });
            }
            continue;
        }

        if let Some(rest) = line.strip_prefix("from ") {
            let mut parts = rest.split_whitespace();
            let Some(module) = parts.next() else {
                continue;
            };
            if parts.next() == Some("import") {
                let module = module.trim_start_matches('.');
                if !module.is_empty() {
                    imports.push(PythonImport {
                        module: module.to_string(),
                    });
                }
            }
        }
    }

    imports
}

fn normalize_import_module(part: &str) -> Option<String> {
    let module = part
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .trim_start_matches('.');
    if module.is_empty() {
        None
    } else {
        Some(module.to_string())
    }
}

fn strip_python_comment(line: &str) -> &str {
    line.split_once('#').map(|(code, _)| code).unwrap_or(line)
}

fn python_module_map(files: &[DiscoveredFile]) -> BTreeMap<String, PathBuf> {
    let mut modules = BTreeMap::new();
    for file in files {
        if file_ext(&file.path).as_deref() != Some(".py") {
            continue;
        }

        if let Some(stem) = file.path.file_stem().and_then(|stem| stem.to_str()) {
            modules
                .entry(stem.to_string())
                .or_insert_with(|| file.path.clone());
        }
        if file.path.file_name().and_then(|name| name.to_str()) == Some("__init__.py") {
            if let Some(package) = file
                .path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
            {
                modules
                    .entry(package.to_string())
                    .or_insert_with(|| file.path.clone());
            }
        }
    }
    modules
}

fn resolve_python_import(module: &str, modules: &BTreeMap<String, PathBuf>) -> Option<PathBuf> {
    let root = module.split('.').next().unwrap_or(module);
    modules.get(module).or_else(|| modules.get(root)).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gpu_search_core_deps_{name}_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root).expect("temp root should be created");
        root
    }

    fn write(root: &Path, relative: &str, content: &str) -> DiscoveredFile {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("parent should be created");
        }
        fs::write(&path, content).expect("file should be written");
        DiscoveredFile {
            path,
            size: content.len() as u64,
            modified_ns: 1,
        }
    }

    #[test]
    fn parses_python_import_and_from_import_lines() {
        let imports = parse_python_imports(
            "import os, services.user_service as users\nfrom controllers import home\nfrom .local import thing\n",
        );

        assert_eq!(
            imports,
            vec![
                PythonImport {
                    module: "os".to_string()
                },
                PythonImport {
                    module: "services.user_service".to_string()
                },
                PythonImport {
                    module: "controllers".to_string()
                },
                PythonImport {
                    module: "local".to_string()
                },
            ]
        );
    }

    #[test]
    fn builds_python_dependency_edges_with_reasons() {
        let root = temp_root("edges");
        let service = write(&root, "service.py", "class Service: pass\n");
        let controller = write(&root, "controller.py", "import service\n");
        let readme = write(&root, "README.md", "import service\n");

        let graph = DependencyGraph::from_files(&[service.clone(), controller.clone(), readme])
            .expect("graph should build");

        assert_eq!(graph.edges().len(), 1);
        assert_eq!(graph.edges()[0].from, controller.path);
        assert_eq!(graph.edges()[0].to, service.path);
        assert_eq!(graph.edges()[0].reason, "imports module service");
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn impact_returns_reverse_dependencies_with_hops() {
        let root = temp_root("impact");
        let model = write(&root, "model.py", "class Model: pass\n");
        let service = write(&root, "service.py", "from model import Model\n");
        let controller = write(&root, "controller.py", "import service\n");

        let graph =
            DependencyGraph::from_files(&[model.clone(), service.clone(), controller.clone()])
                .expect("graph should build");
        let impacted = graph.impact(&model.path);

        assert_eq!(impacted.len(), 2);
        assert_eq!(impacted[0].file, service.path);
        assert_eq!(impacted[0].hops, 1);
        assert_eq!(impacted[0].reason.as_deref(), Some("imports module model"));
        assert_eq!(impacted[1].file, controller.path);
        assert_eq!(impacted[1].hops, 2);
        assert_eq!(
            impacted[1].reason.as_deref(),
            Some("imports module service")
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn unresolved_imports_do_not_create_edges() {
        let root = temp_root("unresolved");
        let app = write(&root, "app.py", "import missing_module\n");

        let graph = DependencyGraph::from_files(&[app]).expect("graph should build");

        assert!(graph.edges().is_empty());
        fs::remove_dir_all(root).ok();
    }
}
