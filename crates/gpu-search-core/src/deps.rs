//! Heuristic dependency graph primitives for the experimental Rust core.
//!
//! This module starts Phase 6 with lightweight Python, JS/TS, and C# import /
//! reference parsing. It intentionally mirrors the project stance: dependency
//! impact is best-effort and not compiler accurate. Later PRs can add richer
//! resolvers.

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
        let js_modules = js_module_map(files);
        let csharp_symbols = csharp_symbol_map(files)?;

        for file in files {
            let text = fs::read_to_string(&file.path)
                .map_err(|source| DependencyGraphError::new(&file.path, source))?;

            match file_ext(&file.path).as_deref() {
                Some(".py") => {
                    for import in parse_python_imports(&text) {
                        if let Some(target) = resolve_python_import(&import.module, &python_modules)
                        {
                            graph.add_edge(DependencyEdge {
                                from: file.path.clone(),
                                to: target,
                                reason: format!("imports module {}", import.module),
                            });
                        }
                    }
                }
                Some(".js" | ".jsx" | ".ts" | ".tsx") => {
                    for import in parse_js_imports(&text) {
                        if let Some(target) =
                            resolve_js_import(&file.path, &import.module, files, &js_modules)
                        {
                            graph.add_edge(DependencyEdge {
                                from: file.path.clone(),
                                to: target,
                                reason: format!("imports module {}", import.module),
                            });
                        }
                    }
                }
                Some(".cs") => {
                    for edge in csharp_edges_for_file(file, &text, &csharp_symbols) {
                        graph.add_edge(edge);
                    }
                }
                _ => {}
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct JsImport {
    module: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CSharpSymbol {
    name: String,
    namespace: Option<String>,
    file: PathBuf,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct CSharpFileInfo {
    namespace: Option<String>,
    usings: Vec<String>,
    declared_types: Vec<String>,
    base_types: Vec<String>,
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

fn parse_js_imports(text: &str) -> Vec<JsImport> {
    let mut imports = Vec::new();

    for line in text.lines() {
        let line = strip_js_comment(line).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(module) = parse_static_js_import(line) {
            imports.push(JsImport { module });
            continue;
        }

        if let Some(module) = parse_require_import(line) {
            imports.push(JsImport { module });
        }
    }

    imports
}

fn parse_static_js_import(line: &str) -> Option<String> {
    if !line.starts_with("import ") {
        return None;
    }

    if let Some(module) = quoted_after(line, " from ") {
        return Some(module);
    }

    quoted_after(line, "import ")
}

fn parse_require_import(line: &str) -> Option<String> {
    let require_index = line.find("require(")?;
    quoted_after(&line[require_index..], "require(")
}

fn quoted_after(line: &str, marker: &str) -> Option<String> {
    let marker_index = line.find(marker)?;
    let rest = line[marker_index + marker.len()..].trim_start();
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let value = &rest[quote.len_utf8()..];
    let end = value.find(quote)?;
    let module = &value[..end];
    if module.is_empty() {
        None
    } else {
        Some(module.to_string())
    }
}

fn strip_js_comment(line: &str) -> &str {
    line.split_once("//").map(|(code, _)| code).unwrap_or(line)
}

fn js_module_map(files: &[DiscoveredFile]) -> BTreeMap<String, PathBuf> {
    let mut modules = BTreeMap::new();
    for file in files {
        if !is_js_like_file(&file.path) {
            continue;
        }

        let without_ext = strip_known_js_ext(&file.path);
        modules
            .entry(path_key(&without_ext))
            .or_insert_with(|| file.path.clone());

        if let Some(stem) = file.path.file_stem().and_then(|stem| stem.to_str()) {
            modules
                .entry(stem.to_string())
                .or_insert_with(|| file.path.clone());
        }

        if file.path.file_stem().and_then(|stem| stem.to_str()) == Some("index") {
            if let Some(parent) = file.path.parent() {
                modules
                    .entry(path_key(parent))
                    .or_insert_with(|| file.path.clone());
                if let Some(package) = parent.file_name().and_then(|name| name.to_str()) {
                    modules
                        .entry(package.to_string())
                        .or_insert_with(|| file.path.clone());
                }
            }
        }
    }
    modules
}

fn resolve_js_import(
    importer: &Path,
    module: &str,
    files: &[DiscoveredFile],
    modules: &BTreeMap<String, PathBuf>,
) -> Option<PathBuf> {
    if module.starts_with('.') {
        let base = importer.parent().unwrap_or_else(|| Path::new(""));
        let joined = normalize_path(base.join(module));
        return resolve_js_path(&joined, files);
    }

    modules.get(module).cloned()
}

fn resolve_js_path(candidate: &Path, files: &[DiscoveredFile]) -> Option<PathBuf> {
    for file in files {
        if !is_js_like_file(&file.path) {
            continue;
        }

        let without_ext = strip_known_js_ext(&file.path);
        if without_ext == candidate {
            return Some(file.path.clone());
        }

        let index_candidate = candidate.join("index");
        if without_ext == index_candidate {
            return Some(file.path.clone());
        }
    }
    None
}

fn strip_known_js_ext(path: &Path) -> PathBuf {
    match file_ext(path).as_deref() {
        Some(".js" | ".jsx" | ".ts" | ".tsx") => path.with_extension(""),
        _ => path.to_path_buf(),
    }
}

fn is_js_like_file(path: &Path) -> bool {
    matches!(
        file_ext(path).as_deref(),
        Some(".js" | ".jsx" | ".ts" | ".tsx")
    )
}

fn normalize_path(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn path_key(path: &Path) -> String {
    path.components()
        .filter_map(|component| match component {
            std::path::Component::Normal(value) => Some(value.to_string_lossy()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn csharp_symbol_map(
    files: &[DiscoveredFile],
) -> Result<BTreeMap<String, Vec<CSharpSymbol>>, DependencyGraphError> {
    let mut symbols: BTreeMap<String, Vec<CSharpSymbol>> = BTreeMap::new();
    for file in files {
        if file_ext(&file.path).as_deref() != Some(".cs") {
            continue;
        }

        let text = fs::read_to_string(&file.path)
            .map_err(|source| DependencyGraphError::new(&file.path, source))?;
        let info = parse_csharp_file(&text);
        for name in info.declared_types {
            symbols.entry(name.clone()).or_default().push(CSharpSymbol {
                name,
                namespace: info.namespace.clone(),
                file: file.path.clone(),
            });
        }
    }
    Ok(symbols)
}

fn csharp_edges_for_file(
    file: &DiscoveredFile,
    text: &str,
    symbols: &BTreeMap<String, Vec<CSharpSymbol>>,
) -> Vec<DependencyEdge> {
    let info = parse_csharp_file(text);
    let mut edges_by_target: BTreeMap<PathBuf, DependencyEdge> = BTreeMap::new();

    for using_namespace in &info.usings {
        for symbol in symbols.values().flatten() {
            if symbol.file == file.path {
                continue;
            }
            if symbol.namespace.as_deref() == Some(using_namespace.as_str()) {
                insert_csharp_edge(
                    &mut edges_by_target,
                    file,
                    &symbol.file,
                    format!("imports namespace {using_namespace}"),
                );
            }
        }
    }

    for base_type in &info.base_types {
        for symbol in resolve_csharp_type(base_type, symbols) {
            if symbol.file == file.path {
                continue;
            }
            let reason = if base_type.starts_with('I') {
                format!("implements interface {base_type}")
            } else {
                format!("inherits from {base_type}")
            };
            insert_csharp_edge(&mut edges_by_target, file, &symbol.file, reason);
        }
    }

    for type_name in csharp_type_references(text, symbols) {
        if info
            .declared_types
            .iter()
            .any(|declared| declared == &type_name)
        {
            continue;
        }
        for symbol in resolve_csharp_type(&type_name, symbols) {
            if symbol.file == file.path {
                continue;
            }
            insert_csharp_edge(
                &mut edges_by_target,
                file,
                &symbol.file,
                format!("references type {type_name}"),
            );
        }
    }

    edges_by_target.into_values().collect()
}

fn insert_csharp_edge(
    edges_by_target: &mut BTreeMap<PathBuf, DependencyEdge>,
    file: &DiscoveredFile,
    target: &Path,
    reason: String,
) {
    let target = target.to_path_buf();
    let replace = edges_by_target
        .get(&target)
        .map(|existing| csharp_reason_priority(&reason) > csharp_reason_priority(&existing.reason))
        .unwrap_or(true);
    if replace {
        edges_by_target.insert(
            target.clone(),
            DependencyEdge {
                from: file.path.clone(),
                to: target,
                reason,
            },
        );
    }
}

fn csharp_reason_priority(reason: &str) -> u8 {
    if reason.starts_with("inherits from ") || reason.starts_with("implements interface ") {
        4
    } else if reason.starts_with("references type ") {
        3
    } else if reason.starts_with("imports namespace ") {
        2
    } else {
        1
    }
}

fn parse_csharp_file(text: &str) -> CSharpFileInfo {
    let mut info = CSharpFileInfo::default();
    for line in text.lines() {
        let line = strip_csharp_comment(line).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(namespace) = parse_csharp_namespace(line) {
            info.namespace = Some(namespace);
        }
        if let Some(using_namespace) = parse_csharp_using(line) {
            info.usings.push(using_namespace);
        }
        if let Some((declared_type, base_types)) = parse_csharp_type_declaration(line) {
            info.declared_types.push(declared_type);
            info.base_types.extend(base_types);
        }
    }
    info
}

fn parse_csharp_namespace(line: &str) -> Option<String> {
    let rest = line.strip_prefix("namespace ")?;
    let namespace = rest
        .split(|ch: char| ch == '{' || ch == ';' || ch.is_whitespace())
        .next()
        .unwrap_or_default();
    if namespace.is_empty() {
        None
    } else {
        Some(namespace.to_string())
    }
}

fn parse_csharp_using(line: &str) -> Option<String> {
    let rest = line.strip_prefix("using ")?;
    if rest.starts_with("static ") || rest.contains('=') {
        return None;
    }
    let namespace = rest.trim_end_matches(';').trim();
    if namespace.is_empty() {
        None
    } else {
        Some(namespace.to_string())
    }
}

fn parse_csharp_type_declaration(line: &str) -> Option<(String, Vec<String>)> {
    const TYPE_KEYWORDS: [&str; 5] = ["class", "interface", "record", "struct", "enum"];
    let tokens = csharp_identifier_tokens(line);
    let (keyword_index, _) = tokens
        .iter()
        .enumerate()
        .find(|(_, token)| TYPE_KEYWORDS.contains(&token.as_str()))?;
    let name = tokens.get(keyword_index + 1)?.to_string();

    let base_types = line
        .split_once(':')
        .map(|(_, rest)| {
            rest.split(|ch: char| ch == ',' || ch == '{' || ch.is_whitespace())
                .filter_map(clean_csharp_type_name)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Some((name, base_types))
}

fn csharp_type_references(
    text: &str,
    symbols: &BTreeMap<String, Vec<CSharpSymbol>>,
) -> Vec<String> {
    let known: BTreeSet<&str> = symbols.keys().map(String::as_str).collect();
    let mut references = BTreeSet::new();
    for token in csharp_identifier_tokens(text) {
        if known.contains(token.as_str()) {
            references.insert(token);
        }
    }
    references.into_iter().collect()
}

fn resolve_csharp_type<'a>(
    type_name: &str,
    symbols: &'a BTreeMap<String, Vec<CSharpSymbol>>,
) -> Vec<&'a CSharpSymbol> {
    symbols
        .get(type_name)
        .map(|values| values.iter().collect())
        .unwrap_or_default()
}

fn clean_csharp_type_name(value: &str) -> Option<String> {
    let value = value
        .trim()
        .trim_end_matches(';')
        .trim_end_matches('{')
        .split('<')
        .next()
        .unwrap_or_default();
    if value.is_empty() || !is_csharp_identifier(value) {
        None
    } else {
        Some(value.to_string())
    }
}

fn csharp_identifier_tokens(text: &str) -> Vec<String> {
    text.split(|ch: char| !(ch == '_' || ch.is_ascii_alphanumeric()))
        .filter(|token| is_csharp_identifier(token))
        .map(ToString::to_string)
        .collect()
}

fn is_csharp_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn strip_csharp_comment(line: &str) -> &str {
    line.split_once("//").map(|(code, _)| code).unwrap_or(line)
}

#[cfg(test)]
#[path = "deps_tests.rs"]
mod tests;
