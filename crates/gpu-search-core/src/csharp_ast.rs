//! Tree-sitter C# parsing helpers for the experimental Rust core.
//!
//! This is the first small Phase 7 step. It wires `tree-sitter-c-sharp` into the
//! Rust core and exposes a lightweight summary of important C# syntax nodes. The
//! existing heuristic dependency graph remains the active dependency engine.

use std::fmt;

use tree_sitter::{Node, Parser, Tree};

/// C# AST item extracted from Tree-sitter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CSharpAstItem {
    /// Tree-sitter node kind, e.g. `class_declaration`.
    pub kind: String,
    /// Best-effort symbol/name text for the node.
    pub name: Option<String>,
    /// 1-based start line.
    pub line: usize,
}

/// Error returned when the C# parser cannot be initialized or parse source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CSharpAstParseError {
    message: String,
}

impl CSharpAstParseError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CSharpAstParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C# Tree-sitter parse error: {}", self.message)
    }
}

impl std::error::Error for CSharpAstParseError {}

/// Parse C# source and return a compact syntax summary.
pub fn parse_csharp_ast_summary(source: &str) -> Result<Vec<CSharpAstItem>, CSharpAstParseError> {
    let tree = parse_tree(source)?;
    let mut items = Vec::new();
    collect_items(tree.root_node(), source.as_bytes(), &mut items);
    Ok(items)
}

/// Best-effort regex-like C# summary used only as a fallback when Tree-sitter is unavailable.
pub fn parse_csharp_regex_summary(source: &str) -> Vec<CSharpAstItem> {
    let mut items = Vec::new();
    let mut current_class: Option<String> = None;

    for (idx, line) in source.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = strip_line_comment(line).trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with("using ") {
            items.push(CSharpAstItem {
                kind: "using_directive".to_string(),
                name: trimmed
                    .trim_start_matches("using ")
                    .trim_end_matches(';')
                    .split('=')
                    .next_back()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToString::to_string),
                line: line_no,
            });
            continue;
        }

        if let Some(name) = fallback_namespace_name(trimmed) {
            items.push(CSharpAstItem {
                kind: "namespace_declaration".to_string(),
                name: Some(name),
                line: line_no,
            });
            continue;
        }

        if let Some((kind, name, relationships)) = fallback_type_declaration(trimmed) {
            current_class = (kind == "class_declaration").then(|| name.clone());
            items.push(CSharpAstItem {
                kind: kind.to_string(),
                name: Some(name),
                line: line_no,
            });
            for (idx, relationship) in relationships.into_iter().enumerate() {
                let relationship_kind = if kind == "interface_declaration" {
                    "inherits_from"
                } else if idx == 0 && !looks_like_interface_name(&relationship) {
                    "inherits_from"
                } else {
                    "implements_interface"
                };
                items.push(CSharpAstItem {
                    kind: relationship_kind.to_string(),
                    name: Some(relationship),
                    line: line_no,
                });
            }
            continue;
        }

        if let Some((kind, name)) = fallback_member(trimmed, current_class.as_deref()) {
            items.push(CSharpAstItem {
                kind: kind.to_string(),
                name: Some(name.clone()),
                line: line_no,
            });
            if kind == "method_declaration"
                && current_class
                    .as_deref()
                    .is_some_and(|class| class.ends_with("Controller"))
            {
                items.push(CSharpAstItem {
                    kind: "controller_action".to_string(),
                    name: Some(name),
                    line: line_no,
                });
            }
        }
    }

    items
}

fn parse_tree(source: &str) -> Result<Tree, CSharpAstParseError> {
    let mut parser = Parser::new();
    let language: tree_sitter::Language = tree_sitter_c_sharp::LANGUAGE.into();
    parser
        .set_language(&language)
        .map_err(|err| CSharpAstParseError::new(err.to_string()))?;
    parser
        .parse(source, None)
        .ok_or_else(|| CSharpAstParseError::new("parser returned no tree"))
}

fn collect_items(node: Node<'_>, source: &[u8], items: &mut Vec<CSharpAstItem>) {
    if is_interesting_csharp_node(node.kind()) {
        let name = node_name(node, source);
        items.push(CSharpAstItem {
            kind: node.kind().to_string(),
            name: name.clone(),
            line: line_number(source, node.start_byte()),
        });

        if node.kind() == "method_declaration" && is_controller_action(node, source) {
            items.push(CSharpAstItem {
                kind: "controller_action".to_string(),
                name,
                line: line_number(source, node.start_byte()),
            });
        }

        if is_type_declaration_node(node.kind()) {
            collect_type_relationship_items(node, source, items);
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_items(child, source, items);
    }
}

fn is_interesting_csharp_node(kind: &str) -> bool {
    matches!(
        kind,
        "using_directive"
            | "namespace_declaration"
            | "file_scoped_namespace_declaration"
            | "class_declaration"
            | "interface_declaration"
            | "record_declaration"
            | "struct_declaration"
            | "enum_declaration"
            | "method_declaration"
            | "constructor_declaration"
            | "property_declaration"
    )
}

fn is_type_declaration_node(kind: &str) -> bool {
    matches!(
        kind,
        "class_declaration" | "interface_declaration" | "record_declaration" | "struct_declaration"
    )
}

fn node_name(node: Node<'_>, source: &[u8]) -> Option<String> {
    if let Some(name) = node.child_by_field_name("name") {
        return node_text(name, source);
    }

    let mut cursor = node.walk();
    node.children(&mut cursor)
        .filter(|child| child.kind() == "identifier")
        .filter_map(|child| node_text(child, source))
        .last()
}

fn collect_type_relationship_items(node: Node<'_>, source: &[u8], items: &mut Vec<CSharpAstItem>) {
    let Some(text) = node_text(node, source) else {
        return;
    };
    let Some((_, after_colon)) = text
        .split(['{', ';'])
        .next()
        .unwrap_or_default()
        .split_once(':')
    else {
        return;
    };

    for (idx, type_name) in after_colon
        .split(',')
        .filter_map(clean_relationship_type_name)
        .enumerate()
    {
        let kind = if node.kind() == "interface_declaration" {
            "inherits_from"
        } else if idx == 0 && !looks_like_interface_name(&type_name) {
            "inherits_from"
        } else {
            "implements_interface"
        };

        items.push(CSharpAstItem {
            kind: kind.to_string(),
            name: Some(type_name),
            line: line_number(source, node.start_byte()),
        });
    }
}

fn clean_relationship_type_name(raw: &str) -> Option<String> {
    let without_constraints = raw.split("where").next().unwrap_or(raw).trim();
    let name = without_constraints
        .chars()
        .skip_while(|ch| !is_type_name_char(*ch))
        .take_while(|ch| is_type_name_char(*ch))
        .collect::<String>();

    (!name.is_empty()).then_some(name)
}

fn is_type_name_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '_' | '.' | '<' | '>')
}

fn looks_like_interface_name(name: &str) -> bool {
    name.rsplit('.').next().is_some_and(|short| {
        short.starts_with('I') && short.chars().nth(1).is_some_and(char::is_uppercase)
    })
}

fn is_controller_action(node: Node<'_>, source: &[u8]) -> bool {
    node.parent()
        .and_then(|parent| nearest_parent_of_kind(parent, "class_declaration"))
        .and_then(|class_node| node_name(class_node, source))
        .is_some_and(|name| name.ends_with("Controller"))
}

fn nearest_parent_of_kind<'tree>(mut node: Node<'tree>, kind: &str) -> Option<Node<'tree>> {
    loop {
        if node.kind() == kind {
            return Some(node);
        }
        node = node.parent()?;
    }
}

fn node_text(node: Node<'_>, source: &[u8]) -> Option<String> {
    node.utf8_text(source).ok().map(ToString::to_string)
}

fn line_number(source: &[u8], byte_offset: usize) -> usize {
    source[..byte_offset.min(source.len())]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        + 1
}

fn strip_line_comment(line: &str) -> &str {
    line.split_once("//").map(|(code, _)| code).unwrap_or(line)
}

fn fallback_namespace_name(trimmed: &str) -> Option<String> {
    trimmed
        .strip_prefix("namespace ")
        .map(|value| {
            value
                .split(['{', ';'])
                .next()
                .unwrap_or_default()
                .trim()
                .to_string()
        })
        .filter(|value| !value.is_empty())
}

fn fallback_type_declaration(trimmed: &str) -> Option<(&'static str, String, Vec<String>)> {
    let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
    let (kind, idx) = tokens
        .iter()
        .enumerate()
        .find_map(|(idx, token)| match *token {
            "class" => Some(("class_declaration", idx)),
            "interface" => Some(("interface_declaration", idx)),
            "record" => Some(("record_declaration", idx)),
            "struct" => Some(("struct_declaration", idx)),
            "enum" => Some(("enum_declaration", idx)),
            _ => None,
        })?;
    let name = tokens
        .get(idx + 1)?
        .trim_matches(|ch: char| !(ch == '_' || ch.is_ascii_alphanumeric()))
        .to_string();
    if name.is_empty() {
        return None;
    }

    let relationships = trimmed
        .split(['{', ';'])
        .next()
        .unwrap_or_default()
        .split_once(':')
        .map(|(_, after_colon)| {
            after_colon
                .split(',')
                .filter_map(clean_relationship_type_name)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Some((kind, name, relationships))
}

fn fallback_member(trimmed: &str, current_class: Option<&str>) -> Option<(&'static str, String)> {
    if !trimmed.contains('(') {
        let before_accessor = trimmed.split('{').next().unwrap_or_default().trim();
        let name = before_accessor
            .split_whitespace()
            .last()
            .map(|value| value.trim_matches(|ch: char| !(ch == '_' || ch.is_ascii_alphanumeric())))
            .filter(|value| !value.is_empty())?;
        return Some(("property_declaration", name.to_string()));
    }

    let before_params = trimmed.split('(').next().unwrap_or_default().trim();
    let name = before_params
        .split_whitespace()
        .last()
        .map(|value| value.trim_matches(|ch: char| !(ch == '_' || ch.is_ascii_alphanumeric())))
        .filter(|value| !value.is_empty())?;
    if current_class == Some(name) {
        Some(("constructor_declaration", name.to_string()))
    } else {
        Some(("method_declaration", name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_csharp_core_symbols() {
        let source = r#"
using MyApp.Services;

namespace MyApp.Controllers;

public interface IUserController { }
public record UserDto(string Name);
public class UserController : ControllerBase, IUserController
{
    public UserController(UserService service) { }
    public string Name { get; set; }
    public IActionResult GetUser(int id) => null;
}
"#;

        let items = parse_csharp_ast_summary(source).expect("C# source should parse");

        assert!(items.iter().any(|item| item.kind == "using_directive"));
        assert!(
            items
                .iter()
                .any(|item| item.kind == "file_scoped_namespace_declaration"
                    && item.name.as_deref() == Some("MyApp.Controllers"))
        );
        assert!(items.iter().any(|item| {
            item.kind == "interface_declaration" && item.name.as_deref() == Some("IUserController")
        }));
        assert!(items.iter().any(
            |item| item.kind == "record_declaration" && item.name.as_deref() == Some("UserDto")
        ));
        assert!(items.iter().any(|item| {
            item.kind == "class_declaration" && item.name.as_deref() == Some("UserController")
        }));
        assert!(items.iter().any(|item| {
            item.kind == "constructor_declaration" && item.name.as_deref() == Some("UserController")
        }));
        assert!(items.iter().any(
            |item| item.kind == "property_declaration" && item.name.as_deref() == Some("Name")
        ));
        assert!(items.iter().any(
            |item| item.kind == "method_declaration" && item.name.as_deref() == Some("GetUser")
        ));
        assert!(items.iter().any(
            |item| item.kind == "controller_action" && item.name.as_deref() == Some("GetUser")
        ));
        assert!(
            items.iter().any(|item| item.kind == "inherits_from"
                && item.name.as_deref() == Some("ControllerBase"))
        );
        assert!(items.iter().any(|item| {
            item.kind == "implements_interface" && item.name.as_deref() == Some("IUserController")
        }));
    }

    #[test]
    fn parses_csharp_struct_and_enum() {
        let source = "namespace MyApp.Models { public struct Point { } public enum Color { Red } }";
        let items = parse_csharp_ast_summary(source).expect("C# source should parse");

        assert!(
            items
                .iter()
                .any(|item| item.kind == "struct_declaration"
                    && item.name.as_deref() == Some("Point"))
        );
        assert!(
            items.iter().any(
                |item| item.kind == "enum_declaration" && item.name.as_deref() == Some("Color")
            )
        );
    }

    #[test]
    fn regex_fallback_summarizes_csharp_core_symbols() {
        let source = r#"
using MyApp.Services;
namespace MyApp.Controllers;
public interface IUserController { }
public class UserController : ControllerBase, IUserController
{
    public UserController(UserService service) { }
    public string Name { get; set; }
    public IActionResult GetUser(int id) => null;
}
"#;

        let items = parse_csharp_regex_summary(source);

        assert!(items.iter().any(|item| item.kind == "using_directive"));
        assert!(items.iter().any(|item| {
            item.kind == "namespace_declaration"
                && item.name.as_deref() == Some("MyApp.Controllers")
        }));
        assert!(items.iter().any(|item| {
            item.kind == "interface_declaration" && item.name.as_deref() == Some("IUserController")
        }));
        assert!(items.iter().any(|item| {
            item.kind == "class_declaration" && item.name.as_deref() == Some("UserController")
        }));
        assert!(items.iter().any(|item| {
            item.kind == "inherits_from" && item.name.as_deref() == Some("ControllerBase")
        }));
        assert!(items.iter().any(|item| {
            item.kind == "implements_interface" && item.name.as_deref() == Some("IUserController")
        }));
        assert!(items.iter().any(|item| {
            item.kind == "constructor_declaration" && item.name.as_deref() == Some("UserController")
        }));
        assert!(items.iter().any(|item| {
            item.kind == "property_declaration" && item.name.as_deref() == Some("Name")
        }));
        assert!(items.iter().any(|item| {
            item.kind == "method_declaration" && item.name.as_deref() == Some("GetUser")
        }));
        assert!(items.iter().any(|item| {
            item.kind == "controller_action" && item.name.as_deref() == Some("GetUser")
        }));
    }

    #[test]
    fn marks_only_controller_methods_as_controller_actions() {
        let source = r#"
public class UserController : ControllerBase
{
    public IActionResult Index() => null;
}

public class UserService
{
    public string Index() => "ok";
}
"#;
        let items = parse_csharp_ast_summary(source).expect("C# source should parse");

        let controller_actions = items
            .iter()
            .filter(|item| item.kind == "controller_action")
            .collect::<Vec<_>>();

        assert_eq!(controller_actions.len(), 1);
        assert_eq!(controller_actions[0].name.as_deref(), Some("Index"));
    }

    #[test]
    fn summarizes_csharp_inheritance_and_interface_relationships() {
        let source = r#"
public interface IAuditedRepository : IRepository { }
public class UserRepository : BaseRepository<User>, MyApp.Contracts.IUserRepository, IDisposable
{
}
"#;
        let items = parse_csharp_ast_summary(source).expect("C# source should parse");

        assert!(items.iter().any(
            |item| item.kind == "inherits_from" && item.name.as_deref() == Some("IRepository")
        ));
        assert!(items.iter().any(|item| {
            item.kind == "inherits_from" && item.name.as_deref() == Some("BaseRepository<User>")
        }));
        assert!(items.iter().any(|item| {
            item.kind == "implements_interface"
                && item.name.as_deref() == Some("MyApp.Contracts.IUserRepository")
        }));
        assert!(items.iter().any(|item| {
            item.kind == "implements_interface" && item.name.as_deref() == Some("IDisposable")
        }));
    }
}
