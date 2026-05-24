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
        items.push(CSharpAstItem {
            kind: node.kind().to_string(),
            name: node_name(node, source),
            line: line_number(source, node.start_byte()),
        });
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
}
