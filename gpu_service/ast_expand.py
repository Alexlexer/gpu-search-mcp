"""AST-aware block expansion and skeleton mode using tree-sitter."""
import os
from pathlib import Path
from typing import Optional

# Cached (parser, config) per file extension — None means unsupported
_parsers: dict = {}

_PY_CONFIG = {
    "foldable":   {"function_definition"},
    "transparent": {"class_definition", "decorated_definition"},
    "body_types":  {"block"},
    "containers":  {"function_definition", "class_definition", "decorated_definition"},
}

_TS_CONFIG = {
    "foldable": {
        "function_declaration", "function_expression", "arrow_function",
        "method_definition", "generator_function_declaration",
    },
    "transparent": {"class_declaration", "class_expression"},
    "body_types":  {"statement_block"},
    "containers": {
        "function_declaration", "function_expression", "arrow_function",
        "method_definition", "generator_function_declaration",
        "class_declaration", "class_expression",
    },
}


def _get_parser(ext: str):
    """Return (parser, config) or (None, None) if the extension is unsupported."""
    if ext in _parsers:
        return _parsers[ext]
    result = (None, None)
    try:
        from tree_sitter import Language, Parser
        if ext == ".py":
            import tree_sitter_python as m
            result = (Parser(Language(m.language())), _PY_CONFIG)
        elif ext in (".ts", ".tsx", ".js", ".jsx"):
            import tree_sitter_typescript as m
            lang = m.language_typescript() if ext in (".ts", ".tsx") else m.language_tsx()
            result = (Parser(Language(lang)), _TS_CONFIG)
    except Exception:
        pass
    _parsers[ext] = result
    return result


def _find_innermost_container(node, target_row: int, containers: set) -> Optional[object]:
    """DFS: return the deepest container node whose range includes target_row."""
    if not (node.start_point.row <= target_row <= node.end_point.row):
        return None
    best = node if node.type in containers else None
    for child in node.children:
        deeper = _find_innermost_container(child, target_row, containers)
        if deeper is not None:
            best = deeper
    return best


def expand_match(filepath: str, match_line: int) -> Optional[tuple[int, int]]:
    """
    Return (start_line, end_line) 1-indexed for the enclosing function/class block.
    Returns None if the file type is unsupported or parse fails.
    match_line is 1-indexed.
    """
    ext = Path(filepath).suffix.lower()
    parser, cfg = _get_parser(ext)
    if parser is None:
        return None
    try:
        src = Path(filepath).read_bytes()
        tree = parser.parse(src)
        target = match_line - 1  # 0-indexed
        node = _find_innermost_container(tree.root_node, target, cfg["containers"])
        if node is None or node.type == tree.root_node.type:
            return None
        return (node.start_point.row + 1, node.end_point.row + 1)
    except Exception:
        return None


def read_block(filepath: str, match_line: int, fallback_window: int = 20) -> tuple[str, int, int]:
    """
    Return (code_text, start_line, end_line) for the block containing match_line.
    Falls back to ±fallback_window lines if AST expansion unavailable.
    Lines are 1-indexed.
    """
    try:
        lines = Path(filepath).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except Exception as e:
        return f"[read error: {e}]", match_line, match_line

    n = len(lines)
    bounds = expand_match(filepath, match_line)
    if bounds:
        start, end = bounds
    else:
        start = max(1, match_line - fallback_window)
        end = min(n, match_line + fallback_window)

    block = "".join(lines[start - 1:end])
    return block, start, end


# ---------------------------------------------------------------------------
# Skeleton mode
# ---------------------------------------------------------------------------

def _body_fold_range(body) -> Optional[tuple[int, int, int]]:
    """
    Return (start_row, end_row, indent_col) for the foldable content of a body node.
    Uses first/last named children so the opening { and closing } lines remain visible.
    Returns None if the body has no content to fold.
    """
    named = [c for c in body.children if c.is_named]
    if not named:
        return None
    return (named[0].start_point.row, named[-1].end_point.row, named[0].start_point.column)


def _collect_folds(node, match_rows: set, cfg: dict, folds: list):
    """
    Recursively identify body block ranges to fold.
    Appends (start_row, end_row, indent_col) to folds for bodies with no match lines.
    """
    foldable    = cfg["foldable"]
    transparent = cfg["transparent"]
    body_types  = cfg["body_types"]

    if node.type in foldable:
        body = next((c for c in node.children if c.type in body_types), None)
        if body is not None:
            body_rows = set(range(body.start_point.row, body.end_point.row + 1))
            if not (body_rows & match_rows):
                fr = _body_fold_range(body)
                if fr:
                    folds.append(fr)
                return  # Don't recurse into a folded body
            # Match is inside — recurse to fold nested functions
            _collect_folds(body, match_rows, cfg, folds)
        return

    # Transparent containers (classes) and everything else: recurse into children
    for child in node.children:
        _collect_folds(child, match_rows, cfg, folds)


def skeleton_file(filepath: str, match_lines: Optional[list[int]] = None) -> Optional[str]:
    """
    Return a skeleton of the file with unexpanded function bodies replaced by
    '...  # N lines'.  Bodies containing a match_line are kept fully expanded.
    Returns None if the file type is unsupported or parse fails.
    match_lines is 1-indexed.
    """
    ext = Path(filepath).suffix.lower()
    parser, cfg = _get_parser(ext)
    if parser is None:
        return None
    try:
        src_text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        src_bytes = src_text.encode("utf-8", errors="replace")
        tree = parser.parse(src_bytes)
    except Exception:
        return None

    match_rows: set[int] = set()
    if match_lines:
        match_rows = {ln - 1 for ln in match_lines}  # 0-indexed

    folds: list[tuple[int, int, int]] = []
    _collect_folds(tree.root_node, match_rows, cfg, folds)
    folds.sort()

    lines = src_text.splitlines(keepends=True)
    out: list[str] = []
    fold_idx = 0
    i = 0
    while i < len(lines):
        if fold_idx < len(folds) and folds[fold_idx][0] == i:
            start_row, end_row, indent_col = folds[fold_idx]
            n_hidden = end_row - start_row + 1
            noun = "line" if n_hidden == 1 else "lines"
            out.append(" " * indent_col + f"...  # {n_hidden} {noun}\n")
            i = end_row + 1
            fold_idx += 1
        else:
            out.append(lines[i])
            i += 1

    return "".join(out)
