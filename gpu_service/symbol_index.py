"""Language-neutral symbol graph with a dependency-free C# extractor.

The C# parser is deliberately conservative.  It provides useful symbol results
when tree-sitter extras are unavailable and records that provenance on every
edge so callers can distinguish heuristic relationships from AST-backed ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
import re
import threading


PARSER = "csharp-heuristic"
PARSER_VERSION = "1"
TYPE_KINDS = {"class", "struct", "interface", "enum", "record"}
_SKIP_DIRS = {
    ".git", ".gpu-search-cache", ".venv", ".vs", "bin", "build",
    "dist", "node_modules", "obj", "target", "__pycache__",
}
_MODIFIERS = (
    "public|private|protected|internal|static|abstract|sealed|partial|readonly|"
    "virtual|override|async|extern|unsafe|new|required|const"
)


def _stable_id(prefix: str, *parts: object) -> str:
    value = "\x1f".join(str(part or "") for part in parts)
    return f"{prefix}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


@dataclass(frozen=True, slots=True)
class Symbol:
    id: str
    name: str
    qualified_name: str
    kind: str
    language: str
    file_path: str
    line_start: int
    line_end: int
    parent_symbol_id: str | None = None
    signature: str = ""
    visibility: str = ""
    modifiers: tuple[str, ...] = ()
    parser: str = PARSER
    parser_version: str = PARSER_VERSION

    def as_dict(self) -> dict:
        result = asdict(self)
        result["modifiers"] = list(self.modifiers)
        return result


@dataclass(frozen=True, slots=True)
class SymbolEdge:
    id: str
    kind: str
    source_symbol_id: str
    target_symbol_id: str | None
    target_name: str
    file_path: str
    line: int
    confidence: float
    provenance: str
    parser: str = PARSER
    parser_version: str = PARSER_VERSION

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _RawEdge:
    kind: str
    source_symbol_id: str
    target_name: str
    file_path: str
    line: int
    confidence: float
    provenance: str


def _line(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def _scrub(text: str) -> str:
    """Blank comments and literals while preserving offsets and newlines."""
    out = list(text)
    i = 0
    state = "code"
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                out[i] = out[i + 1] = " "
                i += 2
                state = "line"
                continue
            if ch == "/" and nxt == "*":
                out[i] = out[i + 1] = " "
                i += 2
                state = "block"
                continue
            if ch == '@' and nxt == '"':
                out[i] = out[i + 1] = " "
                i += 2
                state = "verbatim"
                continue
            if ch in {'"', "'"}:
                out[i] = " "
                state = "string" if ch == '"' else "char"
        elif state == "line":
            if ch == "\n":
                state = "code"
            else:
                out[i] = " "
        elif state == "block":
            if ch == "*" and nxt == "/":
                out[i] = out[i + 1] = " "
                i += 2
                state = "code"
                continue
            if ch != "\n":
                out[i] = " "
        elif state == "verbatim":
            if ch == '"' and nxt == '"':
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if ch == '"':
                out[i] = " "
                state = "code"
            elif ch != "\n":
                out[i] = " "
        else:
            if ch == "\\":
                out[i] = " "
                if i + 1 < len(text) and text[i + 1] != "\n":
                    out[i + 1] = " "
                    i += 2
                    continue
            elif (state == "string" and ch == '"') or (state == "char" and ch == "'"):
                out[i] = " "
                state = "code"
            elif ch != "\n":
                out[i] = " "
        i += 1
    return "".join(out)


def _matching_brace(text: str, opening: int) -> int:
    depth = 0
    for pos in range(opening, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return pos
    return len(text) - 1


def _depth_at(text: str, position: int) -> int:
    return text.count("{", 0, position) - text.count("}", 0, position)


def _modifiers(value: str) -> tuple[str, ...]:
    return tuple(word for word in value.split() if word)


def _visibility(modifiers: tuple[str, ...], default: str = "private") -> str:
    for value in ("public", "protected", "internal", "private"):
        if value in modifiers:
            return value
    return default


_TYPE_RE = re.compile(
    rf"(?m)^[ \t]*(?P<attrs>(?:\[[^\]\n]+\][ \t]*\n[ \t]*)*)"
    rf"(?P<mods>(?:(?:{_MODIFIERS})\s+)*)"
    r"(?P<kind>class|struct|interface|enum|record)\s+(?P<name>[A-Za-z_]\w*)"
    r"(?:\s*<[^>{;\n]+>)?(?:\s*:\s*(?P<bases>[^\{;\n]+))?"
)
_METHOD_RE = re.compile(
    rf"(?m)^[ \t]*(?P<attrs>(?:\[[^\]\n]+\][ \t]*\n[ \t]*)*)"
    rf"(?P<mods>(?:(?:{_MODIFIERS})\s+)*)"
    r"(?P<return>[A-Za-z_][\w.<>,?\[\]]*)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)\s*"
    r"(?P<tail>\{|=>|;)"
)
_PROPERTY_RE = re.compile(
    rf"(?m)^[ \t]*(?P<mods>(?:(?:{_MODIFIERS})\s+)*)"
    r"(?P<type>[A-Za-z_][\w.<>,?\[\]]*)\s+(?P<name>[A-Za-z_]\w*)\s*\{"
)
_FIELD_RE = re.compile(
    rf"(?m)^[ \t]*(?P<mods>(?:(?:{_MODIFIERS})\s+)*)"
    r"(?P<type>[A-Za-z_][\w.<>,?\[\]]*)\s+(?P<name>[A-Za-z_]\w*)"
    r"(?:\s*=\s*[^;\n]+)?\s*;"
)


def _extract_csharp(path: str, text: str) -> tuple[list[Symbol], list[_RawEdge]]:
    clean = _scrub(text)
    symbols: list[Symbol] = []
    raw_edges: list[_RawEdge] = []
    namespace_match = re.search(r"(?m)^\s*namespace\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", clean)
    namespace = namespace_match.group(1) if namespace_match else ""
    namespace_id = None
    if namespace_match:
        namespace_id = _stable_id("sym", "csharp", path, namespace, "namespace", "")
        symbols.append(Symbol(
            namespace_id, namespace.rsplit(".", 1)[-1], namespace, "namespace", "csharp",
            path, _line(text, namespace_match.start()), _line(text, namespace_match.end()),
            visibility="public",
        ))

    module_name = os.path.splitext(os.path.basename(path))[0]
    module_qname = (
        f"{namespace}.{module_name}#module" if namespace else f"{module_name}#module"
    )
    module_id = _stable_id("sym", "csharp", path, module_qname, "module", "")
    symbols.append(Symbol(
        module_id, module_name, module_qname, "module", "csharp", path, 1,
        max(1, text.count("\n") + 1), namespace_id, visibility="internal",
    ))

    for match in _TYPE_RE.finditer(clean):
        opening = clean.find("{", match.end())
        if opening < 0:
            continue
        terminator = clean.find(";", match.end(), opening)
        if terminator >= 0:
            continue
        closing = _matching_brace(clean, opening)
        kind = match.group("kind")
        name = match.group("name")
        qname = f"{namespace}.{name}" if namespace else name
        mods = _modifiers(match.group("mods"))
        type_id = _stable_id("sym", "csharp", path, qname, kind, "")
        symbols.append(Symbol(
            type_id, name, qname, kind, "csharp", path, _line(text, match.start()),
            _line(text, closing), namespace_id or module_id,
            text[match.start():opening].strip(), _visibility(mods, "internal"), mods,
        ))
        for base in (match.group("bases") or "").split(","):
            target = re.sub(r"<.*>", "", base).strip().split()[-1:]
            if target:
                target_name = target[0].split(".")[-1]
                edge_kind = "implements" if target_name.startswith("I") else "inherits"
                raw_edges.append(_RawEdge(
                    edge_kind, type_id, target_name, path, _line(text, match.start()),
                    0.86 if edge_kind == "implements" else 0.78, "declaration",
                ))

        body_depth = _depth_at(clean, opening) + 1
        body = clean[opening + 1:closing]
        offset = opening + 1
        occupied: set[tuple[int, str]] = set()

        # Constructors need a dedicated expression because they have no return type.
        ctor_re = re.compile(
            rf"(?m)^[ \t]*(?P<attrs>(?:\[[^\]\n]+\][ \t]*\n[ \t]*)*)"
            rf"(?P<mods>(?:(?:{_MODIFIERS})\s+)*){re.escape(name)}\s*"
            r"\((?P<params>[^)]*)\)\s*(?P<tail>\{|=>|;)"
        )
        member_matches: list[tuple[re.Match, str]] = [
            *((m, "constructor") for m in ctor_re.finditer(body)),
            *((m, "method") for m in _METHOD_RE.finditer(body)),
        ]
        member_matches.sort(key=lambda item: item[0].start())
        for member, candidate_kind in member_matches:
            absolute = offset + member.start()
            if _depth_at(clean, absolute) != body_depth:
                continue
            member_name = name if candidate_kind == "constructor" else member.group("name")
            key = (absolute, member_name)
            if key in occupied:
                continue
            occupied.add(key)
            attrs = member.groupdict().get("attrs") or ""
            mods = _modifiers(member.group("mods"))
            params = " ".join(member.group("params").split())
            signature = text[absolute:offset + member.end()].strip()
            symbol_kind = candidate_kind
            if re.search(r"\[(?:Fact|Theory|Test|TestMethod)\b", attrs):
                symbol_kind = "test"
            member_qname = f"{qname}.{member_name}"
            member_id = _stable_id("sym", "csharp", path, member_qname, symbol_kind, params)
            member_end = offset + member.end()
            if member.group("tail") == "{":
                brace = clean.rfind("{", absolute, member_end)
                member_end = _matching_brace(clean, brace)
            symbols.append(Symbol(
                member_id, member_name, member_qname, symbol_kind, "csharp", path,
                _line(text, absolute), _line(text, member_end), type_id, signature,
                _visibility(mods, "public" if kind == "interface" else "private"), mods,
            ))
            if re.search(r"\[Http(?:Get|Post|Put|Delete|Patch|Head|Options)\b", attrs):
                endpoint_qname = f"{member_qname}#endpoint"
                endpoint_id = _stable_id("sym", "csharp", path, endpoint_qname, "endpoint", params)
                symbols.append(Symbol(
                    endpoint_id, member_name, endpoint_qname, "endpoint", "csharp", path,
                    _line(text, absolute), _line(text, member_end), member_id, signature,
                    _visibility(mods, "public"), mods,
                ))
            member_text = clean[absolute:member_end + 1]
            for creation in re.finditer(r"\bnew\s+([A-Za-z_]\w*)\s*\(", member_text):
                raw_edges.append(_RawEdge(
                    "instantiates", member_id, creation.group(1), path,
                    _line(text, absolute + creation.start()), 0.93, "new-expression",
                ))
            for call in re.finditer(r"(?:\b[A-Za-z_]\w*\s*\.)?\b([A-Za-z_]\w*)\s*\(", member_text):
                called = call.group(1)
                if called not in {"if", "for", "foreach", "while", "switch", "catch", "lock", "using", "new", member_name}:
                    raw_edges.append(_RawEdge(
                        "calls", member_id, called, path,
                        _line(text, absolute + call.start()), 0.58, "call-shape",
                    ))

        for prop in _PROPERTY_RE.finditer(body):
            absolute = offset + prop.start()
            if _depth_at(clean, absolute) != body_depth:
                continue
            opening_prop = clean.find("{", absolute, offset + prop.end() + 1)
            closing_prop = _matching_brace(clean, opening_prop)
            accessors = clean[opening_prop + 1:closing_prop]
            if not re.search(r"\b(?:get|set|init)\s*(?:;|\{)", accessors):
                continue
            prop_name = prop.group("name")
            mods = _modifiers(prop.group("mods"))
            prop_qname = f"{qname}.{prop_name}"
            prop_id = _stable_id("sym", "csharp", path, prop_qname, "property", prop.group("type"))
            symbols.append(Symbol(
                prop_id, prop_name, prop_qname, "property", "csharp", path,
                _line(text, absolute), _line(text, closing_prop), type_id,
                text[absolute:offset + prop.end()].strip(), _visibility(mods), mods,
            ))

        for field in _FIELD_RE.finditer(body):
            absolute = offset + field.start()
            if _depth_at(clean, absolute) != body_depth:
                continue
            field_name = field.group("name")
            mods = _modifiers(field.group("mods"))
            field_kind = "constant" if "const" in mods else "field"
            field_qname = f"{qname}.{field_name}"
            field_id = _stable_id("sym", "csharp", path, field_qname, field_kind, field.group("type"))
            symbols.append(Symbol(
                field_id, field_name, field_qname, field_kind, "csharp", path,
                _line(text, absolute), _line(text, offset + field.end()), type_id,
                text[absolute:offset + field.end()].strip(), _visibility(mods), mods,
            ))

    # ASP.NET Core service registrations are useful even when Program.cs uses
    # top-level statements and therefore has no enclosing method symbol.
    for registration in re.finditer(
        r"\bAdd(?:Scoped|Transient|Singleton)\s*<\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*>",
        clean,
    ):
        raw_edges.append(_RawEdge(
            "configured_by", module_id, registration.group(2), path,
            _line(text, registration.start()), 0.97, "aspnet-di-registration",
        ))

    unique_symbols = {symbol.id: symbol for symbol in symbols}
    return list(unique_symbols.values()), raw_edges


class SymbolIndex:
    """Thread-safe in-memory symbol and relationship index."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._symbols: dict[str, Symbol] = {}
        self._raw_edges: list[_RawEdge] = []
        self._edges: list[SymbolEdge] = []
        self._files: set[str] = set()
        self._roots: list[str] = []

    def reset(self) -> None:
        with self._lock:
            self._symbols.clear()
            self._raw_edges.clear()
            self._edges.clear()
            self._files.clear()
            self._roots.clear()

    def index_directory(self, directory: str, append: bool = False) -> dict:
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            raise FileNotFoundError(directory)
        paths: list[str] = []
        for current, dirs, files in os.walk(directory):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
            paths.extend(os.path.join(current, name) for name in sorted(files) if name.endswith(".cs"))
        with self._lock:
            if not append:
                self._symbols.clear()
                self._raw_edges.clear()
                self._files.clear()
                self._roots = []
            if directory not in self._roots:
                self._roots.append(directory)
            for path in paths:
                self._remove_file(path)
                self._add_file(path)
            self._resolve_edges()
            return self.stats()

    def update_file(self, filepath: str) -> dict:
        filepath = os.path.abspath(filepath)
        with self._lock:
            self._remove_file(filepath)
            if os.path.isfile(filepath) and filepath.endswith(".cs"):
                self._add_file(filepath)
            self._resolve_edges()
            return self.stats()

    def _add_file(self, filepath: str) -> None:
        try:
            with open(filepath, encoding="utf-8-sig", errors="replace") as handle:
                text = handle.read()
        except OSError:
            return
        symbols, edges = _extract_csharp(filepath, text)
        self._symbols.update((symbol.id, symbol) for symbol in symbols)
        self._raw_edges.extend(edges)
        self._files.add(filepath)

    def _remove_file(self, filepath: str) -> None:
        filepath = os.path.abspath(filepath)
        stale = {symbol_id for symbol_id, symbol in self._symbols.items() if os.path.abspath(symbol.file_path) == filepath}
        for symbol_id in stale:
            self._symbols.pop(symbol_id, None)
        self._raw_edges = [edge for edge in self._raw_edges if os.path.abspath(edge.file_path) != filepath]
        self._files.discard(filepath)

    def _resolve_edges(self) -> None:
        by_name: dict[str, list[Symbol]] = {}
        for symbol in self._symbols.values():
            by_name.setdefault(symbol.name.casefold(), []).append(symbol)
            by_name.setdefault(symbol.qualified_name.casefold(), []).append(symbol)
        resolved: list[SymbolEdge] = []
        seen: set[str] = set()
        preferred = {
            "implements": {"interface"}, "inherits": TYPE_KINDS,
            "instantiates": {"class", "struct", "record"},
            "calls": {"method", "constructor", "endpoint", "test"},
            "configured_by": TYPE_KINDS,
        }
        for raw in self._raw_edges:
            candidates = by_name.get(raw.target_name.casefold(), [])
            filtered = [candidate for candidate in candidates if candidate.kind in preferred.get(raw.kind, set())]
            target = (filtered or candidates or [None])[0]
            target_id = target.id if target else None
            edge_id = _stable_id("edge", raw.kind, raw.source_symbol_id, target_id, raw.target_name, raw.file_path, raw.line)
            if edge_id in seen:
                continue
            seen.add(edge_id)
            resolved.append(SymbolEdge(
                edge_id, raw.kind, raw.source_symbol_id, target_id, raw.target_name,
                raw.file_path, raw.line, raw.confidence, raw.provenance,
            ))
        self._edges = resolved

    def stats(self) -> dict:
        with self._lock:
            return {
                "files": len(self._files), "symbols": len(self._symbols),
                "edges": len(self._edges), "base_dir": self._roots[0] if self._roots else None,
                "roots": list(self._roots), "parser": PARSER,
                "parser_version": PARSER_VERSION,
            }

    def symbols(self) -> list[Symbol]:
        with self._lock:
            return sorted(self._symbols.values(), key=lambda value: (value.file_path, value.line_start, value.kind))

    def edges(self) -> list[SymbolEdge]:
        with self._lock:
            return list(self._edges)

    def find_symbol(self, query: str, kind: str | None = None, top_k: int = 20) -> list[Symbol]:
        needle = query.strip().casefold()
        if not needle:
            return []
        with self._lock:
            ranked: list[tuple[int, str, Symbol]] = []
            for symbol in self._symbols.values():
                if kind and symbol.kind != kind:
                    continue
                name = symbol.name.casefold()
                qualified = symbol.qualified_name.casefold()
                if needle == name or needle == qualified:
                    score = 0
                elif name.startswith(needle) or qualified.endswith("." + needle):
                    score = 1
                elif needle in name or needle in qualified:
                    score = 2
                else:
                    continue
                ranked.append((score, qualified, symbol))
            ranked.sort(key=lambda item: (item[0], item[1], item[2].file_path, item[2].line_start))
            return [item[2] for item in ranked[:max(1, min(top_k, 100))]]

    def find_implementations(self, query: str, top_k: int = 20) -> list[dict]:
        targets = self.find_symbol(query, top_k=100)
        target_ids = {symbol.id for symbol in targets if symbol.kind in TYPE_KINDS}
        target_names = {symbol.name.casefold() for symbol in targets}
        target_names.add(query.strip().casefold().split(".")[-1])
        with self._lock:
            results: list[dict] = []
            used: set[str] = set()
            for edge in self._edges:
                if edge.kind not in {"implements", "inherits"}:
                    continue
                if edge.target_symbol_id not in target_ids and edge.target_name.casefold() not in target_names:
                    continue
                source = self._symbols.get(edge.source_symbol_id)
                if source and source.id not in used:
                    used.add(source.id)
                    results.append({"symbol": source, "edge": edge})
            results.sort(key=lambda result: (result["symbol"].qualified_name, result["symbol"].file_path))
            return results[:max(1, min(top_k, 100))]


__all__ = ["Symbol", "SymbolEdge", "SymbolIndex", "PARSER", "PARSER_VERSION"]
