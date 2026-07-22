"""Deterministic, budget-aware context bundles for agent change planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import re

try:
    from redact import redact
except ImportError:  # pragma: no cover - package import fallback
    from .redact import redact


SECTION_ORDER = (
    "primary_implementation",
    "parent_context",
    "direct_callers",
    "direct_dependencies",
    "implementations_overrides",
    "configuration_documentation",
    "tests_coverage",
    "git_changes",
)
SECTION_LABELS = {
    "primary_implementation": "Primary implementation",
    "parent_context": "Parent class/module context",
    "direct_callers": "Direct callers",
    "direct_dependencies": "Direct dependencies",
    "implementations_overrides": "Implementations and overrides",
    "configuration_documentation": "Configuration and documentation",
    "tests_coverage": "Tests and coverage",
    "git_changes": "Relevant Git changes",
}
_SECTION_RANK = {name: rank for rank, name in enumerate(SECTION_ORDER)}
_STOPWORDS = {
    "add", "adjust", "and", "change", "create", "delete", "feature", "fix",
    "for", "from", "implement", "in", "make", "modify", "of", "remove",
    "rename", "the", "this", "to", "update", "with",
}
_CONFIG_NAMES = {
    "dockerfile", "makefile", "readme", "license", "pyproject.toml",
    "package.json", "appsettings.json", "appsettings.development.json",
}
_CONFIG_SUFFIXES = {
    ".cfg", ".conf", ".config", ".ini", ".json", ".md", ".rst",
    ".toml", ".yaml", ".yml",
}
_SKIP_PARTS = {
    ".agents", ".git", ".gpu-search-cache", ".venv", ".vs",
    "__pycache__", "build", "dist", "node_modules", "target",
}


def estimate_tokens(text: str) -> int:
    """Stable approximation used for budgets; intentionally tokenizer-free."""
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


@dataclass(frozen=True, slots=True)
class PlanItem:
    section: str
    title: str
    file_path: str
    line_start: int
    line_end: int
    reason: str
    confidence: float
    content: str
    estimated_tokens: int
    symbol_id: str | None = None
    symbol_kind: str | None = None
    git_boost: float = 0.0
    truncated: bool = False
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OmittedItem:
    section: str
    title: str
    file_path: str
    line_start: int
    reason: str
    estimated_tokens: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChangePlan:
    request: str
    top_k: int
    max_context_tokens: int
    tokens_used: int
    items: tuple[PlanItem, ...]
    omitted: tuple[OmittedItem, ...]
    risks: tuple[str, ...]
    unknowns: tuple[str, ...]
    inspection_order: tuple[str, ...]
    likely_change_set: tuple[str, ...]
    index_status: dict

    def as_dict(self) -> dict:
        return {
            "request": self.request,
            "top_k": self.top_k,
            "max_context_tokens": self.max_context_tokens,
            "tokens_used": self.tokens_used,
            "items": [item.as_dict() for item in self.items],
            "omitted": [item.as_dict() for item in self.omitted],
            "risks": list(self.risks),
            "unknowns": list(self.unknowns),
            "inspection_order": list(self.inspection_order),
            "likely_change_set": list(self.likely_change_set),
            "index_status": self.index_status,
        }

    def to_markdown(self, base_dir: str | None = None) -> str:
        lines = [
            f"# Change plan: {self.request}",
            "",
            f"Context budget: {self.tokens_used}/{self.max_context_tokens} estimated tokens",
        ]
        grouped = {section: [] for section in SECTION_ORDER}
        for item in self.items:
            grouped[item.section].append(item)
        for section in SECTION_ORDER:
            lines.extend(["", f"## {SECTION_LABELS[section]}"])
            if not grouped[section]:
                lines.append("- No included context.")
                continue
            for item in grouped[section]:
                path = _relative(item.file_path, base_dir)
                location = f"{path}:{item.line_start}"
                if item.line_end > item.line_start:
                    location += f"-{item.line_end}"
                truncated = "; truncated" if item.truncated else ""
                lines.append(
                    f"- **{item.title}** — `{location}` "
                    f"(confidence {item.confidence:.2f}; {item.reason}{truncated})"
                )
                if item.content:
                    lines.extend(["", "```", item.content, "```"])
        lines.extend(["", "## Match reasons and confidence"])
        for item in self.items:
            lines.append(
                f"- {item.title}: {item.reason}; confidence={item.confidence:.2f}; "
                f"git_boost={item.git_boost:.2f}"
            )
        lines.extend(["", "## Likely change set"])
        if self.likely_change_set:
            lines.extend(
                f"- `{_relative(path, base_dir)}`"
                for path in self.likely_change_set
            )
        else:
            lines.append("- No likely edits identified.")
        lines.extend(["", "## Risks"])
        lines.extend(f"- {risk}" for risk in self.risks) if self.risks else lines.append("- None identified.")
        lines.extend(["", "## Unknowns"])
        lines.extend(f"- {value}" for value in self.unknowns) if self.unknowns else lines.append("- None identified.")
        lines.extend(["", "## Inspection order"])
        lines.extend(
            f"{index}. `{_relative(path, base_dir)}`"
            for index, path in enumerate(self.inspection_order, 1)
        )
        lines.extend(["", "## Omitted items"])
        if self.omitted:
            for item in self.omitted:
                lines.append(
                    f"- {SECTION_LABELS.get(item.section, item.section)}: "
                    f"`{_relative(item.file_path, base_dir)}:{item.line_start}` — "
                    f"{item.reason} ({item.estimated_tokens} estimated tokens)"
                )
        else:
            lines.append("- None.")
        return "\n".join(lines)


@dataclass(slots=True)
class _Candidate:
    section: str
    title: str
    file_path: str
    line_start: int
    line_end: int
    reason: str
    confidence: float
    content: str
    priority: int
    symbol_id: str | None = None
    symbol_kind: str | None = None
    git_boost: float = 0.0
    metadata: dict = field(default_factory=dict)

    def sort_key(self) -> tuple:
        # Priority is evaluated before Git state so an exact symbol always wins.
        return (
            _SECTION_RANK[self.section], self.priority, -self.confidence,
            -self.git_boost, self.file_path.casefold(), self.line_start, self.title,
        )


class ChangePlanner:
    """Compose an ordered plan from exact, semantic, graph, and Git evidence."""

    def __init__(self, pattern, semantic, deps, symbols, git_state) -> None:
        self.pattern = pattern
        self.semantic = semantic
        self.deps = deps
        self.symbols = symbols
        self.git_state = git_state

    def plan_change(
        self,
        request: str,
        top_k: int = 8,
        max_context_tokens: int = 6000,
    ) -> ChangePlan:
        request = (request or "").strip()
        if not request:
            raise ValueError("request must not be empty")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if not 256 <= max_context_tokens <= 100_000:
            raise ValueError("max_context_tokens must be between 256 and 100000")

        p_stats = self._stats(self.pattern)
        s_stats = self._stats(self.semantic)
        d_stats = self._stats(self.deps)
        y_stats = self._stats(self.symbols)
        base = (
            y_stats.get("base_dir") or p_stats.get("base_dir")
            or s_stats.get("base_dir") or d_stats.get("base_dir")
        )
        terms = _query_terms(request)
        candidates: list[_Candidate] = []
        symbol_seeds = self._symbol_candidates(terms, top_k, candidates)
        pattern_candidates = self._pattern_candidates(terms, top_k, candidates)
        self._semantic_candidates(request, top_k, candidates)

        primary_paths = []
        for candidate in sorted(candidates, key=lambda item: item.sort_key()):
            if candidate.section == "primary_implementation" and candidate.file_path not in primary_paths:
                primary_paths.append(candidate.file_path)
        primary_paths = primary_paths[:top_k]

        self._parent_candidates(symbol_seeds, candidates)
        self._relationship_candidates(symbol_seeds, top_k, candidates)
        self._dependency_candidates(primary_paths, top_k, candidates)
        self._configuration_candidates(pattern_candidates, candidates)
        self._test_candidates(symbol_seeds, top_k, candidates)
        self._git_candidates(candidates)

        items, omitted, tokens_used = self._allocate(
            request, candidates, top_k, max_context_tokens,
        )
        included_sections = {item.section for item in items}
        risks: list[str] = []
        unknowns: list[str] = []
        if "primary_implementation" not in included_sections:
            risks.append("No primary implementation fit the context bundle.")
        if "tests_coverage" not in included_sections:
            risks.append("No related tests were identified; coverage may be missing.")
        low_confidence = [item for item in items if item.confidence < 0.70]
        if low_confidence:
            risks.append("Some relationships are heuristic and below 0.70 confidence.")
        if omitted:
            risks.append("The token budget omitted relevant context; inspect omissions before editing.")
        if not d_stats.get("files", 0):
            unknowns.append("Dependency graph is unavailable, so file-level callers/dependencies may be incomplete.")
        if not y_stats.get("symbols", 0):
            unknowns.append("Symbol index is unavailable, so structural relationships may be incomplete.")
        if not s_stats.get("chunks", 0):
            unknowns.append("Semantic index is unavailable; the plan relies on exact and graph evidence.")
        if not symbol_seeds and pattern_candidates:
            unknowns.append("No exact symbol matched; primary ranking is based on text matches.")

        inspection_order = tuple(dict.fromkeys(item.file_path for item in items if item.file_path))
        likely_sections = {
            "primary_implementation", "implementations_overrides",
            "configuration_documentation", "tests_coverage",
        }
        likely_change_set = tuple(dict.fromkeys(
            item.file_path for item in items
            if item.file_path and item.section in likely_sections
        ))
        return ChangePlan(
            request=request,
            top_k=top_k,
            max_context_tokens=max_context_tokens,
            tokens_used=tokens_used,
            items=tuple(items),
            omitted=tuple(omitted),
            risks=tuple(risks),
            unknowns=tuple(unknowns),
            inspection_order=inspection_order,
            likely_change_set=likely_change_set,
            index_status={
                "pattern_ready": bool(p_stats.get("files", 0)),
                "semantic_ready": bool(s_stats.get("chunks", 0)),
                "dependency_ready": bool(d_stats.get("files", 0)),
                "symbol_ready": bool(y_stats.get("symbols", 0)),
                "base_dir": base,
            },
        )

    @staticmethod
    def _stats(service) -> dict:
        try:
            return service.stats()
        except Exception:
            return {}

    def _boost(self, filepath: str) -> float:
        try:
            return float(self.git_state.boost(filepath))
        except Exception:
            return 0.0

    def _symbol_candidates(self, terms, top_k, candidates) -> list:
        seeds = []
        seen: set[str] = set()
        if not self._stats(self.symbols).get("symbols", 0):
            return seeds
        for term in terms:
            try:
                matches = self.symbols.find_symbol(term, top_k=top_k)
            except Exception:
                continue
            matches = [
                symbol for symbol in matches
                if not _is_skipped_path(symbol.file_path)
            ]
            declaration_matches = [
                symbol for symbol in matches
                if symbol.kind not in {"module", "namespace"}
            ]
            if declaration_matches:
                matches = declaration_matches
            matches = sorted(
                matches,
                key=lambda symbol: (
                    symbol.name.casefold() != term.casefold()
                    and symbol.qualified_name.casefold() != term.casefold(),
                    symbol.qualified_name.casefold(),
                    symbol.file_path,
                    symbol.line_start,
                ),
            )
            for symbol in matches:
                if symbol.id in seen:
                    continue
                exact = term.casefold() in {
                    symbol.name.casefold(), symbol.qualified_name.casefold(),
                }
                if not exact and seeds:
                    continue
                seen.add(symbol.id)
                seeds.append(symbol)
                candidates.append(self._from_symbol(
                    "primary_implementation", symbol,
                    "exact symbol match" if exact else "partial symbol match",
                    1.0 if exact else 0.90, 0 if exact else 1,
                ))
                if len(seeds) >= top_k:
                    return seeds
        return seeds

    def _pattern_candidates(self, terms, top_k, candidates) -> list[_Candidate]:
        found: list[_Candidate] = []
        if not self._stats(self.pattern).get("files", 0):
            return found
        seen: set[tuple[str, int]] = set()
        for term_rank, term in enumerate(terms[:6]):
            try:
                results = self.pattern.search(term)
            except Exception:
                continue
            for result in results[:top_k]:
                path = os.path.abspath(result["file"])
                if _is_skipped_path(path):
                    continue
                for match in result.get("matches", [])[:2]:
                    line = int(match.get("line") or 1)
                    key = (os.path.normcase(path), line)
                    if key in seen:
                        continue
                    seen.add(key)
                    content = redact(str(match.get("content") or ""))[:2000]
                    candidate = _Candidate(
                        "primary_implementation", os.path.basename(path), path,
                        line, line, f"exact text match for {term}", 0.95, content,
                        1 + term_rank,
                        git_boost=self._boost(path), metadata={"term": term},
                    )
                    found.append(candidate)
                    candidates.append(candidate)
        return found

    def _semantic_candidates(self, request, top_k, candidates) -> None:
        if not self._stats(self.semantic).get("chunks", 0):
            return
        try:
            results = self.semantic.search(request, top_k=top_k)
        except Exception:
            return
        for result in results[:top_k]:
            path = os.path.abspath(result["file"])
            if _is_skipped_path(path):
                continue
            candidates.append(_Candidate(
                "primary_implementation", os.path.basename(path), path,
                int(result.get("start_line") or 1), int(result.get("end_line") or 1),
                "semantic match", float(result.get("score") or 0.0),
                redact(str(result.get("snippet") or ""))[:4000], 2,
                git_boost=self._boost(path),
            ))

    def _from_symbol(self, section, symbol, reason, confidence, priority) -> _Candidate:
        return _Candidate(
            section, f"{symbol.kind} {symbol.qualified_name}",
            os.path.abspath(symbol.file_path), symbol.line_start, symbol.line_end,
            reason, confidence,
            _read_excerpt(symbol.file_path, symbol.line_start, symbol.line_end),
            priority, symbol.id, symbol.kind, self._boost(symbol.file_path),
        )

    def _parent_candidates(self, seeds, candidates) -> None:
        try:
            by_id = {symbol.id: symbol for symbol in self.symbols.symbols()}
        except Exception:
            return
        seen: set[str] = set()
        for symbol in seeds:
            parent = by_id.get(symbol.parent_symbol_id)
            if parent and parent.id not in seen:
                seen.add(parent.id)
                candidates.append(self._from_symbol(
                    "parent_context", parent, "parent symbol context", 0.98, 0,
                ))

    def _relationship_candidates(self, seeds, top_k, candidates) -> None:
        seen: set[tuple[str, str]] = set()
        for seed in seeds:
            try:
                callers = self.symbols.find_callers(seed.name, top_k=top_k)
                implementations = self.symbols.find_implementations(seed.name, top_k=top_k)
                overrides = self.symbols.find_references(
                    seed.name, kinds={"overrides"}, top_k=top_k,
                )
            except Exception:
                continue
            for result in callers:
                symbol, edge = result.get("symbol"), result["edge"]
                if symbol and ("caller", symbol.id) not in seen:
                    seen.add(("caller", symbol.id))
                    candidates.append(self._from_symbol(
                        "direct_callers", symbol, edge.provenance,
                        edge.confidence, 1,
                    ))
            for result in implementations + overrides:
                symbol, edge = result.get("symbol"), result["edge"]
                if symbol and ("implementation", symbol.id) not in seen:
                    seen.add(("implementation", symbol.id))
                    candidates.append(self._from_symbol(
                        "implementations_overrides", symbol,
                        edge.kind + ": " + edge.provenance, edge.confidence, 1,
                    ))

    def _dependency_candidates(self, primary_paths, top_k, candidates) -> None:
        if not self._stats(self.deps).get("files", 0):
            return
        seen: set[tuple[str, str]] = set()
        for primary in primary_paths:
            try:
                dependencies = self.deps.direct_imports(primary)
                impacted = self.deps.impact(primary)
            except Exception:
                continue
            for path in dependencies[:top_k]:
                if _is_skipped_path(path):
                    continue
                key = ("dependency", os.path.normcase(path))
                if key not in seen:
                    seen.add(key)
                    candidates.append(_file_candidate(
                        "direct_dependencies", path, "direct import", 0.90,
                        self._boost(path),
                    ))
            for result in impacted:
                if int(result.get("hops") or 0) != 1:
                    continue
                path = result["file"]
                if _is_skipped_path(path):
                    continue
                key = ("caller", os.path.normcase(path))
                if key not in seen:
                    seen.add(key)
                    candidates.append(_file_candidate(
                        "direct_callers", path,
                        result.get("reason") or "direct importer", 0.88,
                        self._boost(path),
                    ))

    def _configuration_candidates(self, pattern_candidates, candidates) -> None:
        for candidate in pattern_candidates:
            if _is_configuration_path(candidate.file_path):
                candidates.append(_Candidate(
                    "configuration_documentation", candidate.title,
                    candidate.file_path, candidate.line_start, candidate.line_end,
                    "configuration/documentation text match", candidate.confidence,
                    candidate.content, 1, git_boost=candidate.git_boost,
                ))

    def _test_candidates(self, seeds, top_k, candidates) -> None:
        queries = list(seeds)
        try:
            for seed in seeds:
                queries.extend(
                    result["symbol"]
                    for result in self.symbols.find_implementations(seed.name, top_k=top_k)
                )
        except Exception:
            pass
        seen: set[str] = set()
        for target in queries:
            try:
                tests = self.symbols.find_tests(target.name, top_k=top_k)
            except Exception:
                continue
            for result in tests:
                symbol, edge = result["symbol"], result["edge"]
                if symbol.id not in seen:
                    seen.add(symbol.id)
                    candidates.append(self._from_symbol(
                        "tests_coverage", symbol, edge.provenance,
                        edge.confidence, 1,
                    ))

    def _git_candidates(self, candidates) -> None:
        seen: set[str] = set()
        snapshot = list(candidates)
        for candidate in snapshot:
            path = candidate.file_path
            if not path or path in seen or candidate.git_boost <= 0:
                continue
            seen.add(path)
            candidates.append(_Candidate(
                "git_changes", os.path.basename(path), path,
                candidate.line_start, candidate.line_start,
                "recent or modified Git state", min(1.0, 0.6 + candidate.git_boost),
                f"Git relevance boost: {candidate.git_boost:.2f}", 3,
                git_boost=candidate.git_boost,
            ))

    @staticmethod
    def _allocate(request, candidates, top_k, budget):
        used = estimate_tokens(request) + 48
        items: list[PlanItem] = []
        omitted: list[OmittedItem] = []
        section_counts = {section: 0 for section in SECTION_ORDER}
        seen: set[tuple[str, str, int, str | None]] = set()
        for candidate in sorted(candidates, key=lambda item: item.sort_key()):
            key = (
                candidate.section, os.path.normcase(candidate.file_path),
                candidate.line_start, candidate.symbol_id,
            )
            if key in seen:
                continue
            seen.add(key)
            token_cost = estimate_tokens(
                candidate.title + candidate.reason + candidate.content
            ) + 12
            if section_counts[candidate.section] >= top_k:
                omitted.append(_omitted(candidate, "top_k limit", token_cost))
                continue
            remaining = budget - used
            content = candidate.content
            truncated = False
            if token_cost > remaining:
                if (
                    candidate.section == "primary_implementation"
                    and remaining >= 64 and content
                ):
                    content = content[:max(1, (remaining - 16) * 4)]
                    token_cost = min(remaining, estimate_tokens(
                        candidate.title + candidate.reason + content
                    ) + 12)
                    truncated = True
                else:
                    omitted.append(_omitted(candidate, "token budget", token_cost))
                    continue
            used += token_cost
            section_counts[candidate.section] += 1
            items.append(PlanItem(
                candidate.section, candidate.title, candidate.file_path,
                candidate.line_start, candidate.line_end, candidate.reason,
                round(candidate.confidence, 4), content, token_cost,
                candidate.symbol_id, candidate.symbol_kind,
                round(candidate.git_boost, 4), truncated, candidate.metadata,
            ))
        return items, omitted, used


def _omitted(candidate: _Candidate, reason: str, tokens: int) -> OmittedItem:
    return OmittedItem(
        candidate.section, candidate.title, candidate.file_path,
        candidate.line_start, reason, tokens,
    )


def _query_terms(request: str) -> list[str]:
    raw = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", request)
    preferred = [
        token for token in raw
        if token.casefold() not in _STOPWORDS
        and ("_" in token or "." in token or any(ch.isupper() for ch in token[1:]))
    ]
    fallback = sorted(
        (token for token in raw if token.casefold() not in _STOPWORDS and len(token) > 2),
        key=lambda token: (-len(token), token.casefold()),
    )
    return list(dict.fromkeys(preferred + fallback))[:8] or [request]


def _read_excerpt(path: str, line_start: int = 1, line_end: int = 80) -> str:
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return ""
    start = max(1, line_start)
    end = max(start, min(line_end, start + 119, len(lines)))
    return redact("".join(lines[start - 1:end]))[:6000].rstrip()


def _file_candidate(section, path, reason, confidence, git_boost) -> _Candidate:
    path = os.path.abspath(path)
    return _Candidate(
        section, os.path.basename(path), path, 1, 80, reason, confidence,
        _read_excerpt(path), 2, git_boost=git_boost,
    )


def _is_skipped_path(path: str) -> bool:
    parts = {part.casefold() for part in Path(path).parts}
    return bool(parts & _SKIP_PARTS)


def _is_configuration_path(path: str) -> bool:
    name = os.path.basename(path).casefold()
    return name in _CONFIG_NAMES or os.path.splitext(name)[1] in _CONFIG_SUFFIXES


def _relative(path: str, base_dir: str | None) -> str:
    if not path:
        return "unknown"
    if base_dir:
        try:
            return os.path.relpath(path, base_dir)
        except ValueError:
            pass
    return path


__all__ = [
    "ChangePlan", "ChangePlanner", "OmittedItem", "PlanItem",
    "SECTION_ORDER", "estimate_tokens",
]
