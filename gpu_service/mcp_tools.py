"""MCP tool and prompt registrations for gpu-search.

All tools are registered via register(mcp), called from mcp_server after global
state is fully initialised. Tools access shared state through a module reference
obtained inside register() to avoid circular imports at the module level.
"""
import os
import sys
import threading


def register(mcp) -> dict:
    """Register all MCP tools and prompts onto mcp. Returns exposed tool functions."""
    # Late import inside the function: mcp_server is still being loaded when
    # register() is called, but all global state (index, semantic, deps, helpers)
    # is defined before this point.  By the time any tool is *invoked*, the
    # module is fully initialised.
    import mcp_server as _app
    from mcp.server.fastmcp import Context

    # -----------------------------------------------------------------------
    # Search tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    def search_code(query: str, top_k: int = 5, mode: str = "auto",
                    context_mode: str = "normal", ctx: Context = None) -> str:
        """Search code by exact identifier or natural language.
        mode: 'auto' (default) routes identifier→pattern, prose→semantic;
              'hybrid' runs both in parallel and merges results;
              'pattern' or 'semantic' forces a specific engine.
        context_mode: 'compact' returns file/line/snippet/reason; 'normal' expands
              small AST blocks; 'full' is the least compressed.
        Use this for ALL searches."""
        semantic_candidate = mode in {"semantic", "hybrid"} or (
            mode == "auto" and _app._is_natural_language(query)
        )

        if ctx is not None and semantic_candidate:
            _app._auto_load_semantic(ctx)

        pattern_ready = False
        if mode in {"pattern", "hybrid"} or not semantic_candidate:
            pattern_ready = _app.index.stats()["files"] > 0

        semantic_ready = False
        if semantic_candidate:
            semantic_stats = _app.semantic.stats()
            semantic_ready = semantic_stats["chunks"] > 0

        if mode == "auto":
            effective = "semantic" if _app._is_natural_language(query) and semantic_ready else "pattern"
        else:
            effective = mode

        # ── Hybrid ────────────────────────────────────────────────────────
        if effective == "hybrid":
            if not pattern_ready and not semantic_ready:
                return "No index found. Call gpu_index and gpu_semantic_index first."

            p_results: list = []
            s_results: list = []

            def _run_p():
                if pattern_ready:
                    p_results.extend(_app._do_pattern_search(query))

            def _run_s():
                if semantic_ready:
                    try:
                        s_results.extend(_app._do_semantic_search(query, top_k))
                    except Exception:
                        pass

            pt = threading.Thread(target=_run_p)
            st = threading.Thread(target=_run_s)
            pt.start()
            st.start()
            pt.join()
            st.join()

            out = _app._format_hybrid_results(
                p_results, s_results, query,
                _app.index.stats(), _app.semantic.stats(), context_mode,
            )
            if out and p_results:
                out = _app._append_blast_radius(out, p_results)
            return out or f"No results for '{query}'"

        # ── Semantic ──────────────────────────────────────────────────────
        if effective == "semantic":
            if not semantic_ready:
                return _app.semantic.semantic_unavailable_message()
            try:
                results = _app.semantic.search(query, top_k=top_k)
            except Exception:
                return _app.semantic.semantic_unavailable_message()
            for r in results:
                r["score"] = round(r["score"] + _app.git_state.boost(r["file"]), 4)
            results.sort(key=lambda r: r["score"], reverse=True)
            out = _app._format_semantic_results(results, query, _app.semantic.stats(),
                                                context_mode=context_mode)
            return out or f"No semantic matches for '{query}'"

        # ── Pattern (default) ─────────────────────────────────────────────
        if not pattern_ready:
            return ("No index found. Call gpu_index (and optionally gpu_semantic_index)"
                    " with your project directory first.")
        results = _app._do_pattern_search(query)
        results.sort(key=lambda r: _app.git_state.boost(r["file"]), reverse=True)
        for r in results:
            r["reason"] = "exact token match" + (
                " + recent git activity" if _app.git_state.boost(r["file"]) else ""
            )
        out = _app._format_pattern_results(results, _app.index.stats(), context_mode=context_mode)
        if not out:
            return f"No matches for '{query}'"
        out = _app._append_blast_radius(out, results)
        return out

    @mcp.tool()
    def gpu_search(query: str, case_sensitive: bool = False) -> str:
        """Exact-text pattern search. Use only when case_sensitive control is needed; otherwise use search_code."""
        stats = _app.index.stats()
        if stats["files"] == 0:
            return "No files indexed. Call gpu_index with your project directory first."
        results = _app.index.search(query, case_sensitive=case_sensitive)
        out = _app._format_pattern_results(results, stats)
        return out or f"No matches for '{query}'"

    @mcp.tool()
    async def gpu_index(directory: str, append: bool = False) -> str:
        """Load a directory into GPU VRAM for pattern search. Runs in background; call gpu_stats to check. append=True for multi-root."""
        if not os.path.isdir(directory):
            return f"Directory not found: {directory}"

        def _do():
            _app._bg_status["pattern"] = f"indexing {directory}..."
            stats = _app.index.index_directory(directory, append=append,
                                                allow_env_files=_app._ALLOW_ENV_FILES)
            _app._bg_status["pattern"] = f"done: {stats['indexed']} files ({stats['vram_mb']} MB)"

        threading.Thread(target=_do, daemon=True).start()
        return f"Pattern indexing started for {directory} — call gpu_stats to check progress."

    @mcp.tool()
    def gpu_stats() -> str:
        """Show index status and VRAM usage for all indexes."""
        p = _app.index.stats()
        s = _app.semantic.stats()
        d = _app.deps.stats()
        g_modified = len(_app.git_state._modified)
        g_recent = len(_app.git_state._recent)
        lines = [
            f"Pattern index:  {p['files']} files, {p['vram_mb']} MB, cache={p.get('cache', 'n/a')}  ({p['base_dir'] or 'none'})",
            f"Semantic index: {s['chunks']} chunks, {s['vram_mb']} MB  ({s['base_dir'] or 'not built'})",
            f"Dep graph:      {d['files']} files, {d['edges']} edges, cache={d.get('cache', 'n/a')}  ({d['base_dir'] or 'not built'})",
            f"Git state:      {g_modified} modified, {g_recent} recently-committed files",
        ]
        if _app._bg_status["pattern"]:
            lines.append(f"Pattern status:   {_app._bg_status['pattern']}")
        if _app._bg_status["deps"]:
            lines.append(f"Deps status:      {_app._bg_status['deps']}")
        if _app._bg_status["semantic"]:
            lines.append(f"Semantic status:  {_app._bg_status['semantic']}")
        if s.get("chunks_capped"):
            lines.append(f"Semantic WARNING: chunk cap ({_app.MAX_CHUNKS:,}) hit — index is partial")
        if s.get("embed_progress"):
            lines.append(f"Embed progress:   {s['embed_progress']}")
        if s.get("last_error"):
            lines.append(f"Semantic ERROR:   {s['last_error']}")
        elif s.get("model_error"):
            lines.append(f"Semantic ERROR:   {s['model_error']}")
        return "\n".join(lines)

    @mcp.tool()
    def gpu_update_file(filepath: str) -> str:
        """Re-index a specific file after editing it (keeps VRAM in sync)."""
        _app.index.update_file(filepath)
        return f"Updated: {filepath}"

    @mcp.tool()
    def gpu_read_block(filepath: str, line: int) -> str:
        """Read the AST-expanded block (function/class) that contains the given line number. Pass a line from search results to get the full syntactically-complete context instead of a raw snippet."""
        from ast_expand import read_block
        if not os.path.isfile(filepath):
            return f"File not found: {filepath}"
        code, start, end = read_block(filepath, line)
        base = _app.index.stats().get("base_dir") or os.path.dirname(filepath)
        rel = os.path.relpath(filepath, base)
        return f"{rel} L{start}–{end}:\n```\n{code}```"

    @mcp.tool()
    def gpu_skeleton(filepath: str, match_lines: list[int] = None) -> str:
        """Return a code skeleton of a file with unexpanded function bodies folded to '...'. Pass match_lines (from search results) to keep those blocks fully expanded. Useful for understanding a large file's structure without reading all N thousand lines."""
        from ast_expand import skeleton_file
        if not os.path.isfile(filepath):
            return f"File not found: {filepath}"
        result = skeleton_file(filepath, match_lines)
        if result is None:
            try:
                lines = open(filepath, encoding="utf-8", errors="replace").readlines()
                return f"No AST parser for this file type ({len(lines)} lines). Use Read tool to view it directly."
            except Exception as e:
                return f"Could not read {filepath}: {e}"
        base = _app.index.stats().get("base_dir") or os.path.dirname(filepath)
        rel = os.path.relpath(filepath, base)
        return f"Skeleton of {rel}:\n```\n{result}```"

    @mcp.tool()
    async def gpu_semantic_index(directory: str, append: bool = False, force: bool = False) -> str:
        """Build semantic embedding cache for a directory (bge-small-en-v1.5). Runs in background; cache persists across restarts. append=True for multi-root, force=True to rebuild."""
        if not os.path.isdir(directory):
            return f"Directory not found: {directory}"

        if not append:
            _app.semantic.reset(base_dir=os.path.abspath(directory))

        def _do():
            try:
                _app._bg_status["semantic"] = f"embedding {directory}..."
                stats = _app.semantic.index_directory(directory, append=append, force=force)
                _app._bg_status["semantic"] = f"done: {stats['chunks']} chunks ({stats['vram_mb']} MB)"
                print(
                    f"[gpu-search] Semantic index ready: {stats['chunks']} chunks ({stats['vram_mb']} MB VRAM)",
                    file=sys.stderr, flush=True,
                )
            except Exception as e:
                _app._bg_status["semantic"] = f"ERROR: {e}"
                print(f"[gpu-search] Semantic index FAILED: {e}", file=sys.stderr, flush=True)
                if not _app.semantic.stats().get("model_error"):
                    import traceback
                    traceback.print_exc(file=sys.stderr)

        threading.Thread(target=_do, daemon=True).start()
        return f"Semantic indexing started for {directory} — call gpu_stats to check progress."

    @mcp.tool()
    async def gpu_add_directory(directory: str) -> str:
        """Add a directory to the permanent startup config so it auto-indexes on every future launch. Also indexes immediately."""
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            return f"Directory not found: {directory}"

        _app._save_config_dirs([directory])

        def _do_pattern():
            _app.git_state.add_root(directory)
            append = _app.index.stats()["files"] > 0
            _app._bg_status["pattern"] = f"indexing {directory}..."
            stats = _app.index.index_directory(directory, append=append,
                                                allow_env_files=_app._ALLOW_ENV_FILES)
            _app._bg_status["pattern"] = f"done: {stats['indexed']} files ({stats['vram_mb']} MB)"

        def _do_deps():
            try:
                append = _app.deps.stats()["files"] > 0
                _app._bg_status["deps"] = f"indexing {directory}..."
                dep_stats = _app.deps.index_directory(directory, append=append)
                _app._bg_status["deps"] = f"done: {dep_stats['files']} files, {dep_stats['edges']} edges"
            except Exception as e:
                _app._bg_status["deps"] = f"ERROR: {e}"
                print(f"[gpu-search] Dep index FAILED: {e}", file=sys.stderr, flush=True)

        def _do_semantic():
            s = _app.semantic.try_load_cache(directory)
            if s is None:
                _app._bg_status["semantic"] = (
                    f"no semantic cache for {directory} — run gpu_semantic_index to build it"
                )
            else:
                _app._bg_status["semantic"] = f"done: {s['chunks']} chunks ({s['vram_mb']} MB)"
                _app._loaded_roots.add(directory)

        threading.Thread(target=_do_pattern, daemon=True).start()
        threading.Thread(target=_do_deps, daemon=True).start()
        threading.Thread(target=_do_semantic, daemon=True).start()
        return f"Added '{directory}' to startup config. Indexing started — call gpu_stats to check progress."

    # -----------------------------------------------------------------------
    # Dependency tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    def dep_impact(filepath: str) -> str:
        """CALL BEFORE EDITING. Returns every file that transitively imports the given file, grouped by hop distance. Call dep_index first."""
        s = _app.deps.stats()
        if s["files"] == 0:
            if _app._bg_status.get("deps", "").startswith("indexing "):
                return f"Dependency graph is still building: {_app._bg_status['deps']}"
            return "Dependency graph not built. Call dep_index with your project directory first."

        results = _app.deps.impact(filepath)
        if not results:
            rel = os.path.relpath(filepath, s["base_dir"]) if s["base_dir"] else filepath
            return f"Nothing in the project imports '{rel}' — safe to change."

        base = s["base_dir"]
        by_hop: dict[int, list[str]] = {}
        for r in results:
            by_hop.setdefault(r["hops"], []).append(r)

        _MAX_PER_HOP = 20
        lines = [f"Impact of changing '{os.path.relpath(filepath, base)}' ({len(results)} affected files):"]
        for hop in sorted(by_hop):
            files = by_hop[hop]
            label = "Direct importers" if hop == 1 else f"Indirect (depth {hop})"
            shown = files[:_MAX_PER_HOP]
            lines.append(f"\n{label} ({len(files)} files):")
            for item in shown:
                reason = item.get("reason")
                suffix = f" — {reason}" if reason else ""
                lines.append(f"  {os.path.relpath(item['file'], base)}{suffix}")
            if len(files) > _MAX_PER_HOP:
                lines.append(f"  ... {len(files) - _MAX_PER_HOP} more")
        return "\n".join(lines)

    @mcp.tool()
    def dep_imports(filepath: str) -> str:
        """Show all project files directly imported by the given file."""
        s = _app.deps.stats()
        if s["files"] == 0:
            if _app._bg_status.get("deps", "").startswith("indexing "):
                return f"Dependency graph is still building: {_app._bg_status['deps']}"
            return "Dependency graph not built. Call dep_index with your project directory first."
        imports = _app.deps.direct_imports(filepath)
        base = s["base_dir"]
        rel = os.path.relpath(filepath, base) if base else filepath
        if not imports:
            return f"'{rel}' has no tracked project imports."
        lines = [f"'{rel}' directly imports:"]
        for f in imports:
            lines.append(f"  {os.path.relpath(f, base) if base else f}")
        return "\n".join(lines)

    @mcp.tool()
    async def dep_index(directory: str, append: bool = False) -> str:
        """Build import dependency graph (Python/JS/TS/Go/Rust/Java/C#/Ruby). Runs in background. Required before dep_impact/dep_imports. append=True for multi-root."""
        if not os.path.isdir(directory):
            return f"Directory not found: {directory}"

        def _do():
            try:
                _app._bg_status["deps"] = f"indexing {directory}..."
                s = _app.deps.index_directory(directory, append=append)
                _app._bg_status["deps"] = f"done: {s['files']} files, {s['edges']} edges"
            except Exception as e:
                _app._bg_status["deps"] = f"ERROR: {e}"
                print(f"[gpu-search] Dep index FAILED: {e}", file=sys.stderr, flush=True)

        threading.Thread(target=_do, daemon=True).start()
        return f"Dep graph indexing started for {directory} — call gpu_stats to check progress."

    # -----------------------------------------------------------------------
    # Signal scan tool
    # -----------------------------------------------------------------------

    @mcp.tool()
    def scan_repository_signals(
        categories: list[str] = None,
        top_k_per_signal: int = 5,
        context_mode: str = "compact",
    ) -> str:
        """Scan the repository for common legacy .NET, config, SQL, async-risk, exception-risk, DI, and test signals.
        Returns a categorized audit summary. Useful for audit and onboarding workflows such as LegacyLens.
        categories: optional list to filter (e.g. ['legacy-dotnet', 'sql']). Omit for all categories.
        top_k_per_signal: max matches per signal (capped at 20).
        """
        if _app.index.stats()["files"] == 0:
            return "No pattern index found. Call gpu_index first."

        top_k = min(top_k_per_signal, 20)
        signals_to_run = _app._BUILTIN_SIGNALS
        if categories:
            signals_to_run = [s for s in _app._BUILTIN_SIGNALS if s["category"] in categories]

        lines = ["Repository signal scan:"]
        current_cat = None
        total_signals = 0
        total_matches = 0

        for signal in signals_to_run:
            try:
                matches = _app._run_signal(signal, top_k, context_mode)
            except Exception as exc:
                lines.append(f"  [WARN] {signal['id']}: {exc}")
                continue
            if not matches:
                continue
            if signal["category"] != current_cat:
                current_cat = signal["category"]
                lines.append(f"\n[{current_cat}]")
            total_signals += 1
            total_matches += len(matches)
            n = len(matches)
            lines.append(f"  {signal['label']} ({n} match{'es' if n != 1 else ''}):")
            for m in matches[:3]:
                snippet = (m.get("snippet") or "")[:100]
                lines.append(f"    {m['file']} L{m['lineStart']}: {snippet}")
            if n > 3:
                lines.append(f"    ... {n - 3} more matches")

        if total_signals == 0:
            lines.append("\nNo signals detected.")
        else:
            lines.append(f"\nTotal: {total_signals} signals detected, {total_matches} matches.")
        lines.append("Note: Signal scan is heuristic. Absence of a signal does not prove absence in the repository.")
        return "\n".join(lines)

    @mcp.tool()
    def gpu_semantic_search(query: str, top_k: int = 5) -> str:
        """Semantic search by meaning (GPU cosine similarity). Use search_code for most queries; use this when you need explicit top_k control."""
        s = _app.semantic.stats()
        if s["chunks"] == 0:
            if s.get("model_error"):
                return s["model_error"]
            return "No semantic index found. Call gpu_semantic_index with your project directory first."
        try:
            results = _app.semantic.search(query, top_k=top_k)
        except Exception:
            return _app.semantic.semantic_unavailable_message()
        if not results:
            return f"No results for '{query}'"
        base = s["base_dir"]
        lines = [f"Semantic: {len(results)} matches for '{query}':"]
        for r in results:
            rel = os.path.relpath(r["file"], base) if base else r["file"]
            lines.append(f"\n[{r['score']}] {rel} L{r['start_line']}–{r['end_line']}")
            from redact import redact
            lines.append(redact(r["snippet"][:_app._SNIPPET_CHARS]))
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Prompts
    # -----------------------------------------------------------------------

    @mcp.prompt()
    def search_codebase(query: str) -> str:
        """Search the indexed codebase for any identifier, symbol, or concept."""
        return (
            f"Use search_code('{query}') to find relevant code. "
            "If the results are exact matches, read the surrounding context with the Read tool. "
            "If no matches are found, try a broader or natural-language rephrasing."
        )

    @mcp.prompt()
    def before_edit(filepath: str) -> str:
        """Understand the blast radius of a file before changing it."""
        return (
            f"Before editing '{filepath}', call dep_impact('{filepath}') to see every file "
            "that transitively imports it. Review the direct importers (hop 1) carefully — "
            "those are the files most likely to break. Then make your edit and check those files for regressions."
        )

    @mcp.prompt()
    def explore_feature(description: str) -> str:
        """Find where a feature or concept is implemented across the codebase."""
        return (
            f"To locate '{description}' in the codebase:\n"
            f"1. Call search_code('{description}') — this will route to semantic search if it reads as natural language.\n"
            "2. For the top results, use dep_imports(filepath) to understand what each file depends on.\n"
            "3. Use dep_impact(filepath) on key files to see what else references them.\n"
            "This gives you both the implementation site and its full call graph."
        )

    return {
        "search_code": search_code,
        "gpu_search": gpu_search,
        "gpu_index": gpu_index,
        "gpu_stats": gpu_stats,
        "gpu_update_file": gpu_update_file,
        "gpu_read_block": gpu_read_block,
        "gpu_skeleton": gpu_skeleton,
        "gpu_semantic_index": gpu_semantic_index,
        "gpu_add_directory": gpu_add_directory,
        "dep_impact": dep_impact,
        "dep_imports": dep_imports,
        "dep_index": dep_index,
        "scan_repository_signals": scan_repository_signals,
        "gpu_semantic_search": gpu_semantic_search,
    }
