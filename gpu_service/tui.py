"""gpu-search-tui: live dashboard for index size, resources and usage.

Two modes:

* attach  -- ``gpu-search-tui --url http://127.0.0.1:8765`` polls a server that
  is already running (``gpu-search-mcp --http``).
* embedded -- ``gpu-search-tui --directory PATH`` starts the same HTTP server on
  a background thread in this process, so one command shows a loaded index
  without a second terminal.

Data comes only from the public HTTP surface (``/stats`` and ``/runtime``), so
the dashboard cannot drift from what the API reports. ``--json`` and ``--once``
work without ``rich``; the live view needs the ``tui`` extra.
"""
from __future__ import annotations

import argparse
from collections import deque
import contextlib
import http.client
import json
import sys
import threading
import time
from urllib.parse import urlsplit

DEFAULT_URL = "http://127.0.0.1:8765"
MIN_INTERVAL = 0.2

UNICODE_GLYPHS = {"sep": " \u00b7 ", "arrow": "\u2192", "dash": "\u2014", "spark": "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"}
ASCII_GLYPHS = {"sep": " | ", "arrow": "->", "dash": "-", "spark": "._-=+*#@"}
_GLYPHS = dict(UNICODE_GLYPHS)
_GLYPHS_READY = False


def _ensure_glyphs(stream=None) -> None:
    """Fall back to ASCII decorations when the terminal cannot encode them.

    Windows consoles often default to cp1252/cp437, where the sparkline blocks
    and the arrow are unencodable; rich would otherwise crash mid-render.
    """
    global _GLYPHS_READY
    if _GLYPHS_READY:
        return
    _GLYPHS_READY = True
    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        for character in "\u00b7\u2192\u2581\u2014":
            character.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        _GLYPHS.update(ASCII_GLYPHS)



# ---------------------------------------------------------------------------
# Snapshot collection
# ---------------------------------------------------------------------------

def fetch_json(url: str, timeout: float = 5.0) -> dict:
    """GET a JSON document from the local API.

    Uses ``http.client`` instead of ``urllib.request`` on purpose: urllib asks the
    platform for proxy settings on every call, and a system proxy must never see
    loopback repository queries.
    """
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"unsupported URL: {url}")
    connection_class = (
        http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_class(parts.hostname, parts.port, timeout=timeout)
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    try:
        connection.request("GET", path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        body = response.read()
        if response.status >= 400:
            raise ValueError(f"HTTP {response.status} from {url}")
        return json.loads(body.decode("utf-8"))
    finally:
        connection.close()


def collect_snapshot(base_url: str, timeout: float = 5.0) -> dict:
    """Return one dashboard snapshot; connection failures never raise."""
    base = base_url.rstrip("/")
    snapshot: dict = {
        "url": base,
        "fetchedAt": time.time(),
        "error": None,
        "warning": None,
        "stats": {},
        "runtime": {},
    }
    try:
        snapshot["stats"] = fetch_json(base + "/stats", timeout)
    except (OSError, ValueError, http.client.HTTPException) as error:
        snapshot["error"] = _describe_error(error, base)
        return snapshot
    try:
        snapshot["runtime"] = fetch_json(base + "/runtime", timeout)
    except (OSError, ValueError, http.client.HTTPException) as error:
        snapshot["warning"] = f"runtime unavailable: {type(error).__name__}"
    return snapshot


def _describe_error(error: BaseException, base: str) -> str:
    reason = getattr(error, "reason", None)
    detail = f"{type(error).__name__}: {reason or error}"
    return f"{detail} ({base})"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_bytes(value) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            precision = 0 if unit == "B" else 1
            return f"{size:.{precision}f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def format_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return _GLYPHS["dash"]


def format_ms(seconds) -> str:
    try:
        return f"{float(seconds) * 1000:.1f} ms"
    except (TypeError, ValueError):
        return _GLYPHS["dash"]


def format_percent(fraction) -> str:
    try:
        return f"{float(fraction):.1f}%"
    except (TypeError, ValueError):
        return _GLYPHS["dash"]


def _megabytes(value) -> float:
    try:
        return max(0.0, float(value)) * 1024 * 1024
    except (TypeError, ValueError):
        return 0.0


def format_duration(seconds) -> str:
    try:
        total = int(max(0, float(seconds)))
    except (TypeError, ValueError):
        return _GLYPHS["dash"]
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sparkline(values) -> str:
    """Render a compact latency history; an empty history renders as blanks."""
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return _GLYPHS["spark"][0] * 8
    lowest, highest = min(numbers), max(numbers)
    span = highest - lowest
    if span <= 0:
        return _GLYPHS["spark"][-1] * len(numbers)
    return "".join(
        _GLYPHS["spark"][
            min(len(_GLYPHS["spark"]) - 1, int((value - lowest) / span * (len(_GLYPHS["spark"]) - 1)))
        ]
        for value in numbers
    )


def _dig(source: dict, *path, default=None):
    """Read a nested key without raising; kept for ad-hoc snapshot probes."""

    current = source
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _require_rich():
    try:
        from rich.console import Console  # noqa: F401
        from rich.panel import Panel  # noqa: F401
        from rich.table import Table  # noqa: F401
        from rich.text import Text  # noqa: F401
    except ImportError as error:  # pragma: no cover - exercised by packaging
        raise SystemExit(
            "gpu-search-tui needs the optional 'tui' extra:\n"
            '  pip install "gpu-search-mcp[tui]"'
        ) from error


def _panel(title: str, rows: list[tuple[str, str]], border: str = "cyan"):
    """Build a two-column label/value panel."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    table = Table.grid(padding=(0, 2, 0, 2), expand=True)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(style="bold", no_wrap=False)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(style="bold", no_wrap=False)
    row: list = []
    for label, value in rows:
        row.extend([Text(label), Text(str(value))])
        if len(row) >= 4:
            table.add_row(*row)
            row = []
    if row:
        row.extend([Text("")] * (4 - len(row)))
        table.add_row(*row)
    return Panel(table, title=title, title_align="left", border_style=border)


def render_dashboard(
    snapshot: dict,
    log_lines: list[str] | None = None,
    footer: bool = False,
    height: int | None = None,
):
    """Build the rich renderable for one snapshot (pure; used by tests).

    ``height`` is the terminal height, when known. A short terminal gets a
    compact summary instead of the full stack of panels, because the view is
    cropped rather than scrolled.
    """
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    _ensure_glyphs()

    stats = snapshot.get("stats") or {}
    runtime = snapshot.get("runtime") or {}
    device = stats.get("device") or {}
    usage = stats.get("usage") or {}

    header_bits = [
        f"v{stats.get('version') or '?'}",
        str(device.get("backend") or "unknown"),
        snapshot.get("url", ""),
        f"up {format_duration(usage.get('uptimeSeconds'))}",
        time.strftime("%H:%M:%S", time.localtime(snapshot.get("fetchedAt") or time.time())),
    ]
    header = Panel(
        Text(_GLYPHS["sep"].join(bit for bit in header_bits if bit), style="bold white on grey15"),
        title="gpu-search",
        title_align="left",
        border_style="grey50",
    )

    if snapshot.get("error"):
        return Group(
            header,
            Panel(
                Text(str(snapshot["error"]), style="bold red")
                + Text(
                    "\nStart a server with: gpu-search-mcp --http --directory PATH"
                    "\nor run: gpu-search-tui --directory PATH",
                    style="dim",
                ),
                title="No server",
                title_align="left",
                border_style="red",
            ),
        )

    footer_text = Text(
        f"Ctrl+C quits{_GLYPHS['sep']}--once prints one snapshot{_GLYPHS['sep']}--json for scripts",
        style="dim",
    )

    if height is not None and height < 34:
        compact: list = [header, _compact_panel(stats, runtime, usage)]
        if footer:
            compact.append(footer_text)
        return Group(*compact)

    panels: list = [header, _overview_panel(stats, usage), _index_panel(stats)]
    last_query = _last_query_panel(stats)
    if last_query is not None:
        panels.append(last_query)
    panels.append(_resources_panel(stats, runtime))
    panels.append(_usage_panel(usage))
    if snapshot.get("warning"):
        panels.append(
            Panel(Text(str(snapshot["warning"]), style="yellow"),
                  title="Warning", title_align="left", border_style="yellow")
        )
    if log_lines:
        panels.append(
            Panel(
                Text("\n".join(log_lines[-6:]), style="dim"),
                title="Server log",
                title_align="left",
                border_style="grey50",
            )
        )
    if footer:
        panels.append(footer_text)
    return Group(*panels)


def _compact_panel(stats: dict, runtime: dict, usage: dict):
    """One panel with every headline number, for terminals too short for the full view."""
    pattern = stats.get("pattern") or {}
    semantic = stats.get("semantic") or {}
    dependency = stats.get("dependency") or {}
    symbols = stats.get("symbols") or {}
    caps = stats.get("capabilities") or {}
    candidate_index = pattern.get("candidate_index") or {}
    metrics = pattern.get("last_query") or {}
    latency = usage.get("latencyMs") or {}
    watcher = runtime.get("watcher") or {}
    buffers = runtime.get("pattern_buffers") or {}
    selector = str(pattern.get("candidate_selector") or _GLYPHS["dash"]).replace(
        "CandidateSelector", ""
    )
    rows = [
        (
            "index",
            f"{format_int(pattern.get('files'))} files{_GLYPHS['sep']}"
            f"{format_int(pattern.get('chunks'))} chunks{_GLYPHS['sep']}"
            f"{format_bytes(pattern.get('corpus_bytes'))}",
        ),
        ("cache", f"{pattern.get('cache') or _GLYPHS['dash']}{_GLYPHS['sep']}{selector}"),
        (
            "trigrams",
            f"{format_int(candidate_index.get('indexed_keys'))} keys{_GLYPHS['sep']}"
            f"{format_bytes(candidate_index.get('index_bytes'))}",
        ),
        (
            "graph",
            f"{format_int(dependency.get('files'))} files / "
            f"{format_int(dependency.get('edges'))} edges{_GLYPHS['sep']}"
            f"{format_int(symbols.get('symbols'))} symbols",
        ),
        (
            "semantic",
            "off" if not caps.get("semanticSearch") else f"{format_int(semantic.get('chunks'))} chunks",
        ),
        (
            "last query",
            f"{format_ms(metrics.get('total_query_seconds'))}{_GLYPHS['sep']}"
            f"{format_int(metrics.get('candidate_chunks'))}/"
            f"{format_int(metrics.get('number_of_chunks'))} candidates{_GLYPHS['sep']}"
            f"{format_bytes(metrics.get('bytes_read_from_storage'))} read",
        ),
        (
            "usage",
            f"{format_int(usage.get('requests'))} requests{_GLYPHS['sep']}"
            f"{format_int(usage.get('searches'))} searches{_GLYPHS['sep']}"
            f"p50 {latency.get('p50', 0):.1f} ms",
        ),
        (
            "watcher",
            f"{format_int(watcher.get('pending'))} pending{_GLYPHS['sep']}"
            f"{format_int(watcher.get('coalesced'))} coalesced{_GLYPHS['sep']}"
            f"{format_int(buffers.get('available_buffers'))} free buffers",
        ),
    ]
    return _panel("Summary", rows, border="green")


def _overview_panel(stats: dict, usage: dict):
    pattern = stats.get("pattern") or {}
    dependency = stats.get("dependency") or {}
    symbols = stats.get("symbols") or {}
    caps = stats.get("capabilities") or {}
    ready = sum(1 for value in caps.values() if value is True)
    return _panel(
        "Overview",
        [
            ("files", format_int(pattern.get("files"))),
            ("chunks", format_int(pattern.get("chunks"))),
            ("corpus", format_bytes(pattern.get("corpus_bytes"))),
            ("requests", format_int(usage.get("requests"))),
            ("semantic", "ready" if caps.get("semanticSearch") else "off"),
            ("deps", format_int(dependency.get("files"))),
            ("symbols", format_int(symbols.get("symbols"))),
            ("features", f"{ready}/{len(caps)}"),
        ],
        border="green",
    )


def _index_panel(stats: dict):
    pattern = stats.get("pattern") or {}
    semantic = stats.get("semantic") or {}
    dependency = stats.get("dependency") or {}
    symbols = stats.get("symbols") or {}
    selector = str(pattern.get("candidate_selector") or _GLYPHS["dash"])
    selector = selector.replace("CandidateSelector", "")
    candidate_index = pattern.get("candidate_index") or {}
    rows = [
        ("pattern", f"{format_int(pattern.get('files'))} files{_GLYPHS['sep']}{format_int(pattern.get('chunks'))} chunks"),
        ("cache", str(pattern.get("cache") or _GLYPHS["dash"])),
        ("selector", selector),
        ("storage", str(pattern.get("storage_backend") or _GLYPHS["dash"])),
        ("trigrams", f"{format_int(candidate_index.get('indexed_keys'))} keys{_GLYPHS['sep']}{format_bytes(candidate_index.get('index_bytes'))}"),
        ("semantic", f"{format_int(semantic.get('chunks'))} chunks{_GLYPHS['sep']}{format_bytes(_megabytes(semantic.get('vram_mb')))}"),
        ("deps", f"{format_int(dependency.get('files'))} files{_GLYPHS['sep']}{format_int(dependency.get('edges'))} edges"),
        ("symbols", f"{format_int(symbols.get('symbols'))} symbols{_GLYPHS['sep']}{format_int(symbols.get('edges'))} edges"),
    ]
    base = pattern.get("base_dir")
    if base:
        rows.append(("root", str(base)))
    return _panel("Indexes", rows)


def _last_query_panel(stats: dict):
    metrics = (stats.get("pattern") or {}).get("last_query") or {}
    if not metrics or not metrics.get("number_of_chunks"):
        return None
    candidates = (
        f"{format_int(metrics.get('candidate_chunks'))}/{format_int(metrics.get('number_of_chunks'))}"
        f" ({format_percent(metrics.get('candidate_percentage'))})"
    )
    pipeline = "pipeline" if metrics.get("pipeline_enabled") else "synchronous"
    return _panel(
        "Last exact query",
        [
            ("latency", format_ms(metrics.get("total_query_seconds"))),
            ("candidates", candidates),
            ("bytes read", format_bytes(metrics.get("bytes_read_from_storage"))),
            ("corpus read", format_percent(metrics.get("corpus_percentage_physically_read"))),
            ("storage", format_ms(metrics.get("storage_read_seconds"))),
            ("kernel", format_ms(metrics.get("gpu_search_seconds"))),
            (f"host{_GLYPHS['arrow']}gpu", format_ms(metrics.get("host_to_gpu_seconds"))),
            ("mode", f"{pipeline}{_GLYPHS['sep']}{format_int(metrics.get('prefetched_chunks'))} prefetched"),
        ],
        border="blue",
    )


def _resources_panel(stats: dict, runtime: dict):
    process = runtime.get("process") or {}
    buffers = runtime.get("pattern_buffers") or {}
    watcher = runtime.get("watcher") or {}
    pattern = stats.get("pattern") or {}
    cuda_allocated = process.get("cuda_allocated_bytes")
    rows = [
        ("rss", format_bytes(process.get("resident_bytes"))),
        ("peak rss", format_bytes(process.get("peak_resident_bytes"))),
        ("vram", format_bytes(cuda_allocated)),
        ("host staging", format_bytes(buffers.get("host_staging_bytes"))),
        ("device bytes", format_bytes(buffers.get("device_bytes"))),
        ("pool", f"{format_int(buffers.get('available_buffers'))} free{_GLYPHS['sep']}{format_int(buffers.get('leased_buffers'))} leased"),
        ("buffer", f"{format_bytes(pattern.get('chunk_size'))} x {format_int(pattern.get('buffer_count'))}"),
        ("budget", format_bytes(buffers.get("max_owned_bytes"))),
        ("watcher", f"{format_int(watcher.get('pending'))} pending{_GLYPHS['sep']}{format_int(watcher.get('coalesced'))} coalesced"),
        ("watcher calls", f"{format_int(watcher.get('completed'))} ok{_GLYPHS['sep']}{format_int(watcher.get('failed'))} failed"),
        ("threads", format_int(process.get("python_threads"))),
        ("pid", format_int(process.get("pid"))),
    ]
    return _panel("Resources", rows, border="magenta")


def _usage_panel(usage: dict):
    latency = usage.get("latencyMs") or {}
    by_endpoint = usage.get("byEndpoint") or {}
    by_engine = usage.get("bySearchEngine") or {}
    top_endpoints = sorted(by_endpoint.items(), key=lambda item: -item[1])[:4]
    engines = ", ".join(f"{name} {count}" for name, count in sorted(by_engine.items())) or _GLYPHS["dash"]
    last = usage.get("lastRequest") or {}
    rows = [
        ("requests", format_int(usage.get("requests"))),
        ("errors", format_int(usage.get("errors"))),
        ("searches", format_int(usage.get("searches"))),
        ("p50 / p95", f"{latency.get('p50', 0):.1f} / {latency.get('p95', 0):.1f} ms"),
        ("max", f"{latency.get('max', 0):.1f} ms"),
        ("engines", engines),
        ("endpoints", ", ".join(f"{path} {count}" for path, count in top_endpoints) or _GLYPHS["dash"]),
        ("last", f"{last.get('endpoint', '—')} {last.get('status', '')} {last.get('milliseconds', 0):.1f} ms"),
        ("latency", sparkline(usage.get("samplesMs") or [])),
        ("uptime", format_duration(usage.get("uptimeSeconds"))),
    ]
    return _panel("Usage (HTTP API)", rows, border="yellow")


# ---------------------------------------------------------------------------
# Embedded server (same code path as `gpu-search-mcp --http`)
# ---------------------------------------------------------------------------

class LogBuffer:
    """Tiny stderr sink so server logs never garble the live view."""

    # The dashboard itself polls these every tick; keeping their access log
    # lines would push the actually useful startup messages out of the panel.
    _POLL_MARKERS = ('"GET /stats', '"GET /runtime', '"GET /health')

    def __init__(self, max_lines: int = 400):
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._lock = threading.Lock()

    def write(self, data) -> int:
        text = str(data)
        with self._lock:
            for line in text.splitlines():
                if line.strip() and not any(marker in line for marker in self._POLL_MARKERS):
                    self._lines.append(line[-400:])
        return len(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def lines(self, limit: int = 6) -> list[str]:
        with self._lock:
            return list(self._lines)[-limit:]


@contextlib.contextmanager
def embedded_server(args):
    """Start an in-process server and yield ``(url, log_buffer)``."""
    try:
        from .mcp_server import _build_http_server, _parse_args, _prepare_startup
    except ImportError:  # pragma: no cover - only when running as a script
        from mcp_server import _build_http_server, _parse_args, _prepare_startup

    argv: list[str] = []
    for directory in args.directories:
        argv += ["--directory", directory]
    argv += ["--http", "--host", args.host, "--port", str(args.port)]
    serve_args = _parse_args(argv)

    buffer = LogBuffer()
    previous_stderr = sys.stderr
    sys.stderr = buffer  # type: ignore[assignment]
    httpd = None
    try:
        cli_targets, all_targets = _prepare_startup(serve_args)
        httpd = _build_http_server(serve_args, cli_targets, all_targets)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, name="gpu-search-tui-http", daemon=True)
        thread.start()
        yield f"http://{args.host}:{port}", buffer
    finally:
        sys.stderr = previous_stderr
        if httpd is not None:
            with contextlib.suppress(Exception):
                httpd.shutdown()
                httpd.server_close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-search-tui",
        description="Live dashboard for gpu-search indexes, resources and usage.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="server to attach to (default: %(default)s)")
    parser.add_argument(
        "--directory", action="append", default=[], dest="directories",
        help="index this directory in-process instead of attaching to a running server (repeatable)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind host for --directory mode")
    parser.add_argument("--port", type=int, default=8765, help="bind port for --directory mode (0 = ephemeral)")
    parser.add_argument("--interval", type=float, default=1.0, help="refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--json", action="store_true", dest="as_json", help="print the raw snapshot as JSON")
    return parser


def _wait_ready(url: str, timeout: float = 60.0) -> None:
    """Poll until the pattern index reports files.

    Only used by the one-shot modes: a live view can show indexing progress, but
    ``--once``/``--json`` on an embedded server would otherwise print an empty
    index while the background startup threads are still working.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = collect_snapshot(url)
        stats = snapshot.get("stats") or {}
        status = str((stats.get("status") or {}).get("pattern", ""))
        if status.startswith(("done", "ERROR")):
            return
        if not status and (stats.get("pattern") or {}).get("files"):
            return
        time.sleep(0.25)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.directories and args.url != DEFAULT_URL:
        parser.error("--url cannot be combined with --directory")
    if args.interval < MIN_INTERVAL:
        parser.error(f"--interval must be >= {MIN_INTERVAL}")
    if args.directories:
        with embedded_server(args) as (url, buffer):
            if args.once or args.as_json:
                _wait_ready(url)
            return _run(url, args, buffer)
    return _run(args.url, args, None)


def _run(url: str, args, log_buffer: LogBuffer | None) -> int:
    if args.as_json:
        snapshot = collect_snapshot(url)
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        return 0 if snapshot.get("error") is None else 1

    _require_rich()
    if args.once:
        from rich.console import Console

        snapshot = collect_snapshot(url)
        Console().print(render_dashboard(snapshot, _log_lines(log_buffer)))
        return 0 if snapshot.get("error") is None else 1
    return _run_live(url, args.interval, log_buffer)


def live_options() -> dict:
    """Live-render options that keep the dashboard fixed instead of scrolling.

    ``screen`` switches to the alternate buffer (htop-style), and
    ``vertical_overflow="crop"`` stops a short terminal from scrolling content
    off the top when the dashboard is taller than the viewport.
    """
    return {"refresh_per_second": 4, "screen": True, "vertical_overflow": "crop"}


def _run_live(url: str, interval: float, log_buffer: LogBuffer | None) -> int:
    """Full-screen dashboard.

    ``screen=True`` switches to the terminal's alternate buffer, so frames update
    in place like htop instead of appending to the scrollback, and
    ``vertical_overflow="crop"`` keeps a short terminal from scrolling the view.
    A non-terminal target (pipe, redirect) falls back to line-oriented frames.
    """
    from rich.console import Console
    from rich.live import Live

    console = Console()
    snapshot = collect_snapshot(url)
    if not console.is_terminal:
        try:
            while True:
                console.print(render_dashboard(snapshot, _log_lines(log_buffer)))
                console.print()
                time.sleep(interval)
                snapshot = collect_snapshot(url)
        except KeyboardInterrupt:
            return 0
    try:
        with Live(
            render_dashboard(
                snapshot, _log_lines(log_buffer), footer=True, height=console.height
            ),
            console=console,
            **live_options(),
        ) as live:
            while True:
                time.sleep(interval)
                snapshot = collect_snapshot(url)
                live.update(
                    render_dashboard(
                        snapshot,
                        _log_lines(log_buffer),
                        footer=True,
                        height=console.height,
                    )
                )
    except KeyboardInterrupt:
        console.print("[dim]gpu-search-tui stopped[/dim]")
        return 0
    return 0


def _log_lines(buffer: LogBuffer | None) -> list[str]:
    return buffer.lines(6) if buffer is not None else []


if __name__ == "__main__":
    raise SystemExit(main())
