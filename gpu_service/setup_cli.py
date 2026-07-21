"""Safe, idempotent local setup for supported MCP clients."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .semantic_model_manager import get_semantic_model_status

SUPPORTED_CLIENTS = ("claude", "codex")


class SetupError(RuntimeError):
    """Raised when setup cannot safely preserve an existing configuration."""


@dataclass(frozen=True)
class PendingWrite:
    path: Path
    content: str
    label: str


def detect_clients(home: Path | None = None) -> list[str]:
    """Detect supported clients without modifying their configuration."""
    root = home or Path.home()
    detected: list[str] = []
    if (root / ".claude.json").exists() or (root / ".claude").exists() or shutil.which("claude"):
        detected.append("claude")
    if (root / ".codex").exists() or shutil.which("codex"):
        detected.append("codex")
    return detected


def _load_json_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupError(f"Refusing to overwrite invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SetupError(f"Refusing to overwrite non-object JSON at {path}")
    return value


def _json_text(value: dict) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def _toml_string(value: str) -> str:
    # JSON strings use the TOML-compatible escaping needed for paths emitted here.
    return json.dumps(value, ensure_ascii=False)


def _codex_block(command: str, args: list[str]) -> list[str]:
    rendered_args = ", ".join(_toml_string(value) for value in args)
    return [
        '[mcp_servers."gpu-search"]',
        f"command = {_toml_string(command)}",
        f"args = [{rendered_args}]",
    ]


def _update_codex_toml(existing: str, command: str, args: list[str]) -> str:
    """Replace only the gpu-search table while preserving unrelated TOML text."""
    lines = existing.splitlines()
    headers = {'[mcp_servers."gpu-search"]', "[mcp_servers.gpu-search]"}
    start = next((index for index, line in enumerate(lines) if line.strip() in headers), None)
    block = _codex_block(command, args)

    if start is None:
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            lines.append("")
        lines.extend(block)
    else:
        end = start + 1
        while end < len(lines) and not lines[end].lstrip().startswith("["):
            end += 1
        lines[start:end] = block
    return "\n".join(lines) + "\n"


def _server_entry(executable: str, directories: list[str], index_enabled: bool) -> tuple[str, list[str]]:
    args = ["-m", "gpu_service.mcp_server"]
    if index_enabled:
        for directory in directories:
            args.extend(("--directory", directory))
    return executable, args


def build_setup_plan(
    clients: list[str],
    directories: list[str],
    *,
    home: Path | None = None,
    executable: str | None = None,
    index_enabled: bool = True,
) -> list[PendingWrite]:
    """Build a complete write plan, validating all existing files up front."""
    root = home or Path.home()
    python = executable or sys.executable
    command, server_args = _server_entry(python, directories, index_enabled)
    writes: list[PendingWrite] = []

    if "claude" in clients:
        path = root / ".claude.json"
        config = _load_json_object(path)
        servers = config.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            raise SetupError(f"Refusing to replace non-object mcpServers in {path}")
        servers["gpu-search"] = {"command": command, "args": server_args}
        content = _json_text(config)
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            writes.append(PendingWrite(path, content, "Claude MCP configuration"))

    if "codex" in clients:
        path = root / ".codex" / "config.toml"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        content = _update_codex_toml(existing, command, server_args)
        if existing != content:
            writes.append(PendingWrite(path, content, "Codex MCP configuration"))

    if index_enabled:
        path = root / ".gpu-search-config.json"
        config = _load_json_object(path)
        existing_dirs = config.get("directories", [])
        if not isinstance(existing_dirs, list) or not all(isinstance(value, str) for value in existing_dirs):
            raise SetupError(f"Refusing to replace invalid directories in {path}")
        config["directories"] = list(dict.fromkeys(existing_dirs + directories))
        content = _json_text(config)
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            writes.append(PendingWrite(path, content, "startup index configuration"))

    return writes


def _next_backup_path(path: Path) -> Path:
    candidate = Path(f"{path}.bak")
    counter = 1
    while candidate.exists():
        candidate = Path(f"{path}.bak.{counter}")
        counter += 1
    return candidate


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def apply_setup_plan(writes: list[PendingWrite]) -> list[Path]:
    """Back up existing files and atomically apply the prepared plan."""
    backups: list[Path] = []
    for write in writes:
        if write.path.exists():
            backup = _next_backup_path(write.path)
            shutil.copy2(write.path, backup)
            backups.append(backup)
        _atomic_write(write.path, write.content)
    return backups


def _resolve_directories(values: list[str] | None) -> list[str]:
    resolved: list[str] = []
    for raw in values or [os.getcwd()]:
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            raise SetupError(f"Index directory does not exist or is not a directory: {path}")
        value = str(path)
        if value not in resolved:
            resolved.append(value)
    return resolved


def run_setup(
    args,
    *,
    home: Path | None = None,
    executable: str | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    """Run setup. Model inspection is local-only and no download is attempted."""
    root = home or Path.home()
    clients = list(dict.fromkeys(args.clients or detect_clients(root)))
    if not clients:
        print("No supported client detected. Pass --client claude or --client codex.", file=sys.stderr)
        return 2

    try:
        directories = [] if args.no_index else _resolve_directories(args.directories)
        writes = build_setup_plan(
            clients,
            directories,
            home=root,
            executable=executable,
            index_enabled=not args.no_index,
        )
    except SetupError as exc:
        print(f"Setup aborted: {exc}", file=sys.stderr)
        return 2

    print("gpu-search-mcp setup")
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  Python   : {executable or sys.executable} ({platform.python_version()})")
    print(f"  Clients  : {', '.join(clients)}")
    print(f"  Index    : {'skipped' if args.no_index else ', '.join(directories)}")
    if args.no_model:
        print("  Model    : check skipped; no model was downloaded")
    else:
        status = get_semantic_model_status()
        state = "cached" if status.get("cached") else "not cached"
        print(f"  Model    : {status.get('modelId', 'unknown')} ({state}; no download attempted)")

    if not writes:
        print("Already configured; no changes needed.")
        return 0

    print("  Changes:")
    for write in writes:
        print(f"    - {write.label}: {write.path}")

    if args.dry_run:
        print("Dry run complete; no files changed.")
        return 0

    if not args.yes:
        answer = input_fn("Apply these changes? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Setup cancelled; no files changed.")
            return 1

    try:
        backups = apply_setup_plan(writes)
    except OSError as exc:
        print(f"Setup failed while writing configuration: {exc}", file=sys.stderr)
        return 2

    for backup in backups:
        print(f"  Backup   : {backup}")
    print("Setup complete. Restart the selected client to activate gpu-search.")
    return 0
