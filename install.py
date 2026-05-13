#!/usr/bin/env python3
"""
gpu-search-mcp installer
Installs dependencies and wires the MCP server into Claude Code and Codex.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent.resolve()
SERVER_SCRIPT = REPO_DIR / "gpu_service" / "mcp_server.py"
VENV_DIR = REPO_DIR / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
UV_BIN = shutil.which("uv")

CLAUDE_MD_CONTENT = """\
# Search

Prefer `gpu-search` MCP tools for codebase search when available. Fall back to grep/rg if the MCP server is not running.

- **`search_code(query)`** — Exact identifier → pattern search (sub-ms). Natural language → semantic search.
- **`dep_impact(filepath)`** — Understand what imports a file before editing it.
- **`gpu_add_directory(dir)`** — Add a new project (persists across restarts).
- **`gpu_semantic_index(dir)`** — Build semantic cache once per project.
"""

CODEX_INSTRUCTIONS = (
    "Prefer gpu-search MCP for codebase search when available; use grep/rg as fallback. "
    "search_code(query): exact identifier→pattern search, natural language→semantic search. "
    "dep_impact(filepath): call before editing. gpu_add_directory(dir): add new project."
)


def run(cmd: list[str]):
    subprocess.check_call(cmd)


def choose_installer(mode: str) -> str:
    if mode == "auto":
        return "uv" if UV_BIN else "pip"
    if mode == "uv" and not UV_BIN:
        raise RuntimeError("uv was requested but is not installed or not on PATH.")
    return mode


def server_python() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def venv_has_pip() -> bool:
    if not VENV_PYTHON.exists():
        return False
    result = subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "--version"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def ensure_venv(installer: str):
    if VENV_PYTHON.exists() and venv_has_pip():
        return
    if VENV_DIR.exists():
        print("[1/4] Recreating broken local virtualenv...")
        shutil.rmtree(VENV_DIR)
    else:
        print(f"[1/4] Creating local virtualenv with {installer}...")
    if installer == "uv":
        run([UV_BIN, "venv", "--python", sys.executable, str(VENV_DIR)])
    else:
        run([sys.executable, "-m", "venv", str(VENV_DIR)])


def install_deps(installer: str):
    ensure_venv(installer)
    system = platform.system()
    if installer == "uv":
        pip = [UV_BIN, "pip", "install", "--python", server_python()]
    else:
        pip = [server_python(), "-m", "pip", "install"]
    py_version = sys.version_info[:2]

    if system == "Darwin":
        print(f"[2/4] Installing PyTorch (MPS — Apple Silicon/Intel) via {installer}...")
        run(pip + ["torch"])
    else:
        has_cuda = False
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            has_cuda = result.returncode == 0
        except FileNotFoundError:
            pass

        if has_cuda:
            if py_version >= (3, 13):
                print(f"[2/4] Installing PyTorch (CUDA-capable build from default index for Python 3.13+) via {installer}...")
                run(pip + ["torch"])
            else:
                print(f"[2/4] Installing PyTorch (CUDA 12.1) via {installer}...")
                run(pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cu121"])
        else:
            print(f"[2/4] No NVIDIA GPU found — installing PyTorch (CPU) via {installer}...")
            run(pip + ["torch"])

    print(f"[3/4] Installing server dependencies via {installer}...")
    run(pip + ["-r", str(REPO_DIR / "requirements.txt")])


def save_startup_config(project_dirs: list[str]):
    """Write directories to ~/.gpu-search-config.json for auto-indexing on server startup."""
    config_path = Path.home() / ".gpu-search-config.json"
    existing: list[str] = []
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8")).get("directories", [])
        except Exception:
            pass
    merged = list(dict.fromkeys(existing + project_dirs))
    config_path.write_text(json.dumps({"directories": merged}, indent=2), encoding="utf-8")
    print(f"  Startup config → {config_path}")


def patch_claude_code(project_dirs: list[str]):
    """Wire MCP server into Claude Code and write global CLAUDE.md instructions."""
    # MCP server registration (~/.claude.json)
    config_path = Path.home() / ".claude.json"
    config: dict = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    args = [str(SERVER_SCRIPT)]
    for d in project_dirs:
        args += ["--directory", d]

    config.setdefault("mcpServers", {})
    config["mcpServers"]["gpu-search"] = {
        "command": server_python(),
        "args": args,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"  MCP server → {config_path}")

    # Global CLAUDE.md so Claude uses search_code instead of grep
    claude_dir = Path.home() / ".claude"
    claude_dir.mkdir(exist_ok=True)
    claude_md = claude_dir / "CLAUDE.md"
    claude_md.write_text(CLAUDE_MD_CONTENT, encoding="utf-8")
    print(f"  Instructions → {claude_md}")


def patch_codex(project_dirs: list[str]):
    """Wire MCP server into Codex (config.toml preferred, config.yaml fallback)."""
    codex_dir = Path.home() / ".codex"
    entry = {
        "command": server_python(),
        "args": [str(SERVER_SCRIPT)] + [a for d in project_dirs for a in ("--directory", d)],
    }

    # Try config.toml first (newer Codex)
    toml_config = codex_dir / "config.toml"
    if toml_config.exists():
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                tomllib = None

        try:
            import tomli_w  # type: ignore
            has_toml_write = True
        except ImportError:
            has_toml_write = False

        if tomllib is not None and has_toml_write:
            try:
                cfg = tomllib.loads(toml_config.read_text(encoding="utf-8"))
                cfg.setdefault("mcpServers", {})["gpu-search"] = entry
                cfg["instructions"] = CODEX_INSTRUCTIONS
                toml_config.write_text(tomli_w.dumps(cfg), encoding="utf-8")
                print(f"  Codex config → {toml_config}")
                return
            except Exception as e:
                print(f"  TOML patch failed ({e}); falling back...")
        else:
            # Patch instructions manually via text since we lack toml libs
            text = toml_config.read_text(encoding="utf-8")
            import re
            instr_line = f'instructions = """{CODEX_INSTRUCTIONS}"""\n'
            if "instructions" in text:
                text = re.sub(r'instructions\s*=\s*""".*?"""', instr_line.strip(), text, flags=re.DOTALL)
            else:
                text = instr_line + text
            toml_config.write_text(text, encoding="utf-8")
            print(f"  Codex instructions → {toml_config}")

    # Try config.yaml
    yaml_config = codex_dir / "config.yaml"
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None

    if yaml is not None and yaml_config.exists():
        try:
            cfg = yaml.safe_load(yaml_config.read_text(encoding="utf-8")) or {}
            cfg.setdefault("mcpServers", {})["gpu-search"] = entry
            cfg["instructions"] = CODEX_INSTRUCTIONS
            yaml_config.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
            print(f"  Codex config → {yaml_config}")
            return
        except Exception as e:
            print(f"  YAML patch failed ({e}); trying CLI...")

    # Try Codex CLI
    codex = shutil.which("codex")
    if codex:
        cmd = [codex, "mcp", "add", "gpu-search", "--", server_python(), str(SERVER_SCRIPT)]
        for d in project_dirs:
            cmd += ["--directory", d]
        subprocess.run([codex, "mcp", "remove", "gpu-search"], capture_output=True, text=True)
        run(cmd)
        print("  Registered via Codex CLI → gpu-search")
        return

    # Last resort: create config.yaml
    codex_dir.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        cfg = {"mcpServers": {"gpu-search": entry}, "instructions": CODEX_INSTRUCTIONS}
        yaml_config.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
        print(f"  Created Codex config → {yaml_config}")
    else:
        cfg_json = json.dumps({"mcpServers": {"gpu-search": entry}}, indent=2)
        yaml_config.write_text(cfg_json, encoding="utf-8")
        print(f"  Created Codex config (JSON) → {yaml_config}")


def prompt_dirs() -> list[str]:
    print("\nEnter project directories to index.")
    print("Press Enter with no input when done (blank = current directory).\n")
    dirs: list[str] = []
    while True:
        try:
            raw = input(f"  Directory {len(dirs) + 1} (Enter to finish): ").strip()
        except EOFError:
            if not dirs:
                default_dir = os.getcwd()
                print(f"  No stdin available; defaulting to current directory: {default_dir}")
                dirs.append(default_dir)
            break
        if not raw:
            if not dirs:
                dirs.append(os.getcwd())
            break
        resolved = str(Path(raw).expanduser().resolve())
        if not Path(resolved).is_dir():
            print(f"  Not found: {resolved}")
            continue
        dirs.append(resolved)
        print(f"  Added: {resolved}")
    return dirs


def check_python():
    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ required (you have {sys.version.split()[0]})")
        if platform.system() == "Darwin":
            print("\nInstall a newer Python via Homebrew:")
            print("  brew install python@3.12")
            print("  /opt/homebrew/bin/python3.12 install.py")
        sys.exit(1)


def backup_file(path: Path) -> None:
    """Create a .bak copy of a config file before modifying it."""
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        import shutil as _shutil
        _shutil.copy2(path, bak)
        print(f"  Backup: {bak}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install gpu-search-mcp and register it with Claude Code/Codex.")
    parser.add_argument(
        "--installer",
        choices=("auto", "uv", "pip"),
        default="auto",
        help="Package installer backend to use. Defaults to uv when available.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be changed without making any modifications.",
    )
    parser.add_argument(
        "--no-claude", action="store_true",
        help="Skip registering the MCP server with Claude Code.",
    )
    parser.add_argument(
        "--no-codex", action="store_true",
        help="Skip registering the MCP server with Codex.",
    )
    parser.add_argument(
        "--backup-configs", action="store_true",
        help="Create .bak copies of config files before modifying them (default: enabled).",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip interactive directory prompt; use the current directory.",
    )
    return parser.parse_args()


def _dry_run_report(dirs: list[str]) -> None:
    print("\n[DRY RUN] The following files would be modified:")
    print(f"  ~/.gpu-search-config.json  (add directories: {dirs})")
    config_path = Path.home() / ".claude.json"
    print(f"  {config_path}  (add mcpServers.gpu-search entry)")
    claude_md = Path.home() / ".claude" / "CLAUDE.md"
    print(f"  {claude_md}  (write gpu-search instructions)")
    codex_dir = Path.home() / ".codex"
    toml = codex_dir / "config.toml"
    yaml = codex_dir / "config.yaml"
    if toml.exists():
        print(f"  {toml}  (add mcpServers.gpu-search and instructions)")
    elif yaml.exists():
        print(f"  {yaml}  (add mcpServers.gpu-search and instructions)")
    else:
        print(f"  {yaml}  (create with mcpServers.gpu-search entry)")
    print("\nNo files were changed. Remove --dry-run to apply.")


def main():
    args = parse_args()
    installer = choose_installer(args.installer)

    print("=" * 50)
    print("  gpu-search-mcp installer")
    print("=" * 50)
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  Python   : {sys.executable} ({sys.version.split()[0]})")
    print(f"  Installer: {installer}")
    print(f"  Repo     : {REPO_DIR}")
    if args.dry_run:
        print("  Mode     : DRY RUN (no changes will be made)")
    print()

    check_python()

    if args.yes:
        dirs = [os.getcwd()]
        print(f"  Using current directory: {dirs[0]}")
    else:
        dirs = prompt_dirs()

    if args.dry_run:
        _dry_run_report(dirs)
        return

    install_deps(installer)
    print(f"  Server Python : {server_python()}")

    do_backup = args.backup_configs or True  # backup is on by default

    if not args.no_claude:
        print("\n[4/4] Wiring into Claude Code...")
        if do_backup:
            backup_file(Path.home() / ".claude.json")
            backup_file(Path.home() / ".claude" / "CLAUDE.md")
        patch_claude_code(dirs)

    if not args.no_codex:
        print("      Wiring into Codex...")
        codex_dir = Path.home() / ".codex"
        if do_backup:
            backup_file(codex_dir / "config.toml")
            backup_file(codex_dir / "config.yaml")
        patch_codex(dirs)

    save_startup_config(dirs)

    print()
    print("Done! Restart Claude Code or Codex to activate gpu-search.")
    if len(dirs) == 1:
        print(f"Indexed directory: {dirs[0]}")
    else:
        print(f"Indexed directories: {', '.join(dirs)}")


if __name__ == "__main__":
    main()
