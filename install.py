#!/usr/bin/env python3
"""
gpu-search-mcp installer
Installs dependencies and wires the MCP server into Claude Code and Codex.
"""
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

CLAUDE_MD_CONTENT = """\
# Search

The `gpu-search` MCP server is always available. Use it instead of grep, ripgrep, or find for any code search task.

- **`search_code(query)`** — use this for ALL searches. It auto-routes: exact identifiers go to GPU pattern search (sub-ms), natural language goes to semantic search (meaning-based). Never use Grep, Bash grep/rg, or the Glob tool when `search_code` can answer the question.
- **`dep_impact(filepath)`** — call this before editing any file to see what else could break.
- **`dep_index(directory)`** — build the dependency graph for a project (run once, persists across restarts).
- **`gpu_semantic_index(directory)`** — build the semantic embedding cache for a project (run once per project, then auto-loads on restart).

Prefer `search_code` over all other search mechanisms. It is faster than ripgrep and works from VRAM with no disk I/O.
"""

CODEX_INSTRUCTIONS = (
    "The gpu-search MCP server is always available. Use it instead of grep, ripgrep, or find. "
    "search_code(query) — use for ALL searches. Exact identifiers → GPU pattern search (sub-ms). "
    "Natural language → semantic search. Never use shell grep/rg when search_code can answer. "
    "dep_impact(filepath) — call before editing any file to see what else could break."
)


def run(cmd: list[str]):
    subprocess.check_call(cmd)


def server_python() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def ensure_venv():
    if VENV_PYTHON.exists():
        return
    print("[1/4] Creating local virtualenv...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])


def install_deps():
    ensure_venv()
    system = platform.system()
    pip = [server_python(), "-m", "pip", "install"]

    if system == "Darwin":
        print("[2/4] Installing PyTorch (MPS — Apple Silicon/Intel)...")
        run(pip + ["torch", "torchvision"])
    else:
        has_cuda = False
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            has_cuda = result.returncode == 0
        except FileNotFoundError:
            pass

        if has_cuda:
            print("[2/4] Installing PyTorch (CUDA 12.1)...")
            run(pip + ["torch", "torchvision",
                       "--index-url", "https://download.pytorch.org/whl/cu121"])
        else:
            print("[2/4] No NVIDIA GPU found — installing PyTorch (CPU)...")
            run(pip + ["torch", "torchvision"])

    print("[3/4] Installing server dependencies...")
    run(pip + ["-r", str(REPO_DIR / "requirements.txt")])


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
        raw = input(f"  Directory {len(dirs) + 1} (Enter to finish): ").strip()
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


def main():
    print("=" * 50)
    print("  gpu-search-mcp installer")
    print("=" * 50)
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  Python   : {sys.executable} ({sys.version.split()[0]})")
    print(f"  Repo     : {REPO_DIR}")
    print()

    check_python()
    install_deps()
    print(f"  Server Python : {server_python()}")

    dirs = prompt_dirs()

    print("\n[4/4] Wiring into Claude Code...")
    patch_claude_code(dirs)
    print("      Wiring into Codex...")
    patch_codex(dirs)

    print()
    print("Done! Restart Claude Code or Codex to activate gpu-search.")
    if len(dirs) == 1:
        print(f"Indexed directory: {dirs[0]}")
    else:
        print(f"Indexed directories: {', '.join(dirs)}")


if __name__ == "__main__":
    main()
