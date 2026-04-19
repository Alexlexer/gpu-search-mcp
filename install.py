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


def run(cmd: list[str]):
    subprocess.check_call(cmd)


def server_python() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def ensure_venv():
    if VENV_PYTHON.exists():
        return
    print("[1/3] Creating local virtualenv...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])


def install_deps():
    ensure_venv()
    system = platform.system()
    pip = [server_python(), "-m", "pip", "install"]

    if system == "Darwin":
        # Apple Silicon / Intel Mac — plain PyPI torch has MPS support
        print("[2/3] Installing PyTorch (MPS)...")
        run(pip + ["torch", "torchvision"])
    else:
        # Windows / Linux — prefer CUDA if nvidia-smi is present
        has_cuda = False
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            has_cuda = result.returncode == 0
        except FileNotFoundError:
            pass

        if has_cuda:
            print("[2/3] Installing PyTorch (CUDA 12.1)...")
            run(pip + ["torch", "torchvision",
                       "--index-url", "https://download.pytorch.org/whl/cu121"])
        else:
            print("[2/3] No NVIDIA GPU found — installing PyTorch (CPU)...")
            run(pip + ["torch", "torchvision"])

    print("[3/3] Installing server dependencies...")
    run(pip + ["-r", str(REPO_DIR / "requirements.txt")])


def patch_claude_json(project_dirs: list[str]):
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
    print(f"  Wrote MCP config → {config_path}")


def patch_codex_mcp(project_dirs: list[str]):
    """Patch ~/.codex/config.yaml directly — works without the Codex CLI."""
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None

    codex_config = Path.home() / ".codex" / "config.yaml"

    entry = {
        "command": server_python(),
        "args": [str(SERVER_SCRIPT)] + [a for d in project_dirs for a in ("--directory", d)],
    }

    if yaml is not None and codex_config.exists():
        try:
            cfg = yaml.safe_load(codex_config.read_text(encoding="utf-8")) or {}
            cfg.setdefault("mcpServers", {})["gpu-search"] = entry
            codex_config.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
            print(f"  Wrote Codex config → {codex_config}")
            return
        except Exception as e:
            print(f"  YAML patch failed ({e}); trying CLI fallback...")

    # Fallback: use the Codex CLI if available
    codex = shutil.which("codex")
    if not codex:
        # Write a minimal config.yaml even if the dir doesn't exist yet
        codex_config.parent.mkdir(parents=True, exist_ok=True)
        if yaml is not None:
            cfg = {"mcpServers": {"gpu-search": entry}}
            codex_config.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
            print(f"  Created Codex config → {codex_config}")
        else:
            # No yaml module, no CLI — write JSON-in-YAML (valid YAML superset)
            import json
            cfg_json = json.dumps({"mcpServers": {"gpu-search": entry}}, indent=2)
            codex_config.write_text(cfg_json, encoding="utf-8")
            print(f"  Created Codex config (JSON) → {codex_config}")
        return

    cmd = [codex, "mcp", "add", "gpu-search", "--", server_python(), str(SERVER_SCRIPT)]
    for d in project_dirs:
        cmd += ["--directory", d]
    subprocess.run([codex, "mcp", "remove", "gpu-search"], capture_output=True, text=True)
    run(cmd)
    print("  Registered MCP server in Codex via CLI → gpu-search")


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

    print("\nWiring into Claude Code...")
    patch_claude_json(dirs)
    print("Wiring into Codex...")
    patch_codex_mcp(dirs)

    print()
    print("Done! Restart Claude Code or Codex to activate gpu-search.")
    if len(dirs) == 1:
        print(f"Indexed directory: {dirs[0]}")
    else:
        print(f"Indexed directories: {', '.join(dirs)}")


if __name__ == "__main__":
    main()
