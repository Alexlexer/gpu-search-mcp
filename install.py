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
    codex = shutil.which("codex")
    if not codex:
        print("  Codex CLI not found in PATH; skipping Codex MCP setup.")
        return

    cmd = [codex, "mcp", "add", "gpu-search", "--", server_python(), str(SERVER_SCRIPT)]
    for d in project_dirs:
        cmd += ["--directory", d]

    # Replace an existing registration if present.
    subprocess.run([codex, "mcp", "remove", "gpu-search"], capture_output=True, text=True)
    run(cmd)
    print("  Registered MCP server in Codex → gpu-search")


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
