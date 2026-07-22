#!/usr/bin/env python3
"""Validate a built wheel from an isolated environment outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tempfile
import venv
from zipfile import ZipFile


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    allowed_codes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode not in allowed_codes:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _assert_wheel_contents(wheel: Path) -> None:
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
    if "gpu_service/mcp_server.py" not in names:
        raise RuntimeError("wheel does not contain gpu_service/mcp_server.py")
    forbidden = (".agents/", ".git/", "crates/")
    leaked = sorted(name for name in names if name.startswith(forbidden) or name.endswith(".rs"))
    if leaked:
        raise RuntimeError(f"wheel contains repository-only files: {leaked[:5]}")


def _assert_within(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"installed module resolved outside venv: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path, help="Wheel produced by python -m build")
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error(f"wheel not found: {wheel}")

    _assert_wheel_contents(wheel)

    with tempfile.TemporaryDirectory(prefix="gpu-search-package-smoke-") as temp:
        root = Path(temp)
        environment = root / "venv"
        work = root / "outside-checkout"
        home = root / "home"
        work.mkdir()
        home.mkdir()

        venv.EnvBuilder(with_pip=True).create(environment)
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        cli = scripts / ("gpu-search-mcp.exe" if os.name == "nt" else "gpu-search-mcp")

        # Reuse the caller's already-installed runtime dependencies without
        # exposing its editable project checkout to the smoke environment.
        if os.name == "nt":
            child_site = environment / "Lib" / "site-packages"
        else:
            version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
            child_site = environment / "lib" / version_dir / "site-packages"
        parent_sites = [Path(item).resolve() for item in site.getsitepackages()]
        dependency_paths = [item for item in parent_sites if item.is_dir()]
        (child_site / "caller-runtime-dependencies.pth").write_text(
            "\n".join(
                f"import site; site.addsitedir({str(item)!r})"
                for item in dependency_paths
            )
            + "\n",
            encoding="utf-8",
        )

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(
            {
                "HOME": str(home),
                "USERPROFILE": str(home),
                "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "GPU_SEARCH_DEVICE": "cpu",
            }
        )

        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                str(wheel),
            ],
            cwd=work,
            env=env,
        )

        location = _run(
            [
                str(python),
                "-c",
                (
                    "import importlib.util, pathlib; "
                    "print(pathlib.Path(importlib.util.find_spec("
                    "'gpu_service.mcp_server').origin).resolve())"
                ),
            ],
            cwd=work,
            env=env,
        ).stdout.strip()
        _assert_within(Path(location), environment)

        version = _run(
            [
                str(python),
                "-c",
                "from importlib.metadata import version; print(version('gpu-search-mcp'))",
            ],
            cwd=work,
            env=env,
        ).stdout.strip()
        cli_version = _run([str(cli), "--version"], cwd=work, env=env).stdout.strip()
        if version not in cli_version:
            raise RuntimeError(f"CLI version mismatch: metadata={version!r}, output={cli_version!r}")

        setup = _run(
            [
                str(cli),
                "setup",
                "--client",
                "codex",
                "--no-index",
                "--no-model",
                "--dry-run",
            ],
            cwd=work,
            env=env,
        )
        if "Dry run complete; no files changed." not in setup.stdout:
            raise RuntimeError("installed setup dry-run did not complete safely")
        if any(home.iterdir()):
            raise RuntimeError("setup dry-run wrote files under the isolated home")

        doctor = _run(
            [str(cli), "doctor", "--json"],
            cwd=work,
            env=env,
            allowed_codes=(0, 1),
        )
        report = json.loads(doctor.stdout)
        if report.get("version") != version:
            raise RuntimeError("doctor version does not match package metadata")

    print(f"[PASS] wheel install outside checkout: {wheel.name}")
    print("[PASS] base CLI works without semantic or AST extras")
    print("[PASS] setup dry-run and doctor JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
