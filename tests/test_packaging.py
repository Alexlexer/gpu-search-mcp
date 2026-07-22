from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from gpu_service.server_config import VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]


def _project_metadata() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def _requirement_names(requirements: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()
        for requirement in requirements
    }


def test_base_install_keeps_semantic_and_ast_dependencies_optional():
    project = _project_metadata()
    base = _requirement_names(project["dependencies"])
    extras = project["optional-dependencies"]

    assert "torch" in base
    assert "sentence-transformers" not in base
    assert "tree-sitter" not in base
    assert "sentence-transformers" in _requirement_names(extras["semantic"])
    assert "tree-sitter" in _requirement_names(extras["ast"])


def test_packaging_exposes_expected_extras_and_entry_points():
    project = _project_metadata()
    extras = project["optional-dependencies"]

    assert {"semantic", "ast", "cuda", "test", "all"} <= extras.keys()
    assert "torch" in _requirement_names(extras["cuda"])
    assert _requirement_names(extras["semantic"]) <= _requirement_names(extras["all"])
    assert _requirement_names(extras["ast"]) <= _requirement_names(extras["all"])
    assert project["scripts"]["gpu-search-mcp"] == "gpu_service.mcp_server:cli_main"
    assert project["scripts"]["gpu-search-bench"] == "gpu_service.bench:main"


def test_package_metadata_version_matches_runtime():
    assert _project_metadata()["version"] == VERSION
