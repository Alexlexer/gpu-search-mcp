from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

from ast_expand import read_block, skeleton_file
from gpu_dep_index import DepIndex


def test_csharp_read_block_fallback(tmp_path: Path):
    src = tmp_path / "UserController.cs"
    src.write_text(
        """
using Microsoft.AspNetCore.Mvc;

namespace Demo.Api;

public class UserController : ControllerBase
{
    public IActionResult GetUser(int id)
    {
        return Ok(id);
    }
}
""".lstrip(),
        encoding="utf-8",
    )
    block, start, end = read_block(str(src), 8)
    assert "GetUser" in block
    assert start <= 7 <= end


def test_csharp_dependency_uses_namespace_and_base_type(tmp_path: Path):
    contract = tmp_path / "IUserService.cs"
    contract.write_text("namespace Demo.Services; public interface IUserService {}", encoding="utf-8")
    impl = tmp_path / "UserService.cs"
    impl.write_text(
        "using Demo.Services; namespace Demo.App; public class UserService : IUserService {}",
        encoding="utf-8",
    )

    deps = DepIndex()
    stats = deps.index_directory(str(tmp_path))
    assert stats["files"] == 2
    imports = deps.direct_imports(str(impl))
    assert str(contract) in imports


def test_csharp_skeleton_returns_content_or_fallback(tmp_path: Path):
    src = tmp_path / "Widget.cs"
    src.write_text("namespace Demo; public record Widget(int Id);", encoding="utf-8")
    # tree-sitter-c-sharp may not be installed in CI; the important behavior is no crash.
    result = skeleton_file(str(src), [1])
    assert result is None or "Widget" in result
