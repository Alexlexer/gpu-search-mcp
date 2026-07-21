from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

from ast_expand import read_block, skeleton_file
import gpu_dep_index
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


def test_csharp_dependency_impact_includes_type_reference_reason(tmp_path: Path):
    service = tmp_path / "UserService.cs"
    service.write_text("namespace Demo.Services; public class UserService {}", encoding="utf-8")
    controller = tmp_path / "UserController.cs"
    controller.write_text(
        """
using Demo.Services;
namespace Demo.Api;
public class UserController
{
    private readonly UserService _service;
}
""".lstrip(),
        encoding="utf-8",
    )

    deps = DepIndex()
    deps.index_directory(str(tmp_path))
    impact = deps.impact(str(service))
    controller_hit = next(item for item in impact if item["file"] == str(controller))
    assert controller_hit["hops"] == 1
    assert controller_hit["reason"] == "references type UserService"


def test_csharp_dependency_impact_includes_interface_reason(tmp_path: Path):
    contract = tmp_path / "IUserService.cs"
    contract.write_text("namespace Demo.Services; public interface IUserService {}", encoding="utf-8")
    impl = tmp_path / "UserService.cs"
    impl.write_text(
        "namespace Demo.Services; public class UserService : IUserService {}",
        encoding="utf-8",
    )

    deps = DepIndex()
    deps.index_directory(str(tmp_path))
    impact = deps.impact(str(contract))
    impl_hit = next(item for item in impact if item["file"] == str(impl))
    assert impl_hit["reason"] == "implements interface IUserService"


def test_python_dependency_impact_includes_module_import_reason(tmp_path: Path):
    module = tmp_path / "settings.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    consumer = tmp_path / "app.py"
    consumer.write_text("from settings import VALUE\nprint(VALUE)\n", encoding="utf-8")

    deps = DepIndex()
    deps.index_directory(str(tmp_path))
    impact = deps.impact(str(module))
    consumer_hit = next(item for item in impact if item["file"] == str(consumer))
    assert consumer_hit["reason"] == "imports module settings"


def test_csharp_skeleton_returns_content_or_fallback(tmp_path: Path):
    src = tmp_path / "Widget.cs"
    src.write_text("namespace Demo; public record Widget(int Id);", encoding="utf-8")
    # tree-sitter-c-sharp may not be installed in CI; the important behavior is no crash.
    result = skeleton_file(str(src), [1])
    assert result is None or "Widget" in result

def test_cpu_dependency_impact_avoids_sparse_matmul(tmp_path: Path, monkeypatch):
    (tmp_path / "leaf.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "middle.py").write_text("import leaf\n", encoding="utf-8")
    (tmp_path / "root.py").write_text("import middle\n", encoding="utf-8")
    monkeypatch.setattr(gpu_dep_index, "DEVICE", gpu_dep_index.torch.device("cpu"))
    monkeypatch.setattr(
        gpu_dep_index.torch.sparse,
        "mm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sparse.mm called")),
    )

    deps = DepIndex()
    deps.index_directory(str(tmp_path), force_rebuild=True)
    impact = deps.impact(str(tmp_path / "leaf.py"))

    by_file = {Path(item["file"]).name: item["hops"] for item in impact}
    assert by_file == {"middle.py": 1, "root.py": 2}
