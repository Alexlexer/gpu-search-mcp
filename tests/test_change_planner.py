from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu_service"))

from change_planner import ChangePlanner, SECTION_ORDER  # noqa: E402
from symbol_index import SymbolIndex  # noqa: E402


class FakePattern:
    def __init__(self, root: Path, results: dict[str, list[dict]]):
        self.root = root
        self.results = results

    def stats(self):
        return {"files": len(list(self.root.iterdir())), "base_dir": str(self.root)}

    def search(self, query):
        return self.results.get(query, [])


class FakeSemantic:
    def stats(self):
        return {"chunks": 0, "base_dir": None}

    def search(self, query, top_k=8):
        return []


class FakeDeps:
    def __init__(self, root: Path):
        self.root = root

    def stats(self):
        return {"files": 6, "edges": 2, "base_dir": str(self.root)}

    def direct_imports(self, filepath):
        if Path(filepath).name == "IUserService.cs":
            return [str(self.root / "Models.cs")]
        return []

    def impact(self, filepath):
        if Path(filepath).name == "IUserService.cs":
            return [{
                "file": str(self.root / "UsersController.cs"),
                "hops": 1,
                "reason": "direct importer",
            }]
        return []


class FakeGit:
    def boost(self, filepath):
        name = Path(filepath).name
        if name == "Recent.cs":
            return 0.4
        if name == "Program.cs":
            return 0.15
        return 0.0


FILES = {
    "IUserService.cs": """
namespace Demo;

public interface IUserService
{
    User GetUser(int id);
}
""",
    "UserService.cs": """
namespace Demo;

public class UserService : IUserService
{
    public User GetUser(int id) => new User();
}
""",
    "UsersController.cs": """
namespace Demo;

public class UsersController
{
    private readonly IUserService _service;
    public UsersController(IUserService service) { _service = service; }
    public User Get(int id) => _service.GetUser(id);
}
""",
    "UserServiceTests.cs": """
namespace Demo.Tests;

public class UserServiceTests
{
    [Fact]
    public void Returns_user()
    {
        var service = new UserService();
        service.GetUser(1);
    }
}
""",
    "Models.cs": "namespace Demo; public record User;\n",
    "Program.cs": "services.AddScoped<IUserService, UserService>();\n",
    "Recent.cs": "public class RecentlyChangedHelper {}\n",
    "appsettings.json": '{"service": "IUserService"}\n',
}


def _pattern_result(path: Path, line: int, content: str) -> dict:
    return {"file": str(path), "matches": [{"line": line, "content": content}]}


def _planner(tmp_path: Path) -> ChangePlanner:
    for name, content in FILES.items():
        (tmp_path / name).write_text(content.strip() + "\n", encoding="utf-8")
    hidden = tmp_path / ".agents"
    hidden.mkdir()
    (hidden / "plan.txt").write_text("IUserService token budget", encoding="utf-8")
    symbols = SymbolIndex()
    symbols.index_directory(str(tmp_path))
    results = {
        "IUserService": [
            _pattern_result(hidden / "plan.txt", 1, "IUserService"),
            _pattern_result(tmp_path / "Recent.cs", 1, "RecentlyChangedHelper"),
            _pattern_result(tmp_path / "IUserService.cs", 3, "IUserService"),
            _pattern_result(tmp_path / "Program.cs", 1, "IUserService"),
            _pattern_result(tmp_path / "appsettings.json", 1, "IUserService"),
        ],
    }
    return ChangePlanner(
        FakePattern(tmp_path, results),
        FakeSemantic(),
        FakeDeps(tmp_path),
        symbols,
        FakeGit(),
    )


def test_plan_change_builds_ordered_complete_bundle(tmp_path):
    planner = _planner(tmp_path)
    plan = planner.plan_change(
        "change IUserService response behavior",
        top_k=8,
        max_context_tokens=6000,
    )

    assert plan.items[0].section == "primary_implementation"
    assert "interface Demo.IUserService" in plan.items[0].title
    assert plan.items[0].reason == "exact symbol match"
    assert plan.items[0].git_boost == 0.0

    ranks = [SECTION_ORDER.index(item.section) for item in plan.items]
    assert ranks == sorted(ranks)
    sections = {item.section for item in plan.items}
    assert {
        "primary_implementation",
        "parent_context",
        "direct_callers",
        "direct_dependencies",
        "implementations_overrides",
        "configuration_documentation",
        "tests_coverage",
        "git_changes",
    } <= sections
    assert plan.tokens_used <= plan.max_context_tokens
    assert plan.inspection_order[0].endswith("IUserService.cs")
    assert all(".agents" not in Path(item.file_path).parts for item in plan.items)
    assert "Semantic index is unavailable" in " ".join(plan.unknowns)


def test_exact_symbol_is_not_outranked_by_git_state(tmp_path):
    plan = _planner(tmp_path).plan_change(
        "change IUserService", top_k=8, max_context_tokens=3000,
    )
    primary = [item for item in plan.items if item.section == "primary_implementation"]

    assert primary[0].title.startswith("interface Demo.IUserService")
    assert any(item.file_path.endswith("Recent.cs") and item.git_boost == 0.4 for item in primary)


def test_budget_is_deterministic_and_reports_omissions(tmp_path):
    planner = _planner(tmp_path)
    first = planner.plan_change("change IUserService", top_k=8, max_context_tokens=256)
    second = planner.plan_change("change IUserService", top_k=8, max_context_tokens=256)

    assert first.as_dict() == second.as_dict()
    assert first.tokens_used <= 256
    assert first.omitted
    assert {item.reason for item in first.omitted} <= {"token budget", "top_k limit"}
    assert "## Omitted items" in first.to_markdown(str(tmp_path))
    assert any("token budget" in risk.lower() for risk in first.risks)


@pytest.mark.parametrize(
    ("change_request", "top_k", "budget", "message"),
    [
        ("", 8, 6000, "request"),
        ("change service", 0, 6000, "top_k"),
        ("change service", 8, 100, "max_context_tokens"),
    ],
)
def test_plan_change_validates_inputs(tmp_path, change_request, top_k, budget, message):
    with pytest.raises(ValueError, match=message):
        _planner(tmp_path).plan_change(change_request, top_k=top_k, max_context_tokens=budget)


def test_mcp_plan_change_formats_the_bundle(tmp_path, monkeypatch):
    import mcp_server

    planner = _planner(tmp_path)
    monkeypatch.setattr(mcp_server, "planner", planner)

    output = mcp_server.plan_change(
        "change IUserService", top_k=4, max_context_tokens=2000,
    )
    assert output.startswith("# Change plan: change IUserService")
    assert "## Primary implementation" in output
    assert "## Risks" in output
    assert "## Inspection order" in output
    assert "## Omitted items" in output
