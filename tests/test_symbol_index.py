from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu_service"))

from symbol_index import SymbolIndex  # noqa: E402


CSHARP_FIXTURE = {
    "IUserService.cs": """
namespace Demo.Services;

public interface IUserService
{
    User GetUser(int id);
}
""",
    "UserService.cs": """
namespace Demo.Services;

public sealed class UserService : IUserService
{
    private readonly UserRepository _repository;
    public UserRepository Repository { get; init; }

    public UserService(UserRepository repository)
    {
        _repository = repository;
    }

    public User GetUser(int id) => _repository.GetUser(id);
}
""",
    "UsersController.cs": """
using Microsoft.AspNetCore.Mvc;
using Demo.Services;

namespace Demo.Controllers;

[ApiController]
[Route("api/[controller]")]
public class UsersController : ControllerBase
{
    private readonly IUserService _service;

    public UsersController(IUserService service)
    {
        _service = service;
    }

    [HttpGet("{id}")]
    public IActionResult Get(int id)
    {
        return Ok(_service.GetUser(id));
    }
}
""",
    "Program.cs": """
using Demo.Services;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddScoped<IUserService, UserService>();
""",
    "UserServiceTests.cs": """
namespace Demo.Tests;

public class UserServiceTests
{
    [Fact]
    public void Returns_user()
    {
        var service = new UserService(new UserRepository());
        service.GetUser(1);
    }
}
""",
}


def _build_fixture(tmp_path: Path) -> SymbolIndex:
    for name, content in CSHARP_FIXTURE.items():
        (tmp_path / name).write_text(content.strip() + "\n", encoding="utf-8")
    index = SymbolIndex()
    index.index_directory(str(tmp_path))
    return index


def test_csharp_symbol_model_extracts_aspnet_shapes(tmp_path):
    index = _build_fixture(tmp_path)
    symbols = index.symbols()
    kinds = {symbol.kind for symbol in symbols}

    assert {"namespace", "module", "interface", "class", "method"} <= kinds
    assert {"constructor", "property", "field", "endpoint", "test"} <= kinds
    assert index.find_symbol("Demo.Services.IUserService")[0].kind == "interface"
    assert index.find_symbol("Repository", kind="property")[0].signature
    assert index.find_symbol("Get", kind="endpoint")[0].qualified_name.endswith("#endpoint")


def test_edges_capture_implementations_calls_construction_and_di(tmp_path):
    index = _build_fixture(tmp_path)
    implementations = index.find_implementations("IUserService")

    assert [item["symbol"].name for item in implementations] == ["UserService"]
    assert implementations[0]["edge"].confidence >= 0.8

    edges = index.edges()
    edge_kinds = {edge.kind for edge in edges}
    assert {"implements", "inherits", "calls", "instantiates", "configured_by"} <= edge_kinds
    assert all(edge.parser == "csharp-heuristic" for edge in edges)
    assert all(edge.parser_version == "1" for edge in edges)


def test_symbol_ids_are_deterministic_and_updates_replace_stale_data(tmp_path):
    index = _build_fixture(tmp_path)
    first = {symbol.qualified_name: symbol.id for symbol in index.symbols()}
    index.index_directory(str(tmp_path))
    second = {symbol.qualified_name: symbol.id for symbol in index.symbols()}
    assert first == second

    service_path = tmp_path / "UserService.cs"
    service_path.write_text(
        service_path.read_text(encoding="utf-8").replace("GetUser", "FindUser"),
        encoding="utf-8",
    )
    index.update_file(str(service_path))
    assert index.find_symbol("FindUser", kind="method")
    assert not any(
        symbol.file_path == str(service_path) and symbol.name == "GetUser"
        for symbol in index.symbols()
    )


def test_serialized_contracts_are_json_friendly(tmp_path):
    index = _build_fixture(tmp_path)
    symbol = index.find_symbol("UserService", kind="class")[0].as_dict()
    edge = index.find_implementations("IUserService")[0]["edge"].as_dict()

    assert isinstance(symbol["modifiers"], list)
    assert symbol["language"] == "csharp"
    assert edge["target_symbol_id"]
    assert 0.0 <= edge["confidence"] <= 1.0
    assert edge["provenance"] == "declaration"
