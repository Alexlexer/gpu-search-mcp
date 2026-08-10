"""Hidden deterministic validation for the initial real .NET agent tasks."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
from xml.sax.saxutils import escape


PROGRAMS = {
    "dotnet-expired-promotion": r"""
using Storefront.App.Pricing;

static void Equal(decimal expected, decimal actual, string message)
{
    if (expected != actual) throw new Exception($"{message}: expected {expected}, got {actual}");
}

var now = new DateTimeOffset(2026, 8, 10, 12, 0, 0, TimeSpan.Zero);
var calculator = new DiscountCalculator();
Equal(100m, calculator.Apply(100m, new Promotion(.20m, now.AddDays(-5), now.AddMinutes(-1)), now), "expired promotion");
Equal(80m, calculator.Apply(100m, new Promotion(.20m, now.AddDays(-1), now.AddMinutes(1)), now), "active promotion");
Equal(100m, calculator.Apply(100m, new Promotion(.20m, now.AddMinutes(1), now.AddDays(1)), now), "future promotion");
Equal(100m, calculator.Apply(100m, new Promotion(.20m, now.AddDays(-1), now), now), "exclusive expiration boundary");
""",
    "dotnet-external-reference": r"""
using Storefront.App.Domain;
using Storefront.App.Infrastructure;
using Storefront.App.Services;

var order = new Order("1", "customer", "Web-ABC-42", OrderStatus.Pending, DateTimeOffset.UtcNow);
var repository = new InMemoryOrderRepository(new[] { order });
var found = await repository.FindByExternalReferenceAsync("web-abc-42", CancellationToken.None);
if (found != order) throw new Exception("repository lookup must be ordinal case-insensitive");
var service = new OrderService(repository);
var fromService = await service.FindByExternalReferenceAsync("WEB-ABC-42", CancellationToken.None);
if (fromService != order) throw new Exception("service lookup must delegate to the repository");
if (await repository.FindByExternalReferenceAsync("missing", CancellationToken.None) is not null)
    throw new Exception("missing external reference must return null");
using var cancelled = new CancellationTokenSource();
cancelled.Cancel();
try
{
    await repository.FindByExternalReferenceAsync("web-abc-42", cancelled.Token);
    throw new Exception("cancelled lookup did not throw");
}
catch (OperationCanceledException)
{
}
""",
    "dotnet-di-retry-options": r"""
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Storefront.App.Bootstrap;
using Storefront.App.Configuration;
using Storefront.App.Infrastructure;
using Storefront.App.Time;

var values = new Dictionary<string, string?>
{
    ["Storefront:Retry:MaxAttempts"] = "7",
    ["Storefront:Retry:BaseDelayMilliseconds"] = "250",
};
var configuration = new ConfigurationBuilder().AddInMemoryCollection(values).Build();
var services = new ServiceCollection();
services.AddStorefront(configuration);
using var provider = services.BuildServiceProvider();
var options = provider.GetRequiredService<IOptions<RetryOptions>>().Value;
if (options.MaxAttempts != 7 || options.BaseDelayMilliseconds != 250)
    throw new Exception("retry options were not bound from Storefront:Retry");
var firstClock = provider.GetRequiredService<IClock>();
var secondClock = provider.GetRequiredService<IClock>();
if (!ReferenceEquals(firstClock, secondClock))
    throw new Exception("IClock must be registered as a singleton");
var gateway = provider.GetRequiredService<OrderGateway>();
if (gateway.MaxAttempts != 7 || gateway.BaseDelayMilliseconds != 250)
    throw new Exception("OrderGateway did not receive bound retry options");
""",
    "dotnet-open-orders-endpoint": r"""
using Storefront.App.Api;
using Storefront.App.Domain;
using Storefront.App.Infrastructure;
using Storefront.App.Services;

var now = DateTimeOffset.UtcNow;
var orders = new[]
{
    new Order("pending-old", "c1", "P-1", OrderStatus.Pending, now.AddHours(-3)),
    new Order("processing-new", "c1", "P-2", OrderStatus.Processing, now.AddHours(-1)),
    new Order("completed", "c1", "P-3", OrderStatus.Completed, now),
    new Order("cancelled", "c1", "P-4", OrderStatus.Cancelled, now.AddHours(-2)),
    new Order("other-customer", "c2", "P-5", OrderStatus.Pending, now.AddMinutes(-1)),
};
var endpoint = new OrderSummaryEndpoint(
    new OrderService(new InMemoryOrderRepository(orders)));
var response = await endpoint.GetOpenOrdersAsync("c1", CancellationToken.None);
var ids = response.Orders.Select(order => order.Id).ToArray();
if (response.Total != 2)
    throw new Exception($"expected two open orders, got {response.Total}");
if (!ids.SequenceEqual(new[] { "processing-new", "pending-old" }))
    throw new Exception($"unexpected open-order sequence: {string.Join(",", ids)}");
""",
    "dotnet-cancellation-propagation": r"""
using Storefront.App.Api;
using Storefront.App.Contracts;
using Storefront.App.Domain;
using Storefront.App.Services;

var repository = new RecordingRepository();
var endpoint = new OrderSummaryEndpoint(new OrderService(repository));
using var source = new CancellationTokenSource();
await endpoint.GetOpenOrdersAsync("customer", source.Token);
if (repository.ObservedToken != source.Token)
    throw new Exception("request cancellation token did not reach the repository");

sealed class RecordingRepository : IOrderRepository
{
    public CancellationToken ObservedToken { get; private set; }

    public Task<IReadOnlyList<Order>> GetByCustomerAsync(
        string customerId,
        CancellationToken cancellationToken)
    {
        ObservedToken = cancellationToken;
        return Task.FromResult<IReadOnlyList<Order>>(Array.Empty<Order>());
    }
}
""",
}


def validate(task_id: str, workspace: Path) -> int:
    source = PROGRAMS.get(task_id)
    if source is None:
        raise ValueError(f"unknown task: {task_id}")
    app = (
        workspace
        / "benchmarks"
        / "fixtures"
        / "agent_eval_dotnet"
        / "Storefront.App"
        / "Storefront.App.csproj"
    ).resolve()
    if not app.is_file():
        raise FileNotFoundError(f"fixture project not found: {app}")

    version = subprocess.run(
        ["dotnet", "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    major = max(8, int(version.split(".", 1)[0]))
    target_framework = f"net{major}.0"

    with tempfile.TemporaryDirectory(prefix=f"gpu-search-{task_id}-") as temp:
        root = Path(temp)
        (root / "Validation.csproj").write_text(
            (
                '<Project Sdk="Microsoft.NET.Sdk">\n'
                "  <PropertyGroup>\n"
                "    <OutputType>Exe</OutputType>\n"
                f"    <TargetFramework>{target_framework}</TargetFramework>\n"
                "    <ImplicitUsings>enable</ImplicitUsings>\n"
                "    <Nullable>enable</Nullable>\n"
                "    <RestoreIgnoreFailedSources>true</RestoreIgnoreFailedSources>\n"
                "  </PropertyGroup>\n"
                "  <ItemGroup>\n"
                f'    <ProjectReference Include="{escape(str(app))}" />\n'
                '    <FrameworkReference Include="Microsoft.AspNetCore.App" />\n'
                "  </ItemGroup>\n"
                "</Project>\n"
            ),
            encoding="utf-8",
        )
        (root / "Program.cs").write_text(source.strip() + "\n", encoding="utf-8")
        completed = subprocess.run(
            [
                "dotnet",
                "run",
                "--project",
                str(root / "Validation.csproj"),
                "--configuration",
                "Release",
                "--nologo",
            ],
            cwd=workspace,
            check=False,
        )
        return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", choices=sorted(PROGRAMS))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    return validate(args.task_id, args.workspace.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
