"""Server-wide constants, built-in audit signal definitions, and config I/O.

No imports from other gpu_service modules — safe to import from anywhere.
"""
import json
import os
import sys
from pathlib import Path

VERSION = "0.1.0"
CONFIG_PATH = Path.home() / ".gpu-search-config.json"

INDEXED_EXTS: set = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.c', '.cpp', '.h',
    '.hpp', '.java', '.cs', '.rb', '.php', '.swift', '.kt', '.json', '.yaml',
    '.yml', '.toml', '.md', '.txt', '.html', '.css', '.scss', '.sql', '.sh',
    '.bat', '.ps1', '.cfg', '.ini', '.xml',
    # .env excluded by default — pass --allow-env-files to opt in
}

SKIP_DIRS: set = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build',
    '.next', '.nuxt', 'target', 'bin', 'obj', '.idea', '.vscode', '.mypy_cache',
}

_DEP_EXTS: set = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".cs", ".rb"}

MAX_CHUNKS = 500_000

_DEP_LIMITATIONS: list[str] = [
    "Dependency impact is based on import/type/name heuristics and is not compiler-accurate.",
    "C# analysis does not use Roslyn — namespace, type, and base/interface name heuristics are used instead.",
]

_GLOBAL_LIMITATIONS: list[str] = [
    "Dependency impact is heuristic, not compiler-accurate.",
    "Secret redaction is best-effort pattern matching, not a DLP scanner.",
    "CPU fallback is slower than CUDA/MPS for large repositories.",
    "Semantic search requires model download/cache on first use.",
    "HTTP mode is local-first — do not expose to the public internet.",
]

_SIGNAL_SCAN_LIMITATIONS: list[str] = [
    "Signal scan is heuristic and search-based, not compiler-accurate.",
    "Absence of a signal does not prove absence in the repository.",
    "Pattern matching may produce false positives (e.g., commented-out code, test fixtures).",
    "Multi-query signals run separate searches; results are deduplicated by file+line.",
]

_BUILTIN_SIGNALS: list[dict] = [
    # ── legacy-dotnet ──────────────────────────────────────────────────────
    {
        "id": "legacy-web-config",
        "category": "legacy-dotnet",
        "label": "web.config present",
        "description": "Repository contains ASP.NET/.NET Framework web.config files.",
        "confidence": "high",
        "queries": ["web.config"],
    },
    {
        "id": "legacy-global-asax",
        "category": "legacy-dotnet",
        "label": "Global.asax present",
        "description": "Repository contains Global.asax (ASP.NET Framework application lifecycle file).",
        "confidence": "high",
        "queries": ["Global.asax"],
    },
    {
        "id": "legacy-packages-config",
        "category": "legacy-dotnet",
        "label": "packages.config present",
        "description": "Repository uses NuGet packages.config (pre-SDK-style package management).",
        "confidence": "high",
        "queries": ["packages.config"],
    },
    {
        "id": "legacy-system-web",
        "category": "legacy-dotnet",
        "label": "System.Web usage",
        "description": "Repository references System.Web (ASP.NET Framework, not ASP.NET Core).",
        "confidence": "medium",
        "queries": ["System.Web"],
    },
    {
        "id": "legacy-system-web-mvc",
        "category": "legacy-dotnet",
        "label": "System.Web.Mvc usage",
        "description": "Repository references System.Web.Mvc (ASP.NET MVC Framework).",
        "confidence": "medium",
        "queries": ["System.Web.Mvc"],
    },
    {
        "id": "legacy-app-start",
        "category": "legacy-dotnet",
        "label": "App_Start directory",
        "description": "Repository contains App_Start/ folder (ASP.NET MVC/WebAPI bootstrap convention).",
        "confidence": "high",
        "queries": ["App_Start"],
    },
    # ── config ─────────────────────────────────────────────────────────────
    {
        "id": "appsettings-json",
        "category": "config",
        "label": "appsettings.json present",
        "description": "Repository contains ASP.NET Core appsettings.json configuration files.",
        "confidence": "high",
        "queries": ["appsettings.json"],
    },
    {
        "id": "connection-strings",
        "category": "config",
        "label": "connectionStrings usage",
        "description": "Repository contains connectionStrings configuration entries.",
        "confidence": "medium",
        "queries": ["connectionStrings"],
    },
    {
        "id": "app-config",
        "category": "config",
        "label": "app.config present",
        "description": "Repository contains app.config (desktop/console .NET Framework configuration).",
        "confidence": "high",
        "queries": ["app.config"],
    },
    # ── sql ────────────────────────────────────────────────────────────────
    {
        "id": "sql-connection",
        "category": "sql",
        "label": "SqlConnection usage",
        "description": "Repository uses SqlConnection (direct ADO.NET SQL Server access).",
        "confidence": "high",
        "queries": ["SqlConnection"],
    },
    {
        "id": "raw-sql-execute",
        "category": "sql",
        "label": "Raw SQL execution",
        "description": "Repository uses raw SQL via EF Core methods (ExecuteSql/FromSql variants).",
        "confidence": "medium",
        "queries": ["ExecuteSql", "ExecuteSqlRaw", "FromSqlRaw", "FromSql"],
    },
    {
        "id": "sql-command",
        "category": "sql",
        "label": "SqlCommand usage",
        "description": "Repository uses SqlCommand (raw ADO.NET command execution).",
        "confidence": "high",
        "queries": ["SqlCommand"],
    },
    # ── async-risk ─────────────────────────────────────────────────────────
    {
        "id": "sync-over-async-result",
        "category": "async-risk",
        "label": ".Result (sync-over-async)",
        "description": "Repository uses .Result to synchronously block on async tasks (deadlock risk).",
        "confidence": "medium",
        "queries": [".Result"],
    },
    {
        "id": "sync-over-async-wait",
        "category": "async-risk",
        "label": ".Wait() (sync-over-async)",
        "description": "Repository uses .Wait() to synchronously block on async tasks (deadlock risk).",
        "confidence": "medium",
        "queries": [".Wait()"],
    },
    # ── exception-risk ─────────────────────────────────────────────────────
    {
        "id": "broad-catch-exception",
        "category": "exception-risk",
        "label": "catch (Exception) usage",
        "description": "Repository has broad catch blocks catching all exceptions.",
        "confidence": "medium",
        "queries": ["catch (Exception"],
    },
    {
        "id": "empty-catch-candidate",
        "category": "exception-risk",
        "label": "catch block candidate",
        "description": "Repository has catch blocks (review for swallowed exceptions).",
        "confidence": "low",
        "queries": ["catch {"],
    },
    # ── di ─────────────────────────────────────────────────────────────────
    {
        "id": "add-singleton",
        "category": "di",
        "label": "AddSingleton usage",
        "description": "Repository registers singleton services in DI container.",
        "confidence": "high",
        "queries": ["AddSingleton"],
    },
    {
        "id": "add-scoped",
        "category": "di",
        "label": "AddScoped usage",
        "description": "Repository registers scoped services in DI container.",
        "confidence": "high",
        "queries": ["AddScoped"],
    },
    {
        "id": "add-transient",
        "category": "di",
        "label": "AddTransient usage",
        "description": "Repository registers transient services in DI container.",
        "confidence": "high",
        "queries": ["AddTransient"],
    },
    {
        "id": "service-locator-getservice",
        "category": "di",
        "label": "GetService (service locator)",
        "description": "Repository uses GetService (service locator anti-pattern).",
        "confidence": "medium",
        "queries": ["GetService"],
    },
    {
        "id": "service-locator-getrequiredservice",
        "category": "di",
        "label": "GetRequiredService (service locator)",
        "description": "Repository uses GetRequiredService (service locator pattern).",
        "confidence": "medium",
        "queries": ["GetRequiredService"],
    },
    # ── tests ──────────────────────────────────────────────────────────────
    {
        "id": "test-project",
        "category": "tests",
        "label": "Test project present",
        "description": "Repository contains test projects or test framework references.",
        "confidence": "medium",
        "queries": [".Tests", "xunit", "NUnit", "MSTest"],
    },
]


def _load_config_dirs() -> list[str]:
    """Read directories from ~/.gpu-search-config.json."""
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return [d for d in data.get("directories", []) if os.path.isdir(d)]
    except Exception:
        pass
    return []


def _save_config_dirs(dirs: list[str]):
    """Persist directory list to ~/.gpu-search-config.json."""
    try:
        data = {}
        existing: list[str] = []
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            existing = data.get("directories", [])
        merged = list(dict.fromkeys(existing + dirs))
        data["directories"] = merged
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[gpu-search] Could not save config: {e}", file=sys.stderr, flush=True)
