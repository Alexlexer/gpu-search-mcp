# Repository Signal Scan

`POST /scan/signals` runs a bounded, categorized scan of common legacy and risk signals in one call. It is designed for audit and onboarding tools such as LegacyLens that need broad repository evidence without issuing dozens of separate search queries.

---

## How it works

Signal scan reuses the existing GPU pattern search engine. For each built-in signal, it runs one or more pattern queries against the indexed repository, deduplicates matches by file and line, and returns capped, redacted results.

- **Heuristic, not compiler-accurate.** Pattern matching may match commented-out code, test fixtures, or string literals. Treat results as advisory context for review.
- **Absence of a signal does not prove absence.** The signal was not found by pattern search; it may exist in code paths the heuristic did not cover.
- **Read-only.** No files are written or modified.
- **Bounded.** `topKPerSignal` defaults to 5 and is capped at 20. Total matches per request are capped at ~200.
- **Redacted.** All snippets pass through the same credential redaction layer as all other endpoints.

For compiler-accurate facts (type resolution, call graph, overrides), use Roslyn or a language server in the client (e.g., LegacyLens). gpu-search-mcp handles broad repository retrieval; the client handles interpretation.

---

## Request

```json
{
  "categories": ["legacy-dotnet", "sql"],
  "topKPerSignal": 5,
  "includeSnippets": true,
  "contextMode": "compact"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `categories` | string[] | (all) | Limit to specific categories. Omit to scan all. |
| `topKPerSignal` | integer | 5 | Max matches per signal (capped at 20). |
| `includeSnippets` | boolean | true | Include redacted code snippets in matches. |
| `contextMode` | string | `"compact"` | Snippet length (`compact` / `normal` / `full`). |

---

## Response

```json
{
  "result": "Signal scan: 3 signals with matches, 7 total matches across 2 categories.",
  "categories": ["legacy-dotnet", "sql"],
  "summary": {
    "signalCount": 3,
    "matchCount": 7,
    "categories": {
      "legacy-dotnet": 4,
      "sql": 3
    }
  },
  "signals": [
    {
      "id": "legacy-web-config",
      "category": "legacy-dotnet",
      "label": "web.config present",
      "description": "Repository contains ASP.NET/.NET Framework web.config files.",
      "confidence": "high",
      "query": "web.config",
      "matches": [
        {
          "file": "src/Web/web.config",
          "absoluteFile": "D:\\repos\\app\\src\\Web\\web.config",
          "lineStart": 1,
          "lineEnd": 1,
          "score": 1.0,
          "reason": "pattern match: web.config",
          "snippet": "<configuration>",
          "engine": "pattern"
        }
      ]
    }
  ],
  "limitations": [
    "Signal scan is heuristic and search-based, not compiler-accurate.",
    "Absence of a signal does not prove absence in the repository."
  ],
  "warnings": []
}
```

**Signal confidence:**
- `high` — direct file-name or exact token evidence (e.g., `web.config` file present, `SqlConnection` exact token).
- `medium` — strong pattern evidence that may have contextual false positives.
- `low` — weak or fallback evidence (e.g., generic `catch {` block presence).

---

## Built-in signals

### Category: `legacy-dotnet`

| Signal ID | Query | Confidence | Meaning |
|---|---|---|---|
| `legacy-web-config` | `web.config` | high | ASP.NET Framework config file present |
| `legacy-global-asax` | `Global.asax` | high | Application lifecycle file present |
| `legacy-packages-config` | `packages.config` | high | Pre-SDK-style NuGet package management |
| `legacy-system-web` | `System.Web` | medium | ASP.NET Framework namespace reference |
| `legacy-system-web-mvc` | `System.Web.Mvc` | medium | ASP.NET MVC Framework reference |
| `legacy-app-start` | `App_Start` | high | MVC/WebAPI bootstrap convention folder |

### Category: `config`

| Signal ID | Query | Confidence | Meaning |
|---|---|---|---|
| `appsettings-json` | `appsettings.json` | high | ASP.NET Core configuration file |
| `connection-strings` | `connectionStrings` | medium | Connection string configuration entries |
| `app-config` | `app.config` | high | Desktop/console .NET Framework config |

### Category: `sql`

| Signal ID | Query | Confidence | Meaning |
|---|---|---|---|
| `sql-connection` | `SqlConnection` | high | Direct ADO.NET SQL Server access |
| `raw-sql-execute` | `ExecuteSql OR ExecuteSqlRaw OR FromSqlRaw OR FromSql` | medium | Raw SQL via EF Core |
| `sql-command` | `SqlCommand` | high | Raw ADO.NET command execution |

### Category: `async-risk`

| Signal ID | Query | Confidence | Meaning |
|---|---|---|---|
| `sync-over-async-result` | `.Result` | medium | Sync-over-async deadlock risk |
| `sync-over-async-wait` | `.Wait()` | medium | Sync-over-async deadlock risk |

### Category: `exception-risk`

| Signal ID | Query | Confidence | Meaning |
|---|---|---|---|
| `broad-catch-exception` | `catch (Exception` | medium | Broad catch — all exceptions swallowed |
| `empty-catch-candidate` | `catch {` | low | Catch block present — review for empty body |

### Category: `di`

| Signal ID | Query | Confidence | Meaning |
|---|---|---|---|
| `add-singleton` | `AddSingleton` | high | Singleton DI registration |
| `add-scoped` | `AddScoped` | high | Scoped DI registration |
| `add-transient` | `AddTransient` | high | Transient DI registration |
| `service-locator-getservice` | `GetService` | medium | Service locator anti-pattern |
| `service-locator-getrequiredservice` | `GetRequiredService` | medium | Service locator pattern |

### Category: `tests`

| Signal ID | Query | Confidence | Meaning |
|---|---|---|---|
| `test-project` | `.Tests OR xunit OR NUnit OR MSTest` | medium | Test project or framework present |

---

## Limitations

- Signal scan is heuristic and search-based, not compiler-accurate.
- Absence of a signal does not prove absence in the repository.
- Pattern matching may produce false positives (e.g., commented-out code, test fixtures, string literals containing a keyword).
- Multi-query signals (OR) run separate searches; matches are deduplicated by file+line.
- The pattern index must be built (`gpu_index` or `POST /` startup) before signals can be detected.

For compiler-accurate analysis, use Roslyn or a language server in the calling tool. gpu-search-mcp is a broad retrieval backend; interpretation and verification belong in the client.

---

## MCP tool

The `scan_repository_signals` MCP tool is available for Claude/Codex workflows:

```
scan_repository_signals(
    categories=["legacy-dotnet", "sql"],
    top_k_per_signal=5,
    context_mode="compact"
)
```

Returns a human-readable audit summary grouped by category.
