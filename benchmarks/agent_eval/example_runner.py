"""Deterministic protocol smoke adapter; not an agent-effectiveness benchmark."""
import json
from pathlib import Path
import sys

request = json.loads(sys.stdin.readline())
workspace = Path(request["workspace"])
target = "benchmarks/fixtures/csharp/src/Auth/JwtValidator.cs"
print(json.dumps({
    "type": "tool_call", "elapsed_ms": 1, "tool": "read",
    "category": "file_read", "file_paths": [target],
    "arguments": {"path": target},
}))
content = (workspace / target).read_text(encoding="utf-8")
print(json.dumps({
    "type": "tool_result", "elapsed_ms": 2, "category": "file_read",
    "file_paths": [target], "result_size_bytes": len(content.encode("utf-8")),
}))
if request["gpu_search_enabled"]:
    print(json.dumps({
        "type": "tool_call", "elapsed_ms": 3, "tool": "search_code",
        "category": "gpu_search", "file_paths": [target],
        "arguments": {"query": "JwtValidator"},
    }))
    print(json.dumps({
        "type": "tool_result", "elapsed_ms": 4, "category": "gpu_search",
        "file_paths": [target], "result_size_bytes": 256,
    }))
marker = "// agent-eval-smoke\n"
if marker not in content:
    (workspace / target).write_text(marker + content, encoding="utf-8")
print(json.dumps({
    "type": "tool_call", "elapsed_ms": 5, "tool": "edit",
    "category": "edit", "file_paths": [target],
}))
print(json.dumps({
    "type": "usage", "elapsed_ms": 6,
    "usage_semantics": "cumulative",
    "token_usage": {"input_tokens": 100, "output_tokens": 20},
}))
print(json.dumps({"type": "final", "elapsed_ms": 7, "data": {"completed": True}}))
