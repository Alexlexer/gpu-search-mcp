"""Semantic embedding model configuration, preflight, and explicit download helpers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_SEMANTIC_MODEL = "BAAI/bge-small-en-v1.5"
SEMANTIC_MODEL_ENV = "GPU_SEARCH_SEMANTIC_MODEL"
PROVIDER = "sentence-transformers"


@dataclass
class SemanticModelStatus:
    modelId: str
    available: bool
    cached: bool
    provider: str = PROVIDER
    requiresDownload: bool = True
    cachePath: str | None = None
    error: str | None = None
    message: str = ""
    device: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


def _read_config_model(config_path: str | Path | None = None) -> str | None:
    if config_path is None:
        config_path = Path.home() / ".gpu-search-config.json"
    try:
        path = Path(config_path)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get("semanticModel") or data.get("semantic_model")
        return str(value).strip() if value else None
    except Exception:
        return None


def resolve_semantic_model_id(
    cli_model: str | None = None,
    config_model: str | None = None,
) -> str:
    """Resolve semantic model by priority: CLI > env > config > default."""
    if cli_model and str(cli_model).strip():
        return str(cli_model).strip()
    env_model = os.environ.get(SEMANTIC_MODEL_ENV)
    if env_model and env_model.strip():
        return env_model.strip()
    if config_model and str(config_model).strip():
        return str(config_model).strip()
    cfg = _read_config_model()
    if cfg:
        return cfg
    return DEFAULT_SEMANTIC_MODEL


def get_configured_semantic_model_id() -> str:
    return resolve_semantic_model_id()


def set_configured_semantic_model_id(model_id: str) -> None:
    os.environ[SEMANTIC_MODEL_ENV] = model_id


def _download_command(model_id: str) -> str:
    return f"gpu-search-mcp --semantic-model {model_id} --download-semantic-model"


def _status_unavailable(model_id: str, device: str | None, error: Exception | str | None = None) -> SemanticModelStatus:
    err_text = str(error) if error else None
    return SemanticModelStatus(
        modelId=model_id,
        available=False,
        cached=False,
        requiresDownload=True,
        error=err_text,
        device=device,
        message=(
            "Semantic model is not available locally. Run: "
            f"{_download_command(model_id)}"
        ),
    )


def get_semantic_model_status(model_id: str | None = None, device: str | None = None) -> dict[str, Any]:
    """Best-effort local-only model preflight. Never downloads."""
    resolved = model_id or get_configured_semantic_model_id()
    try:
        from huggingface_hub import snapshot_download

        cache_path = snapshot_download(resolved, local_files_only=True)
        return SemanticModelStatus(
            modelId=resolved,
            available=True,
            cached=True,
            requiresDownload=False,
            cachePath=str(cache_path),
            device=device,
            message="Semantic embedding model is available locally.",
        ).as_dict()
    except Exception as hub_error:
        # Fallback for environments where HuggingFace internals differ. This
        # must still be local-only to avoid network on status checks.
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(resolved, device=device or "cpu", local_files_only=True)
            cache_path = getattr(model, "cache_folder", None) or getattr(model, "_cache_folder", None)
            return SemanticModelStatus(
                modelId=resolved,
                available=True,
                cached=True,
                requiresDownload=False,
                cachePath=str(cache_path) if cache_path else None,
                device=device,
                message="Semantic embedding model is available locally.",
            ).as_dict()
        except Exception as st_error:
            return _status_unavailable(resolved, device, st_error or hub_error).as_dict()


def download_semantic_model(
    model_id: str | None = None,
    device: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Explicitly download/preload the configured sentence-transformers model."""
    resolved = model_id or get_configured_semantic_model_id()
    try:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {"device": device or "cpu"}
        if force:
            # SentenceTransformer does not expose a stable force-download knob
            # across versions. Loading without local_files_only is enough to
            # refresh/download missing artifacts when the backend supports it.
            kwargs["local_files_only"] = False
        model = SentenceTransformer(resolved, **kwargs)
        cache_path = getattr(model, "cache_folder", None) or getattr(model, "_cache_folder", None)
        return SemanticModelStatus(
            modelId=resolved,
            available=True,
            cached=True,
            requiresDownload=False,
            cachePath=str(cache_path) if cache_path else None,
            device=device or "cpu",
            message="Semantic embedding model downloaded/preloaded successfully.",
        ).as_dict()
    except Exception as exc:
        status = _status_unavailable(resolved, device or "cpu", exc)
        status.message = f"Semantic model download/preload failed: {exc}"
        return status.as_dict()
