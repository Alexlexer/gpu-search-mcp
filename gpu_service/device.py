"""Centralized torch device resolver for gpu-search-mcp.

Priority order (auto): cuda → mps → cpu.
Override via GPU_SEARCH_DEVICE env var or --device CLI flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeviceInfo:
    name: str         # "cuda" | "mps" | "cpu"
    torch_device: str # same as name for these backends
    backend: str      # same as name
    available: bool
    reason: str
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "torchDevice": self.torch_device,
            "reason": self.reason,
            "warnings": list(self.warnings),
        }


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch
        mps = getattr(torch.backends, "mps", None)
        if mps is None:
            return False
        is_built = getattr(mps, "is_built", lambda: False)
        is_avail = getattr(mps, "is_available", lambda: False)
        return bool(is_built() and is_avail())
    except Exception:
        return False


def _auto_best(*, cuda: bool, mps: bool) -> DeviceInfo:
    if cuda:
        return DeviceInfo("cuda", "cuda", "cuda", True, "CUDA available (NVIDIA GPU)")
    if mps:
        return DeviceInfo("mps", "mps", "mps", True, "Apple Silicon MPS backend available")
    return DeviceInfo("cpu", "cpu", "cpu", True, "CUDA and MPS unavailable; using CPU fallback")


def resolve_torch_device(preferred: str | None = None) -> DeviceInfo:
    """Return the best available torch device, optionally constrained by *preferred*.

    preferred: "auto" | "cuda" | "mps" | "cpu" | None  (None treated as "auto")
    Falls back gracefully with a warning when the requested backend is unavailable.
    Never raises — always returns a usable DeviceInfo.
    """
    pref = (preferred or "auto").lower().strip()

    cuda = _cuda_available()
    mps = _mps_available()

    if pref == "cuda":
        if cuda:
            return DeviceInfo("cuda", "cuda", "cuda", True, "CUDA requested and available")
        fallback = _auto_best(cuda=False, mps=mps)
        fallback.warnings.append(
            f"--device cuda requested but CUDA is not available; using {fallback.backend}"
        )
        return fallback

    if pref == "mps":
        if mps:
            return DeviceInfo("mps", "mps", "mps", True, "MPS requested and available (Apple Silicon)")
        fallback = _auto_best(cuda=cuda, mps=False)
        fallback.warnings.append(
            f"--device mps requested but MPS is not available; using {fallback.backend}"
        )
        return fallback

    if pref == "cpu":
        return DeviceInfo("cpu", "cpu", "cpu", True, "CPU forced by --device cpu")

    # "auto" or any unrecognised value
    return _auto_best(cuda=cuda, mps=mps)
