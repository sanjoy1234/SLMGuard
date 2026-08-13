"""Single point where a backend name (from config, CLI flag, or env) becomes
a `ModelBackend` instance. Everything else in the codebase should go through
this, never `import` a specific backend module directly."""

from __future__ import annotations

from slmguard.backends.base import ModelBackend

_BACKENDS: dict[str, type[ModelBackend]] = {}


def register(name: str, backend_cls: type[ModelBackend]) -> None:
    _BACKENDS[name] = backend_cls


def get_backend(name: str) -> ModelBackend:
    if name not in _BACKENDS:
        _register_builtin_backends()
    try:
        return _BACKENDS[name]()
    except KeyError:
        available = ", ".join(sorted(_BACKENDS)) or "(none registered)"
        raise ValueError(f"Unknown model backend '{name}'. Available: {available}") from None


def _register_builtin_backends() -> None:
    from slmguard.backends.cuda_qlora_backend import CudaQLoRABackend
    from slmguard.backends.mlx_backend import MLXBackend

    register("mlx", MLXBackend)
    register("cuda_qlora", CudaQLoRABackend)
