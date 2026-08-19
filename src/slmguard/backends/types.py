"""Backend-agnostic value types passed across the ModelBackend boundary.

None of these types may carry a framework-specific object (an mlx.array, a
torch.Tensor, a HF PreTrainedModel, ...). If a caller outside a backend
implementation needs to inspect one of these, that need has to be met with a
plain Python value here — not by reaching past the interface into the
backend's framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadedModel:
    version_id: str
    backend_name: str
    handle: Any = field(repr=False)
    """Opaque framework-specific handle. Only the owning backend may unwrap it."""


@dataclass(frozen=True)
class RawModelOutput:
    text: str
    schema_valid: bool
    parse_error: str | None = None


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: int
    epochs: int
    learning_rate: float
    batch_size: int
    max_seq_length: int
    grad_checkpoint: bool
    target_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdapterArtifact:
    path: Path
    base_version_id: str
    backend_name: str
    training_data_hash: str


@dataclass(frozen=True)
class ModelVersion:
    version_id: str
    parent_version_id: str | None
    backend_name: str
    fused_weights_path: Path
