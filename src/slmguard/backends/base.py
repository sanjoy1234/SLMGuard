"""The model backend contract.

Non-negotiable project rule (see docs/technical-build-plan-v5.md, "Model
Backend Abstraction"): every model-layer capability the rest of the system
needs is expressed here, and nowhere else. The control plane, specialization
loop, evaluation harness, and model registry call `ModelBackend` methods —
never `mlx_lm`, never `transformers`/`peft`, never any other framework API
directly.

MLX (see mlx_backend.py) is the Phase 1 implementation of this interface,
chosen for unified memory and one-stack coverage of inference + LoRA on
Apple Silicon. It is not the architecture. CUDA + QLoRA (see
cuda_qlora_backend.py) is the confirmed production backend, implementing the
exact same contract. Adding a capability means adding it here first, then to
every backend — a backend growing a method the interface doesn't have is a
sign the abstraction is leaking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from slmguard.backends.types import (
    AdapterArtifact,
    LoadedModel,
    LoRAConfig,
    ModelVersion,
    RawModelOutput,
)
from slmguard.schema import Recommendation


class ModelBackend(ABC):
    """Abstract contract every model runtime (MLX, CUDA/QLoRA, ...) must implement."""

    name: str

    @abstractmethod
    def load_model(self, version: str) -> LoadedModel:
        """Load a model version (base or previously fused) for inference or training."""

    @abstractmethod
    def generate(self, model: LoadedModel, prompt: str) -> RawModelOutput:
        """Generate a raw response. Schema validation against `Recommendation`
        happens in the control plane, not here — a backend's only job is to
        produce text and say whether it self-reports as schema-valid."""

    @abstractmethod
    def fine_tune(self, model: LoadedModel, dataset: Path, config: LoRAConfig) -> AdapterArtifact:
        """Run a LoRA (or QLoRA) fine-tune and return the trained adapter, unfused."""

    @abstractmethod
    def fuse_adapters(self, model: LoadedModel, adapter: AdapterArtifact) -> ModelVersion:
        """Fuse a trained adapter into the base model, producing a new deployable version."""


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced top-level {...} substring in text, or None.
    Models routinely wrap valid JSON in leading/trailing chatter or run on
    past it after the intended answer; this isolates just the object so
    validation isn't defeated by tokens the schema was never meant to cover."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def validate_output(raw: RawModelOutput) -> Recommendation | None:
    """Shared helper: parse a backend's raw output into the schema, or return
    None on failure. Backend-agnostic on purpose — every backend's output goes
    through the same validation path, per the control-plane failure-mode
    contract (schema failure -> hard escalation, never a silent retry)."""
    if not raw.schema_valid:
        return None
    candidate = _extract_json_object(raw.text)
    if candidate is None:
        return None
    try:
        return Recommendation.model_validate_json(candidate)
    except Exception:
        return None
