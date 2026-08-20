"""Backend-agnostic value types passed across the Teacher boundary.

None of these types may carry a provider-specific object (an OpenAI/OpenRouter
SDK response object, an HTTP client, ...). Same rule as
slmguard.backends.types and slmguard.audit.types: plain Python values only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationSpec:
    """What to ask the teacher to produce for one example -- a diversity
    focus plus free-text guidance, never real trace or customer content."""

    diversity_tags: tuple[str, ...]
    guidance: str


@dataclass(frozen=True)
class TeacherMetadata:
    """Provenance for one generated example -- stored on every example so a
    batch's origin is always auditable, never implicit."""

    teacher_name: str
    model_id: str
    generated_at: str


@dataclass(frozen=True)
class TeacherExample:
    scenario: str
    recommendation_json: str
    diversity_tags: tuple[str, ...]
    metadata: TeacherMetadata
