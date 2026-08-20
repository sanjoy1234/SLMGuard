"""The teacher contract: a model more capable than the 3B specialist,
producing synthetic scenario/label pairs for `generate-data`, per
docs/technical-build-plan-v5.md's Distillation Approach ("the larger model
produces the first synthetic scenario/label pairs used to establish the
baseline specialist").

Non-negotiable project rule (enterprise-grade mandate, same pattern as
ModelBackend/AuditStore): OpenRouter (see openrouter_teacher.py) is the
Phase 1 implementation of this interface, not the architecture. Swapping to
a different provider (a direct Claude API teacher, a larger local MLX
model) means implementing this same interface, never leaking a
provider-specific request/response shape into `generate_data.py` or
anywhere else. Nothing outside a concrete Teacher implementation may import
a provider SDK or make an HTTP call to a specific provider's endpoint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from slmguard.teacher.types import GenerationSpec, TeacherExample


class Teacher(ABC):
    """Abstract contract every teacher model (OpenRouter, Claude API, a
    larger local model, ...) must implement."""

    name: str

    @abstractmethod
    def generate(self, spec: GenerationSpec) -> TeacherExample:
        """Produce one scenario/label pair for the given generation spec."""
