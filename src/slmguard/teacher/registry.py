"""Single point where a teacher name (from config) becomes a Teacher
instance. Everything else in the codebase should go through this, never
import a specific teacher module directly."""

from __future__ import annotations

from slmguard.teacher.base import Teacher

_TEACHERS: dict[str, type[Teacher]] = {}


def register(name: str, teacher_cls: type[Teacher]) -> None:
    _TEACHERS[name] = teacher_cls


def get_teacher(name: str, **kwargs) -> Teacher:
    if name not in _TEACHERS:
        _register_builtin_teachers()
    try:
        return _TEACHERS[name](**kwargs)
    except KeyError:
        available = ", ".join(sorted(_TEACHERS)) or "(none registered)"
        raise ValueError(f"Unknown teacher '{name}'. Available: {available}") from None


def _register_builtin_teachers() -> None:
    from slmguard.teacher.openrouter_teacher import OpenRouterTeacher

    register("openrouter", OpenRouterTeacher)
