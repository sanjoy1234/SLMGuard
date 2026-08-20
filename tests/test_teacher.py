"""Proves the Teacher interface holds and OpenRouterTeacher actually parses
what a real OpenRouter chat-completions response looks like -- request
construction, JSON-envelope extraction even with wrapping chatter, clear
failures (missing API key, HTTP error, malformed response) rather than
silent bad data, and the hard "never a paid model" construction-time guard
(rejects paid models and unknown model ids, accepts confirmed-free ones).
No live network call -- urlopen is mocked throughout."""

from __future__ import annotations

import json
import urllib.error

import pytest

from slmguard.teacher import Teacher, get_teacher
from slmguard.teacher.openrouter_teacher import NotFreeModelError, OpenRouterTeacher
from slmguard.teacher.types import GenerationSpec


def _openrouter_response(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")


def _envelope_json(scenario="A synthetic scenario.", action="approve") -> str:
    return json.dumps(
        {
            "scenario": scenario,
            "recommendation": {
                "alert_id": "SYN-0001",
                "action": action,
                "risk_score": 0.2,
                "confidence": 0.9,
                "policy_flags": [],
                "rationale": "Synthetic rationale for teacher-generated example.",
            },
        }
    )


class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_openrouter_teacher_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires an API key"):
        OpenRouterTeacher(model_id="some/model")


def test_openrouter_teacher_parses_response_and_attaches_metadata(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: _FakeHTTPResponse(
            _openrouter_response(_envelope_json())
        ),
    )
    teacher = OpenRouterTeacher(model_id="some/model", api_key="test-key", verify_free=False)

    result = teacher.generate(GenerationSpec(diversity_tags=("edge_case",), guidance="test"))

    assert result.scenario == "A synthetic scenario."
    parsed = json.loads(result.recommendation_json)
    assert parsed["action"] == "approve"
    assert result.diversity_tags == ("edge_case",)
    assert result.metadata.teacher_name == "openrouter"
    assert result.metadata.model_id == "some/model"
    assert result.metadata.generated_at  # non-empty ISO timestamp


def test_openrouter_teacher_extracts_json_despite_wrapping_chatter(monkeypatch):
    wrapped = "Sure, here you go:\n" + _envelope_json() + "\nHope that helps!"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: _FakeHTTPResponse(_openrouter_response(wrapped)),
    )
    teacher = OpenRouterTeacher(model_id="some/model", api_key="test-key", verify_free=False)

    result = teacher.generate(GenerationSpec(diversity_tags=(), guidance="test"))

    assert result.scenario == "A synthetic scenario."


def test_openrouter_teacher_raises_clearly_on_unparseable_response(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: _FakeHTTPResponse(
            _openrouter_response("not json at all")
        ),
    )
    teacher = OpenRouterTeacher(model_id="some/model", api_key="test-key", verify_free=False)

    with pytest.raises(ValueError, match="no JSON object found"):
        teacher.generate(GenerationSpec(diversity_tags=(), guidance="test"))


def test_openrouter_teacher_raises_clearly_on_http_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=__import__("io").BytesIO(b"invalid api key"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    teacher = OpenRouterTeacher(model_id="some/model", api_key="bad-key", verify_free=False)

    with pytest.raises(ValueError, match="OpenRouter API error 401"):
        teacher.generate(GenerationSpec(diversity_tags=(), guidance="test"))


def test_teacher_registry_resolves_openrouter_by_name(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    teacher = get_teacher("openrouter", model_id="some/model", verify_free=False)
    assert isinstance(teacher, Teacher)
    assert isinstance(teacher, OpenRouterTeacher)


def test_teacher_registry_unknown_name_raises_with_available_list():
    with pytest.raises(ValueError, match="openrouter"):
        get_teacher("does-not-exist")


def _models_response(models: list[dict]) -> bytes:
    return json.dumps({"data": models}).encode("utf-8")


def test_construction_rejects_a_paid_model(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: _FakeHTTPResponse(
            _models_response(
                [{"id": "some/paid-model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]
            )
        ),
    )
    with pytest.raises(NotFreeModelError, match="is NOT free"):
        OpenRouterTeacher(model_id="some/paid-model", api_key="test-key")


def test_construction_rejects_a_model_not_found_in_the_catalog(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: _FakeHTTPResponse(_models_response([])),
    )
    with pytest.raises(NotFreeModelError, match="was not found"):
        OpenRouterTeacher(model_id="some/nonexistent-model", api_key="test-key")


def test_construction_accepts_a_confirmed_free_model(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: _FakeHTTPResponse(
            _models_response(
                [{"id": "some/free-model", "pricing": {"prompt": "0", "completion": "0"}}]
            )
        ),
    )
    teacher = OpenRouterTeacher(model_id="some/free-model", api_key="test-key")
    assert teacher._model_id == "some/free-model"
