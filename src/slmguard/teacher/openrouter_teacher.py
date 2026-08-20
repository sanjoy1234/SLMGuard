"""Phase 1 teacher: OpenRouter (https://openrouter.ai), a unified API
gateway to many hosted models. Chosen as the first automated teacher
backend because it needs no per-provider integration work to try different
models and has genuinely free-tier options -- not the project's
architecture. Swapping to a direct Claude API teacher or a larger local
model means implementing `Teacher` again, nothing else in `generate_data.py`
should need to change.

Uses the stdlib `urllib` rather than adding an HTTP client dependency --
this is a single JSON-in/JSON-out POST, not worth a new dependency for.

No real customer or trace data ever reaches this class: `generate_data.py`
has no `AuditStore` dependency at all, so there is nothing real for a
`GenerationSpec`'s free-text `guidance` to leak -- every prompt sent here
is a synthetic-scenario instruction, by construction, not real case data.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

from slmguard.backends.base import extract_json_object
from slmguard.teacher.base import Teacher
from slmguard.teacher.types import GenerationSpec, TeacherExample, TeacherMetadata

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

_PROMPT_TEMPLATE = """You are generating SYNTHETIC training data for a bank fraud-alert triage system. This is entirely fictional -- invent a plausible scenario, and never reference any real person, company, or account.

Diversity focus for this example: {guidance}

Write one realistic fraud-alert scenario in this style: an alert_id (format SYN-<4 digits>), card/account details (tenure, transaction frequency, any prior flags), the event itself (amount, merchant type, timing), and the channel (card-present, card-not-present, ACH, wire).

Then provide the correct triage assessment for your own scenario, using exactly this schema:
{{"alert_id": string, "action": one of ["approve","decline","escalate_l2","request_more_info"], "risk_score": float 0.0-1.0, "confidence": float 0.0-1.0, "policy_flags": array of strings, "rationale": string justifying the action using specifics from the scenario, max 1000 chars}}

Respond with ONLY a single JSON object with exactly two keys, no other text before or after it:
{{"scenario": "<the fraud alert scenario as plain text>", "recommendation": {{<the assessment JSON exactly as specified above>}}}}"""


class NotFreeModelError(ValueError):
    """Raised when a configured model_id is not confirmed free on OpenRouter.
    This project never uses a paid OpenRouter model -- enforced here, not
    just documented as a preference."""


class OpenRouterTeacher(Teacher):
    name = "openrouter"

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        verify_free: bool = True,
    ):
        self._model_id = model_id
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._timeout = timeout
        if not self._api_key:
            raise ValueError(
                "OpenRouterTeacher requires an API key: set OPENROUTER_API_KEY in the "
                "environment, or construct with api_key=... explicitly."
            )
        if verify_free:
            self._assert_model_is_free()

    def _assert_model_is_free(self) -> None:
        """Fail loud and fail closed: refuse to proceed unless the live
        catalog confirms this exact model_id costs nothing. A model that
        isn't found at all is treated the same as a paid one -- cost must be
        confirmed, never assumed."""
        pricing = _fetch_model_pricing(self._model_id, self._api_key, self._timeout)
        if pricing is None:
            raise NotFreeModelError(
                f"OpenRouterTeacher: model '{self._model_id}' was not found in the "
                "live OpenRouter catalog -- refusing to proceed. This project only "
                "ever uses confirmed-free models."
            )
        prompt_cost, completion_cost = pricing
        if prompt_cost != 0.0 or completion_cost != 0.0:
            raise NotFreeModelError(
                f"OpenRouterTeacher: model '{self._model_id}' is NOT free "
                f"(prompt={prompt_cost}/token, completion={completion_cost}/token). "
                "This project only ever uses free OpenRouter models -- refusing to proceed."
            )

    def generate(self, spec: GenerationSpec) -> TeacherExample:
        prompt = _PROMPT_TEMPLATE.format(guidance=spec.guidance)
        response_text = self._call(prompt)
        envelope_json = extract_json_object(response_text)
        if envelope_json is None:
            raise ValueError(
                f"OpenRouterTeacher: no JSON object found in model response: {response_text[:300]!r}"
            )
        envelope = json.loads(envelope_json)
        if "scenario" not in envelope or "recommendation" not in envelope:
            raise ValueError(
                "OpenRouterTeacher: model response JSON is missing 'scenario' and/or "
                f"'recommendation': {envelope_json[:300]!r}"
            )
        scenario = envelope["scenario"]
        recommendation = envelope["recommendation"]
        return TeacherExample(
            scenario=scenario,
            recommendation_json=json.dumps(recommendation),
            diversity_tags=spec.diversity_tags,
            metadata=TeacherMetadata(
                teacher_name=self.name,
                model_id=self._model_id,
                generated_at=datetime.now(timezone.utc).isoformat(),
            ),
        )

    def _call(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self._model_id,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            OPENROUTER_API_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"OpenRouter API error {exc.code}: {detail[:500]}") from exc
        return payload["choices"][0]["message"]["content"]


def _fetch_all_models(api_key: str | None, timeout: float) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(OPENROUTER_MODELS_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload.get("data", [])


def _fetch_model_pricing(
    model_id: str, api_key: str | None, timeout: float
) -> tuple[float, float] | None:
    """Return (prompt_cost, completion_cost) per token for model_id from the
    live catalog, or None if the model isn't found."""
    for m in _fetch_all_models(api_key, timeout):
        if m.get("id") == model_id:
            pricing = m.get("pricing", {})
            return float(pricing.get("prompt", "1")), float(pricing.get("completion", "1"))
    return None


def list_free_models(api_key: str | None = None, timeout: float = 30.0) -> list[dict]:
    """Query OpenRouter's live model catalog and return only entries with
    zero prompt/completion cost. Not called by `generate` itself -- this is
    a discovery helper for picking `model_id`, since OpenRouter's free-tier
    catalog changes over time and shouldn't be guessed/hardcoded."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    return [
        m
        for m in _fetch_all_models(key, timeout)
        if float(m.get("pricing", {}).get("prompt", "1")) == 0.0
        and float(m.get("pricing", {}).get("completion", "1")) == 0.0
    ]
