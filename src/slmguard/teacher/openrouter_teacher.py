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


class OpenRouterTeacher(Teacher):
    name = "openrouter"

    def __init__(self, model_id: str, api_key: str | None = None, timeout: float = 60.0):
        self._model_id = model_id
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._timeout = timeout
        if not self._api_key:
            raise ValueError(
                "OpenRouterTeacher requires an API key: set OPENROUTER_API_KEY in the "
                "environment, or construct with api_key=... explicitly."
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


def list_free_models(api_key: str | None = None, timeout: float = 30.0) -> list[dict]:
    """Query OpenRouter's live model catalog and return only entries with
    zero prompt/completion cost. Not called by `generate` itself -- this is
    a discovery helper for picking `model_id`, since OpenRouter's free-tier
    catalog changes over time and shouldn't be guessed/hardcoded."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    request = urllib.request.Request(OPENROUTER_MODELS_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return [
        m
        for m in payload.get("data", [])
        if float(m.get("pricing", {}).get("prompt", "1")) == 0.0
        and float(m.get("pricing", {}).get("completion", "1")) == 0.0
    ]
