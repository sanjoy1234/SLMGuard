# Policy Rules Engine (v1)

Companion document to `docs/technical-build-plan-v5.md`, "Control Plane Failure Modes" — "Model output is schema-valid but violates a policy rule → The policy engine overrides the model's recommendation... The model is never trusted merely because its output parsed." Implementation: `src/slmguard/policy.py`, applied by the control plane (`run-baseline`, and every evaluation call inside the specialization loop) immediately after schema validation succeeds.

## What "declarative, versioned" means here

A rule is data, not code: a `PolicyRule` is a `rule_id`, a `kind` (from a small, fixed, reviewable `RuleKind` enum), and a numeric `threshold`. Adding or tuning a rule means editing a `PolicyRuleSet` literal — never writing new control flow, and never an `eval()`/expression parser. The rule *space* is deliberately closed; only the rule *set* (which rules are active, at what thresholds) is data.

`PolicyRuleSet.version` is the versioning mechanism — every policy decision records exactly which version produced it (`PolicyDecision.policy_version`), so a rule-set change is auditable after the fact: "this recommendation was checked against `policy-v1`," not just "checked against policy."

**Not behind a swappable backend interface** like `ModelBackend`/`AuditStore`. There's no local-dev-vs-production infrastructure split to abstract over — this is pure computation over a `Recommendation`'s own fields, with no hardware or service dependency. Versioning, not a backend swap, is what makes it auditable over time. Documented here as a deliberate choice, not an oversight of the pattern used elsewhere.

## The v1 rule set (`DEFAULT_POLICY_RULESET`, version `policy-v1`)

| rule_id | kind | threshold | fires when |
|---|---|---|---|
| `forced_escalation_low_confidence` | `CONFIDENCE_BELOW_FORCES_ESCALATION` | 0.5 | `confidence < 0.5` and the model's action isn't already `escalate_l2` |
| `no_silent_approve_high_risk` | `HIGH_RISK_CANNOT_APPROVE` | 0.8 | `risk_score >= 0.8` and action is `approve` |
| `no_low_risk_decline` | `LOW_RISK_CANNOT_DECLINE` | 0.3 | `risk_score < 0.3` and action is `decline` |

Any rule firing forces `final_action = escalate_l2` and is recorded in `violated_rule_ids` — multiple simultaneous violations are all reported, not just the first found. **Policy overrides only ever route to escalation, never to approve or decline** — a rule firing means the model's own call can't be trusted as-is, and the fix is a human/L2 reviewer, never a different automated answer substituted by the policy layer itself.

## A known, honestly-stated limitation

Rules only see the `Recommendation`'s own fields (`action`, `risk_score`, `confidence`) — there is no structured `Alert` schema yet (transaction type, channel, amount, country, ...), so the build plan's own example rule, "certain transaction types always escalate regardless of model output," **isn't implementable yet**. That needs structured intake this project doesn't have; today, every alert is a free-text prompt, not a typed record.

`policy_flags` on the `Recommendation` (the model's own self-reported claims, e.g. `"high_value_transaction"`) are deliberately **not** used as a rule input, even though they look like exactly the kind of structured signal a rule could key off. A model motivated to evade a flag-based rule could simply omit the flag — self-reported claims from the system being policed aren't a sound basis for policing it. Rules are restricted to `risk_score`/`confidence`/`action`, fields with schema-enforced numeric ranges the model can't just leave out.

## Where this is wired in

- **`run-baseline`** (`cli.py`): after `validate_output` succeeds, `apply_policy(recommendation)` runs; the CLI prints a `POLICY OVERRIDE (...)` line whenever it fires, and the audit trace records `policy_version`, `policy_overridden`, and `policy_violated_rule_ids` (JSON-encoded list) alongside the existing fields. `TraceRecord.final_action` is now genuinely the control plane's decision — post-policy — not a mirror of the model's raw claim.
- **`slmguard.specialization.evaluate_case`**: `EvaluatedCase.policy_violated` is `apply_policy(recommendation).overridden` — real, not hardcoded `False`. `predicted_action` deliberately stays the model's *raw* action even when policy overrides it: the accuracy/precision/recall metrics exist to measure whether the *model* improved, and inflating that number via a policy bailout would mask whether specialization is actually working. `policy_violation_rate` (from `docs/evaluation-harness-v1.md`) is exactly "share of recommendations that would have violated policy had the control plane not intervened" — `apply_policy`'s `overridden` flag on the raw recommendation is precisely that measurement.
- **`slmguard.specialization.select_traces`**: now includes policy-overridden traces as a selection criterion, the third of the build plan's three step-2 criteria (low-confidence, escalated, policy-overridden) — the audit lineage carries `policy_overridden` per trace now, so this was a real gap that's now closed, not a leftover placeholder.
- **`slmguard.specialization.convert_trace`**: a policy-overridden trace's training target is the control plane's actual decision (`trace.final_action`), never the model's raw `recommendation_json` — that raw claim is exactly the rule-violating output that got overridden. Training toward it would teach the model to repeat the mistake the policy engine exists to catch. This is a real correctness fix, found while wiring policy-overridden traces into selection for the first time, not a hypothetical concern.

## Audit schema note

Adding `policy_version`/`policy_overridden`/`policy_violated_rule_ids` to `TraceRecord` changed the hash-chained schema `SQLiteAuditStore` writes. Since the hash chain covers every field, a mid-flight field-set change breaks `verify_chain()` for rows written under the old schema — recomputing an old row's hash under the new field set will never match what was actually stored. The pre-policy `data/audit.db` (8 traces from the specialization-loop validation) was renamed to `data/audit_v1_pre_policy_2026-08-20.db.bak` and preserved, not deleted; a fresh `data/audit.db` starts clean under the new schema. This is consistent with SQLite's documented role here as the *local-development* backend, not a production ledger — a real production deployment on the PostgreSQL backend would need an explicit, tested migration path for a schema change like this, not a file rename. That migration path doesn't exist yet and isn't needed until `PostgresAuditStore` is implemented for real.

---

**Status:** v1. Rules operate only on the `Recommendation`'s own numeric fields, which is the actual limitation to know about — not a placeholder, a real ceiling until a structured `Alert` intake schema exists.
