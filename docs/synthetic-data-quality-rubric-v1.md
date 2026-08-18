# Synthetic Data Quality Rubric (v1)

Companion document to `docs/technical-build-plan-v5.md`, "Synthetic Data Quality Process." Defines what makes a synthetic fraud-alert scenario + label pair "good enough" to train on, before bulk generation via `generate-data` begins. A human-reviewed sample alone is not sufficient — this rubric plus an explicit rejection process (below) is the gate.

Applies to both the initial teacher-generated dataset and every subsequent specialization-cycle batch converted from audit traces.

---

## 1. Quality dimensions

An example (a scenario prompt paired with a target `Recommendation`) passes only if all four hold:

1. **Realistic feature combination.** The scenario's transaction/customer details form a plausible real-world pattern — no combinations that couldn't co-occur (e.g. "3 years of transaction history" with "account opened yesterday").
2. **Correct, internally consistent ground-truth label.** The target `Recommendation` is schema-valid, and the `action`, `risk_score`, and `confidence` are consistent with each other and with the scenario (a `decline` paired with a low `risk_score` and no stated justification is inconsistent).
3. **No contradictory or degenerate cases.** The scenario doesn't contain self-contradicting facts, and isn't a trivial/degenerate stub (near-empty, templated filler, or a copy of another accepted example with only the amount changed).
4. **Rationale actually justifies the action.** The `rationale` field references the specific scenario details that drove the decision — not a generic boilerplate sentence that could apply to any alert.

Dimensions 1, 2, and 4 require semantic judgment (human or teacher-model review). Dimension 2's schema-validity and internal-consistency checks, and dimension 3's degenerate/PII-leak checks, are mechanically enforced — see [Automated checks](#3-automated-checks-slmguardrubric).

## 2. Minimum diversity checklist

A batch is diverse enough to train on only if it covers, at minimum:

- **Transaction types and channels** — card-present, card-not-present, ACH/wire, mixed channel sequences (e.g. test-charge-then-large-purchase).
- **Customer/account segments and risk profiles** — long-tenure low-risk, new account, prior-flagged account, high-transaction-volume account.
- **Edge cases** — near-threshold amounts, first-time-pattern transactions, conflicting signals (e.g. high amount but strong account history).
- **Policy-boundary cases** — scenarios sitting exactly on a forced-escalation or hard-decline rule, once the policy engine exists to define those rules.

A batch missing coverage in any category is not automatically rejected outright, but must be flagged and supplemented before it's merged into the specialization pool or the frozen held-out set — both are drawn from rubric-passing *and* diversity-covering data, since a skewed dataset silently invalidates every downstream accuracy number.

## 3. Rejection process

- Every generated batch is scored against this rubric — not spot-reviewed in isolation.
- **Pass rate threshold: 85%** of examples in a batch must pass all four quality dimensions (`slmguard.rubric.DEFAULT_PASS_RATE`). Below that, the batch is rejected and regenerated, never folded in as-is or patched example-by-example.
- Spot-review of accepted batches still happens, but acceptance is gated by the rubric score first — review is a secondary check, not the primary gate.
- Rejected batches and their failure reasons are worth logging once `generate-data` exists, so recurring failure patterns (e.g. the teacher model consistently producing inconsistent low-risk declines) can be fed back into the generation prompt rather than rediscovered every cycle.

## 4. Automated checks (`slmguard.rubric`)

Implemented now, ahead of `generate-data`, because they're pure functions testable without a teacher model or real data:

- Schema validity of the target `Recommendation` (reuses `slmguard.schema.Recommendation`).
- Degenerate-text detection: scenario and rationale minimum length, rationale-is-not-a-copy-of-the-scenario.
- PII-leakage pattern detection: unmasked card numbers, SSNs, email addresses, phone numbers — synthetic scenarios must use placeholder-style identifiers only, never anything that looks like a real one.
- Batch-level diversity coverage against the checklist in §2, via caller-supplied `diversity_tags` per example.

Not automated, and not expected to be: whether a scenario is *realistic* or whether a label is *correct* for that scenario (dimensions 1, 2's consistency, and 4's justification quality) — those need a human or teacher-model judge. The automated checks catch structurally broken or leaking examples before they ever reach that review step; they are a floor, not a replacement for judgment.

---

**Status:** v1, written before any bulk generation has run. Expected to be revised after the first specialization cycle produces real batches to test it against — see `docs/technical-build-plan-v5.md`, "Immediate Next Steps Before Implementation," item 1.
