# Evaluation Harness (v1)

Companion document to `docs/technical-build-plan-v5.md`, "Evaluation Harness (fully specified)" and "Promotion Gate." Makes those sections precise and executable: exact metric formulas, exact construction/freezing rules for the held-out and challenge sets, and an explicit mapping from computed metrics to promotion-gate outcomes, including "no promotion this cycle."

Scope note: this defines the harness and implements everything about it that's mechanically computable (`slmguard.evaluation`) — it does **not** generate the held-out or challenge set's actual case content, run a specialization cycle, or implement the policy engine. Those depend on `generate-data` and the policy engine, neither of which exist yet.

---

## 1. Metrics — precise definitions

All metrics are computed over a set of `EvaluatedCase` records: a held-out or challenge-set case's ground-truth `true_action`, a candidate model's `predicted_action` and `confidence`, and a `policy_violated` flag supplied by the (future) policy engine's check of that specific prediction against the case.

**Overall accuracy**
`accuracy = (# cases where predicted_action == true_action) / total cases`

**Per-class precision / recall** (4-way: `approve`, `decline`, `escalate_l2`, `request_more_info`)
For class `c`: `precision(c) = TP(c) / (TP(c) + FP(c))`, `recall(c) = TP(c) / support(c)` where `support(c)` is the number of cases whose true action is `c`. Undefined (zero-denominator) values are reported as `0.0`, not dropped — a class with no support or no correct predictions is a real signal, not a gap in the report. `decline` and `escalate_l2` are safety-critical: a miss there (especially recall) matters more than the overall accuracy number and must be read on its own, never averaged away.

**Policy violation rate**
`policy_violation_rate = (# evaluated cases with policy_violated=True) / total cases`
This measures what *would have* reached the customer/case owner had the control plane not intervened — it does not measure the control plane's own override behavior, which is a separate operational metric outside this harness.

**Confidence calibration — Expected Calibration Error (ECE)**
Confidence scores are bucketed into `M=10` equal-width bins over `[0, 1]`. For bin `b` with `|b|` cases: `acc(b)` = accuracy within the bin, `conf(b)` = mean confidence within the bin. `ECE = Σ_b (|b| / n) · |acc(b) − conf(b)|`. The binned `(range, avg_confidence, accuracy, count)` table is retained as the reliability diagram — the primary view; ECE is the single-number summary of it.

**Confidence–correctness correlation (secondary, simpler signal)**
Point-biserial correlation between `confidence` (continuous) and `correct` (binary 0/1) across all cases — i.e. the Pearson correlation coefficient treating `correct` as 0/1. Degenerate case (zero variance in either series, e.g. every prediction correct or every prediction wrong) is reported as `0.0` — no signal, not an error.

## 2. Held-out set — construction rules

- **Frozen once, before any training touches the data**, and never regenerated or silently replaced for the life of Phase 1. A new held-out set is a new version, not a mutation of the old one.
- **Size:** 150–300 cases, drawn from the 800–1,500-example total dataset.
- **Stratification:** every one of the 4 action classes must have at least one case.
- **Policy-boundary coverage:** at least one case must be marked `policy_boundary=True` (a scenario sitting exactly on a forced-escalation or hard-decline rule) — a held-out set of only easy/representative cases invalidates itself.
- **Source:** only rubric-passing data (`slmguard.rubric.score_batch(...).accepted`), per the Synthetic Data Quality Rubric.

`slmguard.evaluation.validate_held_out_set(cases)` enforces the first three rules mechanically and returns every violation found, not just the first — a set can be rejected before it's ever frozen.

## 3. Challenge / regression set — construction rules

- **Curated, not sampled:** hand-picked hard cases plus every real failure found during development or evaluation.
- **Append-only.** A case is never silently dropped once added — that would mask a known failure mode. Growing the set is a union on `alert_id`, never a replacement.
- **Only grows over time**, across cycles — there is no "prune the challenge set" operation in this harness by design.

`slmguard.evaluation.grow_challenge_set(current, new_cases)` implements this as an actual operation: it returns a new `ChallengeSet` with de-duplicated new cases unioned in, and there is deliberately no function anywhere in the module that removes a case from one.

## 4. "No regression" — the fixed threshold

Given the held-out set's small size (150–300 cases), a statistical significance test would produce wide, not-very-actionable confidence intervals. Phase 1 instead uses a fixed, explicit absolute-drop threshold, defined in code rather than left implicit:

`slmguard.evaluation.PromotionThresholds.max_accuracy_drop_pct = 2.0` (percentage points) — a candidate's accuracy on the frozen held-out set must not fall more than 2 points below the current production baseline's accuracy on that same set. This is a conservative starting default for small-sample Phase 1, expected to be replaced by a statistically rigorous comparison once evaluation volume grows in Phase 2 — not a permanent design commitment.

## 5. Promotion gate — mapping table

| Gate | Criterion | Computed by | Blocking from cycle 1? |
|---|---|---|---|
| Accuracy | Candidate accuracy ≥ baseline accuracy − `max_accuracy_drop_pct` | `evaluation.summarize` (accuracy) vs stored baseline | Yes |
| Policy / safety compliance | `policy_violation_rate == 0.0` — zero tolerance | `evaluation.summarize` (policy_violation_rate) | Yes |
| Confidence calibration | ECE tracked always; becomes a hard gate (`ece ≤ max_ece`) only once `PromotionThresholds.ece_hard_gate=True` | `evaluation.summarize` (ece) | No — monitored only, until a stable calibration baseline exists across cycles. Flipping `ece_hard_gate` on is a manual, later decision, not automated here. |
| Regression / challenge set | Zero new failures on the challenge set | Caller runs the candidate against the challenge set, counts failures | Yes |

`slmguard.evaluation.evaluate_promotion(candidate, baseline, challenge_set_new_failures, thresholds)` runs all four gates and returns a `PromotionDecision` with `promoted: bool`, a populated `reason` string, and every individual `GateResult` — never just a bare boolean. **"No promotion this cycle" is the explicit, always-populated outcome whenever any gate fails** — it is a first-class return value, not the absence of one. (A *separate*, loop-level reason for no-promotion — "this cycle didn't produce enough new specialization signal to bother evaluating" — is a decision the specialization loop makes before it would even call `evaluate_promotion`; that loop doesn't exist yet, so it's out of scope here, but the gate machinery it will eventually call already exists and is already tested.)

---

**Status:** v1. Implements the metric/threshold/gate machinery ahead of the specialization loop and `generate-data`, mirroring how `slmguard.rubric` was built before bulk generation existed — testable now against small hand-built case sets, ready to run against real data once the pipeline that produces it exists.
