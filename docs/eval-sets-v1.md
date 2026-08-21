# Held-Out and Challenge Sets (v1)

Companion document to `docs/evaluation-harness-v1.md`'s held-out/challenge set construction rules and `docs/technical-build-plan-v5.md`'s framing of the held-out set as "frozen once, before any training touches the data." Covers the actual Phase 1 sets: `eval_sets/held_out_set.json` and `eval_sets/challenge_set.json`, built 2026-08-20.

## Labeling method — stated honestly

**Every `true_action` label in both files comes from `nvidia/nemotron-3-super-120b-a12b:free`'s own assessment of a scenario it also invented.** This is not:

- Human-labeled or human-reviewed. No fraud analyst, no second person, has looked at any of these cases.
- Derived from real bank policy or real historical outcomes. Every scenario is fictional, invented by the teacher model from a diversity-focused instruction (`slmguard.heldout_construction.ACTION_GUIDANCE`), never from real trace/customer data.
- Independently verified for correctness. There is no ground-truth check beyond the teacher's own reasoning — its `rationale` field is the entire justification on record.

**What it is:** the build plan's own Phase 1 approach — "the larger model produces the first synthetic scenario/label pairs used to establish the baseline specialist" (Distillation Approach). A single large model's judgment is being used as the training-and-evaluation signal, exactly as the plan describes, not something dressed up as more authoritative than that. Compared to the earlier all-Claude-authored bootstrap set (`bootstrap-v1-not-real-phase1-set`, 8 cases, used for the first specialization-loop validation), this is a genuine step up — a second, larger, independent model is now the source of ground truth rather than a single agent's own unreviewed judgment while writing the scenario — but "teacher-generated" must never be read as "verified." Any promotion decision made against these sets inherits this limitation and should be read accordingly.

**Guidance vs. ground truth, explicitly:** each generation spec asks the teacher to aim for a specific action (`ACTION_GUIDANCE`), to get roughly balanced coverage across all four classes. The teacher does not always agree with that target for the scenario it ends up writing — when it disagrees, its own decision is what gets recorded as `true_action`, never the guidance target. This is why the final action counts don't exactly mirror the per-chunk generation targets (see Construction Run below) — it's the mechanism working as designed, not drift or error.

## Construction rules met (`slmguard.evaluation.validate_held_out_set`)

| Rule | Requirement | Actual |
|---|---|---|
| Size | 150–300 | 166 |
| All 4 action classes represented | ≥1 each | approve 40, decline 39, escalate_l2 44, request_more_info 43 |
| Policy-boundary coverage | ≥1 case | 41 |

`validate_held_out_set(cases).valid == True`, `violations == ()` — the first held-out set in this project to actually pass its own structural gate, rather than being a deliberately-small bootstrap set the CLI warns about.

## Construction run (2026-08-20)

Built via `slmguard.heldout_construction.stratified_generation_specs(n_per_action=45)` → 180 specs (45 per action × 4 actions, cycling through the 4 rubric diversity categories within each action) → `slmguard.generate_data.generate_batch` per action-chunk (bounding the cost of a rubric rejection to one 45-example chunk, not the full 180) → `generated_batches_to_cases`.

| Target action (chunk) | Accepted / generated | Attempts |
|---|---|---|
| approve | 41 / 45 | 1 |
| decline | 41 / 45 | 1 |
| escalate_l2 | 41 / 45 | 1 |
| request_more_info | 43 / 45 | 1 |

166 total accepted (rubric-passing) examples, converted to `LabeledCase`s. No chunk needed a second attempt — every chunk's first-pass rubric score cleared the 85% threshold.

## Challenge set

`eval_sets/challenge_set.json` — same construction mechanism (`stratified_generation_specs`), fewer per action (5 vs. 45), and every generation spec has an added instruction to make the case "genuinely hard... include at least one signal that would tempt a less careful triage system toward the wrong action." This is the automated approximation of the plan's "hand-picked hard cases" — it is not literally hand-picked, and does not yet include any real accumulated failure cases (the plan's other named source: "any real failure found during development or evaluation") since no specialization cycle has run against real production traffic yet. `slmguard.evaluation.grow_challenge_set` exists for appending real discovered failures later, append-only, per the plan's "never silently dropped" rule — this set is v1, expected to grow, not a finished artifact.

**Construction run (2026-08-20):** 4 chunks of 5 specs each (one per target action), all 20/20 accepted on the first attempt (100% rubric pass rate). Final distribution: approve 4, decline 5, escalate_l2 6, request_more_info 5; 4 of the 20 marked `policy_boundary`.

## What "frozen" means here, and what it doesn't yet

These files are checked into the repository (`eval_sets/`, not the gitignored `data/`) precisely so they function as a frozen reference — anyone running `slmguard specialize` against them gets the same held-out/challenge content every time, and a diff shows if that ever changes. That is freezing in the *file-immutability* sense. It is not yet freezing in the *process* sense the build plan intends: there is no versioning discipline beyond the file's `version` field (`phase1-v1-teacher-generated`), no approval gate before a new version replaces this one, and no automated check preventing accidental edits. `slmguard specialize --held-out-set eval_sets/held_out_set.json` will use whatever is on disk at run time — treat committing a new version deliberately, the same discipline as any other source change, as the actual freezing mechanism for now.

---

**Status:** v1. First held-out set to structurally pass `validate_held_out_set`. Labels are single-teacher-model-generated, not human-reviewed — the honest ceiling on how much these numbers should be trusted until a human-review or multi-model-consensus step exists.
