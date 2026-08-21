# Specialization Loop (v1)

Companion document to `docs/technical-build-plan-v5.md`, "Specialization Loop." First implementation: `src/slmguard/specialization.py`, wired into the CLI as `slmguard specialize`. Scoped tight on purpose, per explicit instruction: one working loop path, no extra platform features.

## The six steps, as actually implemented

1. **Select traces** (`select_traces`) — pulls recent traces from the `AuditStore` and filters to the ones that matter: schema failures (escalated cases) and schema-valid cases below a confidence threshold.
2. **Convert to training examples** (`convert_trace`, `build_specialization_pool`) — a trace with a valid label becomes a `SyntheticExample` (scenario redacted for PII, paired with its recorded recommendation); a schema-failure trace has no valid label to imitate and is counted, not converted. Every candidate then goes through `slmguard.rubric.score_batch` — only rubric-passing examples enter the pool, same discipline the plan requires for the held-out set.
3. **Fine-tune with safe LoRA settings** (`write_training_dataset` + `ModelBackend.fine_tune`) — the pool is split into `train.jsonl`/`valid.jsonl` and handed to the configured backend, using the `batch_size≤2` / `grad_checkpoint=true` defaults validated in `docs/lora-memory-footprint-spike-v1.md`.
4. **Fuse** (`ModelBackend.fuse_adapters`) — merges the trained adapter into a new deployable model version.
5. **Evaluate** (`evaluate_model`, `evaluate_challenge_set`) — runs both the pre-fine-tune base model and the fused candidate against the same held-out set (a fair, cycle-local baseline — there's no persisted prior-cycle baseline yet, since this is cycle one) and against the challenge set. The challenge-set check measures *true regression*: a case the baseline already gets wrong isn't blamed on the candidate — see "Fix: challenge-set regression semantics" below.
6. **Promote** (`evaluate_promotion`, unchanged from `docs/evaluation-harness-v1.md`) — the four-gate decision, returning an explicit `PromotionDecision` where "no promotion this cycle" is a first-class, reasoned outcome.

`run_cycle` wires all six into one function and always returns a `CycleResult` with a populated `reason` — whether it ran the full pipeline or stopped early.

## Deliberate, documented simplifications

These are gaps against the full plan, flagged here rather than glossed over — each has a clear reason tied to something not yet built:

- **No policy-override selection.** The plan's trace-selection criteria include "cases where the policy engine overrode the model." There is no policy engine yet, so this criterion isn't selectable. Only low-confidence and schema-failure (escalated) traces are selected.
- **Schema-failure traces are surfaced, not converted.** The plan's real answer for an escalated case is a larger model or human resolving it, and that resolution becoming the training example. That teacher-model/human-review path doesn't exist yet, so a schema-failure trace has no correct answer to train toward — it's counted (`SpecializationPool.traces_unconvertible`) and skipped, never silently dropped or converted into a self-imitating example.
- **Privacy-safe conversion is mechanical, not LLM-based.** `redact_pii` reuses `slmguard.rubric.PII_PATTERNS` (SSN/card-number/email/phone regexes), not the "careful rewriting" the plan names as the eventual Phase 1 approach. Matches the plan's own admission that this step is "careful rewriting plus review... not yet a formal privacy guarantee" — the mechanism just isn't LLM-based yet.
- **`policy_violated` is always `False`.** There's no policy engine to compute it for real; the evaluation harness's policy-compliance gate is exercised mechanically but is not yet a meaningful signal.
- **`run_cycle` trusts its `HeldOutSet`/`ChallengeSet` inputs.** It does not call `slmguard.evaluation.validate_held_out_set` itself — freezing and validating a held-out set is a separate, one-time curation step (not yet its own command) that should happen before a set is ever handed to this loop, not be re-checked on every cycle run. The CLI (`slmguard specialize`) does call it and prints a warning if the supplied set doesn't meet the frozen-set requirements, so non-compliant (e.g. bootstrap/demo) sets are never used silently.
- **No cycle-level audit record.** Individual model recommendations are still traced via the existing `AuditStore` (each evaluation call against held-out/challenge cases is not itself logged, to avoid polluting the trace store meant for real triage decisions), but there's no new "cycle outcome" table. Out of scope for this pass — an explicit "extra platform feature" the instruction said to avoid.

## Two real bugs found and fixed while wiring this for real

Both were pre-existing gaps in `MLXBackend`, invisible until something actually called `fine_tune` with a real directory-shaped dataset end to end:

1. **`_hash_file` couldn't hash a dataset directory.** `mlx_lm.lora --data` requires a directory of `{train,valid,test}.jsonl` (confirmed in the memory-footprint spike), but `fine_tune`'s lineage-hash helper opened `dataset` as a single file — `IsADirectoryError` on first real use. Replaced with `_hash_dataset_dir`, which hashes every file in the directory (sorted, for determinism).
2. **A plain 80/20 train/valid split could leave too few validation examples.** `mlx_lm.lora` requires the validation set to have at least `batch_size` examples; a small pool's 80/20 split routinely left exactly 1, which crashed training (`ValueError: Dataset must have at least batch_size=2 examples but only has 1`) on the very first real cycle run. `write_training_dataset` now takes `min_valid_size` (set to the LoRA config's `batch_size` by `run_cycle`) and falls back to using the same examples for both train and valid when the pool is too small to split properly — a documented small-pool bootstrap tradeoff, not real held-out validation, but a working loop instead of a crash.

`MLXBackend.fine_tune` also now passes `--val-batches -1` (use the entire validation set) rather than mlx_lm's own batching default, for the same small-pool robustness reason.

## Fix: challenge-set regression semantics (2026-08-20)

A rigorous validation pass (real fine-tune, real fuse, real independent re-checks) surfaced a third real bug: the `regression_challenge_set` promotion gate is named for *regression* — a case the baseline got right that the candidate now gets wrong — but `count_challenge_failures` (the original implementation) only ever evaluated the candidate and counted its absolute failures. In the validated run, both the base model and the candidate got both challenge cases wrong; that's not a regression, it's a pre-existing gap neither model version closes, but the old code blocked promotion on it exactly as if the candidate had broken something that used to work.

Fixed: `count_challenge_failures` is replaced by `evaluate_challenge_set`, which runs **both** the baseline and the candidate against the challenge set and returns a `ChallengeSetReport` with:
- `new_failures` (primary signal fed into `evaluate_promotion`) — cases the baseline got right that the candidate gets wrong.
- `baseline_failures` / `candidate_failures` (secondary, absolute detail) — kept and surfaced (CLI prints both) because a candidate that's simply bad on the challenge set is still worth seeing, even when none of its failures are technically "new."

The gate's own name and criterion in `docs/evaluation-harness-v1.md` ("no new failures on the curated set") were already correct — the bug was entirely in how `specialization.py` computed the integer it fed into that already-correct gate. Locked in by three new tests in `tests/test_specialization.py`, including one that reproduces the exact validated scenario (both models fail the same case) and asserts promotion is no longer blocked by it.

## First real cycle result (2026-08-20)

Run against the real MLX backend and real audit store (no mocks), via:

```
slmguard specialize --held-out-set <bootstrap set, 8 cases> \
                     --challenge-set <bootstrap set, 2 cases> \
                     --confidence-threshold 0.95
```

The held-out/challenge sets used were a small, hand-built **bootstrap set** (`version: "bootstrap-v1-not-real-phase1-set"`) — not the real, frozen 150–300-case Phase 1 held-out set, which still depends on `generate-data`. The CLI correctly printed a warning that the set doesn't meet `validate_held_out_set`'s size requirement and proceeded anyway, exactly as designed.

Result:

```
Pool: 7 rubric-passing example(s) from 8 selected trace(s) (8 scanned, 1 unconvertible).
Baseline accuracy: 0.500
Candidate accuracy: 0.500
Decision: no promotion this cycle: failed gate(s): regression_challenge_set
```

Every step executed against real infrastructure: 8 real traces (generated via real `run-baseline` calls) scanned and filtered, 7 converted and rubric-passed, a real `mlx_lm.lora` fine-tune ran for 3 iterations (train loss 2.835, val loss 2.911 → 2.730), a real `mlx_lm.fuse` merge produced a candidate model, both the base and candidate models were evaluated against the same 8-case held-out set and 2-case challenge set, and the promotion gate withheld promotion — a legitimate, well-reasoned "no promotion this cycle," not a crash or a short-circuit on missing infrastructure. The audit chain (`verify_chain()`) remained intact throughout.

**Correction, found during a rigorous follow-up validation pass:** at the time, this was reported as "withheld because of a challenge-set regression." Independently re-checking both models against each challenge case afterward showed the base model failed both cases too — so this was never a true regression, it was the pre-existing `count_challenge_failures` bug described above (fixed the same day).

Re-ran the identical cycle after the fix, same audit store, same held-out/challenge sets. Training reproduced bit-for-bit (same train loss 2.835, same adapter/fused-model weight hashes — `mlx_lm.lora`'s default `seed=0` makes the whole pipeline deterministic given unchanged inputs). The promotion outcome changed:

```
promoted: True
reason: all gates passed
  gate=accuracy                 passed=True  detail=candidate=0.5000 baseline=0.5000 max_drop=0.0200
  gate=policy_safety_compliance passed=True  detail=policy_violation_rate=0.0000 (zero tolerance)
  gate=confidence_calibration   passed=True  detail=ece=0.4062 (monitored only, not yet a hard gate)
  gate=regression_challenge_set passed=True  detail=new_failures=0
```

Worth being explicit about *why* this now promotes despite candidate accuracy not improving (0.500 vs. 0.500, unchanged): the accuracy gate is a **no-regression** check (`candidate >= baseline - max_drop`), not a **must-improve** check, and both challenge cases were already-broken for the baseline too, so there's genuinely nothing here for the gate to catch. That's the gate behaving exactly as specified, not a new bug — but it's also a fair signal that a tie isn't the same as progress, and "promoted" alone shouldn't be read as "this cycle made the model better." A production system would want that distinction surfaced more sharply than the current gate does (e.g. a strict-improvement mode, or a minimum-improvement threshold) — noted as a candidate follow-up, not built here.

That the candidate didn't clearly beat the baseline (0.500 vs. 0.500) is expected and not concerning on its own: 7 examples over 3 iterations with `num_layers=4` is a minimal smoke-scale run, not a real specialization attempt — the plan's own "no promotion" outcome exists precisely for cycles like this one.

---

## Second real cycle: on the improved path, against the real eval sets (2026-08-20)

Priority 4 of the post-validation build order: re-run the cycle end to end with everything the earlier passes fixed — corrected challenge-set semantics, the policy engine wired into selection/conversion/evaluation, and, most importantly, the real `eval_sets/held_out_set.json` (166 teacher-labeled, stratified cases, `validate_held_out_set`-passing) and `eval_sets/challenge_set.json` (20 cases) in place of the earlier 8-case Claude-authored bootstrap set.

Ran via `slmguard specialize`-equivalent orchestration (`run_cycle` called directly, same code path) against the real audit store (8 fresh real traces from `run-baseline`, all rubric-passing) and the real MLX backend. This meant 166 + 166 (baseline + candidate held-out evaluation) + 20 + 20 (baseline + candidate challenge evaluation) = 372 real inference calls, plus a real fine-tune and fuse — roughly 32 minutes wall-clock to baseline-eval-through-fuse, ~70 minutes total. Every filesystem artifact (`dataset/{train,valid}.jsonl`, `adapters/.../adapters.safetensors`, `adapters/fused/.../model.safetensors`) confirmed present afterward; the fused model was independently reloaded in a fresh process and generated correctly; the audit chain (`verify_chain()`) remained intact.

```
Pool: 8/8 rubric-passing examples, 8 traces scanned, 0 unconvertible.
Baseline accuracy:  0.3434 (n=166)
Candidate accuracy: 0.3373 (n=166)
Challenge set: 16/20 failed for both baseline and candidate; 0 new failures.
Decision: promoted, all gates passed.
  gate=accuracy                 candidate=0.3373 baseline=0.3434 max_drop=0.0200 -- passed
  gate=policy_safety_compliance policy_violation_rate=0.0000 (zero tolerance)    -- passed
  gate=confidence_calibration   ece=0.4548 (monitored only, not a hard gate)     -- passed
  gate=regression_challenge_set new_failures=0                                  -- passed
```

**The headline number is sobering, and that's the point of using a real, adequately-sized set.** Both the 8-case and earlier 166-case-free bootstrap runs showed 50-100% accuracy — an artifact of tiny, non-representative samples. Against 166 real stratified cases, both the base 3B model and the lightly fine-tuned candidate sit at **~34% overall accuracy** — a genuinely useful, more trustworthy number precisely because the sample is now large enough to mean something.

**The single most important finding from this run: both models have 0% recall and 0% precision on `decline`, out of 39 real decline cases.** Neither the base model nor the candidate ever correctly identifies a case that should be declined — a total blind spot on one of the two safety-critical action classes (`docs/evaluation-harness-v1.md` explicitly flags `decline`/`escalate_l2` as needing individual attention, never averaged away). The model instead leans heavily on `request_more_info` as a hedge: 86-88% recall on that class but only ~33% precision, meaning it's catching most genuine `request_more_info` cases by over-using that label broadly, including on cases that should have been declined outright. This was invisible in every earlier run in this project — the 8-case and 20-case bootstrap sets never had enough `decline` cases to expose it. It is the clearest evidence yet that this specialization pool (8 examples, 3 iterations, `num_layers=4`) is nowhere near enough signal to fix a systemic gap like this, and that real specialization work needs either a much larger, decline-focused trace pool or a policy rule that forces escalation on cases the model is inclined to wave through as `request_more_info` when risk signals are actually decline-shaped.

**Why "promoted" here doesn't mean "improved":** candidate accuracy (0.3373) is very slightly *below* baseline (0.3434) — a real difference, but well inside the 2.0-percentage-point no-regression tolerance. The accuracy gate is (correctly, per its own spec) a no-regression check, not a must-improve check — see the note on this same distinction in the first cycle's write-up above. Both models also failed the identical 16/20 challenge cases, so the corrected regression semantics correctly report 0 new failures rather than penalizing the candidate for a pre-existing gap it didn't create. This is the plan's "no promotion is a valid outcome" framing's mirror image: a *promoted* cycle that didn't actually help, which is just as important to be able to say plainly as a rejected one.

---

**Status:** v2. The mechanism has now been proven twice end to end — once on a small bootstrap set (mechanism-only proof), once on the real, `validate_held_out_set`-passing 166-case set (a real, if still sobering, accuracy signal). Not yet exercised: teacher-model resolution of escalated traces, LLM-based privacy rewriting, or a specialization pool large enough to actually move accuracy (8 examples was enough to prove the pipeline, not enough to fix the decline blind spot this run surfaced).
