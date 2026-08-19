# Specialization Loop (v1)

Companion document to `docs/technical-build-plan-v5.md`, "Specialization Loop." First implementation: `src/slmguard/specialization.py`, wired into the CLI as `slmguard specialize`. Scoped tight on purpose, per explicit instruction: one working loop path, no extra platform features.

## The six steps, as actually implemented

1. **Select traces** (`select_traces`) — pulls recent traces from the `AuditStore` and filters to the ones that matter: schema failures (escalated cases) and schema-valid cases below a confidence threshold.
2. **Convert to training examples** (`convert_trace`, `build_specialization_pool`) — a trace with a valid label becomes a `SyntheticExample` (scenario redacted for PII, paired with its recorded recommendation); a schema-failure trace has no valid label to imitate and is counted, not converted. Every candidate then goes through `slmguard.rubric.score_batch` — only rubric-passing examples enter the pool, same discipline the plan requires for the held-out set.
3. **Fine-tune with safe LoRA settings** (`write_training_dataset` + `ModelBackend.fine_tune`) — the pool is split into `train.jsonl`/`valid.jsonl` and handed to the configured backend, using the `batch_size≤2` / `grad_checkpoint=true` defaults validated in `docs/lora-memory-footprint-spike-v1.md`.
4. **Fuse** (`ModelBackend.fuse_adapters`) — merges the trained adapter into a new deployable model version.
5. **Evaluate** (`evaluate_model`, `count_challenge_failures`) — runs both the pre-fine-tune base model and the fused candidate against the same held-out set (a fair, cycle-local baseline — there's no persisted prior-cycle baseline yet, since this is cycle one) and against the challenge set.
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

Every step executed against real infrastructure: 8 real traces (generated via real `run-baseline` calls) scanned and filtered, 7 converted and rubric-passed, a real `mlx_lm.lora` fine-tune ran for 3 iterations (train loss 2.835, val loss 2.911 → 2.730), a real `mlx_lm.fuse` merge produced a candidate model, both the base and candidate models were evaluated against the same 8-case held-out set and 2-case challenge set, and the promotion gate correctly withheld promotion because of a challenge-set regression — a legitimate, well-reasoned "no promotion this cycle," not a crash or a short-circuit on missing infrastructure. The audit chain (`verify_chain()`) remained intact throughout.

That the candidate didn't clearly beat the baseline (0.500 vs. 0.500) is expected and not concerning on its own: 7 examples over 3 iterations with `num_layers=4` is a minimal smoke-scale run, not a real specialization attempt — the plan's own "no promotion" outcome exists precisely for cycles like this one.

---

**Status:** v1. The mechanism is real and proven end to end. Not yet exercised: policy-engine-driven selection, teacher-model resolution of escalated traces, LLM-based privacy rewriting, or a run against the real (not bootstrap) held-out/challenge sets — all blocked on components this pass deliberately left out of scope.
