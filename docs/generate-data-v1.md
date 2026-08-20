# generate-data (v1)

Companion document to `docs/technical-build-plan-v5.md`'s Distillation Approach ("the larger model produces the first synthetic scenario/label pairs used to establish the baseline specialist") and `docs/synthetic-data-quality-rubric-v1.md`'s rejection process. Implementation: `src/slmguard/teacher/` (the interface) and `src/slmguard/generate_data.py` (the orchestration), wired into the CLI as `slmguard generate-data`.

## The `Teacher` interface

Same swappable-backend pattern as `ModelBackend`/`AuditStore`: `src/slmguard/teacher/base.py` defines `Teacher.generate(spec: GenerationSpec) -> TeacherExample`, and `openrouter_teacher.py` is the Phase 1 implementation, not the architecture. Swapping to a direct Claude API teacher or a larger local model later means implementing `Teacher` again — nothing in `generate_data.py` should need to change, since it only ever calls the interface.

**OpenRouter** (openrouter.ai) is the chosen Phase 1 backend: a unified gateway to many hosted models with genuinely free-tier options, needing one integration instead of one per provider. Uses stdlib `urllib` rather than adding an HTTP client dependency — a single JSON-in/JSON-out POST isn't worth a new package.

## No real customer data — how this is actually guaranteed, not just promised

`generate_data.py` has **no `AuditStore` dependency anywhere in the module** — it literally cannot construct a prompt from real trace/case content, because it never has a handle to real data in the first place. Every `GenerationSpec` sent to the teacher is one of four static, diversity-focused instruction strings (`DEFAULT_CATEGORY_GUIDANCE`), covering the rubric's required diversity categories. This is an architectural guarantee (the module's import graph proves it), not a runtime check that could be bypassed by a bug.

## Rubric enforcement: reject and regenerate, never patch

`generate_batch(teacher, specs, max_attempts=3)`:
1. Calls `teacher.generate(spec)` once per spec, producing a full candidate batch.
2. Scores the whole batch via `slmguard.rubric.score_batch` (the same rubric `generate_data`'s CLI stub always pointed at).
3. If the batch clears `DEFAULT_PASS_RATE` (85%), every rubric-*passing* example in it is accepted — a batch doesn't have to be 100% clean, individual failing examples within an otherwise-accepted batch are still dropped.
4. If the batch doesn't clear the threshold, the **entire batch is discarded and regenerated from scratch** — fresh teacher calls, not a patch of the failing examples. This matches the rubric doc's explicit rule: "rejected and regenerated, never folded in as-is."
5. After `max_attempts` regenerations without clearing the threshold, the command fails outright (`SystemExit(1)`, no output file written) rather than silently accepting substandard data.

## Provenance: teacher metadata on every example

Every accepted example carries a `TeacherMetadata` (`teacher_name`, `model_id`, `generated_at`) written alongside it in the output JSONL — never stored separately from the data it describes, so a downstream consumer can always answer "which model produced this, and when" without a join.

## Output format

`slmguard generate-data` writes `data/generated/batch_<UTC-timestamp>.jsonl`, one JSON object per accepted example:

```json
{
  "scenario": "...",
  "recommendation_json": "{...}",
  "diversity_tags": ["edge_case"],
  "teacher": {"teacher_name": "openrouter", "model_id": "...", "generated_at": "2026-08-20T..."}
}
```

This is deliberately the same shape `slmguard.rubric.SyntheticExample` and `slmguard.specialization.convert_trace`'s output already use (`scenario`/`recommendation_json`/`diversity_tags`) plus the `teacher` provenance block — a generated batch and a trace-converted batch are interchangeable inputs to the same downstream rubric-scoring and pool-building code.

## Never a paid model — enforced, not just documented

`OpenRouterTeacher.__init__` queries the live catalog (`_assert_model_is_free`) and refuses to construct if the configured `model_id` isn't confirmed to cost exactly zero for both prompt and completion tokens — a model missing from the catalog is treated the same as a paid one, fail closed. This is a hard runtime guard per explicit user instruction ("no paid model from OpenRouter ever"), not a config convention someone could accidentally violate by editing YAML.

## Chosen model

**`nvidia/nemotron-3-super-120b-a12b:free`**, chosen 2026-08-20 by querying the live catalog (`list_free_models()`) for NVIDIA Nemotron variants, per explicit user preference. 8 free Nemotron variants existed at query time, from `nemotron-3.5-lightning` (30B total/3B active) up to `nemotron-3-ultra-550b-a55b` (550B total/55B active, described as NVIDIA's "frontier-reasoning" model). Deliberately did **not** pick the largest (`ultra-550b-a55b:free`): its free-tier variant's `supported_parameters` list is missing `response_format`/`structured_outputs`, while `nemotron-3-super-120b-a12b:free` (120B total/12B active, "complex multi-agent applications") supports both on the free tier — for a task that's entirely about producing clean, parseable JSON, that reliability property mattered more than the extra raw size. `OpenRouterTeacher._call` requests `response_format: {"type": "json_object"}` accordingly.

**Live-verified**, not just configured: ran `slmguard generate-data --n-per-category 1` for real against the live API. Result: 4/4 examples accepted on the first attempt (100% rubric pass rate), one per required diversity category, in well under a minute total. Sample quality was genuinely strong — e.g. the `transaction_type` example correctly modeled a card-testing pattern (a $0.99 probe charge followed 10 minutes later by a $2,450 purchase at the same online merchant) with a coherent `decline` rationale citing the specific velocity/amount/time signals; the `policy_boundary` example explicitly reasoned about sitting "exactly at the escalate_l2/decline boundary (threshold 0.5)." Each diversity focus was genuinely reflected in its example's content, not just its tag.

---

**Status:** v1. Interface, orchestration, and rubric enforcement built and unit-tested against a `FakeTeacher`/mocked HTTP responses. Live-verified end to end against the real OpenRouter API and the chosen free model — not just unit-tested, actually run.
