# LoRA Fine-Tune Memory Footprint Spike (v1)

Companion document to `docs/technical-build-plan-v5.md`, "Known Technical Risks" — "8GB memory ceiling during fine-tuning specifically... has not yet been validated on-device and is the next hardware check to run before committing to a training batch size." This is that check.

**Machine:** MacBook Air, Apple M1, 8GB unified memory. **Model:** `mlx-community/Qwen2.5-3B-Instruct-4bit`. **Tooling:** `mlx 0.32.0`, `mlx-lm 0.31.3`, driven via the real `mlx_lm.lora` / `mlx_lm.fuse` CLI entry points (the same ones `MLXBackend.fine_tune`/`fuse_adapters` shell out to). Measurement method: `/usr/bin/time -l` around each run, reading macOS's `peak memory footprint` (the reliable number — Apple's own `phys_footprint`, closely tracking `mlx_lm`'s self-reported `Peak mem`) rather than `maximum resident set size`, which under-reports Metal/unified-memory allocations on Apple Silicon. Scope: measurement only — no specialization loop, no bulk data, no config changes applied to the codebase as part of this spike.

**Caveat on the test load:** the machine was carrying its normal desktop session throughout (VS Code, Mail, Safari/WebKit processes, ~3.5–4GB already resident before any run started, confirmed via `top -l 1 -o mem`). Numbers below reflect *real* Phase 1 working conditions, not an idle dedicated benchmark box — which is the actually relevant number for someone iterating on this machine day to day, but means the safe margin against the 8GB ceiling is tighter than the raw peak-footprint numbers alone suggest.

**Dataset:** 24 synthetic train / 6 valid / 3 test fraud-alert scenario→JSON-recommendation pairs, chat-formatted (`{"messages": [...]}`, matching the chat-template fix already in `MLXBackend.generate`). Built solely to exercise real training, not as a rubric-quality dataset — real scenarios average ~300–500 tokens including the schema instructions.

## Results

| Run | Rank | Layers | Batch | Seq cap | Grad-checkpoint | Peak footprint (OS) | `mlx_lm` self-reported | Loss trajectory |
|---|---|---|---|---|---|---|---|---|
| A | 8 | 4 | 1 | 512 | no | 2.80 GB | 2.536 GB | stable |
| B | 8 | 16 | 4 | 512 | no | **5.86 GB** | 6.453 GB | **diverged to NaN by iter 2** |
| C | 8 | 16 | 1 | 512 | no | 3.66 GB | 3.229 GB | stable |
| D | 8 | 16 | 2 | 512 | no | 4.87 GB | 4.397 GB | stable |
| E | 8 | 16 | 2 | 512 | **yes** | 3.82 GB | 3.008 GB | stable, identical to D |
| F | **16** | 16 | 2 | 512 | no | 4.99 GB | 4.493 GB | stable |
| G | 8 | 16 | 2 | **1024** | no | 4.87 GB | 4.397 GB | stable, no change vs. D |
| fuse (on D's adapter) | — | — | — | — | — | 3.23 GB | — | succeeded; fused model verified to load and generate correctly |

## Findings

1. **Batch size, not layer count or rank, is the dominant memory and stability variable.** Going from 1→2→4 layers-16 runs: 3.66 GB → 4.87 GB → 5.86 GB. The jump from batch 2 to batch 4 is disproportionate (+1 GB) and — far more seriously — **the loss diverged to NaN by the second iteration at batch 4** (train loss printed `2150475776.000` then `nan`; `Tokens/sec`/`Trained Tokens` also showed nonsensical overflowed values in that run, a visible symptom of the same numerical blow-up). Batch 1–2 trained normally in every run (monotonically decreasing loss, e.g. 3.23 → 2.23 → 1.79 in run D). This wasn't re-run multiple times to rule out a one-off fluke of this specific tiny/repetitive synthetic dataset, so treat "batch 4 is unsafe" as a strong signal, not a fully isolated root cause — the safe recommendation (avoid it) holds regardless of which factor is truly responsible.

2. **Rank has minor memory cost.** Doubling rank 8→16 (same layers/batch) cost only +0.10–0.12 GB — a good knob to spend on quality without much memory risk, unlike batch size.

3. **`--grad-checkpoint` is a meaningful, correctness-neutral lever.** Same config as run D, same loss trajectory, but 22% less OS-level peak footprint (4.87 GB → 3.82 GB) and 24% less on `mlx_lm`'s own number, for roughly a 15% training-speed cost. Worth defaulting on for any config beyond the most conservative one.

4. **Sequence-length cap showed no measured effect (512 vs. 1024) in this spike — but that's a limitation of the test data, not a real finding.** None of the synthetic examples actually reached 512 tokens, so raising the cap to 1024 never changed the true batch content length. This axis needs re-validation once real, longer rubric-passing scenarios exist (`generate-data`) — don't read "sequence length doesn't matter" out of this result.

5. **`fuse_adapters` works end-to-end and is cheaper than training itself** (3.23 GB peak vs. 3.66–4.87 GB for the layers-16 training runs). The fused model was loaded fresh and generated correctly (`generate()` on a trivial prompt returned the expected output), confirming the LoRA → fuse → deploy path is mechanically sound on this hardware.

6. **Real operational gotcha found and worked around, worth documenting for whoever runs this next:** `mlx_lm.fuse` requires the *complete* Hugging Face snapshot to be cached locally — including files it never actually uses (`.gitattributes`, `README.md`) — because its `save()` path calls `hf_repo_to_path()`, which hardcodes `local_files_only=True`. `mlx_lm.lora`/`generate` never hit this because they fetch a filtered subset via `allow_patterns` and never need those files. First `fuse` attempt failed with `IncompleteSnapshotError` despite network being available; fixed once, permanently, with one `huggingface_hub.snapshot_download(repo_id, local_files_only=False)` call to fill in the missing files. Anyone running `fine_tune` → `fuse_adapters` for the first time against a cache populated only by `run-baseline`/inference will hit this exact failure unless the snapshot is completed first.

## Practical recommendation for a real specialization cycle on this hardware

- **Safe zone:** batch size 1–2, any rank up to at least 16, any layer count up to 16 (the full model), with `--grad-checkpoint` enabled as a standing safety margin. Peak footprint in this zone stays at 3.0–5.0 GB, leaving real headroom against the 8GB ceiling under normal desktop load.
- **Do not use batch size 4** — mlx_lm's own CLI/config default — until the NaN-loss cause is understood. This is the one hard "no" from this spike.
- **Re-validate sequence-length impact** once real (longer) rubric-passing training data exists — this spike's dataset was too short to exercise it honestly.

## A concrete gap this spike exposed in existing code

`MLXBackend.fine_tune` (`src/slmguard/backends/mlx_backend.py`) does not pass `--batch-size`, `--max-seq-length`, or `--grad-checkpoint` to the `mlx_lm.lora` subprocess at all. An actual call today would silently inherit `mlx_lm`'s own default batch size of **4** — the exact setting this spike found unstable and memory-risky on this machine. Not fixed as part of this spike (measurement only, per scope) — flagged here as the concrete next code change before `fine_tune` is ever called for real.

---

**Status:** v1, single-machine measurement spike. Not re-run for variance; not run under memory pressure beyond ordinary desktop load; not run against real (non-synthetic) training data. Sufficient to unblock a batch-size/grad-checkpoint decision, not sufficient to call this hardware's limits fully characterized.
