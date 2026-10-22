# Repository instructions

Keep the extension in `experiments/qwen_probe`; reuse the released `sft` modules. User instructions in the active conversation take precedence. Do not introduce outside-workspace caches or paid infrastructure. Use the root `uv` environment and pinned submodules.

Scientific decisions live in `docs/protocol.md`; engineering rationale lives in `docs/implementation.md`; observations live in `reports/qwen_probe`. Do not duplicate the protocol or turn unrun experiments into results. Record dated deviations before interpreting outcomes.

Preserve closed-study inputs and results when correcting preprocessing. New SFT data must carry the current conversion identity and pass the source-to-parser contract; use a separate artifact root rather than relabeling legacy locks or checkpoints. See `docs/implementation.md` for the model-free conversion checks and `reports/qwen_probe/correction-2026-10-22/paper.md` for the dated correction.

Before real-model parameter updates, push the implementation and preregistration. Preserve the shared persisted deadline across resumptions (six-hour default; this run has the user's dated extension in `docs/protocol.md`). Never reset it because of results or implementation problems. Keep incomplete checkpoints and coverage explicit. Do not reuse the smoke checkpoint as a study arm.

Tests use small random models; CUDA equivalence tests need the local GPU. Run `uv run pytest -q` and inspect full-pipeline logs. Exclude `.env`, caches, raw logs and downloaded weights from Git. Publish only the explicit Hugging Face staging allowlist. Do not run the original broad process-killing server script.

Use modular functions, source-backed comments explaining local operations and their experiment context, timestamped complete model-input/output logs, meaningful commits, and a single correctness review. The project owner has authorized feature PRs, merges and publication of clearly labeled pilot checkpoints.
