# Implementation notes and engineering evidence

This document records engineering decisions. The [protocol](protocol.md) defines scientific decisions; [generated results](../reports/qwen_probe/summary.md) report observations.

## Reuse and compatibility

The extension calls the existing SFT trainer, `DualMLPAdapter`, `SplitMomentAdamW`, ATIF conversion, merge implementation, dataset builder and pinned Harbor CLI. The original SFT/RL workflows remain available in their directories. New behavior is opt-in through trainer flags. The root `uv` environment locks compatible releases; Harbor is pinned to the repository's Git submodule revision rather than upgraded across the source experiment.

The original trainer scales per-example loss by `world_size / effective_batch`; `Accelerator.backward` also divides by its accumulation count. The pilot removes the first divisor via `--correct-accumulation`. This changes the released implementation's scaling, so this is an adaptation, not bit-for-bit reproduction. A regression test uses actual Accelerate backward calls and compares their gradient to a direct batch-mean loss. Split-moment hooks still capture the natural gradient before applying the routing multiplier.

The released merger assumed a sharded safetensors index. Qwen3-0.6B has one weight file and tied embeddings. The merger now derives a weight map from the file when no index exists and preserves the tying configuration. Three tests compare complete logits between live adapters and merged models, including retain-only and forget-only modes. See [safetensors shared tensors](https://huggingface.co/docs/safetensors/torch_shared_tensors).

Liger patches only Qwen3's fused linear cross-entropy forward. It does not replace the custom adapter MLP or attention. Standard-library and supported-library functionality are used for JSON, SQLite caching, subprocess lifecycle, Hugging Face downloads, statistics and PDF generation. The initial CUDA test compares loss and every parameter gradient on a small tied-embedding Qwen3 model against the ordinary Transformers loss. This is a numerical implementation check, not a behavioral result.

## Detector reproduction

The implementation follows the fine-tuned encoder cells, which differ from the notebook's earlier plain-embedding baseline: there is **no query instruction prefix** in the fine-tuned path. The downloaded head loads strictly with 248 positional slots, 1,024 input dimensions, four transformer layers and 256 internal dimensions. LoRA targets q/k/v/o attention projections with rank 16 and alpha 32. The encoder is BF16; pooled embeddings and the head use FP32. All parameters are frozen and dropout is disabled.

Three Drive artifacts are verified against version-controlled SHA-256 hashes before loading. The PyTorch head is loaded with `weights_only=True`. The source repository declares MIT; the Drive weight folder does not provide a separate license statement. Model cards identify Apache-2.0 for Qwen models. Upstream artifact attribution and uncertainty are preserved, and the detector weights are not reuploaded.

Normalized turn text is the only data input to the monitor. Ground-truth labels and source provenance are stored separately. Source labels are used for balanced SFT selection, simulated-monitor controls and heldout measurement, never inserted into the monitor. First-user red-team appendix stripping is shared with the released conversion. Role-tag and tool normalization reproduces the notebook, including its inline-tool suppression behavior. Exact duplicate normalized inputs are audited, but this is not a semantic contamination detector.

The [provenance audit](probe-provenance.md) records what the pinned repository and source dataset cards reveal, and why upstream task/trajectory exposure remains unresolved.

## Local execution and failure handling

Scoring runs in its own process and caches scores and turn embeddings before training. Child exit frees the monitor's GPU memory. A single Transformers worker provides the limited text chat-completions interface Harbor requires; generation uses the official `generate` API, fixed non-thinking settings and complete prompt/response logs. Requests are serialized. The bridge is bound to localhost, and process-group cleanup targets only the server created by the experiment. The upstream broad `pkill` shutdown script is never called.

Docker environment selection occurs before model evaluation. Task build and launch failures are recorded, with deterministic replacement ordering; no model success is used to select tasks. Images are prebuilt before the six-hour clock. Runtime agent errors remain observed failures/ambiguities rather than reasons to replace a task. The existing Harbor context/output-overflow patch allows verifier execution after those agent failures. One episode runs at a time; this is within the maximum concurrency of two.

The persistent `budget.json` deadline cannot be reset by rerunning the command. Trainers check it between microbatches and save only completed optimizer updates. Incomplete checkpoints record completed/planned updates and are not evaluated as completed five-epoch arms. A 30-second signal grace period is for saving/cleanup. Full optimizer-state resumption is not implemented; interrupted runs are preserved, not silently restarted. The primary training pair and its behavioral evaluation precede secondary controls to make bounded compute useful without outcome-based decisions.

The pipeline emits opaque audit packets and deliberately waits for explicit judgments. It does not substitute probe flags, reward scores or missing files for intent labels. The same implementing Codex assistant may audit; this is a limitation relative to independent human blinding. All packet fingerprints and judgments are frozen before a treatment mapping is exported.

Audit packets also contain the server's original messages and raw responses, selected by the recorded episode execution interval. This includes malformed or truncated output that Harbor may omit from its structured trajectory or replace in `response.txt`. The journal is written before generation and updated on completion; absent output remains explicit. Only message text, response text and completion status are copied into blinded packets.

## Setup issues encountered before experiments

- `uv add --editable` initially attempted to nest Harbor's workspace inside the root workspace. Nested workspaces are unsupported. A path dependency in `[tool.uv.sources]` fixes this without modifying the pinned Harbor project.
- Adding Harbor selected a compatible Hugging Face Hub version through the resolver; `uv.lock` is the reproducibility authority. Latest mutually incompatible packages are not forced into the same environment.
- The first 13 protocol, detector, routing, accumulation, CUDA-loss and merge tests passed before real-model training. Further checks and actual run failures belong in the generated experiment reports or dated amendments.
- The user's existing Hugging Face CLI login is reused for publication. Model and tokenization caches remain workspace-local. Credentials are never passed on command lines, put in model inputs, or copied into staged artifacts.

## Dated engineering corrections during the capped run

2026-10-21 22:57 UTC: the first real memory preflight failed before backward or parameter updates because newly inserted adapter Linear modules defaulted to FP32 while the base used BF16. The normal trainer already cast the combined model, but the standalone preflight exposed the more general constructor problem. A failing regression test was added, then adapter insertion was corrected to inherit the base projection's dtype/device after initialization. The common context was not reduced for this non-OOM error, and the original deadline was retained. This does not use behavioral outcomes or change the scientific protocol.

2026-10-21 23:08 UTC: the upstream CPU routing regression exposed an unnecessary assumption that every MLP names its projection `gate_proj`. Placement now comes from the first base parameter using the standard `Module.parameters()` iterator. This preserves Qwen behavior and also supports the upstream toy MLP. No training updates or behavioral evaluations preceded this correction.

## Sources

[Transformers Qwen3](https://huggingface.co/docs/transformers/model_doc/qwen3), [Qwen non-thinking settings](https://huggingface.co/Qwen/Qwen3-0.6B), [Liger Qwen3 patch](https://github.com/linkedin/Liger-Kernel), [Accelerate accumulation](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation), [Harbor](https://github.com/harbor-framework/harbor), [uv projects](https://docs.astral.sh/uv/concepts/projects/), [probe source](https://github.com/farzingkh/reward-hacking/tree/cd4c566824c5961eaeab7b7fce7845240bd12b3d).
