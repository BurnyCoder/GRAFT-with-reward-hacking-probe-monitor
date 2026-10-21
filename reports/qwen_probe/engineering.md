# Engineering checks and observed setup failures

These checks establish implementation behavior, not reward-hacking mitigation. Scientific outcomes are generated separately in [summary](summary.md).

| Check | Observed result | Scope |
|---|---|---|
| CPU suite, `uv run pytest -q -m 'not gpu'`, 2026-10-21 23:15 UTC | 18 passed, 2 CUDA cases deselected | Preprocessing, metadata exclusion, task isolation, threshold freezing, routing, accumulation, three merge modes, audit evidence, statistics and coverage |
| CUDA loss tests, earlier setup run | FP32 and BF16 cases passed | Tiny tied-embedding Qwen3, ordinary versus Liger fused loss and every parameter gradient; numerical tolerances are explicit in `tests/test_training.py` |
| Released `sft/scripts/gr/test_gr_core.py`, 23:08 UTC run | All eight gates passed | Natural/routed moments, anchor gradients, realized multiplier 2, freezing, checkpoint hooks and two-rank CPU Gloo parity |
| Actual Qwen3-0.6B preflight | 16,384 tokens passed; 13.57 seconds; 3,528,999,936 peak PyTorch-allocated bytes | Real forward/backward with adapter gradients and reserved optimizer-moment memory; no optimizer update |
| Static checks after corrections | `ruff check --select F,E9 experiments tests` and `git diff --check` passed | Undefined/unused names, syntax and whitespace; not a complete security proof |

The first memory preflight failed because freshly constructed adapter layers used FP32 while the base used BF16. The next failing regression exposed a Qwen-specific projection-name assumption. Both were corrected before any parameter update; [implementation notes](../../docs/implementation.md) explain the fixes and sources. This was not an out-of-memory failure, so the context limit was not reduced.

The audit review identified a separate evidence problem before generated evaluations: Harbor can replace a truncated raw response with a parser error. A regression was written first, then original server messages/responses were added to blinded packets using the episode's execution interval. The orchestration was restarted at 23:15 UTC to load that committed correction; 475 completed detector scores were preserved. The shared deadline remained 2026-10-22 04:57:02 UTC. There was no restart or selection of a trained checkpoint based on behavior.

The test suite emits PyTorch's warning that nested tensors are a prototype API when exercising the released transformer head. It is recorded as a dependency warning, not silently treated as a failed classifier check. All real model prompts, outputs and monitor results are retained in timestamped local logs; implementation publication precedes real-model optimizer updates.

At 23:30 UTC the first smoke attempt failed before any optimizer update because the installed TRL collator no longer accepts `completion_only_loss` or applies assistant masks. Explicit prebuilt labels fix both the API exception and the potential silent objective change. The full suite subsequently passed 26 cases, including real CUDA FP32/BF16 loss/gradient equivalence, before the additional paired-detector statistic and judge-pool tests were added. Their final validation is recorded in the next implementation commit.

The user added a same-Qwen LLM-judge detector comparison and GRAFT arm, then authorized ten more hours. The [dated protocol amendment](../../docs/protocol.md) preserves the original clock history, data lock, probe threshold and primary comparison; the amended deadline is 2026-10-22 09:30:19.743 UTC. No optimizer update or generated behavioral outcome preceded this amendment. This supersedes the original deadline noted above without erasing it.

The complete expanded suite passed 28 tests at 23:43 UTC. A subsequent real judge startup exposed a duplicate logging keyword (`turns` for text and count) before any prediction. A new full scoring-path regression reproduced it; renaming the text field fixed it, and all four judge tests passed before restart. The calibrated scoring rule and inputs were unchanged. The published implementation was updated before real-model training.

The single correctness review checked the source launcher against adapter widths, scaling, learning rate, accumulation, anchors, split moments, kappa and filtering settings; it also inspected server ownership, complete generation journals, audit blinding and publication allowlists. It identified a reporting clarification: source-compatible anchor draws occur within each unflagged pool, so monitor changes can change anchor identities despite the common seed and 1% rule. The no-intervention arm intentionally disables anchors with routing. This is now explicit in the protocol.
