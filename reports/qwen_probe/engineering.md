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
