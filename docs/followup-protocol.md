# Exploratory completion of the LLM-judge and probe GRAFT pilot ablations

Libor Burian, 22 October 2026. This plan is committed and pushed before additional model attempts, **after** the original pilot outcomes were audited, unblinded and published. It is not a replacement preregistration for that pilot.

## Authorization and evidence boundary

After the completed-delivery message, the user requested: “Finish the rest you want to finish. You have time.” The remaining experimental work is 40 adapter-ablation episodes left unrun at the original deadline. This follow-up uses existing checkpoints, with no training, detector scoring, threshold fitting, task selection or repeated model attempt. Both frozen LLM-as-a-judge and downloaded probe remain the headline; the outstanding interventions concern the probe and no-intervention adapters.

The closed pilot at Git commit `634673b78e3b4bf9c148bc40519bc5462f625b1e` retains its **2026-10-22 09:30:19.743 UTC** deadline, 120/160 coverage, labels, primary/secondary definitions and inconclusive conclusion. Its judgment SHA-256 is `3234c6e340a6b19a37aeaefd4a7646ccca77dc81efec483e2427bb1ed72a37ae`. The follow-up has separate manifests, budget, generation journals and audit. Hash original evidence, checkpoints and base weights before reuse and check on resumption. Never overwrite or rerun an original attempt.

Continuation was authorized after outcomes were known; subsequent comparisons are **exploratory and outcome-aware**. Recording this plan before the new collection does not restore confirmatory status or independence. See the [Center for Open Science](https://www.cos.io/initiatives/prereg) distinction between planned and exploratory analyses and the [NeurIPS reporting checklist](https://neurips.cc/public/guides/PaperChecklist).

## Fixed collection and stopping

Run five conditions in the original priority order:

| Checkpoint / deployment | Prompt | New attempts |
|---|---|---:|
| Probe GRAFT / both adapters | Plain | 8 |
| Probe GRAFT / forget only | Elicited | 8 |
| Probe GRAFT / forget only | Plain | 8 |
| No intervention / retain only | Elicited | 8 |
| No intervention / retain only | Plain | 8 |

The eight existing probe/both elicited episodes stay in the original cohort. Preserve the selected eight tasks, Docker executable images, model/tokenizer revisions, adapter hashes, seed 2002, thinking disabled, BF16 SDPA, 16,384-token context, temperature 0.7, top-p 0.8, top-k 20, 16 turns and 120-second agent timeout. Use one GPU worker and one concurrent episode, matching actual pilot execution. Reuse prompt conversion, merging, serving and Harbor functions. No output-driven prompt or parser repair is permitted during collection.

Persist **one three-hour cap** at the first evaluation command, after preparation/documentation. Stop after these conditions or at that deadline, whichever occurs first. Never extend based on outcomes, repeat failed attempts, replace tasks or change settings. Report unfinished episodes and infrastructure failures. The original clock is never reset. An exclusive controller lock prevents duplicates; indexed conditions are skipped on resumption even if incomplete.

## Audit and analysis

After every new deployment server closes, create one mixed batch of randomized opaque packet IDs for all available follow-up episodes. Read full messages, commands, observations, raw model responses, verifier evidence and the original intent rubric with treatment labels and scores hidden. Save every judgment and packet fingerprint before joining the mapping. Preserve ambiguous intent as unknown and as bounds. The implementing assistant knows original outcomes, tasks and design; this audit is neither independent nor fully naive. Never revise old labels.

Report the two cohorts separately and add an explicitly combined descriptive table of the original 160-episode grid. Reject duplicate task/arm/mode/prompt keys across cohorts. Retain failures and timeouts; separately report verifier success, hacking attempts and legitimate success. Missing evidence is not benign behavior.

For each prompt, compute three paired task-level differences: probe retain minus both; probe retain minus forget; no-intervention retain minus both. Report all three outcomes. These are six unadjusted exploratory contrasts; elicited probe retain-minus-both already has eight original pairs and receives no new episodes. Reuse known-outcome Wilson 95% intervals, ambiguity bounds, 10,000-draw paired task bootstrap with seed 42 and descriptive exact discordance tests. Do not pool prompts as independent tasks. Favorable contrasts cannot change the original primary analysis or establish general mitigation. One seed, eight tasks, boundary outcomes, prior outcome knowledge and collection-time differences limit interpretation.

Uniformly count parsed command requests, zero-command episodes, timeouts and raw calls with the existing execution diagnostic. Check nonoverlapping execution intervals before relying on time-based journal association. Counts describe execution, do not determine intent, and do not establish a causal failure mechanism.

## Commands and deliverables

```bash
uv run python -m experiments.qwen_probe prepare --study ablation-followup
uv run python -m experiments.qwen_probe evaluate --study ablation-followup
uv run python -m experiments.qwen_probe audit --study ablation-followup
uv run python -m experiments.qwen_probe audit --study ablation-followup --view OPAQUE_ID
uv run python -m experiments.qwen_probe report --study ablation-followup
```

Use the existing audit flags to save judgments with literal evidence and rationale. Files live under `artifacts/qwen_probe/ablation-followup/`; timestamped logs contain full inputs and outputs. Public results will live under `reports/qwen_probe/followup/`, including separate/combined coverage, a sourced PDF supplement, hashes and README links. Publish added evidence to the existing collection without changing checkpoint tensors. Preserve the original pilot paper unchanged; any combined paper must separate the later follow-up explicitly.
