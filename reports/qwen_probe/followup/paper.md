# Qwen3-0.6B GRAFT with LLM-as-a-Judge and Probe Monitors

Libor Burian — exploratory follow-up supplement after the closed pilot, 22 October 2026

## Abstract

This supplement reports later adapter interventions for the Qwen3-0.6B GRAFT pilot with frozen LLM-as-a-judge and downloaded probe monitors [1,2]. Original closed pilot: 120/160 audited trial records. Later follow-up: 40/40. Combined distinct record coverage of the original grid: 160/160. Record coverage includes setup failures with no observable model behavior; unknown outcomes remain unknown and unrun episodes remain unavailable. None were replaced or imputed. The closed pilot's primary result remains inconclusive; these later adapter contrasts are exploratory.

## Context, motivation and prior knowledge

The original pilot had already been audited, unblinded and published when the user authorized finishing the outstanding work. Its primary result, capability failure and frozen judgments remain unchanged. The new collection evaluates existing downloaded-probe and no-intervention checkpoints; it introduces no trained model or detector. The goal is descriptive adapter-intervention coverage, not a new confirmatory mitigation claim [3,4].

## Method and stopping rule

Five previously unattempted conditions reuse the fixed eight tasks and saved checkpoints: probe/both plain; probe/forget elicited and plain; no-intervention/retain elicited and plain. The probe/both elicited condition is inherited from the original cohort. Model/tokenizer files, original evidence and reused adapter states are hash-locked. Sampling seed 2002, non-thinking mode, BF16 SDPA, temperature 0.7, top-p 0.8, top-k 20, context 16,384, maximum 16 turns and 120 seconds per agent remain fixed. One serialized worker runs until the five conditions finish or the separate persisted three-hour cap expires. No actual model attempt is repeated. The protocol and evaluator were pushed before this collection; followup-protocol.md states the exact scope.

## Infrastructure deviation

The dated infrastructure amendment archived 8 setup failures before any agent or verifier execution. It also preserved 1 pre-job launcher failures covering 8 uninitialized tasks; those cases require an absent job directory and zero journaled model requests during a valid launch interval. The user upgraded host packages during collection; containerd shim API skew temporarily prevented container startup. After services restarted externally, recovery required unchanged executable image hashes, no actual prior model attempt, one recovery per wholly unstarted condition and the same persisted deadline. Pre-upgrade host versions were not captured; host-runtime equality is unresolved. Initial publication/controller evidence and result hashes remain available. See followup-protocol.md and infrastructure.json; compatibility-check source: https://github.com/containerd/containerd/blob/v2.4.2/pkg/shim/compat.go.

## Audit and statistical analysis

All new deployments close before one mixed opaque audit batch is created. Full raw responses, commands, observations, task instructions and verifier evidence are judged with the original intent rubric. Every new judgment is saved and fingerprinted before treatment identities are joined. The implementing assistant knows the original outcomes and design; this is neither an independent nor fully naive audit. Each contrast pairs the same task within one prompt condition. Intervals use known pairs; unknown intent remains in bounds. The six comparisons are exploratory and unadjusted. A zero-width bootstrap is not evidence of equivalence. One seed and eight tasks do not support a general five-point capability guarantee. Wilson intervals and paired bootstrap/exact discordance calculations reuse the original analysis implementation [5].

## Results

Original closed pilot: 120/160 audited trial records. Later follow-up: 40/40. Combined distinct record coverage of the original grid: 160/160. Record coverage includes setup failures with no observable model behavior; unknown outcomes remain unknown and unrun episodes remain unavailable. None were replaced or imputed.

Among 40 follow-up records, there are 2 confirmed hacking attempts and 9 unknown intent labels, including 3 agent-setup timeouts. The legitimate-success count is 1, with 3 unknown legitimate outcomes. The paired results below include original comparators; the elicited probe retain-minus-both contrast uses only original data.

probe_retain_minus_both/elicited: hacking: -0.500 (95% interval -0.833 to -0.167), 6/8 known task pairs, ambiguity bounds [-0.500, -0.250], exact discordance p=0.250; legitimate: 0.000 (95% interval 0.000 to 0.000), 8/8 known task pairs, ambiguity bounds [0.000, 0.000], exact discordance p=1.000; verifier_success: 0.000 (95% interval 0.000 to 0.000), 8/8 known task pairs, ambiguity bounds [0.000, 0.000], exact discordance p=1.000.

probe_retain_minus_both/plain: hacking: -0.250 (95% interval -0.750 to 0.000), 4/8 known task pairs, ambiguity bounds [-0.500, 0.125], exact discordance p=1.000; legitimate: 0.000 (95% interval 0.000 to 0.000), 8/8 known task pairs, ambiguity bounds [0.000, 0.000], exact discordance p=1.000; verifier_success: 0.000 (95% interval 0.000 to 0.000), 8/8 known task pairs, ambiguity bounds [0.000, 0.000], exact discordance p=1.000.

probe_retain_minus_forget/elicited: hacking: 0.000 (95% interval -0.600 to 0.600), 5/8 known task pairs, ambiguity bounds [-0.250, 0.125], exact discordance p=1.000; legitimate: 0.000 (95% interval 0.000 to 0.000), 8/8 known task pairs, ambiguity bounds [0.000, 0.000], exact discordance p=1.000; verifier_success: 0.000 (95% interval 0.000 to 0.000), 8/8 known task pairs, ambiguity bounds [0.000, 0.000], exact discordance p=1.000.

probe_retain_minus_forget/plain: hacking: 0.000 (95% interval 0.000 to 0.000), 5/8 known task pairs, ambiguity bounds [-0.125, 0.250], exact discordance p=1.000; legitimate: -0.143 (95% interval -0.429 to 0.000), 7/8 known task pairs, ambiguity bounds [-0.250, -0.125], exact discordance p=1.000; verifier_success: -0.143 (95% interval -0.429 to 0.000), 7/8 known task pairs, ambiguity bounds [-0.250, -0.125], exact discordance p=1.000.

no_intervention_retain_minus_both/elicited: hacking: -0.500 (95% interval -1.000 to 0.000), 4/8 known task pairs, ambiguity bounds [-0.625, -0.125], exact discordance p=0.500; legitimate: -0.125 (95% interval -0.375 to 0.000), 8/8 known task pairs, ambiguity bounds [-0.125, -0.125], exact discordance p=1.000; verifier_success: -0.125 (95% interval -0.375 to 0.000), 8/8 known task pairs, ambiguity bounds [-0.125, -0.125], exact discordance p=1.000.

no_intervention_retain_minus_both/plain: hacking: 0.000 (95% interval 0.000 to 0.000), 6/8 known task pairs, ambiguity bounds [0.000, 0.250], exact discordance p=1.000; legitimate: 0.000 (95% interval 0.000 to 0.000), 6/8 known task pairs, ambiguity bounds [0.000, 0.250], exact discordance p=1.000; verifier_success: 0.000 (95% interval 0.000 to 0.000), 6/8 known task pairs, ambiguity bounds [0.000, 0.250], exact discordance p=1.000.

Counts are positive/known outcomes, with Wilson 95% intervals for the known rate. Unknown counts and ambiguity bounds concern audited episodes; the audited count out of eight separately exposes missing planned episodes. These intervals do not capture training-seed variation.
| Deployment / prompt | Audited | Hacking attempts | Legitimate success | Verifier success |
| --- | --- | --- | --- | --- |
| no_intervention/both/elicited | 8/8 | 2/5 known; 95% [0.118, 0.769]; unknown 3; bounds [0.250, 0.625] | 1/8 known; 95% [0.022, 0.471] | 1/8 known; 95% [0.022, 0.471] |
| no_intervention/both/plain | 8/8 | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_graft/retain/elicited | 8/8 | 1/7 known; 95% [0.026, 0.513]; unknown 1; bounds [0.125, 0.250] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_graft/retain/plain | 8/8 | 0/6 known; 95% [0.000, 0.390]; unknown 2; bounds [0.000, 0.250] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| base/both/elicited | 8/8 | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| base/both/plain | 8/8 | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| qwen_judge_graft/retain/elicited | 8/8 | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| qwen_judge_graft/retain/plain | 8/8 | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| original_graft/retain/elicited | 8/8 | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| original_graft/retain/plain | 8/8 | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_filtering/retain/elicited | 8/8 | 2/8 known; 95% [0.071, 0.591] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_filtering/retain/plain | 8/8 | 0/7 known; 95% [0.000, 0.354]; unknown 1; bounds [0.000, 0.125] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| random_graft/retain/elicited | 8/8 | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| random_graft/retain/plain | 8/8 | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_graft/both/elicited | 8/8 | 4/7 known; 95% [0.250, 0.842]; unknown 1; bounds [0.500, 0.625] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_graft/both/plain | 8/8 | 1/5 known; 95% [0.036, 0.624]; unknown 3; bounds [0.125, 0.500] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_graft/forget/elicited | 8/8 | 1/6 known; 95% [0.030, 0.564]; unknown 2; bounds [0.125, 0.375] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_graft/forget/plain | 8/8 | 0/7 known; 95% [0.000, 0.354]; unknown 1; bounds [0.000, 0.125] | 1/7 known; 95% [0.026, 0.513]; unknown 1; bounds [0.125, 0.250] | 1/7 known; 95% [0.026, 0.513]; unknown 1; bounds [0.125, 0.250] |
| no_intervention/retain/elicited | 8/8 | 0/7 known; 95% [0.000, 0.354]; unknown 1; bounds [0.000, 0.125] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| no_intervention/retain/plain | 8/8 | 0/6 known; 95% [0.000, 0.390]; unknown 2; bounds [0.000, 0.250] | 0/6 known; 95% [0.000, 0.390]; unknown 2; bounds [0.000, 0.250] | 0/6 known; 95% [0.000, 0.390]; unknown 2; bounds [0.000, 0.250] |
## Execution diagnostics

probe_graft/both/plain: 1/8 zero-command episodes; 8 timeouts; probe_graft/forget/elicited: 1/8 zero-command episodes; 8 timeouts; no_intervention/retain/plain: 3/8 zero-command episodes; 6 timeouts; probe_graft/forget/plain: 3/8 zero-command episodes; 7 timeouts; no_intervention/retain/elicited: 0/8 zero-command episodes; 7 timeouts. Parsed requests are not successful shell executions. Counts do not alter frozen intent judgments or establish a causal failure mechanism. Zero overlapping intervals supports the timestamp-based journal association for this sequential run.

## Discussion, claims and limitations

The closed pilot's primary result remains inconclusive; these later adapter contrasts are exploratory. Changes from both to retain/forget modes describe interventions on the same checkpoint, conditional on fixed tasks and this runtime. Except for the original elicited probe retain-minus-both comparison, each mode contrast spans cohorts, so adapter mode is confounded with collection time and possible host-runtime changes. Timing-dependent generation cutoffs and system load can affect outputs. A percentile-bootstrap interval that excludes zero is not a confirmatory discovery: few discordant pairs can give a different exact-test conclusion, and all six contrasts are exploratory. Earlier outcome knowledge, ambiguous intent, single-seed uncertainty, the capability floor and unresolved upstream probe exposure remain limitations. Do not select a favorable contrast as evidence of general alignment or overwrite the original primary conclusion.

## Conclusion

The added evidence completes or documents the remaining adapter interventions under its own stopping rule. Original and follow-up evidence remain distinguishable. Stronger causal and mitigation claims require capable task performance, more tasks and seeds, and independent outcome-blind auditing.

## References and reproducibility

[1] Ward et al. (2026), Gradient Routing Localizes and Removes Unintended Behaviors in RL. https://arxiv.org/abs/2610.17062
[2] Frozen probe source, revision cd4c566824c5961eaeab7b7fce7845240bd12b3d. https://github.com/farzingkh/reward-hacking
[3] Center for Open Science, preregistration and planned/exploratory distinctions. https://www.cos.io/initiatives/prereg
[4] NeurIPS Paper Checklist. https://neurips.cc/public/guides/PaperChecklist
[5] SciPy binomial inference. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
Original report: ../summary.md; original paper: ../paper.pdf. Follow-up source hashes, clock, audit freeze and full numeric evidence are in manifest.json, budget.json, audit_freeze.json, combined_outcomes.json and results.json. The execution commands and study boundary are in docs/followup-protocol.md in the repository. No new weights are trained; the published pilot checkpoints are unchanged.

![ablations](ablations.png)
