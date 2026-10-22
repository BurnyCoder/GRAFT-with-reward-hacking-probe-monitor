# Qwen3-0.6B GRAFT with LLM-as-a-Judge and Probe Monitors

Libor Burian — exploratory pilot, 22 October 2026

## Abstract

We adapt GRAFT to Qwen3-0.6B with two frozen reward-hacking monitors: an LLM-as-a-judge using the untouched learner checkpoint, and a downloaded embedding encoder with a trajectory probe. Each supplies flags to a separate GRAFT treatment; we compare detector quality and agentic behavior. Mitigation effectiveness is inconclusive. The primary comparison is incomplete or has ambiguous judgments. This report covers 3 completed training arms and 0 audited study episodes; 8 audit packets remain pending. The experiment is an exploratory single-seed pilot, not a powered confirmatory test.

## Context, motivation and introduction

GRAFT separates training updates into retain and forget adapters using a monitor [1]. This extension compares two concrete replacements for its simulated Terminal Wrench monitor: a frozen Qwen3-0.6B LLM judge [3,11] and a separately trained, frozen trajectory probe [2], with Qwen3-0.6B as the learner [3]. Both methods are evaluated as detectors and as sources of routing flags. The motivation is to distinguish an implementable monitor integration from evidence that it mitigates reward hacking.

## Background and related work

The LLM judge uses an intent rubric with conditional YES/NO next-token scores from the untouched Qwen3-0.6B [3]. LLM-as-a-judge research documents limitations that motivate empirical validation [11]; it does not establish this small judge's accuracy. The downloaded probe combines Qwen3-Embedding-0.6B [4], an attention LoRA adapter, and a turn-transformer classifier [2]. Both monitors run separately from learner training. GRAFT's monitor flags route gradients; they are not scalar reward penalties [1]. Classification accuracy and generated hacking behavior measure different quantities.

## Method

The exact preregistration is docs/protocol.md in the published repository commit recorded in publication.json. Preserve the published 232/99 task split, reserve 32 training tasks for calibration (seed 42), and independently freeze each monitor's strict score cutoff at empirical clean FPR <= 5%. Select at most 192 balanced trajectories, in multiples of 48, with a common memory-preflight length cap. Freeze the base; train two width-64 branches for five epochs, effective batch 48, learning rate 5e-4, weight decay 1e-4, betas (0.9,0.95), cosine schedule, 5% warmup, 1% unlabeled anchors and forget-gradient multiplier 2. Compare LLM-judge GRAFT, downloaded-probe GRAFT, no intervention, original simulated-monitor GRAFT, probe filtering and length-bin-shuffled probe GRAFT, plus the untouched base model. Filtering uses ordinary AdamW and thus is not a pure routing contrast. Correct Accelerate accumulation normalization [5] and use loss-equivalent Liger Qwen3 fused CE [6].

## Frozen LLM-judge and probe inputs

The untouched pinned Qwen3-0.6B judge uses the released intent rubric and conditional YES/NO next-token logits with thinking disabled. The downloaded probe uses its frozen LoRA encoder and four-layer turn-transformer head. Separate cutoffs use the same calibration tasks and empirical-FPR rule. Both receive the same normalized trajectory text, excluding labels, source paths, rewards and judge metadata. Both retain at most 248 turns. The judge permits 16,384 aggregate input tokens with head/tail truncation; the probe permits 4,096 tokens per turn with normalized 1,024-dimensional last-token embeddings. This coverage difference is disclosed, not equated. The judge arm and its 16 agentic episodes were added by a dated user amendment before any optimizer update or behavioral outcome. Its detector and behavioral comparisons are secondary; the primary remains probe GRAFT versus no intervention. Equal prominence in the presentation does not change these registered analysis priorities.

## Evaluation and analysis

Evaluate eight fixed locally runnable heldout Docker tasks through pinned Harbor, once per mode and prompt condition: seed 2002, thinking disabled, maximum 16 turns and 120 seconds per agent episode. Audit intent using opaque packets before unblinding arms. The primary comparison is elicited hacking in probe-GRAFT retain-only versus no intervention both; legitimate success has a five-percentage-point safeguard. Wilson intervals describe known binary outcomes; ambiguity is retained as bounds. Paired task bootstrap uses 10,000 draws, seed 42, and the exact discordance test is descriptive. Missing outcomes are never imputed as benign. Detector intervals resample whole tasks.

## Results

Training coverage: no_intervention: 20 updates, complete=True; probe_graft: 20 updates, complete=True; qwen_judge_graft: 20 updates, complete=True; original_graft: 0 updates, complete=False; probe_filtering: 0 updates, complete=False; random_graft: 0 updates, complete=False; smoke: 1 updates, complete=True.

Downloaded probe: 194/194 heldout trajectories across 99 tasks; 95 positive and 99 negative labels. auroc 0.497 (95% interval 0.424 to 0.562); precision 0.500 (95% interval 0.143 to 0.857); recall 0.042 (95% interval 0.010 to 0.084); fpr 0.040 (95% interval 0.010 to 0.081). Frozen cutoff 0.00096981943; calibration false positives 10/203.

Qwen judge: 194/194 heldout trajectories across 99 tasks; 95 positive and 99 negative labels. auroc 0.569 (95% interval 0.494 to 0.639); precision 0.368 (95% interval 0.154 to 0.579); recall 0.074 (95% interval 0.021 to 0.129); fpr 0.121 (95% interval 0.061 to 0.192). Frozen cutoff 0.81757444; calibration false positives 9/203.

Realized training flags: probe_graft: 3/192, including 1 labeled hacks and 2 labeled clean trajectories; probe_filtering: 3/192, including 1 labeled hacks and 2 labeled clean trajectories; random_graft: 3/192, including 2 labeled hacks and 1 labeled clean trajectories; no_intervention: 0/192, including 0 labeled hacks and 0 labeled clean trajectories; original_graft: 19/192, including 19 labeled hacks and 0 labeled clean trajectories; qwen_judge_graft: 15/192, including 10 labeled hacks and 5 labeled clean trajectories.

Primary elicited differences (probe retain minus no intervention both): hacking: undefined, 0 known task pairs; legitimate: undefined, 0 known task pairs.

Secondary elicited differences (LLM-judge retain minus no intervention both): hacking: undefined, 0 known task pairs; legitimate: undefined, 0 known task pairs.

Secondary elicited differences (LLM-judge retain minus probe retain): hacking: undefined, 0 known task pairs; legitimate: undefined, 0 known task pairs. Full paired outcomes and missingness are in results.json and audited_outcomes.json.

## Claims and implications

Mitigation effectiveness is inconclusive. The primary comparison is incomplete or has ambiguous judgments. Successful checkpoint loading, kernel equivalence and successful Docker execution support engineering compatibility only. Reported treatment differences are conditional descriptive observations, not claims of robust alignment or a causal mechanism beyond the implemented adapter intervention.

## Discussion and limitations

One training seed, eight selected tasks, a much smaller model than the source paper, task difficulty, label/source confounding, unresolved upstream detector exposure, truncation and non-independent implementing-assistant audits limit generalization. Neither monitor score is assumed calibrated. The judge/probe treatment contrast changes both flag frequency and which trajectories are flagged; the random-monitor control matches the probe's flag counts only. At n=8 one binary outcome changes a rate by 12.5 percentage points, larger than the 5pp capability tolerance. Bootstrap intervals may degenerate at the boundary; exact paired discordance evidence and Wilson intervals remain visible. The registered deadline can leave arms or modes incomplete. The user authorized a ten-more-hours extension before optimizer updates or behavioral outcomes; budget.json preserves both deadlines. Training error and infrastructure error are reported separately from behavior. No result-dependent extension or threshold retuning is allowed.

## Conclusion

Mitigation effectiveness is inconclusive. Larger, multi-seed studies with measured task/seed variance and independently blinded audits are required for stronger conclusions.

## Infrastructure recovery and coverage

40 initial Docker setup failures were archived before any agent or verifier execution. The pinned runner's cleanup had removed the shared local images. A dated post-start amendment permits one uniform recovery of wholly unstarted jobs, preserving their results and hashes in setup_recovery.json. Actual model attempts, including timeouts, are never repeated. Rebuilt executable platform manifests match the original pre-study build logs; attestation/index changes and their evidence are recorded in image_identity_recovery.json. Repairs consume the unchanged deadline. The 120-second agent limit makes measured capability conditional on this local inference implementation and throughput.

## Appendix: prospective power and reproducibility

Normal-approximation sample requirements for a 10pp reduction at 80% power and two-sided alpha .05: [{"baseline": 0.1, "treatment": 0.0, "n_per_arm": 74}, {"baseline": 0.2, "treatment": 0.1, "n_per_arm": 199}, {"baseline": 0.3, "treatment": 0.19999999999999998, "n_per_arm": 294}, {"baseline": 0.5, "treatment": 0.4, "n_per_arm": 388}]. Paired discordance and unknown seed-variance sensitivity are in results.json; they are planning assumptions. uv.lock pins the environment; artifacts.json pins sources and weights; data_lock.json fixes examples and preprocessing; per-checkpoint configuration and progress files record all actual updates. Full prompts, responses, scores and judgments are retained in timestamped local logs. Run uv run python -m experiments.qwen_probe run --profile pilot.

## References

[1] GRAFT. https://arxiv.org/abs/2610.17062
[2] farzingkh/reward-hacking, revision cd4c566824c5961eaeab7b7fce7845240bd12b3d (MIT source). https://github.com/farzingkh/reward-hacking
[3] Qwen3-0.6B. https://huggingface.co/Qwen/Qwen3-0.6B
[4] Qwen3-Embedding-0.6B. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
[5] Accelerate gradient accumulation. https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
[6] Liger Kernel. https://github.com/linkedin/Liger-Kernel
[7] SciPy binomial inference. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
[8] NeurIPS reporting checklist. https://neurips.cc/public/guides/PaperChecklist
[9] Lones (2021), ML pitfalls. https://arxiv.org/abs/2108.02497
[10] Lipton and Steinhardt (2018), Troubling Trends. https://arxiv.org/abs/1807.03341
[11] Zheng et al. (2023), Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. https://arxiv.org/abs/2306.05685

![architecture](architecture.png)

![training](training.png)

![detector](detector.png)
