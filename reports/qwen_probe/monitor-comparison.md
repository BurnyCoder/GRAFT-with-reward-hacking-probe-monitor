# Frozen monitor comparison

Downloaded probe: 194/194 heldout trajectories across 99 tasks; 95 positive and 99 negative labels. auroc 0.497 (95% interval 0.424 to 0.562); precision 0.500 (95% interval 0.143 to 0.857); recall 0.042 (95% interval 0.010 to 0.084); fpr 0.040 (95% interval 0.010 to 0.081). Frozen cutoff 0.00096981943; calibration false positives 10/203.

Same-Qwen judge: 194/194 heldout trajectories across 99 tasks; 95 positive and 99 negative labels. auroc 0.569 (95% interval 0.494 to 0.639); precision 0.368 (95% interval 0.154 to 0.579); recall 0.074 (95% interval 0.021 to 0.129); fpr 0.121 (95% interval 0.061 to 0.192). Frozen cutoff 0.81757444; calibration false positives 9/203.

Both monitors use the same locked heldout trajectories and independently frozen calibration cutoffs. The judge uses the untouched learner and a fixed intent rubric; its score is conditional YES/NO odds. Neither monitor is fitted on these evaluation tasks. See the [protocol](../../docs/protocol.md) for exact inputs and context limits.

| Metric | Judge minus probe | Paired task-bootstrap 95% interval |
|---|---:|---|
| auroc | 0.072 | [-0.03281236859809947, 0.176293713966525] |
| precision | -0.132 | [-0.5837938596491229, 0.3111473429951691] |
| recall | 0.032 | [-0.03158734602463606, 0.10309278350515463] |
| fpr | 0.081 | [0.019949494949494957, 0.15151515151515152] |

These secondary intervals are unadjusted and conditional on the fixed thresholds. A calibration FPR below 5% does not guarantee that heldout FPR is below 5%. Precision reflects the selected baseline-clean/rewarded-hack mixture, not deployment prevalence. Training flags and generated behavior measure different quantities; detector validation alone does not establish mitigation.

The upstream probe's training-task exposure remains unresolved ([provenance audit](../../docs/probe-provenance.md)). Source labels, task/source composition and different truncation capacities limit the comparison. Complete coverage and score-derived counts are retained in [results.json](results.json).
