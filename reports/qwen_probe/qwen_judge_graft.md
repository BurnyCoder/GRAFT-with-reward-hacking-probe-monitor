# qwen_judge_graft

Compare a frozen untouched Qwen3-0.6B intent judge with the downloaded probe, both as a detector and a source of GRAFT flags. The independent calibration cutoff and identical training recipe make this a secondary monitor comparison; constrained binary scoring is not a generated rationale.

Updates: 0. Complete planned training: False.

Method and hyperparameter rationale: [preregistered protocol](../../docs/protocol.md). Exact optimizer arguments are in the per-arm configuration JSON when the run has started. The smoke checkpoint is an engineering check and excluded from study comparisons.

Observed audited behavior (an empty object means no unblinded study measurements):

```json
{}
```

Learning: completion of parameter updates alone does not establish hacking mitigation; inspect the blinded behavioral comparison and capability safeguard in [summary](summary.md).
