# original_graft

Measure the small-model GRAFT recipe with the paper's idealized monitor: zero false positives and approximately 20% positive recall after integer rounding. This separates detector limitations from a method/capability floor.

Updates: 0. Complete planned training: False.

Method and hyperparameter rationale: [preregistered protocol](../../docs/protocol.md). Exact optimizer arguments are in the per-arm configuration JSON when the run has started. The smoke checkpoint is an engineering check and excluded from study comparisons.

Observed audited behavior (an empty object means no unblinded study measurements):

```json
{}
```

Learning: completion of parameter updates alone does not establish hacking mitigation; inspect the blinded behavioral comparison and capability safeguard in [summary](summary.md).
