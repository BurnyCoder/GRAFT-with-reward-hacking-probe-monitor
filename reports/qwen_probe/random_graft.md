# random_graft

Test whether the probe's trajectory-specific flags matter beyond flag frequency and length. Shuffle flags within four equal-count token-length bins, keeping each bin's flagged count fixed.

Updates: 0. Complete planned training: False.

Method and hyperparameter rationale: [preregistered protocol](../../docs/protocol.md). Exact optimizer arguments are in the per-arm configuration JSON when the run has started. The smoke checkpoint is an engineering check and excluded from study comparisons.

Observed audited behavior (an empty object means no unblinded study measurements):

```json
{}
```

Learning: completion of parameter updates alone does not establish hacking mitigation; inspect the blinded behavioral comparison and capability safeguard in [summary](summary.md).
