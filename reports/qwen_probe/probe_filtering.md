# probe_filtering

Test discarding flagged learning signal through the existing zero-weight single-adapter filtering baseline. Ordinary AdamW and fewer nonzero-loss examples differ from GRAFT, so this comparison cannot isolate routing alone.

Updates: 0. Complete planned training: False.

Method and hyperparameter rationale: [preregistered protocol](../../docs/protocol.md). Exact optimizer arguments are in the per-arm configuration JSON when the run has started. The smoke checkpoint is an engineering check and excluded from study comparisons.

Observed audited behavior (an empty object means no unblinded study measurements):

```json
{}
```

Learning: completion of parameter updates alone does not establish hacking mitigation; inspect the blinded behavioral comparison and capability safeguard in [summary](summary.md).
