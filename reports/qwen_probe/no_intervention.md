# no_intervention

Establish the effect of learning the matched clean/hack mixture with both adapters and the same split-moment optimizer, without routing. This is the primary comparator, not the untouched base model.

Updates: 0. Complete planned training: False.

Method and hyperparameter rationale: [preregistered protocol](../../docs/protocol.md). Exact optimizer arguments are in the per-arm configuration JSON when the run has started. The smoke checkpoint is an engineering check and excluded from study comparisons.

Observed audited behavior (an empty object means no unblinded study measurements):

```json
{}
```

Learning: completion of parameter updates alone does not establish hacking mitigation; inspect the blinded behavioral comparison and capability safeguard in [summary](summary.md).
