# probe_graft

Test whether the downloaded out-of-domain detector supplies useful routing flags. Its encoder/head remain frozen; no downstream behavior is used to recalibrate the threshold. Removing the forget adapter is the primary deployment intervention.

Updates: 20. Complete planned training: True.

Method and hyperparameter rationale: [preregistered protocol](../../docs/protocol.md). Exact optimizer arguments are in the per-arm configuration JSON when the run has started. The smoke checkpoint is an engineering check and excluded from study comparisons.

Observed audited behavior (an empty object means no unblinded study measurements):

```json
{}
```

Learning: completion of parameter updates alone does not establish hacking mitigation; inspect the blinded behavioral comparison and capability safeguard in [summary](summary.md).
