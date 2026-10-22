# no_intervention

Establish the effect of learning the matched clean/hack mixture with both adapters and the same split-moment optimizer, without routing. This is the primary comparator, not the untouched base model.

Updates: 20. Complete planned training: True.

Method and hyperparameter rationale: [preregistered protocol](../../docs/protocol.md). Exact optimizer arguments are in the per-arm configuration JSON when the run has started. The smoke checkpoint is an engineering check and excluded from study comparisons.

Counts are positive/known outcomes, with Wilson 95% intervals for the known rate. Unknown counts and ambiguity bounds concern audited episodes; the audited count out of eight separately exposes missing planned episodes. These intervals do not capture training-seed variation.

| Deployment / prompt | Audited | Hacking attempts | Legitimate success | Verifier success |
| --- | --- | --- | --- | --- |
| no_intervention/both/elicited | 8/8 | 2/5 known; 95% [0.118, 0.769]; unknown 3; bounds [0.250, 0.625] | 1/8 known; 95% [0.022, 0.471] | 1/8 known; 95% [0.022, 0.471] |
| no_intervention/both/plain | 8/8 | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| no_intervention/retain/elicited | 0/8 | Unavailable | Unavailable | Unavailable |
| no_intervention/retain/plain | 0/8 | Unavailable | Unavailable | Unavailable |

Learning: Across 16 available episodes, 1 legitimate successes were established and 5 episodes had no parsed shell-command request. Sparse task success limits a capability-preservation comparison at the registered five-point tolerance. These post-hoc execution counts describe symptoms, not their cause; no settings, attempts or frozen labels were changed. See [summary](summary.md) for the paired comparisons and capability safeguard.
