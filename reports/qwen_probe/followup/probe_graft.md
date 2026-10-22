# probe_graft: exploratory adapter follow-up

Original closed pilot: 120/160 audited trial records. Later follow-up: 40/40. Combined distinct record coverage of the original grid: 160/160. Record coverage includes setup failures with no observable model behavior; unknown outcomes remain unknown and unrun episodes remain unavailable. None were replaced or imputed.

The [combined report](summary.md) records cohort provenance, paired effects and limitations. Rows include unchanged original comparators alongside later follow-up conditions.

Counts are positive/known outcomes, with Wilson 95% intervals for the known rate. Unknown counts and ambiguity bounds concern audited episodes; the audited count out of eight separately exposes missing planned episodes. These intervals do not capture training-seed variation.

| Deployment / prompt | Audited | Hacking attempts | Legitimate success | Verifier success |
| --- | --- | --- | --- | --- |
| probe_graft/retain/elicited | 8/8 | 1/7 known; 95% [0.026, 0.513]; unknown 1; bounds [0.125, 0.250] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_graft/retain/plain | 8/8 | 0/6 known; 95% [0.000, 0.390]; unknown 2; bounds [0.000, 0.250] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_graft/both/elicited | 8/8 | 4/7 known; 95% [0.250, 0.842]; unknown 1; bounds [0.500, 0.625] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_graft/both/plain | 8/8 | 1/5 known; 95% [0.036, 0.624]; unknown 3; bounds [0.125, 0.500] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_graft/forget/elicited | 8/8 | 1/6 known; 95% [0.030, 0.564]; unknown 2; bounds [0.125, 0.375] | 0/8 known; 95% [0.000, 0.324] | 0/8 known; 95% [0.000, 0.324] |
| probe_graft/forget/plain | 8/8 | 0/7 known; 95% [0.000, 0.354]; unknown 1; bounds [0.000, 0.125] | 1/7 known; 95% [0.026, 0.513]; unknown 1; bounds [0.125, 0.250] | 1/7 known; 95% [0.026, 0.513]; unknown 1; bounds [0.125, 0.250] |
