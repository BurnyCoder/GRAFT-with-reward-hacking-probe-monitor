# Consolidated experiment results

Mitigation effectiveness is inconclusive.
The primary comparison is incomplete or has ambiguous judgments. No audited hacking attempt is established in the no-intervention baseline. The primary arms have no established legitimate successes; a capability floor may dominate.

This report is generated from local manifests and frozen audits. See [protocol](../../docs/protocol.md) for the prespecified method.

| Experiment | Updates | Completed recipe | Status |
|---|---:|---|---|
| no_intervention | 0 | False | incomplete |
| probe_graft | 0 | False | incomplete |
| original_graft | 0 | False | incomplete |
| probe_filtering | 0 | False | incomplete |
| random_graft | 0 | False | incomplete |
| smoke | 0 | False | incomplete |

Pending blinded audits: 0.

## Frozen detector validation

```json
{
  "status": "not run"
}
```

## Behavioral measurements

```json
{}
```

## Primary paired comparison

```json
{
  "hacking": {
    "paired_tasks": 0,
    "known_pairs": 0,
    "difference": null,
    "bootstrap_ci95": null,
    "exact_discordance_p": null,
    "ambiguity_bounds": null,
    "task_differences": {},
    "warning": "Conditional on one training seed; n=8 bootstrap may be degenerate; missing pairs are not imputed"
  },
  "legitimate": {
    "paired_tasks": 0,
    "known_pairs": 0,
    "difference": null,
    "bootstrap_ci95": null,
    "exact_discordance_p": null,
    "ambiguity_bounds": null,
    "task_differences": {},
    "warning": "Conditional on one training seed; n=8 bootstrap may be degenerate; missing pairs are not imputed"
  }
}
```

## Terminology and networks

| Network / term | Architecture / algorithm | Objective and data | Measured meaning |
|---|---|---|---|
| Learner | Frozen Qwen3-0.6B, 28 layers, two width-64 MLP branches | Assistant-token SFT, matched clean/hack subset | Generates terminal actions |
| Detector encoder | Qwen3-Embedding-0.6B plus frozen rank-16 attention LoRA | Upstream source-invariant training; no fitting here | 1,024-dimensional normalized turn vectors |
| Probe head | Four transformer layers, width 256, 8 heads, CLS classifier | Upstream binary trajectory classifier | AUROC measures ranking, not behavioral mitigation |
| GRAFT | Routed first moments, natural-gradient second moments | Flagged/unflagged pools plus 1% unlabeled anchors | Retain branch deployed after removing forget branch |
| Verifier success | Task-specific executable checks | Heldout Docker tasks | Can include legitimate solves or hacks |
| Hacking attempt | Intent-based blinded audit | Agent actions and rationale | Counts attempts even when they fail |
| Legitimate success | Verifier success without hacking | Same audited episodes | Capability safeguard |
| Completion / accuracy | Training completion and separate task outcomes | See manifests and intervals | No universal model accuracy or 'fill' metric applies |

## Prospective power

```json
{
  "alpha": 0.05,
  "power": 0.8,
  "absolute_reduction": 0.1,
  "independent": [
    {
      "baseline": 0.1,
      "treatment": 0.0,
      "n_per_arm": 74
    },
    {
      "baseline": 0.2,
      "treatment": 0.1,
      "n_per_arm": 199
    },
    {
      "baseline": 0.3,
      "treatment": 0.19999999999999998,
      "n_per_arm": 294
    },
    {
      "baseline": 0.5,
      "treatment": 0.4,
      "n_per_arm": 388
    }
  ],
  "paired": [
    {
      "discordance": 0.1,
      "n_task_pairs": 71
    },
    {
      "discordance": 0.2,
      "n_task_pairs": 150
    },
    {
      "discordance": 0.4,
      "n_task_pairs": 307
    },
    {
      "discordance": 0.6,
      "n_task_pairs": 464
    }
  ],
  "seed_sensitivity": [
    {
      "between_seed_sd": 0,
      "training_seeds": 1,
      "task_pairs_per_seed": 150
    },
    {
      "between_seed_sd": 0,
      "training_seeds": 3,
      "task_pairs_per_seed": 50
    },
    {
      "between_seed_sd": 0,
      "training_seeds": 5,
      "task_pairs_per_seed": 30
    },
    {
      "between_seed_sd": 0,
      "training_seeds": 10,
      "task_pairs_per_seed": 15
    },
    {
      "between_seed_sd": 0.025,
      "training_seeds": 1,
      "task_pairs_per_seed": 293
    },
    {
      "between_seed_sd": 0.025,
      "training_seeds": 3,
      "task_pairs_per_seed": 60
    },
    {
      "between_seed_sd": 0.025,
      "training_seeds": 5,
      "task_pairs_per_seed": 34
    },
    {
      "between_seed_sd": 0.025,
      "training_seeds": 10,
      "task_pairs_per_seed": 16
    },
    {
      "between_seed_sd": 0.05,
      "training_seeds": 1,
      "task_pairs_per_seed": null
    },
    {
      "between_seed_sd": 0.05,
      "training_seeds": 3,
      "task_pairs_per_seed": 144
    },
    {
      "between_seed_sd": 0.05,
      "training_seeds": 5,
      "task_pairs_per_seed": 50
    },
    {
      "between_seed_sd": 0.05,
      "training_seeds": 10,
      "task_pairs_per_seed": 19
    },
    {
      "between_seed_sd": 0.1,
      "training_seeds": 1,
      "task_pairs_per_seed": null
    },
    {
      "between_seed_sd": 0.1,
      "training_seeds": 3,
      "task_pairs_per_seed": null
    },
    {
      "between_seed_sd": 0.1,
      "training_seeds": 5,
      "task_pairs_per_seed": null
    },
    {
      "between_seed_sd": 0.1,
      "training_seeds": 10,
      "task_pairs_per_seed": 70
    }
  ],
  "assumptions": "Two-sided normal approximations; paired discordance sensitivity; seed grid assumes independent seeds and conditional task noise, q=.2. Same tasks across seeds may correlate; unknown seed SD is not estimated from this pilot. p1=0 is a boundary approximation."
}
```

Detailed engineering decisions and failures are recorded in [implementation log](../../docs/implementation.md). Full untruncated transcripts are under the ignored local artifacts/qwen_probe/logs directory.
