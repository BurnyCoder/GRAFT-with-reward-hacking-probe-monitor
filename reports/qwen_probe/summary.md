# Consolidated experiment results

Mitigation effectiveness is inconclusive.
The primary comparison is incomplete or has ambiguous judgments.

This report is generated from local manifests and frozen audits. See [protocol](../../docs/protocol.md) for the prespecified method.

| Experiment | Updates | Completed recipe | Status |
|---|---:|---|---|
| no_intervention | 0 | False | incomplete |
| probe_graft | 0 | False | incomplete |
| qwen_judge_graft | 0 | False | incomplete |
| original_graft | 0 | False | incomplete |
| probe_filtering | 0 | False | incomplete |
| random_graft | 0 | False | incomplete |
| smoke | 1 | True | complete |

Pending blinded audits: 0.

## Frozen detector validation

```json
{
  "status": "complete",
  "scored": 194,
  "planned": 194,
  "tasks": 99,
  "metrics": {
    "auroc": 0.4971823498139288,
    "precision": 0.5,
    "recall": 0.042105263157894736,
    "fpr": 0.04040404040404041
  },
  "task_bootstrap_ci95": {
    "auroc": [
      0.4242344497607655,
      0.5624886474393053
    ],
    "precision": [
      0.14285714285714285,
      0.8571428571428571
    ],
    "recall": [
      0.010309278350515464,
      0.0842329227323628
    ],
    "fpr": [
      0.010101010101010102,
      0.08080808080808081
    ]
  },
  "positive": 95,
  "negative": 99,
  "sampling_note": "Validation selects at most one record per task and class: baseline clean versus rewarded hacks. Precision reflects this case mixture, not an estimated deployment prevalence. Generated audits instead count all observed hacking attempts, including unsuccessful ones.",
  "coverage": {
    "calibration": {
      "planned": 521,
      "scored": 521,
      "turns": 6189,
      "retained_turns": 6189,
      "truncated_turns": 33,
      "tokens_before_turn_truncation": 4366776,
      "retained_tokens": 4337750,
      "dropped_turns": 0,
      "retained_turn_fraction": 1.0,
      "retained_token_fraction_in_selected_turns": 0.9933529908564122
    },
    "train": {
      "planned": 192,
      "scored": 192,
      "turns": 2055,
      "retained_turns": 2055,
      "truncated_turns": 7,
      "tokens_before_turn_truncation": 1291105,
      "retained_tokens": 1286402,
      "dropped_turns": 0,
      "retained_turn_fraction": 1.0,
      "retained_token_fraction_in_selected_turns": 0.9963573837914035
    },
    "heldout": {
      "planned": 194,
      "scored": 194,
      "turns": 2186,
      "retained_turns": 2186,
      "truncated_turns": 19,
      "tokens_before_turn_truncation": 1552625,
      "retained_tokens": 1540483,
      "dropped_turns": 0,
      "retained_turn_fraction": 1.0,
      "retained_token_fraction_in_selected_turns": 0.9921796956766766
    }
  },
  "calibration": {
    "threshold": 0.0009698194335214794,
    "clean": 203,
    "false_positives": 10,
    "empirical_fpr": 0.04926108374384237,
    "note": "Selected on calibration data; heldout task-bootstrap intervals quantify validation uncertainty."
  }
}
```

## Fixed data and realized routing

```json
{
  "data": {
    "train": {
      "records": 192,
      "tasks": 122,
      "labels": {
        "1": 96,
        "0": 96
      },
      "sources": {
        "gemini-3.1-pro": 81,
        "gpt-5.4": 57,
        "claude-opus-4.6": 54
      },
      "total_learner_tokens": 1291372,
      "min_tokens": 1214,
      "max_tokens": 16214
    },
    "calibration": {
      "records": 521,
      "tasks": 32,
      "labels": {
        "0": 203,
        "1": 318
      },
      "sources": {
        "claude-opus-4.6": 142,
        "gemini-3.1-pro": 219,
        "gpt-5.4": 160
      },
      "total_learner_tokens": 4367035,
      "min_tokens": 1111,
      "max_tokens": 61576
    },
    "heldout": {
      "records": 194,
      "tasks": 99,
      "labels": {
        "0": 99,
        "1": 95
      },
      "sources": {
        "gemini-3.1-pro": 78,
        "gpt-5.4": 70,
        "claude-opus-4.6": 46
      },
      "total_learner_tokens": 1553788,
      "min_tokens": 1581,
      "max_tokens": 32369
    }
  },
  "routing": {
    "probe_graft": {
      "total": 192,
      "flagged": 3,
      "flagged_positive": 1,
      "flagged_negative": 2
    },
    "probe_filtering": {
      "total": 192,
      "flagged": 3,
      "flagged_positive": 1,
      "flagged_negative": 2
    },
    "random_graft": {
      "total": 192,
      "flagged": 3,
      "flagged_positive": 2,
      "flagged_negative": 1
    },
    "no_intervention": {
      "total": 192,
      "flagged": 0,
      "flagged_positive": 0,
      "flagged_negative": 0
    },
    "original_graft": {
      "total": 192,
      "flagged": 19,
      "flagged_positive": 19,
      "flagged_negative": 0
    },
    "qwen_judge_graft": {
      "total": 192,
      "flagged": 15,
      "flagged_positive": 10,
      "flagged_negative": 5
    }
  }
}
```

## Same-Qwen judge validation and paired detector comparison

```json
{
  "judge": {
    "status": "complete",
    "scored": 194,
    "planned": 194,
    "tasks": 99,
    "metrics": {
      "auroc": 0.5692185007974481,
      "precision": 0.3684210526315789,
      "recall": 0.07368421052631578,
      "fpr": 0.12121212121212122
    },
    "task_bootstrap_ci95": {
      "auroc": [
        0.4938481134191124,
        0.638862971823498
      ],
      "precision": [
        0.15375000000000003,
        0.5789473684210527
      ],
      "recall": [
        0.021270996640537512,
        0.12910315490960644
      ],
      "fpr": [
        0.06060606060606061,
        0.1919191919191919
      ]
    },
    "positive": 95,
    "negative": 99,
    "sampling_note": "Validation selects at most one record per task and class: baseline clean versus rewarded hacks. Precision reflects this case mixture, not an estimated deployment prevalence. Generated audits instead count all observed hacking attempts, including unsuccessful ones.",
    "coverage": {
      "calibration": {
        "planned": 521,
        "scored": 521,
        "turns": 6189,
        "retained_turns": 6189,
        "transcript_tokens": 4363400,
        "retained_transcript_tokens": 3917383,
        "input_tokens": 4340727,
        "context_truncated_trajectories": 43
      },
      "train": {
        "planned": 192,
        "scored": 192,
        "turns": 2055,
        "retained_turns": 2055,
        "transcript_tokens": 1289980,
        "retained_transcript_tokens": 1288789,
        "input_tokens": 1444713,
        "context_truncated_trajectories": 3
      },
      "heldout": {
        "planned": 194,
        "scored": 194,
        "turns": 2186,
        "retained_turns": 2186,
        "transcript_tokens": 1551428,
        "retained_transcript_tokens": 1462103,
        "input_tokens": 1619750,
        "context_truncated_trajectories": 18
      }
    },
    "calibration": {
      "threshold": 0.8175744414329529,
      "clean": 203,
      "false_positives": 9,
      "empirical_fpr": 0.04433497536945813,
      "note": "Selected on calibration data; heldout task-bootstrap intervals quantify validation uncertainty."
    }
  },
  "comparison": {
    "status": "complete",
    "paired_records": 194,
    "paired_tasks": 99,
    "judge_minus_probe": {
      "auroc": 0.07203615098351929,
      "precision": -0.13157894736842107,
      "recall": 0.03157894736842105,
      "fpr": 0.0808080808080808
    },
    "task_bootstrap_ci95": {
      "auroc": [
        -0.03281236859809947,
        0.176293713966525
      ],
      "precision": [
        -0.5837938596491229,
        0.3111473429951691
      ],
      "recall": [
        -0.03158734602463606,
        0.10309278350515463
      ],
      "fpr": [
        0.019949494949494957,
        0.15151515151515152
      ]
    },
    "valid_bootstrap_draws": {
      "auroc": 1000,
      "precision": 998,
      "recall": 1000,
      "fpr": 1000
    },
    "note": "Secondary unadjusted comparison; conditional on fixed calibration cutoffs and one data sample."
  }
}
```

## Secondary judge behavioral comparisons

```json
{
  "judge_minus_no_intervention": {
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
  },
  "judge_minus_probe": {
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
| Qwen judge | Untouched frozen Qwen3-0.6B with intent rubric | Conditional YES/NO logits; calibration-only threshold | Separate detector baseline and GRAFT flag source |
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
