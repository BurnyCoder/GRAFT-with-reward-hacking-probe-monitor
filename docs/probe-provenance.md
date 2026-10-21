# Downloaded detector provenance audit

Checked 2026-10-21 UTC against the pinned [repository tree](https://github.com/farzingkh/reward-hacking/tree/cd4c566824c5961eaeab7b7fce7845240bd12b3d) and its [released notebook](https://github.com/farzingkh/reward-hacking/blob/cd4c566824c5961eaeab7b7fce7845240bd12b3d/notebooks/cs1090b_ms4_main_group29.ipynb). The tree contains `data/.gitkeep`, but no training dataset, split manifest or list of training trajectory identifiers. The notebook loads four external sources and later fine-tunes the encoder and classifier. The local evidence copies are `artifacts/qwen_probe/sources/probe-tree.json` and `probe.ipynb`.

| Notebook source | Available provenance | Limit of this audit |
|---|---|---|
| [PatronusAI TRACE](https://huggingface.co/datasets/PatronusAI/trace-dataset) | Public card describes synthetic, human-verified code-environment trajectories; access to records is gated | The card cannot establish exact task or trajectory non-overlap |
| [METR MALT](https://huggingface.co/datasets/metr-evals/malt-public) | Public card describes agent transcripts across non-internal task families; records are gated | Task-name aliases and selected upstream training records are unresolved |
| [Realistic Reward Hacks](https://huggingface.co/datasets/Jozdien/realistic_reward_hacks) | Public text-conversation dataset, named by notebook loaders | No artifact-specific selected-record manifest is released |
| [BJS reward-hacking corpus](https://github.com/BJS-Innovation-Lab/reward-hacking-corpus) | Named in the pinned notebook | The linked repository returned 404 during this audit; its absence does not establish what was used upstream |

This is an audit of available provenance, not proof of clean pretraining exposure. No independent detector training is performed here. Hash verification establishes which downloaded files were used; it does not establish their training corpus or licensing beyond the available declarations. Upstream code carries the copied MIT license. The Drive folder does not supply a separate weight-license declaration, so the downloaded detector is not redistributed with the pilot adapters.

Our own task isolation and exact normalized-trajectory duplicate checks are recorded separately in the generated `reports/qwen_probe/provenance.json`. They address this pilot's train/calibration/heldout separation only. Semantic duplicates, aliases, common task templates, Qwen pretraining exposure and upstream detector exposure remain unresolved. Consequently, heldout detector metrics are conditional measurements on this task partition, not certified out-of-distribution performance.
