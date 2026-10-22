# Implementation and delivery plan

This checklist tracks delivery. The [preregistered protocol](protocol.md) owns scientific choices and dated amendments; [results](../reports/qwen_probe/summary.md) own measurements. An unchecked item is unfinished, not a negative experimental result.

- [x] Initialize the root uv project, local virtual environment, workspace caches, `.env` configuration and compatible dependency lock; preserve the upstream experiment.
- [x] Implement independently callable preparation, frozen-monitor validation, training, Docker evaluation, blinded audit, report and publication phases behind one run command.
- [x] Pin and verify the downloaded probe artifacts and reproduce the released encoder/head preprocessing, capacity and truncation accounting.
- [x] Add the untouched Qwen3-0.6B LLM judge as both a detector comparator and a separate GRAFT treatment, with its own frozen calibration cutoff.
- [x] **Headline both LLM-as-a-judge and the downloaded probe in the README and paper**, including the abstract and method description. Preserve the registered primary and secondary comparisons.
- [x] Lock task separation, calibration records, the balanced training set, common context limit, record order and initialization seed.
- [x] Verify metadata exclusion, threshold freezing, routing, accumulation scaling, Liger loss/gradient equivalence and small-model adapter merging; retain test and engineering evidence.
- [x] Publish implementation and preregistration before real-model optimizer updates; complete the real download → score → train → merge → Docker → audit → report smoke path.
- [ ] Run the registered training arms and agentic evaluations in their fixed priority order under the persisted amended deadline; explicitly record any unfinished comparisons.
- [ ] Review generated study trajectories in opaque audit packets, save evidence and judgments before unblinding, and preserve unknown intent as bounds.
- [ ] Generate final per-experiment and consolidated reports, terminology/network table, result images and sourced PDF by Libor Burian, including uncertainty, prospective power, contamination limits and capability safeguards.
- [ ] Publish the resulting pilot checkpoints, configurations and model cards to the Hugging Face collection and verify the remote artifacts.
- [ ] Merge the final results/publication PR and verify that README links, documentation and evidence agree.

Completed implementation and presentation PRs: [implementation #1](https://github.com/BurnyCoder/adapting-gradient-routing-adapter-finetuning-with-reward-hacking-probe-and-LLM-as-a-judge-monitors/pull/1), [monitor framing #2](https://github.com/BurnyCoder/adapting-gradient-routing-adapter-finetuning-with-reward-hacking-probe-and-LLM-as-a-judge-monitors/pull/2). An executed engineering smoke demonstrates integration, including a failed task outcome; it does not establish mitigation. See the [engineering report](../reports/qwen_probe/engineering.md) for exact checks and failures.
