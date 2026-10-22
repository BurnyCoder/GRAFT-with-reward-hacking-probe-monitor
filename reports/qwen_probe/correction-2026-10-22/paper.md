# Qwen3-0.6B GRAFT with LLM-as-a-Judge and Probe Monitors: scientific correction

Libor Burian — post-hoc scientific correction, 22 October 2026

## Abstract

A substantial ATIF-to-SFT conversion defect affected every historical training trajectory. Completion tool calls became invalid empty commands, and their completion flags were lost. Corrected copies of all 932 assistant targets now parse and preserve source semantics, but no checkpoint was retrained or reevaluated. Saved training losses and response parsing also show severe degradation in the judge-routed arm. These diagnostics compromise interpretation of behavioral mitigation; the conclusion remains **mitigation effectiveness is inconclusive**. Numerical summaries, annotations and historical reports remain unchanged. [1–4]

## Scope and method

This correction was requested after the pilot and follow-up had closed and outcomes were known. It is post hoc, not a new preregistration or confirmatory experiment. We read the actual pinned source trajectories, the locked saved training pool, raw optimizer metrics and saved response packets. Previous audit conclusions were not evidence for these checks. Only Python conversion, parsing, hashing, tests and descriptive arithmetic ran: no tokenization, model loading, model execution, parameter updates, deployment evaluation or intent relabeling occurred. The published check file records the converter/parser identities, per-trajectory source and target hashes, raw metric hashes and packet hashes. [1,5]

## Conversion defect and repair

Harbor emits shell actions as `bash_command` and completion as a separate `mark_task_complete` call with empty arguments. The historical converter copied every tool's arguments into `commands`, creating `{}` for completion and failing to restore `task_complete: true`. The parser requires `keystrokes` in every command. [2,3]

In the 192 locked training trajectories, 192/932 historical assistant targets were rejected, affecting 192/192 trajectories; no historical target retained the completion signal. Reconstructing targets from the same sources with the corrected converter yields 932/932 parseable targets, 2,626 unchanged shell commands and all 192 completion turns. Fourteen targets retain nonfatal source newline warnings; none loses or changes a command. Parser acceptance is an interface check, not evidence that a command is safe, correct or task-solving. [1]

The implementation now separates commands from completion, rejects unsupported tools, validates new inventory targets, and fingerprints conversion identity and actual converted messages. Preparation/training reject legacy conversion metadata and require a separate artifact root for future experiments. The original converted targets and locks remain historical evidence. This repair does not retroactively validate existing checkpoints or identify how much of the observed behavioral failure the defect caused. [1,5]

## Training and behavioral degradation

The judge-routed arm's recorded unclassified-example mean training loss was 1.37099 at update 1, 12.75614 at update 2, 16.40138 at update 3 and 10.27991 at update 20. All 256 saved responses across its 16 original pilot episodes fail the pinned parser. The original simulated-monitor arm also shows early loss instability: 1.35675 at update 1, a maximum of 13.38178 at update 3, and 1.68587 at update 20; all 232 saved responses across its 16 episodes fail the parser. [1,4]

These are successive batch means over changing examples and routing groups, not losses on a fixed validation set. They are descriptive evidence of unstable/degraded training, especially when combined with total response-parser failure; they do not isolate a causal mechanism or prove that the conversion defect is the sole cause. Completing 20 updates measures scheduled execution, not successful learning. Neither arm demonstrates selective hacking mitigation when it cannot use the task interface. Raw responses remain relevant to intent judgments; parsing failure does not establish either benign intent or a hacking attempt. [1,4]

## Statistical interpretation

The original primary hacking comparison has only 5/8 unambiguous task pairs: the known-pair difference is −40 percentage points, the 95% paired bootstrap interval is [−80, 0] points, and the two-sided exact paired discordance p-value is 0.50. Allowing every assignment of the ambiguous labels across all eight pairs gives bounds [−50, 0] points. The original probe retain-versus-both ablation has 6/8 known pairs, difference −50 points and bootstrap interval [−83.33, −16.67] points, but exact p = 0.25. An interval excluding zero from this tiny, discrete bootstrap sample does not override the nonsignificant exact result or justify a mitigation discovery. These original estimates and analysis definitions are retained. [4,6]

The implementation defect concerns SFT targets. It does not by itself change detector metrics computed from the separately normalized source inputs, and no detector scores or thresholds were edited. Computational agreement does not independently validate intent annotations, establish absence of upstream contamination, or eliminate the disclosed follow-up runtime/cohort confounding. One training seed, eight behavioral tasks, ambiguous outcomes and interface/capability failures still limit scientific inference. [4,6,7]

## Verification, preservation and implications

All 64 selected non-model regression tests passed, covering conversion/parser compatibility, dataset identity, phase boundaries, statistical summaries, audit freezing, follow-up separation and report preservation. The full suite was deliberately not run because it includes random-model forward/backward and CUDA tests. Before/after checksums cover 344 pre-existing report, training-data, checkpoint and audit files; all remain unchanged. The preservation manifest is published separately. Corrected targets were checked in memory and identified by hashes; no historical pool was overwritten. [1,8]

Future evidence would require a separately preregistered study using the corrected conversion, new token-length checks and data locks, interface-capable controls, stable learning diagnostics and adequate task/seed replication. None of those experiments has been run here. The correction supports an implementation finding and narrows interpretation of the existing behavioral pilot; it supports no new mitigation claim. [5,6]

## Evidence and sources

1. [Reproducible supplemental checks](checks.json): all 192 locked source records, source/target hashes, raw losses, saved-response rejection counts and input checksums. Reproduction: [model-free audit command](../../../docs/implementation.md#conversion-correction--22-october-2026).

2. [Pinned Harbor ATIF emitter](https://github.com/harbor-framework/harbor/blob/290ab985edc6915dd60fb1a0d369fedf10d213a1/src/harbor/agents/terminus_2/terminus_2.py#L1482): shell and completion tool calls.

3. [Pinned Harbor JSON parser](https://github.com/harbor-framework/harbor/blob/290ab985edc6915dd60fb1a0d369fedf10d213a1/src/harbor/agents/terminus_2/terminus_json_plain_parser.py): command requirements and completion-dependent warning behavior.

4. [Unchanged pilot results](../results.json), [historical paper](../paper.md) and [historical PDF](../paper.pdf): original learning curves, comparisons and limitations.

5. [Correction implementation](../../../docs/implementation.md#conversion-correction--22-october-2026) and [historical preregistered protocol](../../../docs/protocol.md).

6. [Analysis definitions](../../../docs/analysis.md) and [SciPy exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html): paired discordance calculation, ambiguity bounds and small-sample limits.

7. [Unchanged exploratory follow-up](../followup/summary.md): 160 distinct planned slots in combined coverage, including three unresolved setup failures; runtime/cohort confounding remains disclosed.

8. [Historical-file preservation manifest](preserved-files.json): SHA-256 identities recorded before the correction and verified afterward. Full raw local artifacts are intentionally not committed; the manifest contains paths and hashes only.
