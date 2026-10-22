# Follow-up engineering evidence

2026-10-22, before additional model attempts: six regressions first failed for the absent follow-up implementation and missing condition selector. After implementation they passed. They cover source-clock preservation, source-hash changes, active or unfrozen original evidence, already-attempted conditions, an independent persistent three-hour clock, and plain-only evaluation without repeating the original elicited attempts. The full CPU suite passed **46 tests**, with two previously validated CUDA tests deselected. The known PyTorch nested-tensor prototype warning remains. Ruff F/E9, `git diff --check` and `uv lock --check` passed.

The real prepare command completed at 11:50:35 UTC without any model attempt or budget start. It checked the closed pilot's frozen judgment hash, preserved its clock and copied only the small required manifests. Original base/tokenizer files and the two reused adapter checkpoints are SHA-256 locked in [manifest.json](manifest.json). Shared inputs are read-only by workflow; the follow-up CLI exposes no training or calibration phase. Final coverage and outcomes are in the [follow-up summary](summary.md).

The single correctness pass checked condition ordering, source/pilot separation, checkpoint completeness, process ownership, unchanged generation settings, no-repeat behavior and audit closure. Science and stopping choices are in the [follow-up protocol](../../../docs/followup-protocol.md), rather than inferred from upcoming outputs.

The implementation and plan were pushed as `bf2a09a` before evaluation began at 11:52:23 UTC. The follow-up deadline is 14:52:23 UTC, independently persisted; the original deadline remains unchanged. During collection, reporting-only work added cohort-preserving comparisons and a separate PDF supplement. Three analysis regressions and a nested-publication-link regression failed before their implementations. A synthetic full supplement render verifies that successful reporting leaves the original paper and primary-results files unchanged. These checks use toy records and do not inspect new model responses or outcomes.

## Infrastructure recovery, 22 October 2026

During blinded collection, the user reported a host package upgrade. Eight trials failed at containerd shim startup before agent/verifier execution. Services restarted externally; later containers launched. The dated amendment in [the follow-up protocol](../../../docs/followup-protocol.md) permits one recovery of wholly unstarted conditions within the same deadline and requires unchanged task images. It does not permit repeating actual model attempts. Pre-upgrade host versions were not captured, so runtime equality remains unresolved.

Five new recovery cases first failed because the explicit shim policy was absent, then passed with it. They reject prior agent execution, verifier evidence, trajectory files and unrelated errors, retain the original image-only default, verify archived result hashes and prohibit a second recovery. A synthetic PDF/report check first failed for absent setup-failure accounting, then passed after adding public infrastructure provenance. The focused suite passed 19 tests; the full CPU suite passed 56, with the two CUDA checks deferred until the evaluation worker closes. No new behavioral output was inspected for these changes.

A subsequent technical check found a distinct 12.368-second CLI failure with no job directory and zero journaled model requests during its interval. Before reading any new behavior, the 13:30 UTC clarification defined a separate first-attempt recovery for this case. Four new regressions first failed for the absent option, then passed; they block initialized jobs, model requests, missing intervals and repeated recovery. The combined focused suite passed 23 tests. The underlying CLI error was deliberately left unread until after audit freezing.

## Final execution and audit

The initial evaluator closed at 13:30 UTC. Recovery archived the eight shim failures and the separate pre-job failure at 13:32:57 UTC, after amendment commit `13136be` was pushed. The resumed evaluator finished at 14:27:35 UTC, before the unchanged 14:52:23 UTC cap. All five conditions have eight final trial records. Thirty-seven contain model behavior; three agent setups timed out before any model call or verifier. No actual model attempt was repeated. Setup failures inside partially executed conditions were retained, not recovered.

The complete mixed batch was created at 14:28:11 UTC after all deployment servers closed. Every packet, including malformed or length-truncated model responses preserved in server journals, was read in full before its immutable judgment was saved. All 40 judgments were frozen at 14:49:43 UTC. The [audit session](audit_session.json) discloses prior knowledge of original outcomes; the [pre-reveal integrity proof](audit_integrity.json) verifies packet fingerprints and the judgment hash before the treatment mapping was inspected. This is a Codex-assisted audit, not an independent human replication. The six ambiguous intent cases and three missing-behavior cases remain distinguishable in [combined outcomes](combined_outcomes.json).

After freezing, the initial controller log identified the pre-job error at 12:53:50 UTC: `Docker daemon is not running. Please start Docker and try again.` This supports a temporary host-service interruption, separately from the earlier shim mismatch. The three remaining setup tracebacks all stop while Harbor waits for tmux installation via Docker exec, with a 360-second setup timeout. Their underlying cause is unresolved; neither the package upgrade nor network conditions can be established as the cause from these traces. They are not silently counted as benign behavior or failed legitimate attempts. The 120-second limit applies to agent execution, separately from setup and verifier phases.

The same-task ablations do not isolate adapter removal from every confound. Apart from the entirely original elicited probe retain-versus-both comparison, deployment mode differs together with collection cohort. Host runtime and time-dependent response cutoffs may therefore explain some differences. Task-image and checkpoint hashes rule out changes to those files, not runtime equivalence. The original favorable bootstrap interval also has exact discordance p=0.25 with only six known pairs; it does not establish a robust discovery. All six exploratory contrasts are reported without selecting a favorable result.

## Verification and reproducible paper bundle

After collection, `uv run pytest -q` passed **62 tests**, including both CUDA equivalence checks, at 14:28 UTC. The only warning is PyTorch's existing nested-tensor prototype warning. The reporting path rechecks original source hashes before rendering; original judgment and PDF hashes remain those recorded in the follow-up protocol and manifest. Numerical reporting and diagram checks are separate from evidence of mitigation.

After the final reporting edits, all eight focused reporting/publication tests passed. The five-page supplement and its plot were visually inspected; the combined paper has 12 pages. [Final artifact verification](verification.json) records unchanged original scientific files and all six checkpoint hashes, unique condition coverage, audit-freeze ordering, working local links and the credential-pattern scan. [Failure diagnosis](failure_diagnosis.json) preserves the post-audit technical observations without altering any labels.

The original paper remains byte-for-byte unchanged. The supplement is generated by `report --study ablation-followup`; a convenience bundle uses [Poppler's pdfunite](https://manpages.debian.org/bookworm/poppler-utils/pdfunite.1.en.html), supplied by `poppler-utils` on Debian/Ubuntu:

```bash
pdfunite reports/qwen_probe/paper.pdf reports/qwen_probe/followup/paper.pdf reports/qwen_probe/paper-with-followup.pdf
```

The bundle presents the original paper first and the explicitly labeled later supplement second. It neither rewrites the original stopping rule nor turns exploratory follow-up evidence into preregistered results.
