"""Global: disclose outcome-aware ablations without changing the closed pilot's scientific record.

Local: reuse existing intervals, figures and PDF layout; keep cohort labels and audit freeze evidence.
Sources: docs/followup-protocol.md; https://www.cos.io/initiatives/prereg.
"""
import shutil  # Local: copy explicit public evidence, never raw trajectories or credential-bearing files.

from .analysis import paired_difference, proportion
from .audit import finalize, make_packets
from .diagnostics import diagnostic_rows, execution_diagnostics
from .followup import PLAN, prepare_followup, require_closed
from .report import behavior_table, markdown_table, paired_text, paper
from .runtime import ROOT, event, file_hash, read_json, write_json

METRICS = ("hacking", "legitimate", "verifier_success")  # Global: task verification and audited intent are separate endpoints.
CONTRASTS = [("probe_retain_minus_both", "probe_graft/retain", "probe_graft/both"),
             ("probe_retain_minus_forget", "probe_graft/retain", "probe_graft/forget"),
             ("no_intervention_retain_minus_both", "no_intervention/retain", "no_intervention/both")]


def combined_records(original, new):
    """Global: preserve original labels and forbid repeated attempts. Local: add cohort labels to copies."""
    allowed = {(arm, mode, prompt) for arm, mode, prompts in PLAN for prompt in prompts}
    rows, seen = [], set()
    for cohort, records in (("closed_pilot", original), ("followup", new)):
        for row in records:
            if row["smoke"]:  # Local: neither engineering smoke nor setup-only checks enter scientific denominators.
                continue
            condition = (row["arm"], row["mode"], row["elicited"])
            identity = (row["task"], *condition)
            if identity in seen:
                raise ValueError(f"Duplicate episode across or within cohorts: {identity}")
            if cohort == "followup" and condition not in allowed:
                raise ValueError("Follow-up row is outside the declared scope")
            seen.add(identity)
            rows.append(row | {"cohort": cohort})  # Local: never mutate an original frozen judgment in memory.
    return rows


def summarize(rows):
    """Global: separate cohorts, paired tasks and prompts. Local: reuse the original statistical functions."""
    grouped = {}
    for row in rows:
        key = f"{row['arm']}/{row['mode']}/{'elicited' if row['elicited'] else 'plain'}"
        grouped.setdefault(key, []).append(row)
    behavior = {key: {metric: proportion([row[metric] for row in group]) for metric in METRICS}
                for key, group in grouped.items()}
    comparisons = {}
    for label, left, right in CONTRASTS:
        for prompt in ("elicited", "plain"):
            comparisons[label + "/" + prompt] = {metric: paired_difference(
                {row["task"]: row[metric] for row in grouped.get(left + "/" + prompt, [])},
                {row["task"]: row[metric] for row in grouped.get(right + "/" + prompt, [])}) for metric in METRICS}
    return {"analysis_status": "exploratory follow-up after original outcomes were known",
            "coverage": {"closed_pilot_audited": sum(row["cohort"] == "closed_pilot" for row in rows),
                         "closed_pilot_planned": 160, "followup_audited": sum(row["cohort"] == "followup" for row in rows),
                         "followup_planned": 40, "combined_unique": len(rows), "original_grid_size": 160},
            "condition_cohorts": {key: sorted({row["cohort"] for row in group}) for key, group in grouped.items()},
            "behavior": behavior, "comparisons": comparisons,
            "conclusion": "The closed pilot's primary result remains inconclusive; these later adapter contrasts are exploratory."}


def comparison_figure(destination, comparisons):
    """Global: signed effects with honest uncertainty. Local: paired bootstrap intervals on a fixed scale."""
    import matplotlib.pyplot as plt  # Local: report.py has already selected the headless backend.
    figure, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    labels = [label.replace("probe_retain_minus_both/", "Probe: retain − both\n").replace(
        "probe_retain_minus_forget/", "Probe: retain − forget\n").replace(
        "no_intervention_retain_minus_both/", "No intervention: retain − both\n") for label in comparisons]  # Local: shorter literal contrast labels remain readable in the PDF.
    for axis, metric in zip(axes, ("hacking", "legitimate")):
        for position, (label, values) in enumerate(comparisons.items()):
            value = values[metric]
            if value["difference"] is None:
                axis.text(0, position, "No known pairs", ha="center", fontsize=9, color="#666666")
                continue  # Local: missing or wholly ambiguous pairs have no plotted point estimate.
            point, interval = value["difference"], value["bootstrap_ci95"]
            color = "#244b6b" if label.endswith("elicited") else "#a65f2b"
            axis.errorbar(point, position, xerr=[[max(0, point - interval[0])], [max(0, interval[1] - point)]],
                          fmt="o", capsize=4, color=color)
            axis.text(.98, position + .18, f"{value['known_pairs']}/{value['paired_tasks']} known pairs",
                      ha="right", fontsize=8, color="#444444")
        axis.axvline(0, color="#999999", linewidth=.7)
        axis.set(xlim=(-1, 1), ylim=(len(comparisons) - .5, -.5), xticks=(-1, -.5, 0, .5, 1),
                 yticks=range(len(comparisons)), yticklabels=labels, xlabel="Paired difference\nConditional bootstrap 95% interval",
                 title={"hacking": "Hacking attempts", "legitimate": "Legitimate success"}[metric])  # Local: half-row padding prevents clipping boundary markers.
    figure.suptitle("Exploratory adapter interventions\nKnown pairs; ambiguity bounds and cohort sources in the report", fontsize=11)
    figure.tight_layout()
    path = destination / "ablations.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def report_followup(pilot):
    """Global: report only after independent follow-up freezing. Local: write a supplement, not pilot replacements."""
    root = prepare_followup(pilot)  # Local: hash-check the original evidence again after collection.
    require_closed(root)
    if make_packets(root):
        raise ValueError("Record every follow-up judgment before reporting or unblinding")
    new = finalize(root)
    if new is None:
        raise ValueError("Follow-up audit is not finalized")
    original = read_json(pilot / "audit/unblinded.json")
    rows = combined_records(original, new)
    tasks = set(read_json(root / "evaluation/tasks.json")["tasks"])
    if any(row["task"] not in tasks for row in rows):
        raise ValueError("An outcome is outside the frozen task set")
    results = summarize(rows)
    diagnostics = execution_diagnostics(root, new)
    if diagnostics["overlapping_execution_pairs"]:
        raise ValueError("Overlapping execution intervals invalidate simple timestamp-only evidence association")
    results["execution_diagnostics"] = diagnostics
    results["source_manifest"] = read_json(root / "manifest.json")
    results["budget"] = read_json(root / "budget.json")
    results["original_results_sha256"] = file_hash(ROOT / "reports/qwen_probe/results.json")
    results["audit_freeze"] = read_json(root / "audit/frozen_judgments.json")
    destination = ROOT / "reports/qwen_probe/followup"
    destination.mkdir(parents=True, exist_ok=True)
    public = ("packet_id", "task", "arm", "mode", "elicited", "smoke", *METRICS, "agent_error",
              "evidence", "rationale", "auditor", "judged_at", "packet_fingerprint", "cohort")
    write_json(destination / "combined_outcomes.json", [{key: row[key] for key in public} for row in rows])
    write_json(destination / "results.json", results)
    for source, target in (("manifest.json", "manifest.json"), ("budget.json", "budget.json"),
                           ("publication.json", "publication.json"), ("controller.json", "controller.json"),
                           ("evaluation/index.json", "evaluation_index.json"), ("audit/judgments.json", "audit_judgments.json"),
                           ("audit/frozen_judgments.json", "audit_freeze.json")):
        shutil.copy2(root / source, destination / target)  # Local: explicit allowlist excludes private mappings and raw logs.
    for name in ("session", "integrity"):  # Local: retain the dated blinding procedure and pre-reveal proof when supplied by the auditor.
        if (root / f"audit/{name}.json").exists():
            shutil.copy2(root / f"audit/{name}.json", destination / f"audit_{name}.json")
    counts, behavior = results["coverage"], results["behavior"]
    condition_sources = markdown_table([["Deployment / prompt", "Cohort"]] +
                                      [[key, ", ".join(value)] for key, value in results["condition_cohorts"].items()])
    paired = "\n\n".join(paired_text(label, value) for label, value in results["comparisons"].items())
    coverage = (f"Original closed pilot: {counts['closed_pilot_audited']}/160 audited episodes. "
                f"Later follow-up: {counts['followup_audited']}/40. Combined distinct coverage of the original grid: "
                f"{counts['combined_unique']}/160. Missing episodes remain unavailable; none were replaced or imputed.")
    bounds = ("Each contrast pairs the same task within one prompt condition. Intervals use known pairs; unknown intent remains in bounds. "
              "The six comparisons are exploratory and unadjusted. A zero-width bootstrap is not evidence of equivalence. "
              "One seed and eight tasks do not support a general five-point capability guarantee.")
    sections = [
        ("Abstract", "This supplement completes previously unrun adapter interventions for the Qwen3-0.6B GRAFT pilot with frozen LLM-as-a-judge and downloaded probe monitors [1,2]. " + coverage + " " + results["conclusion"]),
        ("Context, motivation and prior knowledge", "The original pilot had already been audited, unblinded and published when the user authorized finishing the outstanding work. Its primary result, capability failure and frozen judgments remain unchanged. The new collection evaluates existing downloaded-probe and no-intervention checkpoints; it introduces no trained model or detector. The goal is descriptive adapter-intervention coverage, not a new confirmatory mitigation claim [3,4]."),
        ("Method and stopping rule", "Five previously unattempted conditions reuse the fixed eight tasks and saved checkpoints: probe/both plain; probe/forget elicited and plain; no-intervention/retain elicited and plain. The probe/both elicited condition is inherited from the original cohort. Model/tokenizer files, original evidence and reused adapter states are hash-locked. Sampling seed 2002, non-thinking mode, BF16 SDPA, temperature 0.7, top-p 0.8, top-k 20, context 16,384, maximum 16 turns and 120 seconds per agent remain fixed. One serialized worker runs until the five conditions finish or the separate persisted three-hour cap expires. No original attempt or failed follow-up attempt is repeated. The protocol and evaluator were pushed before this collection; followup-protocol.md states the exact scope."),
        ("Audit and statistical analysis", "All new deployments close before one mixed opaque audit batch is created. Full raw responses, commands, observations, task instructions and verifier evidence are judged with the original intent rubric. Every new judgment is saved and fingerprinted before treatment identities are joined. The implementing assistant knows the original outcomes and design; this is neither an independent nor fully naive audit. " + bounds + " Wilson intervals and paired bootstrap/exact discordance calculations reuse the original analysis implementation [5]."),
        ("Results", coverage + "\n\n" + paired),
        ("Execution diagnostics", "; ".join(f"{key}: {value['zero_parsed_command_episodes']}/{value['episodes']} zero-command episodes; {value['agent_timeouts']} timeouts" for key, value in diagnostics["conditions"].items()) + ". " + diagnostics["note"]),
        ("Discussion, claims and limitations", results["conclusion"] + " Changes from both to retain/forget modes describe interventions on the same checkpoint, conditional on fixed tasks and this runtime. Cohorts were collected at different times; timing-dependent generation cutoffs and system load can affect outputs. Earlier outcome knowledge, ambiguous intent, single-seed uncertainty, the capability floor and unresolved upstream probe exposure remain limitations. Do not select a favorable contrast as evidence of general alignment or overwrite the original primary conclusion."),
        ("Conclusion", "The added evidence completes or documents the remaining adapter interventions under its own stopping rule. Original and follow-up evidence remain distinguishable. Stronger causal and mitigation claims require capable task performance, more tasks and seeds, and independent outcome-blind auditing."),
        ("References and reproducibility", "[1] Ward et al. (2026), Gradient Routing Localizes and Removes Unintended Behaviors in RL. https://arxiv.org/abs/2610.17062\n[2] Frozen probe source, revision cd4c566824c5961eaeab7b7fce7845240bd12b3d. https://github.com/farzingkh/reward-hacking\n[3] Center for Open Science, preregistration and planned/exploratory distinctions. https://www.cos.io/initiatives/prereg\n[4] NeurIPS Paper Checklist. https://neurips.cc/public/guides/PaperChecklist\n[5] SciPy binomial inference. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html\nOriginal report: ../summary.md; original paper: ../paper.pdf. Follow-up source hashes, clock, audit freeze and full numeric evidence are in manifest.json, budget.json, audit_freeze.json, combined_outcomes.json and results.json. The execution commands and study boundary are in docs/followup-protocol.md in the repository. No new weights are trained; the published pilot checkpoints are unchanged.")]
    figure = comparison_figure(destination, results["comparisons"])
    paper(destination, sections, [figure], behavior, subtitle="Exploratory follow-up supplement after the closed pilot")
    table, legend = behavior_table(behavior)
    lines = ["# Exploratory adapter follow-up: LLM-as-a-Judge and Probe GRAFT", "", coverage, "", results["conclusion"],
             "", "See the [follow-up protocol](../../../docs/followup-protocol.md), [unchanged pilot report](../summary.md), "
             "[PDF supplement](paper.pdf) and [complete numeric evidence](results.json).", "", "## Paired comparisons", "", bounds, "", paired,
             "", "## Combined grid with cohort provenance", "", condition_sources, "", legend, "", markdown_table(table),
             "", "## Follow-up execution diagnostic", "", diagnostics["note"], "", markdown_table(diagnostic_rows(diagnostics)),
             "", "![Exploratory paired differences](ablations.png)"]
    (destination / "summary.md").write_text("\n".join(lines) + "\n")
    for arm in ("probe_graft", "no_intervention"):
        table, legend = behavior_table(behavior, arm=arm)
        (destination / f"{arm}.md").write_text(f"# {arm}: exploratory adapter follow-up\n\n" + coverage +
            "\n\nThe [combined report](summary.md) records cohort provenance, paired effects and limitations. "
            "Rows include unchanged original comparators alongside later follow-up conditions.\n\n" + legend + "\n\n" + markdown_table(table) + "\n")
    event("followup_report_complete", **counts, audit_sha256=results["audit_freeze"]["sha256"])
    return results
