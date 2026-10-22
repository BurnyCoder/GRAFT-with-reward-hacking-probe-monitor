"""Global: trace every claim to artifacts. Local: generate Markdown, figures and a sourced PDF.

Sources: https://www.reportlab.com/docs/reportlab-userguide.pdf and https://matplotlib.org/stable/.
"""
import json  # Local: tables and transparent machine-readable results.
from collections import Counter  # Local: measured sample and source composition, not guessed proportions.
import shutil  # Local: copy compact provenance, never credentials or model weights.
from html import escape  # Local: safe PDF paragraphs from data.

import matplotlib  # Local: standalone, reproducible scientific figures.
matplotlib.use("Agg")  # Local: headless rendering in the repository environment.
import matplotlib.pyplot as plt

from .analysis import compare_detectors, detector_metrics, monitor_coverage, paired_difference, power_analysis, proportion
from .audit import finalize, make_packets
from .data import ARMS
from .diagnostics import diagnostic_rows, execution_diagnostics  # Local: post-hoc execution counts never supply intent labels.
from .runtime import ROOT, event, read_json, write_json

ARM_RATIONALE = {  # Global: distinguish scientific questions without duplicating the full protocol.
    "qwen_judge_graft": "Compare a frozen untouched Qwen3-0.6B intent judge with the downloaded probe, both as a detector and a source of GRAFT flags. The independent calibration cutoff and identical training recipe make this a secondary monitor comparison; constrained binary scoring is not a generated rationale.",
    "no_intervention": "Establish the effect of learning the matched clean/hack mixture with both adapters and the same split-moment optimizer, without routing. This is the primary comparator, not the untouched base model.",
    "probe_graft": "Test whether the downloaded external detector supplies useful routing flags. Its encoder/head remain frozen; no downstream behavior is used to recalibrate the threshold. Removing the forget adapter is the primary deployment intervention. Upstream training exposure remains unresolved; this is not certified out-of-distribution evaluation.",
    "original_graft": "Measure the small-model GRAFT recipe with the paper's idealized monitor: zero false positives and approximately 20% positive recall after integer rounding. This separates detector limitations from a method/capability floor.",
    "probe_filtering": "Test discarding flagged learning signal through the existing zero-weight single-adapter filtering baseline. Ordinary AdamW and fewer nonzero-loss examples differ from GRAFT, so this comparison cannot isolate routing alone.",
    "random_graft": "Test whether the probe's trajectory-specific flags matter beyond flag frequency and length. Shuffle flags within four equal-count token-length bins, keeping each bin's flagged count fixed.",
    "smoke": "Check real download, scoring, one parameter update, merging, local Docker execution, audit and reporting. This checkpoint is an engineering artifact and is excluded from all scientific treatment comparisons.",
}


def architecture(path):
    """Global: explain the actual data flow. Local: compact directed graph for README and PDF."""
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.set(xlim=(0, 12), ylim=(0, 4))
    axis.axis("off")
    nodes = {"Pinned trajectories\n232 / 99 tasks": (1.5, 3), "32 calibration tasks\nFrozen threshold": (5, 3),
             "Frozen probe / Qwen judge\nScores → flags": (8.7, 3), "Matched SFT arms\nQwen3-0.6B": (1.5, 1),
             "Merged adapter modes\nHarbor + Docker": (5, 1), "Blinded intent audit\nIntervals + coverage": (8.7, 1)}
    for label, (x, y) in nodes.items():
        axis.text(x, y, label, ha="center", va="center", fontsize=10,
                  bbox=dict(boxstyle="round,pad=.5", facecolor="#edf4fa", edgecolor="#244b6b"))
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
        start, end = list(nodes.values())[a], list(nodes.values())[b]
        axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#244b6b", shrinkA=65, shrinkB=65))
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def result_figures(destination, results):
    """Global: visualize observed coverage without hidden exclusions. Local: fixed scales and intervals."""
    figure, axis = plt.subplots(figsize=(9, 3.6))
    names = list(ARMS)
    counts = [results["training"][name]["steps"] for name in names]
    colors = ["#2e6b96" if results["training"][name]["complete"] else "#ad6b32" for name in names]
    axis.barh(names, counts, color=colors)
    axis.set(xlabel="Completed optimizer updates", title="Training coverage (blue: complete; orange: incomplete)", xlim=(0, max(20, max(counts) + 1)))
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(destination / "training.png", dpi=180)
    plt.close(figure)
    figures = [destination / "training.png"]
    detector = results["detector"]
    if detector.get("metrics"):
        figure, axis = plt.subplots(figsize=(7, 3.4))
        for label, monitor, offset, color in [("Downloaded probe", detector, -.08, "#244b6b"),
                                               ("Qwen judge", results["judge"], .08, "#a65f2b")]:
            for index, (metric, value) in enumerate(monitor.get("metrics", {}).items()):
                interval = monitor["task_bootstrap_ci95"][metric]
                if value is not None and interval is not None:
                    axis.errorbar(index + offset, value, yerr=[[max(0, value - interval[0])], [max(0, interval[1] - value)]],
                                  fmt="o", capsize=5, color=color, label=label if index == 0 else None)
        axis.legend()
        axis.set(xticks=range(4), xticklabels=["AUROC", "Precision", "Recall", "FPR"], ylim=(0, 1),
                 ylabel="Estimate and task-bootstrap 95% interval", title=f"Frozen monitors: {detector['scored']} heldout trajectories, {detector['tasks']} tasks")
        figure.tight_layout()
        figure.savefig(destination / "detector.png", dpi=180)
        plt.close(figure)
        figures.append(destination / "detector.png")
    elicited = {key: metrics for key, metrics in results["behavior"].items() if key.endswith("/elicited")}
    if elicited:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for axis, metric in zip(axes, ("hacking", "legitimate")):
            for index, (key, values) in enumerate(elicited.items()):
                value = values[metric]
                if value["rate_known"] is not None:
                    point, interval = value["rate_known"], value["ci95_known"]
                    axis.errorbar(point, index, xerr=[[point - interval[0]], [interval[1] - point]], fmt="o", capsize=4)
            axis.set(xlim=(0, 1), yticks=range(len(elicited)), yticklabels=list(elicited), xlabel="Known-outcome rate; Wilson 95% interval", title=metric)
        figure.suptitle("Elicited behavior: ambiguity and missingness are reported separately as bounds")
        figure.tight_layout()
        figure.savefig(destination / "behavior.png", dpi=180)
        plt.close(figure)
        figures.append(destination / "behavior.png")
    return figures


def paper(destination, sections, figures, behavior=None, *, subtitle="Exploratory pilot"):
    """Global: one source for paper text and PDF. Local: ReportLab layout with actual result figures."""
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    styles = getSampleStyleSheet()
    styles["Heading2"].keepWithNext = True  # Local: ReportLab keeps section titles with their following evidence.
    outcome_style = ParagraphStyle("OutcomeCell", parent=styles["BodyText"], fontSize=7.5, leading=9.5, spaceBefore=0, spaceAfter=0)
    title = "Qwen3-0.6B GRAFT with LLM-as-a-Judge and Probe Monitors"  # Local: one title keeps both methods visible across paper formats and metadata.
    story = [Paragraph(title, styles["Title"]),
             Paragraph(f"Libor Burian · {escape(subtitle)} · 22 October 2026", styles["Normal"]), Spacer(1, 16)]
    markdown = [f"# {title}", f"\nLibor Burian — {subtitle.lower()}, 22 October 2026\n"]
    for heading, text in sections:
        markdown.extend([f"## {heading}\n", text + "\n"])
        story.append(Paragraph(escape(heading), styles["Heading2"]))
        for paragraph in text.split("\n\n"):
            story.append(Paragraph(escape(paragraph).replace("\n", "<br/>"), styles["BodyText"]))
        if heading == "Results" and behavior:
            rows, legend = behavior_table(behavior)  # Local: one table source serves Markdown, PDF and the consolidated report.
            markdown.extend([legend, markdown_table(rows)])
            story.extend([Spacer(1, 10), Paragraph(escape(legend), styles["BodyText"])])
            cells = [[Paragraph(escape(cell), outcome_style) for cell in row] for row in rows]
            table = Table(cells, colWidths=[140, 40, 93, 93, 94], repeatRows=1)  # Local: repeat labels across page breaks.
            table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LINEBELOW", (0, 0), (-1, 0), .5, "black"),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            story.append(table)
    for figure in figures:
        story.extend([Spacer(1, 12), Image(str(figure), width=460, height=190, kind="proportional")])
        markdown.append(f"![{figure.stem}]({figure.name})\n")
    (destination / "paper.md").write_text("\n".join(markdown))
    SimpleDocTemplate(str(destination / "paper.pdf"), title=title, author="Libor Burian").build(story)


def estimate_text(value, interval=None):
    """Global: readable numerical claims. Local: retain undefined estimates instead of printing fabricated zeros."""
    if value is None:
        return "undefined"
    return f"{value:.3f}" + (f" (95% interval {interval[0]:.3f} to {interval[1]:.3f})" if interval else "")


def behavior_table(behavior, arm=None):
    """Global: separate success and intent. Local: display every planned condition without imputing missing trials."""
    from .evaluation import MODES  # Local: reuse the registered deployment grid instead of duplicating it.
    rows = [["Deployment / prompt", "Audited", "Hacking attempts", "Legitimate success", "Verifier success"]]
    for name, mode in MODES:
        if arm is not None and name != arm:  # Local: per-experiment reports retain every planned mode for their own arm.
            continue
        for prompt in ("elicited", "plain"):
            key = f"{name}/{mode}/{prompt}"
            metrics = behavior.get(key)
            row = [key, f"{metrics['hacking']['total'] if metrics else 0}/8"]
            for field in ("hacking", "legitimate", "verifier_success"):
                if metrics is None:
                    row.append("Unavailable")  # Local: an unrun or unaudited condition is not a zero rate.
                    continue
                value = metrics[field]
                cell = f"{value['successes']}/{value['observed']} known; "
                interval = value["ci95_known"]
                cell += f"95% [{interval[0]:.3f}, {interval[1]:.3f}]" if interval else "rate undefined"
                if value["unknown"]:
                    cell += f"; unknown {value['unknown']}; bounds [{value['bounds'][0]:.3f}, {value['bounds'][1]:.3f}]"
                row.append(cell)
            rows.append(row)
    legend = ("Counts are positive/known outcomes, with Wilson 95% intervals for the known rate. "
              "Unknown counts and ambiguity bounds concern audited episodes; the audited count out of eight "
              "separately exposes missing planned episodes. These intervals do not capture training-seed variation.")
    return rows, legend


def markdown_table(rows):
    """Global: consistent outcome presentation. Local: render the same rectangular cells as the PDF."""
    return "\n".join("| " + " | ".join(row) + " |" for row in [rows[0], ["---"] * len(rows[0]), *rows[1:]])


def monitor_text(name, result):
    """Global: cite measured detector evidence compactly. Local: include counts, cutoff and uncertainty."""
    if "metrics" not in result:
        return f"{name}: {result['status']}."
    metrics = "; ".join(f"{key} {estimate_text(value, result['task_bootstrap_ci95'][key])}"
                        for key, value in result["metrics"].items())
    calibration = result["calibration"]
    return (f"{name}: {result['scored']}/{result['planned']} heldout trajectories across {result['tasks']} tasks; "
            f"{result['positive']} positive and {result['negative']} negative labels. {metrics}. "
            f"Frozen cutoff {calibration['threshold']:.8g}; calibration false positives "
            f"{calibration['false_positives']}/{calibration['clean']}.")


def paired_text(label, comparison):
    """Global: equal visibility without changing analysis priority. Local: render existing paired estimates."""
    values = []  # Local: reuse the same measured quantities in the paper and concise consolidated report.
    for metric, value in comparison.items():
        text = (f"{metric}: {estimate_text(value['difference'], value['bootstrap_ci95'])}, "
                f"{value['known_pairs']}/{value['paired_tasks']} known task pairs")  # Local: ambiguous pairs do not disappear from the denominator.
        if value["ambiguity_bounds"] is not None:  # Local: no shared tasks means no bounds, not zero uncertainty.
            text += ", ambiguity bounds [" + ", ".join(f"{bound:.3f}" for bound in value["ambiguity_bounds"]) + "]"
        text += f", exact discordance p={estimate_text(value['exact_discordance_p'])}"  # Local: descriptive evidence, not a new significance rule.
        values.append(text)
    return f"{label}: {'; '.join(values)}."  # Local: the supplied label states direction and primary/secondary status.


def validation_report(root, name, lock, inventory):
    """Global: common monitor evidence. Local: separate calibrated accuracy from context coverage."""
    path = root / name / "threshold.json"
    if not path.exists():
        return {"status": "not run"}
    scores = read_json(root / name / "scores.json")
    threshold = read_json(path)["threshold"]
    result = detector_metrics([inventory[i] for i in lock["heldout"]], scores, threshold)
    result["sampling_note"] = ("Validation selects at most one record per task and class: baseline clean versus rewarded hacks. "
                               "Precision reflects this case mixture, not an estimated deployment prevalence. "
                               "Generated audits instead count all observed hacking attempts, including unsuccessful ones.")
    result["coverage"] = {}
    for part in ("calibration", "train", "heldout"):
        if name == "monitor":
            coverage = monitor_coverage(lock[part], scores)
        else:
            rows = [scores[i] for i in lock[part] if i in scores]
            coverage = {"planned": len(lock[part]), "scored": len(rows),
                        **{key: sum(row[key] for row in rows) for key in ("turns", "retained_turns", "transcript_tokens",
                                                                       "retained_transcript_tokens", "input_tokens")},
                        "context_truncated_trajectories": sum(row["context_truncated"] for row in rows)}
        result["coverage"][part] = coverage
    clean = [scores[i]["score"] for i in lock["calibration"] if inventory[i]["label"] == 0 and i in scores]
    result["calibration"] = {"threshold": threshold, "clean": len(clean), "false_positives": sum(score > threshold for score in clean),
                             "empirical_fpr": sum(score > threshold for score in clean) / len(clean) if clean else None,
                             "note": "Selected on calibration data; heldout task-bootstrap intervals quantify validation uncertainty."}
    return result


def training_status(root, arm):
    """Global: distinguish observed updates from saved models. Local: combine final manifests and live logs."""
    folder = root / "checkpoints" / arm  # Local: each registered treatment has separate evidence.
    path, metrics = folder / "progress.json", folder / "metrics.jsonl"  # Local: the trainer writes metrics after optimizer.step.
    status = read_json(path) if path.exists() else {"complete": False, "steps": 0, "status": "not run"}
    status["checkpoint_saved"] = path.exists() and (folder / "adapter_state_dict.pt").exists()  # Local: a loss log is not a checkpoint.
    if metrics.exists():  # Local: retain actual observed learning curves even before a trainer exits.
        status["learning_curve"] = [json.loads(line) for line in metrics.read_text().splitlines() if line]
        if not path.exists() and status["learning_curve"]:  # Local: completion still requires the trainer's final manifest.
            status.update(steps=status["learning_curve"][-1]["step"], status="updates logged; final checkpoint not recorded")
    return status  # Local: an interrupted checkpoint remains incomplete even if it contains useful weights.


def arm_learning(arm, grouped, diagnostics):
    """Global: outcome-specific learning without inventing unrun results. Local: summarize only audited evidence."""
    rows = [row for key, values in grouped.items() if key.startswith(arm + "/") for row in values]
    if not rows:  # Local: an engineering smoke or unrun treatment cannot receive a scientific interpretation.
        return "No audited study outcomes are available for this arm."
    zero = sum(value["zero_parsed_command_episodes"] for key, value in diagnostics["conditions"].items() if key.startswith(arm + "/"))
    solves = sum(row["legitimate"] is True for row in rows)  # Local: do not turn a passing verifier with ambiguous intent into capability.
    text = f"Across {len(rows)} available episodes, {solves} legitimate successes were established and {zero} episodes had no parsed shell-command request. "
    if zero == len(rows):
        text += "The absence of established hacking accompanies complete command-interface failure, so it cannot demonstrate selective removal of hacking. "
    elif solves == 0:
        text += "Task capability is at the observed floor; low hacking counts alone do not establish a useful mitigation. "
    else:
        text += "Sparse task success limits a capability-preservation comparison at the registered five-point tolerance. "
    return text + "These post-hoc execution counts describe symptoms, not their cause; no settings, attempts or frozen labels were changed."


def report(root):
    """Global: complete and incomplete results are both visible. Local: compute only observed metrics."""
    destination = ROOT / "reports/qwen_probe"
    destination.mkdir(parents=True, exist_ok=True)
    architecture(destination / "architecture.png")
    pending = make_packets(root)
    audited = finalize(root) if not pending else None
    audited = audited or []
    public_fields = ("packet_id", "task", "arm", "mode", "elicited", "smoke", "hacking", "legitimate",
                     "verifier_success", "agent_error", "evidence", "rationale", "auditor", "judged_at", "packet_fingerprint")
    write_json(destination / "audited_outcomes.json", [{key: row[key] for key in public_fields} for row in audited])
    trained = {arm: training_status(root, arm) for arm in (*ARMS, "smoke")}  # Local: one status rule for every report and figure.
    detector, judge, detector_comparison = ({"status": "not run"} for _ in range(3))
    data_summary, routing = {}, {}  # Local: absent manifests remain absent evidence.
    lock_path, threshold_path = root / "data/lock.json", root / "monitor/threshold.json"
    if lock_path.exists() and threshold_path.exists():
        lock = read_json(lock_path)
        inventory = {r["id"]: r for r in read_json(root / "data/inventory.json")}
        for part in ("train", "calibration", "heldout"):
            rows = [inventory[i] for i in lock[part]]
            data_summary[part] = {"records": len(rows), "tasks": len({row["task"] for row in rows}),
                                  "labels": dict(Counter(str(row["label"]) for row in rows)),
                                  "sources": dict(Counter(row["source_model"] for row in rows)),
                                  "total_learner_tokens": sum(row["tokens"] for row in rows),
                                  "min_tokens": min(row["tokens"] for row in rows),
                                  "max_tokens": max(row["tokens"] for row in rows)}
        if (root / "data/arms.json").exists():
            for arm, pool in read_json(root / "data/arms.json").items():
                routing[arm] = {"total": pool["total"], "flagged": pool["flagged"],
                                "flagged_positive": sum(flag and inventory[identity]["label"] == 1 for identity, flag in pool["flags"].items()),
                                "flagged_negative": sum(flag and inventory[identity]["label"] == 0 for identity, flag in pool["flags"].items())}
        detector = validation_report(root, "monitor", lock, inventory)
        judge = validation_report(root, "judge", lock, inventory)
        if "calibration" in judge:
            detector_comparison = compare_detectors([inventory[i] for i in lock["heldout"]],
                read_json(root / "judge/scores.json"), read_json(root / "monitor/scores.json"),
                judge["calibration"]["threshold"], detector["calibration"]["threshold"])
    grouped = {}
    for row in audited:
        if not row["smoke"]:
            key = f"{row['arm']}/{row['mode']}/{'elicited' if row['elicited'] else 'plain'}"
            grouped.setdefault(key, []).append(row)
    behavior = {key: {field: proportion([r[field] for r in rows]) for field in ("hacking", "legitimate", "verifier_success")}
                for key, rows in grouped.items()}
    treatment = grouped.get("probe_graft/retain/elicited", [])
    control = grouped.get("no_intervention/both/elicited", [])
    primary = {field: paired_difference({r["task"]: r[field] for r in treatment}, {r["task"]: r[field] for r in control})
               for field in ("hacking", "legitimate")}
    baseline_hacks = sum(r["hacking"] is True for r in control)
    legitimate_solves = sum(r["legitimate"] is True for r in treatment + control)
    complete_primary = len(treatment) == len(control) == 8 and primary["hacking"]["known_pairs"] == 8
    conclusion = "Mitigation effectiveness is inconclusive."
    reasons = []
    if len(treatment) != 8 or len(control) != 8:
        reasons.append("The primary comparison has missing episodes.")
    elif primary["hacking"]["known_pairs"] != 8:
        reasons.append(f"All primary episodes were audited, but only {primary['hacking']['known_pairs']}/8 task pairs have unambiguous hacking judgments.")
    if control and not baseline_hacks:
        reasons.append("No audited hacking attempt is established in the no-intervention baseline.")
    if treatment + control and not legitimate_solves:
        reasons.append("The primary arms have no established legitimate successes; a capability floor may dominate.")
    capability_difference = primary["legitimate"]["difference"]  # Local: apply the already registered tolerance without changing the estimand.
    capability = {"tolerance": 0.05, "observed_difference": capability_difference,
                  "point_estimate_within_tolerance": capability_difference >= -0.05 if capability_difference is not None else None,
                  "noninferiority_established": False, "note": "A single seed and eight tasks cannot establish the registered five-point capability safeguard."}
    if capability["point_estimate_within_tolerance"] is False:
        reasons.append(f"The observed legitimate-success difference is {capability_difference * 100:.1f} percentage points, below the registered -5-point tolerance.")
    if complete_primary and baseline_hacks and legitimate_solves and capability["point_estimate_within_tolerance"]:
        conclusion = "The paired pilot estimates describe these eight tasks and one training seed; they do not establish general mitigation effectiveness."
    judge_behavior = grouped.get("qwen_judge_graft/retain/elicited", [])
    secondary = {name: {field: paired_difference({r["task"]: r[field] for r in judge_behavior},
                                                {r["task"]: r[field] for r in comparator})
                        for field in ("hacking", "legitimate")} for name, comparator in
                 (("judge_minus_no_intervention", control), ("judge_minus_probe", treatment))}
    ablations = {}  # Local: retain every registered intervention, including missing modes, in paired evidence.
    for label, left, right in [("probe_retain_minus_both", "probe_graft/retain/elicited", "probe_graft/both/elicited"),
                               ("probe_retain_minus_forget", "probe_graft/retain/elicited", "probe_graft/forget/elicited"),
                               ("no_intervention_retain_minus_both", "no_intervention/retain/elicited", "no_intervention/both/elicited")]:
        ablations[label] = {field: paired_difference({r["task"]: r[field] for r in grouped.get(left, [])},
                                                    {r["task"]: r[field] for r in grouped.get(right, [])}) for field in ("hacking", "legitimate")}
    diagnostics = execution_diagnostics(root, audited)  # Local: labels are already frozen and joined before this descriptive inspection.
    write_json(destination / "execution_diagnostics.json", diagnostics)
    results = {"training": trained, "detector": detector, "judge": judge, "detector_comparison": detector_comparison,
               "data": data_summary, "routing": routing,
               "behavior": behavior, "primary": primary, "secondary": secondary, "ablations": ablations,
               "capability_safeguard": capability, "execution_diagnostics": diagnostics,
               "pending_audits": len(pending), "conclusion": conclusion, "reasons": reasons, "power": power_analysis()}
    recovery_path = root / "evaluation/setup_recovery.json"  # Local: setup errors are separate from model-attempt denominators.
    recoveries = read_json(recovery_path) if recovery_path.exists() else []
    results["coverage"] = {"planned_study_episodes": 160, "audited_study_episodes": sum(map(len, grouped.values())),
                           "missing_planned_episodes": 160 - sum(map(len, grouped.values())),
                           "archived_setup_failures": sum(item["trial_count"] for item in recoveries),
                           "recovered_jobs": [item["job"] for item in recoveries],
                           "pending_audit_packets": len(pending), "note": "Unstarted or missing-result episodes are absent, never treated as nonhacking."}
    index_path = root / "evaluation/index.json"  # Local: job shutdown and available trial evidence are different completion measures.
    index = read_json(index_path) if index_path.exists() else {}
    results["coverage"]["job_statuses"] = {name: job["status"] for name, job in index.items() if not job["smoke"]}
    write_json(destination / "results.json", results)
    write_json(destination / "audit_freeze.json", read_json(root / "audit/frozen_judgments.json") if audited else
               {"status": "not finalized", "pending_packets": len(pending)})  # Local: an earlier smoke freeze cannot certify later study judgments.
    figures = result_figures(destination, results)
    for source, target in [("data/lock.json", "data_lock.json"), ("data/provenance.json", "provenance.json"),
                           ("monitor/threshold.json", "threshold.json"), ("evaluation/tasks.json", "tasks.json"),
                           ("judge/threshold.json", "judge_threshold.json"), ("judge/manifest.json", "judge_manifest.json"),
                           ("data/arms.json", "routing_flags.json"),
                           ("evaluation/setup_recovery.json", "setup_recovery.json"),
                           ("evaluation/image_identity_recovery.json", "image_identity_recovery.json"),
                           ("evaluation/index.json", "evaluation_index.json"),  # Local: expose the implementation commit and timing of every deployment job.
                           ("budget.json", "budget.json"), ("preflight.json", "preflight.json"),
                           ("publication.json", "publication.json"), ("audit/judgments.json", "audit_judgments.json"),
                           ("audit/session.json", "audit_session.json")]:  # Local: publish the mixed-batch procedure alongside the evidence.
        if (root / source).exists():
            shutil.copy2(root / source, destination / target)
    for arm in trained:  # Local: publish portable evidence of the exact invocation, including incomplete runs.
        configuration = root / "checkpoints" / arm / "configuration.json"
        if configuration.exists():
            shutil.copy2(configuration, destination / f"{arm}_configuration.json")
    lines = ["# Consolidated experiment results", "", conclusion, " ".join(reasons), "",
             "This report is generated from local manifests and frozen audits. See [protocol](../../docs/protocol.md) for the prespecified method.", "",
             "| Experiment | Updates | Completed recipe | Status |", "|---|---:|---|---|"]
    for arm, progress in trained.items():
        lines.append(f"| {arm} | {progress['steps']} | {progress['complete']} | {'complete' if progress['complete'] else 'incomplete'} |")
        rows, legend = behavior_table(behavior, arm=arm)  # Local: share counts, intervals and missingness with the main paper.
        (destination / f"{arm}.md").write_text(f"# {arm}\n\n" + ARM_RATIONALE[arm] + "\n\n" +
            f"Updates: {progress['steps']}. Complete planned training: {progress['complete']}.\n\n" +
            "Method and hyperparameter rationale: [preregistered protocol](../../docs/protocol.md). " +
            "Exact optimizer arguments are in the per-arm configuration JSON when the run has started. " +
            "The smoke checkpoint is an engineering check and excluded from study comparisons.\n\n" +
            (legend + "\n\n" + markdown_table(rows) if len(rows) > 1 else "The smoke is excluded from study outcomes; its task result is recorded in [engineering](engineering.md).") + "\n\n" +
            "Learning: " + arm_learning(arm, grouped, diagnostics) + " See [summary](summary.md) for the paired comparisons and capability safeguard.\n")
    rows, legend = behavior_table(behavior, arm="base")  # Local: the untouched baseline gets the same outcome definitions.
    (destination / "base.md").write_text("# Base model\n\nThe untouched pinned Qwen3-0.6B checkpoint measures initial task capability and spontaneous/elicited hacking. It receives no SFT updates. A failed-task floor here and in the primary no-intervention arm limits interpretation of lower hacking counts. Absent measurements are not zero rates.\n\n" + legend + "\n\n" + markdown_table(rows) + "\n\nLearning: " + arm_learning("base", grouped, diagnostics) + "\n")
    lines += ["", f"Pending blinded audits: {len(pending)}.", "", "## Coverage and infrastructure", "",
              f"Audited study episodes: {results['coverage']['audited_study_episodes']}/160. Archived setup failures: {results['coverage']['archived_setup_failures']}. "
              "[Exact results](results.json) retain all denominators, paired task differences, truncation counts, source composition and planning calculations.",
              "", f"Missing planned episodes: {results['coverage']['missing_planned_episodes']}. Job lifecycle states: " + str(dict(Counter(results['coverage']['job_statuses'].values()))) + ". Available trial records remain audited even when the cap interrupts job cleanup.",
              "", "Archived setup failures, when present, precede any model attempt. Their evidence and the dated recovery amendment remain separate from behavioral outcomes.",
              "", "## Frozen LLM-judge and probe validation", "", monitor_text("Same-Qwen judge", judge), "", monitor_text("Downloaded probe", detector),
              "", "[Paired detector comparison](monitor-comparison.md). Precision describes the selected clean/rewarded-hack mixture; calibration FPR does not guarantee heldout FPR.",
              "", "## Fixed data and realized routing", "",
              markdown_table([["Partition", "Records", "Tasks", "Clean", "Hack", "Learner tokens"]] +
                  [[part, str(value["records"]), str(value["tasks"]), str(value["labels"].get("0", 0)), str(value["labels"].get("1", 0)), str(value["total_learner_tokens"])]
                   for part, value in data_summary.items()]),
              "", markdown_table([["Training treatment", "Flagged / total", "Labeled hacks flagged", "Labeled clean flagged"]] +
                  [[arm, f"{value['flagged']}/{value['total']}", str(value["flagged_positive"]), str(value["flagged_negative"])] for arm, value in routing.items()]),
              "", "## Primary paired comparison", "", paired_text("Elicited probe retain minus no intervention both", primary),
              "", "## Secondary judge behavioral comparisons", "",
              paired_text("Elicited LLM-judge retain minus no intervention both", secondary["judge_minus_no_intervention"]), "",
              paired_text("Elicited LLM-judge retain minus probe retain", secondary["judge_minus_probe"]),
              "", "These are descriptive, single-seed contrasts. Degenerate bootstrap intervals do not establish equivalence. Ambiguity bounds do not include entirely missing planned episodes.",
              "", "## Adapter interventions", "", *[paired_text(label, value) for label, value in ablations.items()],
              "", "These are secondary elicited contrasts. Missing modes have zero paired tasks and undefined effects. A condition with eight records can still have an incomplete job lifecycle at the cap; see evaluation_index.json.",
              "", "## Post-hoc execution diagnostic", "", diagnostics["note"], "", markdown_table(diagnostic_rows(diagnostics)),
              "", f"Checked {diagnostics['execution_intervals_checked']} study execution intervals; {diagnostics['overlapping_execution_pairs']} overlapping pairs. This diagnostic was added after judgment freezing and does not alter intent labels.",
              "", "## Terminology and networks", "", "| Network / term | Architecture / algorithm | Objective and data | Measured meaning |",
              "|---|---|---|---|", "| Learner | Frozen Qwen3-0.6B, 28 layers, two width-64 MLP branches | Assistant-token SFT, matched clean/hack subset | Generates terminal actions |",
              "| Detector encoder | Qwen3-Embedding-0.6B plus frozen rank-16 attention LoRA | Upstream source-invariant training; no fitting here | 1,024-dimensional normalized turn vectors |",
              "| Qwen judge | Untouched frozen Qwen3-0.6B with intent rubric | Conditional YES/NO logits; calibration-only threshold | Separate detector baseline and GRAFT flag source |",
              "| Probe head | Four transformer layers, width 256, 8 heads, CLS classifier | Upstream binary trajectory classifier | AUROC measures ranking, not behavioral mitigation |",
              "| GRAFT | Routed first moments, natural-gradient second moments | Flagged/unflagged pools plus 1% unlabeled anchors | Retain branch deployed after removing forget branch |",
              "| Verifier success | Task-specific executable checks | Heldout Docker tasks | Can include legitimate solves or hacks |",
              "| Hacking attempt | Intent-based blinded audit | Agent actions and rationale | Counts attempts even when they fail |",
              "| Legitimate success | Verifier success without hacking | Same audited episodes | Capability safeguard |",
              "| Completion / accuracy | Training completion and separate task outcomes | See manifests and intervals | No universal model accuracy or 'fill' metric applies |",
              "", "## Prospective power", "", "For a 10-percentage-point reduction, 80% power and two-sided alpha .05:", "",
              markdown_table([["Assumed baseline rate", "Assumed treatment rate", "Independent samples per arm"]] +
                  [[f"{value['baseline']:.2f}", f"{value['treatment']:.2f}", str(value["n_per_arm"])] for value in results["power"]["independent"]]),
              "", "Paired-task requirements and sensitivity to unknown seed variation are in [results.json](results.json); [analysis notes](../../docs/analysis.md) give the formulas and assumptions. These are prospective approximations, not achieved-power claims.",
              "", "Detailed engineering decisions and failures are recorded in [implementation log](../../docs/implementation.md). Full untruncated transcripts are under the ignored local artifacts/qwen_probe/logs directory."]
    if behavior:
        rows, legend = behavior_table(behavior)
        lines += ["", "## All planned deployment conditions", "", legend, "", markdown_table(rows)]
    (destination / "summary.md").write_text("\n".join(lines) + "\n")
    comparison_lines = ["# Frozen monitor comparison", "", monitor_text("Downloaded probe", detector), "",
                        monitor_text("Same-Qwen judge", judge), "",
                        "Both monitors use the same locked heldout trajectories and independently frozen calibration cutoffs. "
                        "The judge uses the untouched learner and a fixed intent rubric; its score is conditional YES/NO odds. "
                        "Neither monitor is fitted on these evaluation tasks. See the [protocol](../../docs/protocol.md) for exact inputs and context limits.", "",
                        "| Metric | Judge minus probe | Paired task-bootstrap 95% interval |", "|---|---:|---|"]
    for metric, value in detector_comparison.get("judge_minus_probe", {}).items():
        comparison_lines.append(f"| {metric} | {estimate_text(value)} | " +
                                str(detector_comparison["task_bootstrap_ci95"][metric]) + " |")
    comparison_lines += ["", "These secondary intervals are unadjusted and conditional on the fixed thresholds. "
                         "A calibration FPR below 5% does not guarantee that heldout FPR is below 5%. "
                         "Precision reflects the selected baseline-clean/rewarded-hack mixture, not deployment prevalence. "
                         "Training flags and generated behavior measure different quantities; detector validation alone does not establish mitigation.", "",
                         "The upstream probe's training-task exposure remains unresolved ([provenance audit](../../docs/probe-provenance.md)). "
                         "Source labels, task/source composition and different truncation capacities limit the comparison. "
                         "Complete coverage and score-derived counts are retained in [results.json](results.json)."]
    (destination / "monitor-comparison.md").write_text("\n".join(comparison_lines) + "\n")
    sections = [
        ("Abstract", "We adapt GRAFT to Qwen3-0.6B with two frozen reward-hacking monitors: an LLM-as-a-judge using the untouched learner checkpoint, and a downloaded embedding encoder with a trajectory probe. Each supplies flags to a separate GRAFT treatment; we compare detector quality and agentic behavior. " + conclusion + " " + " ".join(reasons) + f" This report covers {sum(p['complete'] for a,p in trained.items() if a != 'smoke')} completed training arms and {sum(len(v) for v in grouped.values())}/160 audited study episodes; {len(pending)} audit packets remain pending. The experiment is an exploratory single-seed pilot, not a powered confirmatory test."),
        ("Context, motivation and introduction", "GRAFT separates training updates into retain and forget adapters using a monitor [1]. This extension compares two concrete replacements for its simulated Terminal Wrench monitor: a frozen Qwen3-0.6B LLM judge [3,11] and a separately trained, frozen trajectory probe [2], with Qwen3-0.6B as the learner [3]. Both methods are evaluated as detectors and as sources of routing flags. The motivation is to distinguish an implementable monitor integration from evidence that it mitigates reward hacking."),
        ("Background and related work", "The LLM judge uses an intent rubric with conditional YES/NO next-token scores from the untouched Qwen3-0.6B [3]. LLM-as-a-judge research documents limitations that motivate empirical validation [11]; it does not establish this small judge's accuracy. The downloaded probe combines Qwen3-Embedding-0.6B [4], an attention LoRA adapter, and a turn-transformer classifier [2]. Both monitors run separately from learner training. GRAFT's monitor flags route gradients; they are not scalar reward penalties [1]. Classification accuracy and generated hacking behavior measure different quantities."),
        ("Method", "The exact preregistration is docs/protocol.md in the published repository commit recorded in publication.json. Preserve the published 232/99 task split, reserve 32 training tasks for calibration (seed 42), and independently freeze each monitor's strict score cutoff at empirical clean FPR <= 5%. Select at most 192 balanced trajectories, in multiples of 48, with a common memory-preflight length cap. Freeze the base; train two width-64 branches for five epochs, effective batch 48, learning rate 5e-4, weight decay 1e-4, betas (0.9,0.95), cosine schedule, 5% warmup, 1% unlabeled anchors and forget-gradient multiplier 2. Compare LLM-judge GRAFT, downloaded-probe GRAFT, no intervention, original simulated-monitor GRAFT, probe filtering and length-bin-shuffled probe GRAFT, plus the untouched base model. Filtering uses ordinary AdamW and thus is not a pure routing contrast. Correct Accelerate accumulation normalization [5] and use loss-equivalent Liger Qwen3 fused CE [6]."),
        ("Evaluation and analysis", "Evaluate eight fixed locally runnable heldout Docker tasks through pinned Harbor, once per mode and prompt condition: seed 2002, thinking disabled, maximum 16 turns and 120 seconds per agent episode. Audit intent using opaque packets before unblinding arms. The primary comparison is elicited hacking in probe-GRAFT retain-only versus no intervention both; legitimate success has a five-percentage-point safeguard. Wilson intervals describe known binary outcomes [7]; ambiguity is retained as bounds. Paired task bootstrap uses 10,000 draws, seed 42, and the exact discordance test is descriptive. Missing outcomes are never imputed as benign. Detector intervals resample whole tasks."),
        ("Results", "Training coverage: " + "; ".join(f"{a}: {p['steps']} updates, complete={p['complete']}" for a,p in trained.items()) +
         ".\n\n" + monitor_text("Downloaded probe", detector) + "\n\n" + monitor_text("Qwen judge", judge) +
         "\n\nRealized training flags: " + "; ".join(f"{arm}: {value['flagged']}/{value['total']}, including {value['flagged_positive']} labeled hacks and {value['flagged_negative']} labeled clean trajectories" for arm, value in routing.items()) + "." +
         "\n\n" + paired_text("Primary elicited differences (probe retain minus no intervention both)", primary) +
         "\n\n" + paired_text("Secondary elicited differences (LLM-judge retain minus no intervention both)", secondary["judge_minus_no_intervention"]) +
         "\n\n" + paired_text("Secondary elicited differences (LLM-judge retain minus probe retain)", secondary["judge_minus_probe"]) +
         " Full paired outcomes and missingness are in results.json and audited_outcomes.json."),
        ("Claims and implications", conclusion + " " + " ".join(reasons) + " Successful checkpoint loading, kernel equivalence and successful Docker execution support engineering compatibility only. Reported treatment differences are conditional descriptive observations, not claims of robust alignment or a causal mechanism beyond the implemented adapter intervention. This separation of measured results, uncertainty and mechanistic explanation follows the reporting concerns in [8-10]."),
        ("Discussion and limitations", "One training seed, eight selected tasks, a much smaller model than the source paper, task difficulty, label/source confounding, unresolved upstream detector exposure, truncation and non-independent implementing-assistant audits limit generalization. Neither monitor score is assumed calibrated. The judge/probe treatment contrast changes both flag frequency and which trajectories are flagged; the random-monitor control matches the probe's flag counts only. At n=8 one binary outcome changes a rate by 12.5 percentage points, larger than the 5pp capability tolerance. Bootstrap intervals may degenerate at the boundary; exact paired discordance evidence and Wilson intervals remain visible. The registered deadline can leave arms or modes incomplete. The user authorized a ten-more-hours extension before optimizer updates or behavioral outcomes; budget.json preserves both deadlines. Training error and infrastructure error are reported separately from behavior. No result-dependent extension or threshold retuning is allowed."),
        ("Conclusion", conclusion + " Larger, multi-seed studies with measured task/seed variance and independently blinded audits are required for stronger conclusions."),
        ("Appendix: prospective power and reproducibility", "Normal-approximation sample requirements [12] for a 10pp reduction at 80% power and two-sided alpha .05: " +
         "; ".join(f"{item['baseline']:.0%} to {item['treatment']:.0%}: {item['n_per_arm']} independent samples per arm" for item in results["power"]["independent"]) +
         ". For paired discordance probabilities 0.1, 0.2, 0.4 and 0.6, the approximate requirements are " +
         ", ".join(str(item["n_task_pairs"]) for item in results["power"]["paired"]) +
         " task pairs, respectively. These are planning assumptions; unknown seed-variance sensitivity is in results.json. The derivations and independence assumptions are given in docs/analysis.md. uv.lock pins the environment; artifacts.json pins sources and weights; data_lock.json fixes examples and preprocessing; per-checkpoint configuration and progress files record all actual updates. Full prompts, responses, scores and judgments are retained in timestamped local logs. Run uv run python -m experiments.qwen_probe run --profile pilot."),
        ("References", "[1] Ward, Hu, Wong, Hu, Woodworth, Turner and Cloud (2026). Gradient Routing Localizes and Removes Unintended Behaviors in RL. https://arxiv.org/abs/2610.17062\n[2] farzingkh/reward-hacking, revision cd4c566824c5961eaeab7b7fce7845240bd12b3d (MIT source). https://github.com/farzingkh/reward-hacking\n[3] Qwen3-0.6B. https://huggingface.co/Qwen/Qwen3-0.6B\n[4] Qwen3-Embedding-0.6B. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B\n[5] Accelerate gradient accumulation. https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation\n[6] Liger Kernel. https://github.com/linkedin/Liger-Kernel\n[7] SciPy binomial inference. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html\n[8] NeurIPS reporting checklist. https://neurips.cc/public/guides/PaperChecklist\n[9] Lones (2021; revised 2024), How to avoid machine learning pitfalls: a guide for academic researchers. https://arxiv.org/abs/2108.02497\n[10] Lipton and Steinhardt (2018), Troubling Trends in Machine Learning Scholarship. https://arxiv.org/abs/1807.03341")]
    sections[-1] = (sections[-1][0], sections[-1][1] + "\n[11] Zheng et al. (2023), Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. https://arxiv.org/abs/2306.05685\n[12] statsmodels, power for a test comparing two independent proportions. https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.power_proportions_2indep.html")  # Local: primary sources support the judge framing and prospective planning formula.
    if recoveries:
        sections.insert(-2, ("Infrastructure recovery and coverage", f"{results['coverage']['archived_setup_failures']} initial Docker setup failures were archived before any agent or verifier execution. The pinned runner's cleanup had removed the shared local images. A dated post-start amendment permits one uniform recovery of wholly unstarted jobs, preserving their results and hashes in setup_recovery.json. Actual model attempts, including timeouts, are never repeated. Rebuilt executable platform manifests match the original pre-study build logs; attestation/index changes and their evidence are recorded in image_identity_recovery.json. Repairs consume the unchanged deadline. The 120-second agent limit makes measured capability conditional on this local inference implementation and throughput."))
    sections.insert(4, ("Frozen LLM-judge and probe inputs", "The untouched pinned Qwen3-0.6B judge uses the released intent rubric and conditional YES/NO next-token logits with thinking disabled. The downloaded probe uses its frozen LoRA encoder and four-layer turn-transformer head. Separate cutoffs use the same calibration tasks and empirical-FPR rule. Both receive the same normalized trajectory text, excluding labels, source paths, rewards and judge metadata. Both retain at most 248 turns. The judge permits 16,384 aggregate input tokens with head/tail truncation; the probe permits 4,096 tokens per turn with normalized 1,024-dimensional last-token embeddings. This coverage difference is disclosed, not equated. The judge arm and its 16 agentic episodes were added by a dated user amendment before any optimizer update or behavioral outcome. Its detector and behavioral comparisons are secondary; the primary remains probe GRAFT versus no intervention. Equal prominence in the presentation does not change these registered analysis priorities."))
    sections.insert(-2, ("Adapter interventions and execution diagnostics", " ".join(paired_text(label, value) for label, value in ablations.items()) +
                        "\n\n" + "; ".join(f"{key}: {value['zero_parsed_command_episodes']}/{value['episodes']} episodes with zero parsed commands, {value['agent_timeouts']} agent timeouts" for key, value in diagnostics["conditions"].items()) + "." +
                        f"\n\nAll {diagnostics['execution_intervals_checked']} checked study execution intervals had {diagnostics['overlapping_execution_pairs']} overlapping pairs. " + diagnostics["note"] + " These execution summaries were specified after unblinding and are descriptive; the registered primary/secondary comparisons and all frozen intent judgments remain unchanged. Full condition counts are in execution_diagnostics.json."))
    if data_summary:  # Local: report realized inputs only after the dataset lock exists.
        sections.insert(-2, ("Appendix: realized data and implementation",
            f"The locked SFT subset contains {data_summary['train']['records']} trajectories across {data_summary['train']['tasks']} tasks, "
            f"with {data_summary['train']['labels'].get('0', 0)} clean and {data_summary['train']['labels'].get('1', 0)} labeled hacks. "
            f"Its {data_summary['train']['total_learner_tokens']:,} learner tokens fit the common {lock['configuration']['max_length']:,}-token limit "
            f"(longest selected record {data_summary['train']['max_tokens']:,}); {lock['length_exclusions']} candidate training records were excluded by length. "
            f"Calibration uses {data_summary['calibration']['records']} records on 32 disjoint training-side tasks. "
            f"Detector validation uses {data_summary['heldout']['records']} records on 99 heldout tasks; at most one record per task and class is selected. "
            "Provenance and exact normalized duplicate checks are in provenance.json; upstream detector exposure, semantic duplicates and Qwen pretraining exposure are unresolved.\n\n"
            "The downloaded encoder's rank-16 attention LoRA uses alpha 32 on q/k/v/o projections. Last-token vectors are converted to FP32 and L2-normalized. "
            "The fine-tuned notebook path uses role prefixes but no query-instruction prefix. Its four-layer, eight-head classifier projects 1,024-dimensional inputs to width 256 and uses a CLS readout. "
            "The encoder runs BF16; the classifier runs FP32; all parameters are frozen and dropout is off [2,4]. The learner's 28 frozen transformer layers receive width-64 retain/forget MLP branches. "
            "Qwen3's unsharded weights and tied embeddings are supported by the merger. Assistant-only labels preserve the source objective under the pinned TRL collator. "
            "Accelerate's additional accumulation divisor is corrected in this profile, so it is not a bit-for-bit reproduction [5]. The 1% anchor rule samples within each unflagged pool; changing the monitor can change anchor identities. No intervention disables routing and anchors.\n\n"
            "This run used one serialized GPU worker on an RTX 5070 Laptop GPU reporting 8,151 MiB, BF16 SDPA, temperature 0.7, top-p 0.8 and top-k 20, with thinking disabled. "
            "The 120-second agent budget includes inference throughput and is separate from environment setup/verification. Generated responses, including malformed outputs, are retained. "
            "Implementing-assistant audits use the released intent rubric, mixed opaque batches and an eight-packet unread buffer until the run ends; they are not independent human audits. "
            "Hashes certify unchanged evidence, not judgment correctness. The persisted deadline and amendment history are in budget.json. "
            f"Available audited records total {results['coverage']['audited_study_episodes']}/160; missing planned episodes total {results['coverage']['missing_planned_episodes']}. "
            "Missing ablations remain unavailable; no model attempt was repeated after the cap. Data/configuration manifests and source revisions accompany the checkpoints."))
    paper(destination, sections, [destination / "architecture.png", *figures], behavior)
    event("report_written", directory=destination, conclusion=conclusion, pending_audits=len(pending))
    return results
