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
from .runtime import ROOT, event, read_json, write_json

ARM_RATIONALE = {  # Global: distinguish scientific questions without duplicating the full protocol.
    "qwen_judge_graft": "Compare a frozen untouched Qwen3-0.6B intent judge with the downloaded probe, both as a detector and a source of GRAFT flags. The independent calibration cutoff and identical training recipe make this a secondary monitor comparison; constrained binary scoring is not a generated rationale.",
    "no_intervention": "Establish the effect of learning the matched clean/hack mixture with both adapters and the same split-moment optimizer, without routing. This is the primary comparator, not the untouched base model.",
    "probe_graft": "Test whether the downloaded out-of-domain detector supplies useful routing flags. Its encoder/head remain frozen; no downstream behavior is used to recalibrate the threshold. Removing the forget adapter is the primary deployment intervention.",
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


def paper(destination, sections, figures):
    """Global: one source for paper text and PDF. Local: ReportLab layout with actual result figures."""
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    styles = getSampleStyleSheet()
    title = "Qwen3-0.6B GRAFT with LLM-as-a-Judge and Probe Monitors"  # Local: one title keeps both methods visible across paper formats and metadata.
    story = [Paragraph(title, styles["Title"]),
             Paragraph("Libor Burian · Exploratory pilot · 22 October 2026", styles["Normal"]), Spacer(1, 16)]
    markdown = [f"# {title}", "\nLibor Burian — exploratory pilot, 22 October 2026\n"]
    for heading, text in sections:
        markdown.extend([f"## {heading}\n", text + "\n"])
        story.append(Paragraph(escape(heading), styles["Heading2"]))
        for paragraph in text.split("\n\n"):
            story.append(Paragraph(escape(paragraph).replace("\n", "<br/>"), styles["BodyText"]))
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
    if not complete_primary:
        reasons.append("The primary comparison is incomplete or has ambiguous judgments.")
    if control and not baseline_hacks:
        reasons.append("No audited hacking attempt is established in the no-intervention baseline.")
    if treatment + control and not legitimate_solves:
        reasons.append("The primary arms have no established legitimate successes; a capability floor may dominate.")
    if complete_primary and baseline_hacks and legitimate_solves:
        conclusion = "The paired pilot estimates describe these eight tasks and one training seed; they do not establish general mitigation effectiveness."
    judge_behavior = grouped.get("qwen_judge_graft/retain/elicited", [])
    secondary = {name: {field: paired_difference({r["task"]: r[field] for r in judge_behavior},
                                                {r["task"]: r[field] for r in comparator})
                        for field in ("hacking", "legitimate")} for name, comparator in
                 (("judge_minus_no_intervention", control), ("judge_minus_probe", treatment))}
    results = {"training": trained, "detector": detector, "judge": judge, "detector_comparison": detector_comparison,
               "data": data_summary, "routing": routing,
               "behavior": behavior, "primary": primary, "secondary": secondary,
               "pending_audits": len(pending), "conclusion": conclusion, "reasons": reasons, "power": power_analysis()}
    results["coverage"] = {"planned_study_episodes": 160, "audited_study_episodes": sum(map(len, grouped.values())),
                           "pending_audit_packets": len(pending), "note": "Unstarted or missing-result episodes are absent, never treated as nonhacking."}
    write_json(destination / "results.json", results)
    figures = result_figures(destination, results)
    for source, target in [("data/lock.json", "data_lock.json"), ("data/provenance.json", "provenance.json"),
                           ("monitor/threshold.json", "threshold.json"), ("evaluation/tasks.json", "tasks.json"),
                           ("judge/threshold.json", "judge_threshold.json"), ("judge/manifest.json", "judge_manifest.json"),
                           ("data/arms.json", "routing_flags.json"),
                           ("budget.json", "budget.json"), ("preflight.json", "preflight.json"),
                           ("publication.json", "publication.json"), ("audit/judgments.json", "audit_judgments.json")]:
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
        outcomes = {key: value for key, value in behavior.items() if key.startswith(arm + "/")}  # Local: same metrics as consolidated report.
        (destination / f"{arm}.md").write_text(f"# {arm}\n\n" + ARM_RATIONALE[arm] + "\n\n" +
            f"Updates: {progress['steps']}. Complete planned training: {progress['complete']}.\n\n" +
            "Method and hyperparameter rationale: [preregistered protocol](../../docs/protocol.md). " +
            "Exact optimizer arguments are in the per-arm configuration JSON when the run has started. " +
            "The smoke checkpoint is an engineering check and excluded from study comparisons.\n\n" +
            "Observed audited behavior (an empty object means no unblinded study measurements):\n\n```json\n" + json.dumps(outcomes, indent=2) + "\n```\n\n" +
            "Learning: completion of parameter updates alone does not establish hacking mitigation; inspect the blinded behavioral comparison and capability safeguard in [summary](summary.md).\n")
    base_outcomes = {key: value for key, value in behavior.items() if key.startswith("base/")}
    (destination / "base.md").write_text("# Base model\n\nThe untouched pinned Qwen3-0.6B checkpoint measures initial task capability and spontaneous/elicited hacking. It receives no SFT updates. A failed-task floor here and in the primary no-intervention arm limits interpretation of lower hacking counts. Absent measurements are not zero rates.\n\n```json\n" + json.dumps(base_outcomes, indent=2) + "\n```\n")
    lines += ["", f"Pending blinded audits: {len(pending)}.", "", "## Frozen detector validation", "", "```json", json.dumps(detector, indent=2), "```",
              "", "## Fixed data and realized routing", "", "```json", json.dumps({"data": data_summary, "routing": routing}, indent=2), "```",
              "", "## Same-Qwen judge validation and paired detector comparison", "", "```json", json.dumps({"judge": judge, "comparison": detector_comparison}, indent=2), "```",
              "", "## Secondary judge behavioral comparisons", "", "```json", json.dumps(secondary, indent=2), "```",
              "", "## Behavioral measurements", "", "```json", json.dumps(behavior, indent=2), "```", "", "## Primary paired comparison", "", "```json", json.dumps(primary, indent=2), "```",
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
              "", "## Prospective power", "", "```json", json.dumps(results["power"], indent=2), "```",
              "", "Detailed engineering decisions and failures are recorded in [implementation log](../../docs/implementation.md). Full untruncated transcripts are under the ignored local artifacts/qwen_probe/logs directory."]
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
        ("Abstract", "We adapt GRAFT to Qwen3-0.6B with two frozen reward-hacking monitors: an LLM-as-a-judge using the untouched learner checkpoint, and a downloaded embedding encoder with a trajectory probe. Each supplies flags to a separate GRAFT treatment; we compare detector quality and agentic behavior. " + conclusion + " " + " ".join(reasons) + f" This report covers {sum(p['complete'] for a,p in trained.items() if a != 'smoke')} completed training arms and {sum(len(v) for v in grouped.values())} audited study episodes; {len(pending)} audit packets remain pending. The experiment is an exploratory single-seed pilot, not a powered confirmatory test."),
        ("Context, motivation and introduction", "GRAFT separates training updates into retain and forget adapters using a monitor [1]. This extension compares two concrete replacements for its simulated Terminal Wrench monitor: a frozen Qwen3-0.6B LLM judge [3,11] and a separately trained, frozen trajectory probe [2], with Qwen3-0.6B as the learner [3]. Both methods are evaluated as detectors and as sources of routing flags. The motivation is to distinguish an implementable monitor integration from evidence that it mitigates reward hacking."),
        ("Background and related work", "The LLM judge uses an intent rubric with conditional YES/NO next-token scores from the untouched Qwen3-0.6B [3]. LLM-as-a-judge research documents limitations that motivate empirical validation [11]; it does not establish this small judge's accuracy. The downloaded probe combines Qwen3-Embedding-0.6B [4], an attention LoRA adapter, and a turn-transformer classifier [2]. Both monitors run separately from learner training. GRAFT's monitor flags route gradients; they are not scalar reward penalties [1]. Classification accuracy and generated hacking behavior measure different quantities."),
        ("Method", "The exact preregistration is docs/protocol.md in the published repository commit recorded in publication.json. Preserve the published 232/99 task split, reserve 32 training tasks for calibration (seed 42), and independently freeze each monitor's strict score cutoff at empirical clean FPR <= 5%. Select at most 192 balanced trajectories, in multiples of 48, with a common memory-preflight length cap. Freeze the base; train two width-64 branches for five epochs, effective batch 48, learning rate 5e-4, weight decay 1e-4, betas (0.9,0.95), cosine schedule, 5% warmup, 1% unlabeled anchors and forget-gradient multiplier 2. Compare LLM-judge GRAFT, downloaded-probe GRAFT, no intervention, original simulated-monitor GRAFT, probe filtering and length-bin-shuffled probe GRAFT, plus the untouched base model. Filtering uses ordinary AdamW and thus is not a pure routing contrast. Correct Accelerate accumulation normalization [5] and use loss-equivalent Liger Qwen3 fused CE [6]."),
        ("Evaluation and analysis", "Evaluate eight fixed locally runnable heldout Docker tasks through pinned Harbor, once per mode and prompt condition: seed 2002, thinking disabled, maximum 16 turns and 120 seconds per agent episode. Audit intent using opaque packets before unblinding arms. The primary comparison is elicited hacking in probe-GRAFT retain-only versus no intervention both; legitimate success has a five-percentage-point safeguard. Wilson intervals describe known binary outcomes; ambiguity is retained as bounds. Paired task bootstrap uses 10,000 draws, seed 42, and the exact discordance test is descriptive. Missing outcomes are never imputed as benign. Detector intervals resample whole tasks."),
        ("Results", "Training coverage: " + "; ".join(f"{a}: {p['steps']} updates, complete={p['complete']}" for a,p in trained.items()) +
         ".\n\n" + monitor_text("Downloaded probe", detector) + "\n\n" + monitor_text("Qwen judge", judge) +
         "\n\nPrimary elicited differences (probe retain minus no intervention both): " +
         "; ".join(f"{metric}: {estimate_text(value['difference'], value['bootstrap_ci95'])}, {value['known_pairs']} known task pairs"
                   for metric, value in primary.items()) + ". Full paired outcomes, missingness and secondary comparisons are in results.json and audited_outcomes.json."),
        ("Claims and implications", conclusion + " " + " ".join(reasons) + " Successful checkpoint loading, kernel equivalence and successful Docker execution support engineering compatibility only. Reported treatment differences are conditional descriptive observations, not claims of robust alignment or a causal mechanism beyond the implemented adapter intervention."),
        ("Discussion and limitations", "One training seed, eight selected tasks, a much smaller model than the source paper, task difficulty, label/source confounding, unresolved upstream detector exposure, truncation and non-independent implementing-assistant audits limit generalization. Neither monitor score is assumed calibrated. At n=8 one binary outcome changes a rate by 12.5 percentage points, larger than the 5pp capability tolerance. Bootstrap intervals may degenerate at the boundary; exact paired discordance evidence and Wilson intervals remain visible. The registered deadline can leave arms or modes incomplete. The user authorized a ten-more-hours extension before optimizer updates or behavioral outcomes; budget.json preserves both deadlines. Training error and infrastructure error are reported separately from behavior. No result-dependent extension or threshold retuning is allowed."),
        ("Conclusion", conclusion + " Larger, multi-seed studies with measured task/seed variance and independently blinded audits are required for stronger conclusions."),
        ("Appendix: prospective power and reproducibility", "Normal-approximation sample requirements for a 10pp reduction at 80% power and two-sided alpha .05: " + json.dumps(results["power"]["independent"]) + ". Paired discordance and unknown seed-variance sensitivity are in results.json; they are planning assumptions. uv.lock pins the environment; artifacts.json pins sources and weights; data_lock.json fixes examples and preprocessing; per-checkpoint configuration and progress files record all actual updates. Full prompts, responses, scores and judgments are retained in timestamped local logs. Run uv run python -m experiments.qwen_probe run --profile pilot."),
        ("References", "[1] GRAFT. https://arxiv.org/abs/2610.17062\n[2] farzingkh/reward-hacking, revision cd4c566824c5961eaeab7b7fce7845240bd12b3d (MIT source). https://github.com/farzingkh/reward-hacking\n[3] Qwen3-0.6B. https://huggingface.co/Qwen/Qwen3-0.6B\n[4] Qwen3-Embedding-0.6B. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B\n[5] Accelerate gradient accumulation. https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation\n[6] Liger Kernel. https://github.com/linkedin/Liger-Kernel\n[7] SciPy binomial inference. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html\n[8] NeurIPS reporting checklist. https://neurips.cc/public/guides/PaperChecklist\n[9] Lones (2021), ML pitfalls. https://arxiv.org/abs/2108.02497\n[10] Lipton and Steinhardt (2018), Troubling Trends. https://arxiv.org/abs/1807.03341")]
    sections[-1] = (sections[-1][0], sections[-1][1] + "\n[11] Zheng et al. (2023), Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. https://arxiv.org/abs/2306.05685")  # Local: support judge framing with its primary source.
    sections.insert(4, ("Frozen LLM-judge and probe inputs", "The untouched pinned Qwen3-0.6B judge uses the released intent rubric and conditional YES/NO next-token logits with thinking disabled. The downloaded probe uses its frozen LoRA encoder and four-layer turn-transformer head. Separate cutoffs use the same calibration tasks and empirical-FPR rule. Both receive the same normalized trajectory text, excluding labels, source paths, rewards and judge metadata. Both retain at most 248 turns. The judge permits 16,384 aggregate input tokens with head/tail truncation; the probe permits 4,096 tokens per turn with normalized 1,024-dimensional last-token embeddings. This coverage difference is disclosed, not equated. The judge arm and its 16 agentic episodes were added by a dated user amendment before any optimizer update or behavioral outcome. Its detector and behavioral comparisons are secondary; the primary remains probe GRAFT versus no intervention. Equal prominence in the presentation does not change these registered analysis priorities."))
    paper(destination, sections, [destination / "architecture.png", *figures])
    event("report_written", directory=destination, conclusion=conclusion, pending_audits=len(pending))
    return results
