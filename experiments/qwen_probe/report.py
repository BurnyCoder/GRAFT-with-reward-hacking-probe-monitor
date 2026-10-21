"""Global: trace every claim to artifacts. Local: generate Markdown, figures and a sourced PDF.

Sources: https://www.reportlab.com/docs/reportlab-userguide.pdf and https://matplotlib.org/stable/.
"""
import json  # Local: tables and transparent machine-readable results.
import shutil  # Local: copy compact provenance, never credentials or model weights.
from html import escape  # Local: safe PDF paragraphs from data.

import matplotlib  # Local: standalone, reproducible scientific figures.
matplotlib.use("Agg")  # Local: headless rendering in the repository environment.
import matplotlib.pyplot as plt

from .analysis import detector_metrics, paired_difference, power_analysis, proportion
from .audit import finalize, make_packets
from .data import ARMS
from .runtime import ROOT, event, read_json, write_json

ARM_RATIONALE = {  # Global: distinguish scientific questions without duplicating the full protocol.
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
             "Frozen encoder + probe\nScores → flags": (8.7, 3), "Matched SFT arms\nQwen3-0.6B": (1.5, 1),
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
        for index, (metric, value) in enumerate(detector["metrics"].items()):
            interval = detector["task_bootstrap_ci95"][metric]
            if value is not None and interval is not None:
                axis.errorbar(index, value, yerr=[[max(0, value - interval[0])], [max(0, interval[1] - value)]],
                              fmt="o", capsize=5, color="#244b6b")
        axis.set(xticks=range(4), xticklabels=["AUROC", "Precision", "Recall", "FPR"], ylim=(0, 1),
                 ylabel="Estimate and task-bootstrap 95% interval", title=f"Frozen detector: {detector['scored']} heldout trajectories, {detector['tasks']} tasks")
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
    story = [Paragraph("Qwen3-0.6B GRAFT with a Frozen Reward-Hacking Probe", styles["Title"]),
             Paragraph("Libor Burian · Exploratory pilot · 22 October 2026", styles["Normal"]), Spacer(1, 16)]
    markdown = ["# Qwen3-0.6B GRAFT with a Frozen Reward-Hacking Probe", "\nLibor Burian — exploratory pilot, 22 October 2026\n"]
    for heading, text in sections:
        markdown.extend([f"## {heading}\n", text + "\n"])
        story.append(Paragraph(escape(heading), styles["Heading2"]))
        for paragraph in text.split("\n\n"):
            story.append(Paragraph(escape(paragraph).replace("\n", "<br/>"), styles["BodyText"]))
    for figure in figures:
        story.extend([Spacer(1, 12), Image(str(figure), width=460, height=190, kind="proportional")])
        markdown.append(f"![{figure.stem}]({figure.name})\n")
    (destination / "paper.md").write_text("\n".join(markdown))
    SimpleDocTemplate(str(destination / "paper.pdf"), title="Qwen3-0.6B GRAFT exploratory pilot", author="Libor Burian").build(story)


def report(root):
    """Global: complete and incomplete results are both visible. Local: compute only observed metrics."""
    destination = ROOT / "reports/qwen_probe"
    destination.mkdir(parents=True, exist_ok=True)
    architecture(destination / "architecture.png")
    pending = make_packets(root)
    audited = finalize(root) if not pending else None
    audited = audited or []
    trained = {}
    for arm in (*ARMS, "smoke"):
        path = root / "checkpoints" / arm / "progress.json"
        trained[arm] = read_json(path) if path.exists() else {"complete": False, "steps": 0, "status": "not run"}
        metrics = root / "checkpoints" / arm / "metrics.jsonl"
        if metrics.exists():
            trained[arm]["learning_curve"] = [json.loads(line) for line in metrics.read_text().splitlines() if line]
    detector = {"status": "not run"}
    lock_path, threshold_path = root / "data/lock.json", root / "monitor/threshold.json"
    if lock_path.exists() and threshold_path.exists():
        lock = read_json(lock_path)
        inventory = {r["id"]: r for r in read_json(root / "data/inventory.json")}
        detector = detector_metrics([inventory[i] for i in lock["heldout"]], read_json(root / "monitor/scores.json"), read_json(threshold_path)["threshold"])
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
    if not baseline_hacks:
        reasons.append("No audited hacking attempt is established in the no-intervention baseline.")
    if not legitimate_solves:
        reasons.append("The primary arms have no established legitimate successes; a capability floor may dominate.")
    if complete_primary and baseline_hacks and legitimate_solves:
        conclusion = "The paired pilot estimates describe these eight tasks and one training seed; they do not establish general mitigation effectiveness."
    results = {"training": trained, "detector": detector, "behavior": behavior, "primary": primary,
               "pending_audits": len(pending), "conclusion": conclusion, "reasons": reasons, "power": power_analysis()}
    results["coverage"] = {"planned_study_episodes": 144, "audited_study_episodes": sum(map(len, grouped.values())),
                           "pending_audit_packets": len(pending), "note": "Unstarted or missing-result episodes are absent, never treated as nonhacking."}
    write_json(destination / "results.json", results)
    figures = result_figures(destination, results)
    for source, target in [("data/lock.json", "data_lock.json"), ("data/provenance.json", "provenance.json"),
                           ("monitor/threshold.json", "threshold.json"), ("evaluation/tasks.json", "tasks.json"),
                           ("budget.json", "budget.json"), ("preflight.json", "preflight.json"),
                           ("publication.json", "publication.json"), ("audit/judgments.json", "audit_judgments.json")]:
        if (root / source).exists():
            shutil.copy2(root / source, destination / target)
    lines = ["# Consolidated experiment results", "", conclusion, " ".join(reasons), "",
             "This report is generated from local manifests and frozen audits. See [protocol](../../docs/protocol.md) for the prespecified method.", "",
             "| Experiment | Updates | Completed recipe | Status |", "|---|---:|---|---|"]
    for arm, progress in trained.items():
        lines.append(f"| {arm} | {progress['steps']} | {progress['complete']} | {'complete' if progress['complete'] else 'incomplete'} |")
        (destination / f"{arm}.md").write_text(f"# {arm}\n\n" + ARM_RATIONALE[arm] + "\n\n" +
            f"Updates: {progress['steps']}. Complete planned training: {progress['complete']}.\n\n" +
            "Method and hyperparameter rationale: [preregistered protocol](../../docs/protocol.md). " +
            "Exact optimizer arguments are in results.json and the checkpoint configuration. " +
            "The smoke checkpoint is an engineering check and excluded from study comparisons.\n\n" +
            "Learning: completion of parameter updates alone does not establish hacking mitigation; inspect the blinded behavioral comparison and capability safeguard in [summary](summary.md).\n")
    (destination / "base.md").write_text("# Base model\n\nThe untouched pinned Qwen3-0.6B checkpoint measures initial task capability and spontaneous/elicited hacking. It receives no SFT updates. A failed-task floor here and in the primary no-intervention arm limits interpretation of lower hacking counts. See [summary](summary.md) for actual episode coverage and audited rates; absent measurements are not zero rates.\n")
    lines += ["", f"Pending blinded audits: {len(pending)}.", "", "## Frozen detector validation", "", "```json", json.dumps(detector, indent=2), "```",
              "", "## Behavioral measurements", "", "```json", json.dumps(behavior, indent=2), "```", "", "## Primary paired comparison", "", "```json", json.dumps(primary, indent=2), "```",
              "", "## Terminology and networks", "", "| Network / term | Architecture / algorithm | Objective and data | Measured meaning |",
              "|---|---|---|---|", "| Learner | Frozen Qwen3-0.6B, 28 layers, two width-64 MLP branches | Assistant-token SFT, matched clean/hack subset | Generates terminal actions |",
              "| Detector encoder | Qwen3-Embedding-0.6B plus frozen rank-16 attention LoRA | Upstream source-invariant training; no fitting here | 1,024-dimensional normalized turn vectors |",
              "| Probe head | Four transformer layers, width 256, 8 heads, CLS classifier | Upstream binary trajectory classifier | AUROC measures ranking, not behavioral mitigation |",
              "| GRAFT | Routed first moments, natural-gradient second moments | Flagged/unflagged pools plus 1% unlabeled anchors | Retain branch deployed after removing forget branch |",
              "| Verifier success | Task-specific executable checks | Heldout Docker tasks | Can include legitimate solves or hacks |",
              "| Hacking attempt | Intent-based blinded audit | Agent actions and rationale | Counts attempts even when they fail |",
              "| Legitimate success | Verifier success without hacking | Same audited episodes | Capability safeguard |",
              "| Completion / accuracy | Training completion and separate task outcomes | See manifests and intervals | No universal model accuracy or 'fill' metric applies |",
              "", "## Prospective power", "", "```json", json.dumps(results["power"], indent=2), "```",
              "", "Detailed engineering decisions and failures are recorded in [implementation log](../../docs/implementation.md). Full untruncated transcripts are under the ignored local artifacts/qwen_probe/logs directory."]
    (destination / "summary.md").write_text("\n".join(lines) + "\n")
    sections = [
        ("Abstract", conclusion + " " + " ".join(reasons) + f" This report covers {sum(p['complete'] for a,p in trained.items() if a != 'smoke')} completed training arms and {sum(len(v) for v in grouped.values())} audited study episodes; {len(pending)} audit packets remain pending. The experiment is an exploratory single-seed pilot, not a powered confirmatory test."),
        ("Context, motivation and introduction", "GRAFT separates training updates into retain and forget adapters using a monitor [1]. This study replaces its simulated Terminal Wrench monitor with a separately trained, frozen trajectory detector [2] while reducing the learner to Qwen3-0.6B [3]. The motivation is to distinguish an implementable detector integration from evidence that the integration mitigates reward hacking."),
        ("Background and related work", "The external detector combines Qwen3-Embedding-0.6B [4], an attention LoRA adapter, and a turn-transformer classifier [2]. Its representations do not share activations with the learner. GRAFT's monitor flags route gradients; they are not scalar reward penalties [1]. External classification accuracy and generated hacking behavior measure different quantities."),
        ("Method", "The exact preregistration is docs/protocol.md in the published repository commit recorded in publication.json. Preserve the published 232/99 task split, reserve 32 training tasks for calibration (seed 42), and freeze a strict score cutoff at empirical clean FPR <= 5%. Select at most 192 balanced trajectories, in multiples of 48, with a common memory-preflight length cap. Freeze the base; train two width-64 branches for five epochs, effective batch 48, learning rate 5e-4, weight decay 1e-4, betas (0.9,0.95), cosine schedule, 5% warmup, 1% unlabeled anchors and forget-gradient multiplier 2. Compare no intervention, original simulated monitor, downloaded probe, probe filtering and length-bin-shuffled probe flags. Filtering uses ordinary AdamW and thus is not a pure routing contrast. Correct Accelerate accumulation normalization [5] and use loss-equivalent Liger Qwen3 fused CE [6]."),
        ("Evaluation and analysis", "Evaluate eight fixed locally runnable heldout Docker tasks through pinned Harbor, once per mode and prompt condition: seed 2002, thinking disabled, maximum 16 turns and 120 seconds per agent episode. Audit intent using opaque packets before unblinding arms. The primary comparison is elicited hacking in probe-GRAFT retain-only versus no intervention both; legitimate success has a five-percentage-point safeguard. Wilson intervals describe known binary outcomes; ambiguity is retained as bounds. Paired task bootstrap uses 10,000 draws, seed 42, and the exact discordance test is descriptive. Missing outcomes are never imputed as benign. Detector intervals resample whole tasks."),
        ("Results", "Training coverage: " + "; ".join(f"{a}: {p['steps']} updates, complete={p['complete']}" for a,p in trained.items()) + ".\n\nDetector: " + json.dumps(detector) + ".\n\nPrimary paired estimates: " + json.dumps(primary)),
        ("Claims and implications", conclusion + " " + " ".join(reasons) + " Successful checkpoint loading, kernel equivalence and successful Docker execution support engineering compatibility only. Reported treatment differences are conditional descriptive observations, not claims of robust alignment or a causal mechanism beyond the implemented adapter intervention."),
        ("Discussion and limitations", "One training seed, eight selected tasks, a much smaller model than the source paper, task difficulty, label/source confounding, unresolved upstream detector exposure, truncation and non-independent implementing-assistant audits limit generalization. A sigmoid score is not assumed calibrated. At n=8 one binary outcome changes a rate by 12.5 percentage points, larger than the 5pp capability tolerance. Bootstrap intervals may degenerate at the boundary; exact paired discordance evidence and Wilson intervals remain visible. The six-hour cap can leave arms or modes incomplete. Training error and infrastructure error are reported separately from behavior. No result-dependent extension or threshold retuning is allowed."),
        ("Conclusion", conclusion + " Larger, multi-seed studies with measured task/seed variance and independently blinded audits are required for stronger conclusions."),
        ("Appendix: prospective power and reproducibility", "Normal-approximation sample requirements for a 10pp reduction at 80% power and two-sided alpha .05: " + json.dumps(results["power"]["independent"]) + ". Paired discordance and unknown seed-variance sensitivity are in results.json; they are planning assumptions. uv.lock pins the environment; artifacts.json pins sources and weights; data_lock.json fixes examples and preprocessing; per-checkpoint configuration and progress files record all actual updates. Full prompts, responses, scores and judgments are retained in timestamped local logs. Run uv run python -m experiments.qwen_probe run --profile pilot."),
        ("References", "[1] GRAFT. https://arxiv.org/abs/2610.17062\n[2] farzingkh/reward-hacking, revision cd4c566824c5961eaeab7b7fce7845240bd12b3d (MIT source). https://github.com/farzingkh/reward-hacking\n[3] Qwen3-0.6B. https://huggingface.co/Qwen/Qwen3-0.6B\n[4] Qwen3-Embedding-0.6B. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B\n[5] Accelerate gradient accumulation. https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation\n[6] Liger Kernel. https://github.com/linkedin/Liger-Kernel\n[7] SciPy binomial inference. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html\n[8] NeurIPS reporting checklist. https://neurips.cc/public/guides/PaperChecklist\n[9] Lones (2021), ML pitfalls. https://arxiv.org/abs/2108.02497\n[10] Lipton and Steinhardt (2018), Troubling Trends. https://arxiv.org/abs/1807.03341")]
    paper(destination, sections, [destination / "architecture.png", *figures])
    event("report_written", directory=destination, conclusion=conclusion, pending_audits=len(pending))
    return results
