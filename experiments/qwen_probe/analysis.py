"""Global: bounded claims with uncertainty. Local: task-level resampling and explicit missingness.

Sources: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
and https://www.statsmodels.org/stable/stats.html#power-and-sample-size-calculations
"""
import math  # Local: prospective integer sample sizes.

import numpy as np  # Local: deterministic paired/cluster bootstrap.
from scipy.stats import binomtest, norm  # Local: library intervals, discordance test and quantiles.
from sklearn.metrics import roc_auc_score  # Local: standard rank-based AUROC.


def proportion(values):
    """Global: ambiguity is not absence. Local: report observed counts and full-sample bounds."""
    known = [bool(v) for v in values if v is not None]
    successes, total, missing = sum(known), len(values), len(values) - len(known)
    interval = binomtest(successes, len(known)).proportion_ci(method="wilson") if known else None
    return {"successes": successes, "observed": len(known), "total": total, "unknown": missing,
            "rate_known": successes / len(known) if known else None,
            "ci95_known": list(interval) if interval else None,
            "bounds": [successes / total, (successes + missing) / total] if total else None}


def paired_difference(treatment, control):
    """Global: pair on task identity. Local: keep ambiguity bounds and bootstrap only known pairs."""
    tasks = sorted(set(treatment) & set(control))
    known = [task for task in tasks if treatment[task] is not None and control[task] is not None]
    lower, upper = [], []
    for task in tasks:
        a, b = treatment[task], control[task]
        lower.append((0 if a is None else int(a)) - (1 if b is None else int(b)))
        upper.append((1 if a is None else int(a)) - (0 if b is None else int(b)))
    differences = np.array([int(treatment[t]) - int(control[t]) for t in known], dtype=float)
    interval, estimate, exact_p = None, None, None
    if len(differences):
        samples = np.random.default_rng(42).choice(differences, size=(10000, len(differences))).mean(1)
        interval, estimate = np.quantile(samples, [0.025, 0.975]).tolist(), float(differences.mean())
        discordant = differences[differences != 0]
        exact_p = float(binomtest(int((discordant > 0).sum()), len(discordant), 0.5).pvalue) if len(discordant) else 1.0
    return {"paired_tasks": len(tasks), "known_pairs": len(known), "difference": estimate,
            "bootstrap_ci95": interval, "exact_discordance_p": exact_p,
            "ambiguity_bounds": [float(np.mean(lower)), float(np.mean(upper))] if tasks else None,
            "task_differences": dict(zip(known, differences.tolist())),
            "warning": "Conditional on one training seed; n=8 bootstrap may be degenerate; missing pairs are not imputed"}


def classification_metrics(rows, scores, threshold):
    """Global: common detector definitions. Local: undefined AUROC or precision remains missing."""
    labels = np.array([row["label"] for row in rows])
    values = np.array([scores[row["id"]]["score"] for row in rows])
    flags = values > threshold  # Local: equality is unflagged for both frozen monitors.
    positive, negative = labels == 1, labels == 0
    return {"auroc": float(roc_auc_score(labels, values)) if len(set(labels)) == 2 else None,
            "precision": float(labels[flags].mean()) if flags.any() else None,
            "recall": float(flags[positive].mean()) if positive.any() else None,
            "fpr": float(flags[negative].mean()) if negative.any() else None}


def detector_metrics(rows, scores, threshold):
    """Global: heldout measurement, no refitting. Local: task-cluster bootstrap for all metrics."""
    available = [r for r in rows if r["id"] in scores]
    if not available:
        return {"status": "incomplete", "scored": 0, "planned": len(rows)}
    values = classification_metrics(available, scores, threshold)
    grouped = {task: [r for r in available if r["task"] == task] for task in sorted({r["task"] for r in available})}
    rng, bootstraps = np.random.default_rng(42), []
    for _ in range(1000):
        sampled = rng.choice(list(grouped), size=len(grouped), replace=True)
        bootstraps.append(classification_metrics([r for task in sampled for r in grouped[task]], scores, threshold))
    intervals = {}
    for metric in values:
        draws = [b[metric] for b in bootstraps if b[metric] is not None]
        intervals[metric] = np.quantile(draws, [0.025, 0.975]).tolist() if draws else None
    return {"status": "complete" if len(available) == len(rows) else "incomplete", "scored": len(available),
            "planned": len(rows), "tasks": len(grouped), "metrics": values, "task_bootstrap_ci95": intervals,
            "positive": sum(r["label"] for r in available), "negative": sum(not r["label"] for r in available)}


def compare_detectors(rows, judge_scores, probe_scores, judge_threshold, probe_threshold):
    """Global: secondary paired measurement. Local: resample identical tasks for both fixed detectors."""
    common = [row for row in rows if row["id"] in judge_scores and row["id"] in probe_scores]
    if not common:
        return {"status": "incomplete", "paired_records": 0}

    def difference(batch):
        """Local: subtract metrics only when both denominators are defined."""
        judge = classification_metrics(batch, judge_scores, judge_threshold)
        probe = classification_metrics(batch, probe_scores, probe_threshold)
        return {metric: judge[metric] - probe[metric] if judge[metric] is not None and probe[metric] is not None
                else None for metric in judge}

    grouped = {task: [row for row in common if row["task"] == task] for task in sorted({row["task"] for row in common})}
    rng, samples = np.random.default_rng(42), []
    for _ in range(1000):
        tasks = rng.choice(list(grouped), size=len(grouped), replace=True)
        samples.append(difference([row for task in tasks for row in grouped[task]]))
    values = difference(common)
    valid = {metric: [sample[metric] for sample in samples if sample[metric] is not None] for metric in values}
    return {"status": "complete" if len(common) == len(rows) else "incomplete", "paired_records": len(common),
            "paired_tasks": len(grouped), "judge_minus_probe": values,
            "task_bootstrap_ci95": {metric: np.quantile(draws, [.025, .975]).tolist() if draws else None
                                     for metric, draws in valid.items()},
            "valid_bootstrap_draws": {metric: len(draws) for metric, draws in valid.items()},
            "note": "Secondary unadjusted comparison; conditional on fixed calibration cutoffs and one data sample."}


def monitor_coverage(identities, scores):
    """Global: expose representation loss. Local: distinguish turn dropping, token truncation and missing scores."""
    observed = [scores[identity] for identity in identities if identity in scores]  # Local: no invented zeros for missing work.
    totals = {key: sum(row[key] for row in observed) for key in
              ("turns", "retained_turns", "truncated_turns", "tokens_before_turn_truncation", "retained_tokens")}
    return {"planned": len(identities), "scored": len(observed), **totals,
            "dropped_turns": totals["turns"] - totals["retained_turns"],
            "retained_turn_fraction": totals["retained_turns"] / totals["turns"] if totals["turns"] else None,
            "retained_token_fraction_in_selected_turns": totals["retained_tokens"] / totals["tokens_before_turn_truncation"]
            if totals["tokens_before_turn_truncation"] else None}  # Local: tokens in dropped turns were never encoded/counted.


def power_analysis():
    """Global: plan a powered follow-up. Local: normal approximations, not achieved-power claims."""
    za, zb, delta = norm.ppf(0.975), norm.ppf(0.8), 0.1
    independent = []
    for p0 in (0.1, 0.2, 0.3, 0.5):
        p1, pooled = p0 - delta, p0 - delta / 2
        n = math.ceil((za * math.sqrt(2 * pooled * (1 - pooled)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / delta ** 2)
        independent.append({"baseline": p0, "treatment": p1, "n_per_arm": n})
    paired = [{"discordance": q, "n_task_pairs": math.ceil((za + zb) ** 2 * (q - delta ** 2) / delta ** 2)} for q in (0.1, 0.2, 0.4, 0.6)]
    seed_sensitivity = []
    target_variance = delta ** 2 / (za + zb) ** 2
    for sigma in (0, 0.025, 0.05, 0.10):
        for seeds in (1, 3, 5, 10):
            denominator = seeds * target_variance - sigma ** 2
            n = math.ceil((0.2 - delta ** 2) / denominator) if denominator > 0 else None
            seed_sensitivity.append({"between_seed_sd": sigma, "training_seeds": seeds, "task_pairs_per_seed": n})
    return {"alpha": 0.05, "power": 0.8, "absolute_reduction": delta, "independent": independent, "paired": paired,
            "seed_sensitivity": seed_sensitivity,
            "assumptions": "Two-sided normal approximations; paired discordance sensitivity; seed grid assumes independent seeds and conditional task noise, q=.2. Same tasks across seeds may correlate; unknown seed SD is not estimated from this pilot. p1=0 is a boundary approximation."}
