"""Global: prespecified allocation. Local: pure split, selection and threshold functions.

Sources: docs/protocol.md; https://numpy.org/doc/stable/reference/random/index.html.
"""
import math  # Local: exact allowed false-positive count; global: empirical FPR constraint.
import random  # Local: independent seeded RNGs; global: decisions do not consume model RNG.

from .runtime import read_json, write_json  # Local: immutable on-disk threshold contract.


def partition_tasks(train, evaluation, seed=42):
    """Global: no task leakage. Local: reserve 32 tasks from the published training split."""
    if set(train) & set(evaluation) or len(set(train)) != 232 or len(set(evaluation)) != 99:
        raise ValueError("Expected disjoint published 232/99 task split")
    calibration = random.Random(seed).sample(sorted(train), 32)  # Local: fixed training-side draw.
    return sorted(set(train) - set(calibration)), sorted(calibration), sorted(evaluation)


def calibrate(clean_scores, max_fpr=0.05):
    """Global: calibration only. Local: smallest observed cutoff satisfying strict-score > cutoff."""
    if not clean_scores or not all(math.isfinite(x) and 0 <= x <= 1 for x in clean_scores):
        raise ValueError("Calibration needs finite clean scores in [0,1]")
    ordered = sorted(clean_scores)  # Local: ties remain below a strict threshold.
    allowed = math.floor(max_fpr * len(ordered))  # Local: never round the FPR budget upward.
    return ordered[len(ordered) - allowed - 1]  # Local: at most allowed scores exceed this value.


def freeze_threshold(path, value):
    """Global: prohibit evaluation tuning. Local: repeated identical calibration is idempotent."""
    if path.exists() and read_json(path) != value:  # Local: detect both data and cutoff changes.
        raise ValueError("Threshold is frozen; use a separately preregistered experiment")
    if not path.exists():  # Local: don't change the freeze timestamp on repeated execution.
        write_json(path, value)


def select_balanced(rows, limit=192, seed=42):
    """Global: matched training data. Local: largest balanced multiple of 48, independent of scores."""
    groups = [[r for r in rows if r["label"] == label] for label in (0, 1)]
    count = min(limit, 2 * min(map(len, groups))) // 48 * 48  # Local: full effective batches only.
    if count < 48:  # Local: fail rather than silently changing the preregistered batch size.
        raise ValueError("Fewer than 48 eligible balanced trajectories")
    rng = random.Random(seed)  # Local: score-blind allocation.
    selected = sum((rng.sample(sorted(group, key=lambda r: r["id"]), count // 2) for group in groups), [])
    return sorted(selected, key=lambda r: r["id"])  # Local: same base order for every arm.


def shuffled_flags(rows, flags, seed=42):
    """Global: length-controlled negative control. Local: shuffle flags within four rank bins."""
    ordered = sorted(rows, key=lambda r: (r["tokens"], r["id"]))  # Local: deterministic tie break.
    rng, output = random.Random(seed), {}  # Local: RNG isolated from training.
    for start in range(0, len(rows), len(rows) // 4):  # Local: balanced size is divisible by four.
        group = ordered[start:start + len(rows) // 4]  # Local: equal-count length bins.
        values = [flags[r["id"]] for r in group]  # Local: preserve each bin's flag count.
        rng.shuffle(values)  # Local: destroy trajectory-specific information within bins.
        output.update(zip((r["id"] for r in group), values))  # Local: stable ID-to-flag mapping.
    return output
