"""Global: prevent leakage and adaptive decisions. Local: toy, deterministic tests.

Sources: https://neurips.cc/public/guides/PaperChecklist and docs/protocol.md.
"""
import json  # Local: inspect persisted thresholds; global: prove immutable calibration.

import pytest  # Local: exception checks; global: fail closed on invalid studies.

from experiments.qwen_probe.protocol import (calibrate, freeze_threshold, partition_tasks,
                                              select_balanced, shuffled_flags)
from experiments.qwen_probe.runtime import fingerprint


def test_task_separation():
    """Global: holdout isolation. Local: calibration is a training-side subset."""
    train, calibration, evaluation = partition_tasks(list(range(232)), list(range(232, 331)))
    assert len(train) == 200 and len(calibration) == 32 and len(evaluation) == 99
    assert not (set(train) & set(calibration) or set(calibration) & set(evaluation))
    assert partition_tasks(list(range(232)), list(range(232, 331))) == (train, calibration, evaluation)


def test_threshold_ties_and_freezing(tmp_path):
    """Global: no evaluation tuning. Local: strict > handles tied quantiles."""
    threshold = calibrate([0.1] * 19 + [0.9], max_fpr=0.05)
    assert sum(s > threshold for s in [0.1] * 19 + [0.9]) / 20 <= 0.05
    path = tmp_path / "threshold.json"
    freeze_threshold(path, {"threshold": threshold, "calibration": "abc"})
    freeze_threshold(path, {"threshold": threshold, "calibration": "abc"})
    with pytest.raises(ValueError, match="frozen"):
        freeze_threshold(path, {"threshold": 0.8, "calibration": "abc"})
    assert json.loads(path.read_text())["threshold"] == threshold
    with pytest.raises(ValueError):
        calibrate([])


def test_balanced_multiple_and_length_matched_random_control():
    """Global: matched data/control. Local: labels and bin counts stay fixed."""
    rows = [{"id": str(i), "label": i % 2, "tokens": 100 + i} for i in range(250)]
    chosen = select_balanced(rows)
    assert len(chosen) == 192 and sum(r["label"] for r in chosen) == 96
    flags = {r["id"]: i % 3 == 0 for i, r in enumerate(chosen)}
    shuffled = shuffled_flags(chosen, flags)
    for bucket in range(4):
        members = sorted(chosen, key=lambda r: (r["tokens"], r["id"]))[bucket * 48:(bucket + 1) * 48]
        assert sum(flags[r["id"]] for r in members) == sum(shuffled[r["id"]] for r in members)


def test_fingerprint_changes_with_every_input():
    """Global: cache validity. Local: any upstream semantic change invalidates a key."""
    base = dict(model="a", tokenizer="b", data="c", preprocessing="d", configuration="e")
    for key in base:
        assert fingerprint(base) != fingerprint(base | {key: "changed"})
