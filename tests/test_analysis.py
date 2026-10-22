"""Global: prevent misleading summaries. Local: missing, ambiguous and paired toy outcomes."""
from experiments.qwen_probe.analysis import monitor_coverage, paired_difference, power_analysis, proportion


def test_missing_is_not_nonhacking():
    """Global: preserve audit uncertainty. Local: one positive, one ambiguous has bounds .5–1."""
    result = proportion([True, None])
    assert result["bounds"] == [0.5, 1.0] and result["observed"] == 1


def test_pairs_and_power_are_explicit():
    """Global: compare like tasks. Local: unmatched/unknown tasks cannot inflate n."""
    result = paired_difference({"a": False, "b": None, "c": True}, {"a": True, "b": False})
    assert result["known_pairs"] == 1 and result["difference"] == -1
    assert result["ambiguity_bounds"] == [-0.5, 0]
    assert power_analysis()["independent"][-1]["n_per_arm"] > 300


def test_monitor_coverage_distinguishes_unscored_and_truncated():
    """Global: absent scores are not zero truncation. Local: aggregate only observed coverage."""
    score = {"turns": 250, "retained_turns": 248, "truncated_turns": 2,
             "tokens_before_turn_truncation": 12000, "retained_tokens": 10000}
    value = monitor_coverage(["a", "b"], {"a": score})
    assert value["planned"] == 2 and value["scored"] == 1
    assert value["dropped_turns"] == 2 and value["truncated_turns"] == 2
    assert value["retained_token_fraction_in_selected_turns"] == 10000 / 12000
    assert monitor_coverage(["a"], {})["retained_turn_fraction"] is None
def test_paired_detector_difference_uses_same_tasks():
    """Global: judge comparison has a fixed direction. Local: opposite rankings have a unit AUROC difference."""
    from experiments.qwen_probe.analysis import compare_detectors
    rows = [{"id": str(index), "task": str(index // 2), "label": index % 2} for index in range(8)]
    judge = {row["id"]: {"score": float(row["label"])} for row in rows}
    probe = {row["id"]: {"score": 1 - float(row["label"])} for row in rows}
    result = compare_detectors(rows, judge, probe, 0.5, 0.5)
    assert result["judge_minus_probe"] == {"auroc": 1.0, "precision": 1.0, "recall": 1.0, "fpr": -1.0}
    assert result["paired_tasks"] == 4
    assert result["task_bootstrap_ci95"]["auroc"] == [1.0, 1.0]
