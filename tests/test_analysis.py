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
