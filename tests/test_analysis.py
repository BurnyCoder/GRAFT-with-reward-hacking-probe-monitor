"""Global: prevent misleading summaries. Local: missing, ambiguous and paired toy outcomes."""
from experiments.qwen_probe.analysis import paired_difference, power_analysis, proportion


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
