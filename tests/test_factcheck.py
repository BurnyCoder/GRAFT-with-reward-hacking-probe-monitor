"""Global: protect the closed study during independent reassessment.

Local: synthetic packets test lossless reading, explicit judgments and complete freezing.
Source: https://docs.python.org/3/library/json.html
"""
import pytest  # Local: failure assertions protect historical evidence.

from experiments.qwen_probe.runtime import read_json, write_json  # Local: use the existing interchange.
from experiments.qwen_probe.factcheck.evidence import encode_strings, decode_strings, freeze_review, save_review


def test_display_recovers_every_character_and_occurrence():
    """Global: compression cannot hide evidence. Local: references reconstruct exact Unicode/newlines."""
    packet = {"messages": [{"content": "same\nline\n"}, {"content": "same\nline\n"}],
              "raw_model_calls": [{"response": "same\nchanged λ\n"}], "missing": None}
    lines, strings = [], []  # Local: shared dictionaries also cover raw responses.
    encoded = encode_strings(packet, lines, strings)
    assert decode_strings(encoded, lines, strings) == packet
    assert lines.count("same\n") == 1  # Local: exact repeats are shared, not dropped.


def test_freeze_requires_every_packet_and_rejects_changed_evidence(tmp_path):
    """Global: no selective review completion. Local: freeze hashes only after all explicit labels exist."""
    packets = {"one": {"packet_id": "one", "verifier_success": False, "messages": ["honest failure"]},
               "two": {"packet_id": "two", "verifier_success": None, "messages": [], "agent_error": "Timeout"}}
    save_review(tmp_path, packets, "one", False, False, "honest failure", "No verifier targeting in the observed work.")
    with pytest.raises(ValueError, match="incomplete"):
        freeze_review(tmp_path, packets)
    save_review(tmp_path, packets, "two", None, None, "Timeout", "No behavior is recorded; intent and success remain unknown.")
    frozen = freeze_review(tmp_path, packets)
    assert frozen["count"] == 2
    assert freeze_review(tmp_path, packets) == frozen  # Local: repeated verification preserves the first timestamp.
    packets["one"]["messages"].append("changed")
    with pytest.raises(ValueError, match="changed"):
        freeze_review(tmp_path, packets)


def test_judgments_cannot_invent_quotes_overwrite_or_claim_invalid_success(tmp_path):
    """Global: labels need evidence. Local: reject common forms of retrospective contamination."""
    packets = {"p": {"packet_id": "p", "verifier_success": False, "messages": ["failed command"]}}
    with pytest.raises(ValueError, match="literal"):
        save_review(tmp_path, packets, "p", False, False, "invented", "unsupported")
    with pytest.raises(ValueError, match="Legitimate"):
        save_review(tmp_path, packets, "p", False, True, "failed command", "incorrect")
    save_review(tmp_path, packets, "p", False, False, "failed command", "Observed unsuccessful task work.")
    before = read_json(tmp_path / "judgments.json")  # Local: preserve the first recorded interpretation.
    with pytest.raises(ValueError, match="already"):
        save_review(tmp_path, packets, "p", True, False, "failed command", "changed")
    assert read_json(tmp_path / "judgments.json") == before
