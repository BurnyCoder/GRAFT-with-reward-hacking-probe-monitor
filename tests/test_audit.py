"""Global: audits cannot silently become labels. Local: exercise an opaque toy packet."""
import pytest  # Local: fail closed when evidence or safeguards are missing.

from experiments.qwen_probe.audit import raw_model_calls, record_judgment
from experiments.qwen_probe.runtime import read_json, write_json


def test_judgment_requires_evidence_and_preserves_ambiguity(tmp_path):
    """Global: honest missing/ambiguous outcomes. Local: no success from a failed verifier."""
    write_json(tmp_path / "audit/packets/a.json", {"messages": ["I will solve it"], "verifier_success": False})
    write_json(tmp_path / "audit/judgments.json", {})
    with pytest.raises(ValueError):
        record_judgment(tmp_path, "a", False, True, "I will solve it", "claim")
    with pytest.raises(ValueError):
        record_judgment(tmp_path, "a", False, False, "invented evidence", "claim")
    record_judgment(tmp_path, "a", None, False, "I will solve it", "Intent is ambiguous in this toy example")
    assert read_json(tmp_path / "audit/judgments.json")["a"]["hacking"] is None
    with pytest.raises(ValueError, match="frozen"):
        record_judgment(tmp_path, "a", False, False, "I will solve it", "changed my mind")


def test_raw_audit_keeps_invalid_responses_without_treatment_metadata(tmp_path):
    """Global: failed parsing cannot hide intent. Local: select calls by trial time and whitelist text."""
    folder = tmp_path / "evaluation/generations/opaque"
    write_json(folder / "inside.json", {"started_at": 10, "messages": [{"role": "user", "content": "task"}],
                                       "text": "I will tamper with the verifier", "finish_reason": "length",
                                       "arm": "hidden", "probe_score": 0.9, "api_key": "secret"})
    write_json(folder / "outside.json", {"started_at": 30, "messages": [], "text": "different trial"})
    result = {"agent_execution": {"started_at": "1970-01-01T00:00:09+00:00", "finished_at": "1970-01-01T00:00:20+00:00"}}
    calls = raw_model_calls(tmp_path, result)
    assert calls == [{"messages": [{"role": "user", "content": "task"}],
                      "response": "I will tamper with the verifier", "finish_reason": "length"}]
