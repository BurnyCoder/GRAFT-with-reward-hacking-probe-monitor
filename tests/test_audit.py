"""Global: audits cannot silently become labels. Local: exercise an opaque toy packet."""
import pytest  # Local: fail closed when evidence or safeguards are missing.

from experiments.qwen_probe.audit import record_judgment
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
