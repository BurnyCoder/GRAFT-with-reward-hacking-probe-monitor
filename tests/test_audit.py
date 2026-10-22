"""Global: audits cannot silently become labels. Local: exercise an opaque toy packet."""
from copy import deepcopy  # Local: prove the display cannot mutate frozen evidence.
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


def test_compact_audit_view_is_lossless_and_keeps_order():
    """Global: manageable audits cannot discard evidence. Local: reconstruct every repeated message."""
    from experiments.qwen_probe.audit_view import compact_packet
    task = {"role": "user", "content": "Task α\nDo the work."}  # Local: preserve Unicode and real line breaks.
    output = {"role": "assistant", "content": "Inspect tests", "tool_calls": [{"command": "cat /tests/check.py"}]}
    packet = {"packet_id": "opaque", "messages": [task, output, task], "verifier_success": False,
              "raw_model_calls": [{"messages": [task], "response": "Inspect tests", "finish_reason": "stop"},
                                  {"messages": [task, output, task], "response": None, "finish_reason": "not recorded"}]}
    original = deepcopy(packet)  # Local: a view must leave packet fingerprints and judgments valid.
    view = compact_packet(packet)
    restored = deepcopy(view["packet"])
    restored["messages"] = [view["message_blocks"][index] for index in restored["messages"]]
    for call in restored["raw_model_calls"]:
        call["messages"] = [view["message_blocks"][index] for index in call["messages"]]
    assert restored == original  # Local: roles, tools, output, ordering, multiplicity and missing values survive.
    assert packet == original
    assert len(view["message_blocks"]) == 2  # Local: only exact duplicate message objects are shared.
    assert compact_packet({"instruction": "No trajectory"})["packet"] == {"instruction": "No trajectory"}


def test_running_jobs_cannot_freeze_packets_or_unblind(tmp_path, monkeypatch):
    """Global: wait for stable evidence. Local: a trial result may precede its job's final journal writes."""
    from experiments.qwen_probe import audit
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Solve the task")  # Local: no real trajectory or treatment is exposed by this regression.
    monkeypatch.setattr(audit, "original_task", lambda _: task)
    calls = [{"messages": [], "response": "late complete output", "finish_reason": "stop"}]
    monkeypatch.setattr(audit, "raw_model_calls", lambda *_: calls)
    events = []  # Local: ordered creation logs must not disclose which treatment received an opaque identity.
    monkeypatch.setattr(audit, "event", lambda kind, **data: events.append((kind, data)))
    metadata = {"arm": "hidden", "mode": "retain", "elicited": True, "smoke": False, "status": "running"}
    write_json(tmp_path / "evaluation/index.json", {"job": metadata})
    write_json(tmp_path / "evaluation/jobs/job/trial/result.json", {"task_name": "task"})
    assert audit.make_packets(tmp_path) == []  # Local: completed-trial files alone do not close an active job.
    assert not list((tmp_path / "audit/packets").glob("*.json"))
    assert audit.finalize(tmp_path) is None  # Local: zero ready packets is not permission to unblind mid-job.
    assert not (tmp_path / "audit/unblinded.json").exists()
    write_json(tmp_path / "evaluation/index.json", {"job": metadata | {"status": "finished"}})
    pending = audit.make_packets(tmp_path)
    assert len(pending) == 1
    packet = read_json(tmp_path / "audit/packets" / f"{pending[0]}.json")
    assert packet["raw_model_calls"] == calls  # Local: the later complete response is the evidence that gets frozen.
    assert not any("packet_id" in data for _, data in events)  # Local: pending IDs are shown only as an unordered-treatment batch.


def test_report_refresh_preserves_the_same_judgment_freeze(tmp_path, monkeypatch):
    """Global: rerendering must not move the evidence freeze. Local: equal content keeps its first timestamp."""
    from experiments.qwen_probe import audit
    from experiments.qwen_probe.runtime import fingerprint
    packet = {"verifier_success": False, "agent_error": None, "messages": ["ordinary failure"]}
    write_json(tmp_path / "audit/packets/a.json", packet)  # Local: a synthetic stable packet avoids any study relabeling.
    write_json(tmp_path / "audit/private_mapping.json", {"a": {"packet_fingerprint": fingerprint(packet)}})
    write_json(tmp_path / "audit/judgments.json", {"a": {"packet_fingerprint": fingerprint(packet), "hacking": False}})
    monkeypatch.setattr(audit, "make_packets", lambda _: [])  # Local: isolate idempotent finalization from packet discovery.
    monkeypatch.setattr(audit.time, "time", lambda: 10)
    audit.finalize(tmp_path)
    frozen = read_json(tmp_path / "audit/frozen_judgments.json")
    monkeypatch.setattr(audit.time, "time", lambda: 20)
    audit.finalize(tmp_path)
    assert read_json(tmp_path / "audit/frozen_judgments.json") == frozen
