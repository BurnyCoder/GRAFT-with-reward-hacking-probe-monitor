"""Global: post-hoc execution counts must not become intent labels. Local: use synthetic frozen packets."""
from experiments.qwen_probe.runtime import write_json  # Local: exercise the same evidence serialization as production.


def test_diagnostics_separates_unparsed_commands_and_overlapping_intervals(tmp_path):
    """Global: report capability and evidence limits honestly. Local: count parsed requests, not shell success."""
    from experiments.qwen_probe.diagnostics import execution_diagnostics
    rows = []  # Local: two toy trials deliberately overlap to test association validation.
    for identity, messages, start, end in [("a", [], 1, 4), ("b", [{"tool_calls": [{"name": "bash_command"}]}], 3, 5)]:
        trial = tmp_path / identity
        write_json(trial / "result.json", {"agent_execution": {"started_at": f"1970-01-01T00:00:0{start}+00:00", "finished_at": f"1970-01-01T00:00:0{end}+00:00"}})
        write_json(tmp_path / "audit/packets" / f"{identity}.json", {"messages": messages, "raw_model_calls": [{"response": "malformed"}], "missing_trajectory": False})
        rows.append({"packet_id": identity, "trial": str(trial), "arm": "toy", "mode": "retain", "elicited": True, "smoke": False, "agent_error": "AgentTimeoutError"})
    result = execution_diagnostics(tmp_path, rows)
    group = result["conditions"]["toy/retain/elicited"]
    assert group["episodes"] == 2 and group["zero_parsed_command_episodes"] == 1
    assert group["parsed_command_requests"] == 1 and group["agent_timeouts"] == 2
    assert result["overlapping_execution_pairs"] == 1  # Local: ambiguous time association cannot masquerade as validated isolation.
    assert result["analysis_status"] == "post-hoc descriptive engineering diagnostic"
