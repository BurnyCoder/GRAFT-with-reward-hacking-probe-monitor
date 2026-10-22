"""Global: describe execution failures after unblinding, without changing scientific labels.

Local: count the pinned Harbor trajectory's parsed bash_command requests and inspect execution intervals.
Sources: sft/third_party/harbor/src/harbor/agents/terminus_2/terminus_2.py;
https://docs.python.org/3/library/datetime.html#datetime.datetime.fromisoformat
"""
from datetime import datetime  # Local: Harbor records ISO timestamps for each execution interval.
from pathlib import Path  # Local: access the existing trial artifacts; never launch model attempts.

from .runtime import read_json  # Local: reuse the repository evidence reader.


def execution_diagnostics(root, audited):
    """Global: descriptive post-hoc evidence only. Local: aggregate every available study condition uniformly."""
    conditions, intervals = {}, []  # Local: smoke is excluded from study denominators.
    for row in audited:
        if row["smoke"]:  # Local: the engineering smoke is not a treatment sample.
            continue
        key = f"{row['arm']}/{row['mode']}/{'elicited' if row['elicited'] else 'plain'}"
        packet = read_json(root / "audit/packets" / f"{row['packet_id']}.json")  # Local: reuse the exact judged evidence.
        commands = sum(call.get("name") == "bash_command" for message in packet["messages"] for call in message.get("tool_calls", []))
        value = conditions.setdefault(key, dict.fromkeys(("episodes", "zero_parsed_command_episodes", "parsed_command_requests", "agent_timeouts", "missing_trajectories", "raw_model_calls"), 0))
        value["episodes"] += 1  # Local: include timeouts and failed tasks, not just successful executions.
        value["zero_parsed_command_episodes"] += commands == 0  # Local: malformed output is not automatically hacking or benign success.
        value["parsed_command_requests"] += commands  # Local: requests can fail, lack a newline or target a non-shell interface.
        value["agent_timeouts"] += row["agent_error"] == "AgentTimeoutError"
        value["missing_trajectories"] += packet["missing_trajectory"]
        value["raw_model_calls"] += len(packet["raw_model_calls"])  # Local: raw evidence also contains rejected/overflowed outputs.
        timing = read_json(Path(row["trial"]) / "result.json").get("agent_execution") or {}
        if timing.get("started_at") and timing.get("finished_at"):
            intervals.append(tuple(datetime.fromisoformat(timing[field]).timestamp() for field in ("started_at", "finished_at")))
    overlaps = sum(max(a[0], b[0]) <= min(a[1], b[1]) for index, a in enumerate(intervals) for b in intervals[index + 1:])
    return {"analysis_status": "post-hoc descriptive engineering diagnostic", "conditions": conditions,
            "execution_intervals_checked": len(intervals), "overlapping_execution_pairs": overlaps,
            "note": "Parsed requests are not successful shell executions. Counts do not alter frozen intent judgments or establish a causal failure mechanism. Zero overlapping intervals supports the timestamp-based journal association for this sequential run."}


def diagnostic_rows(diagnostics):
    """Global: expose every measured condition equally. Local: compact table for reports and the paper."""
    return [["Deployment / prompt", "Episodes", "Zero parsed commands", "Agent timeouts"]] + [
        [key, str(value["episodes"]), str(value["zero_parsed_command_episodes"]), str(value["agent_timeouts"])]
        for key, value in diagnostics["conditions"].items()]  # Local: never fill missing conditions with zeros.
