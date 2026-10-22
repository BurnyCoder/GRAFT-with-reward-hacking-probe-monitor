"""Global: match the released probe representation without metadata leakage.

Local: adapted from notebook cells 45/46/115 (MIT; Bridget Atchison Nevel,
Farzin Golkhosh, Jack McLellan; full notice in LICENSES/reward-hacking-MIT.txt):
https://github.com/farzingkh/reward-hacking/blob/cd4c566824c5961eaeab7b7fce7845240bd12b3d/notebooks/cs1090b_ms4_main_group29.ipynb
"""
import copy  # Local: upstream normalization mutates call dictionaries; isolate caller input.
import re  # Local: reproduce notebook XML tool normalization.

INLINE_CALL = re.compile(r"<(bash|python|submit)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
INLINE_RESULT = re.compile(r"<(bash-output|python-output)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
VERSION = "notebook-45-46-115-ft219-221-atif-v1"  # Global: cache invalidation for this exact mapping.


def trajectory_messages(trajectory):
    """Global: label blindness. Local: whitelist ATIF messages, calls and observations only."""
    messages = []  # Local: no task ID, label, file path, verifier or judge enters this list.
    for step in trajectory.get("steps", []):  # Local: preserve actual chronological order.
        role = "assistant" if step.get("source") == "agent" else step.get("source", "user")
        calls = [{"name": call.get("function_name", ""), "parameters": call.get("arguments", "")}
                 for call in step.get("tool_calls", [])]  # Local: explicit visible-field allowlist.
        messages.append({"role": role, "content": step.get("message", ""), "tool_calls": calls})
        for result in (step.get("observation") or {}).get("results", []):  # Local: actual tool outputs.
            messages.append({"role": "tool", "content": result.get("content", "")})
    return messages


def normalize_messages(messages):
    """Global: notebook equivalence. Local: normalize and prefix each nonempty turn."""
    output = []  # Local: fine-tuned encoder takes raw role-prefixed turns, no query instruction.
    for original in messages:  # Local: one normalized embedding per turn.
        msg = copy.deepcopy(original)  # Local: prevent cache inputs changing during normalization.
        role = (msg.get("role") or "assistant").strip().lower()  # Local: notebook role normalization.
        content = msg.get("content") or ""  # Local: empty content may still contain calls.
        calls, results = msg.get("tool_calls") or [], msg.get("tool_results") or []
        function = msg.get("function_call")  # Local: released MALT representation support.
        if not calls and function and function.get("name"):
            calls = [{"name": function["name"], "parameters": function.get("arguments", "")}]
        name = msg.get("tool_name") or msg.get("name")  # Local: released corpus representation.
        if not calls and role == "assistant" and name:
            calls = [{"name": name, "parameters": msg.get("arguments") or {}}]
        if role == "tool_call":  # Local: stand-alone corpus invocation becomes assistant.
            calls, role = [{"name": msg.get("name") or "unknown", "parameters": msg.get("arguments") or {}}], "assistant"
        if role in ("function", "tool", "toolresult", "tool_output", "tool_result"):
            results, role, content = [{"output": content}], "tool", ""  # Local: notebook result wrapper.
        inline = bool(INLINE_CALL.search(content) or INLINE_RESULT.search(content))
        content = INLINE_CALL.sub(lambda m: f'<tool_call name="{m[1].strip().lower()}">{m[2]}</tool_call>', content)
        content = INLINE_RESULT.sub(r"<tool_result>\2</tool_result>", content)
        if not (content.strip() or calls or results):  # Local: notebook skips empty turns.
            continue
        if not inline:  # Local: preserve notebook's suppression of structured fields after inline rewriting.
            for call in calls:
                name = call.get("name", "")  # Local: normalize names only, not parameters.
                name = name.strip().lower() if isinstance(name, str) else ""
                content += f'\n<tool_call name="{name}">{call.get("parameters", "")}</tool_call>'
            for result in results:
                content += f'\n<tool_result>{result.get("output", "")}</tool_result>'
        output.append(f"{role.upper()}: {content}")  # Local: exact notebook role prefix, preserving whitespace.
    return output
