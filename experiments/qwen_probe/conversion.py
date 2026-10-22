"""Global: version and validate training targets independently of monitor preprocessing.

Local: reuse the released converter and the pinned parser; never load a model or execute a command.
Source: https://github.com/harbor-framework/harbor/blob/290ab985edc6915dd60fb1a0d369fedf10d213a1/src/harbor/agents/terminus_2/terminus_json_plain_parser.py
"""
import json  # Local: inspect exact serialized SFT targets.
import os  # Local: configure legacy paths inside the current workspace.
import runpy  # Local: load only the pinned parser source, without importing an agent runtime.
import sys  # Local: reuse the released scripts package.
from functools import cache  # Local: the stateless parser can be reused across trajectories.

from .runtime import ROOT, file_hash, read_json  # Local: shared source identity and artifact readers.

PARSER_PATH = ROOT / "sft/third_party/harbor/src/harbor/agents/terminus_2/terminus_json_plain_parser.py"


def upstream_conversion(root):
    """Global: avoid a second converter. Local: set workspace paths before importing the released module."""
    os.environ["GRAFT_TW_ROOT"] = str(ROOT / "sft/third_party/terminal-wrench")  # Local: pinned source data.
    os.environ["GRAFT_SFT_DATA_DIR"] = str(root / "data")  # Local: prevent legacy outside-workspace defaults.
    if str(ROOT / "sft") not in sys.path:  # Local: repeated checks need only one import path entry.
        sys.path.insert(0, str(ROOT / "sft"))  # Local: the released package is named scripts.
    from scripts import data_prep  # Local: importing now has no output-directory side effect.
    return data_prep  # Local: callers share exactly one conversion implementation.


def conversion_identity(root):
    """Global: invalidate caches after conversion changes. Local: bind semantic version and actual source bytes."""
    converter = upstream_conversion(root)  # Local: resolve the implementation used for SFT.
    return {"version": converter.CONVERSION_VERSION, "source_sha256": file_hash(converter.__file__)}


def require_current_conversion(root):
    """Global: preserve historical experiments. Local: reject legacy/stale metadata before new preparation or training."""
    current = conversion_identity(root)  # Local: compare actual implementation identity, not monitor text hashes.
    for name in ("provenance.json", "lock.json"):  # Local: both inventory creation and freezing must agree.
        path = root / "data" / name  # Local: inspect metadata only; never rewrite a historical file.
        if path.exists() and read_json(path).get("sft_conversion") != current:
            raise ValueError("Stale or legacy SFT conversion; preserve this run and use a separate GRAFT_ARTIFACTS directory")
    if (root / "data/inventory.json").exists() and not (root / "data/provenance.json").exists():
        raise ValueError("Unversioned inventory; preserve this run and use a separate GRAFT_ARTIFACTS directory")
    return current  # Local: use this same identity in the new dataset fingerprint.


@cache
def target_parser():
    """Global: check against the actual evaluation interface. Local: the pinned parser needs only Python's standard library."""
    return runpy.run_path(str(PARSER_PATH))["TerminusJSONPlainParser"]()  # Local: no Harbor agent/model initialization.


def validate_targets(trajectory, messages):
    """Global: no invalid training targets. Local: require lossless commands/termination and disclose nonfatal source warnings."""
    steps = [s for s in trajectory.get("steps", [])[1:] if s.get("source") == "agent"]  # Local: one target per agent step.
    targets = [m["content"] for m in messages if m["role"] == "assistant"]  # Local: observations are not commands.
    commands, completions, warnings = 0, 0, 0  # Local: warnings describe source input; they do not authorize editing it.
    for step, text in zip(steps, targets, strict=True):  # Local: dropping or adding an assistant turn is an error.
        calls = step.get("tool_calls", [])  # Local: the source tool names determine command/completion roles.
        expected = [c.get("arguments", {}) for c in calls if c.get("function_name") == "bash_command"]
        complete = step.get("task_complete", False)  # Local: preserve supported explicit legacy fields too.
        if isinstance(complete, str):  # Local: match Harbor's documented boolean-string interpretation.
            complete = complete.lower() in ("true", "1", "yes")  # Local: do not treat the string "false" as true.
        complete = any(c.get("function_name") == "mark_task_complete" for c in calls) or complete
        parsed = target_parser().parse_response(text)  # Local: parse text only, never execute shell actions.
        if parsed.error or len(parsed.commands) != len(expected):  # Local: completion must not hide discarded malformed commands.
            raise ValueError(f"Invalid converted target: {parsed.error or parsed.warning}")
        if json.loads(text)["commands"] != expected or parsed.is_task_complete != complete:
            raise ValueError("Converted target changed source command/completion semantics")
        if [(c.keystrokes, c.duration) for c in parsed.commands] != [(c["keystrokes"], c.get("duration", 1.0)) for c in expected]:
            raise ValueError("Parser changed source shell commands")  # Local: catch silently discarded or defaulted commands.
        commands += len(expected)  # Local: count source commands, including mixed completion turns.
        completions += bool(complete)  # Local: count completion turns rather than inferred task success.
        warnings += bool(parsed.warning)  # Local: preserve newline warnings without modifying source keystrokes.
    return {"assistant_turns": len(targets), "shell_commands": commands, "completion_turns": completions,
            "parser_warning_turns": warnings}  # Local: parsing success does not imply the source command solves a task.
