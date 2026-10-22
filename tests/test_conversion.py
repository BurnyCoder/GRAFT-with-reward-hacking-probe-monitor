"""Global: preserve the source agent interface without executing any model.

Local: pass converted targets through the actual pinned Harbor parser, including completion.
Sources: https://github.com/harbor-framework/harbor/blob/290ab985edc6915dd60fb1a0d369fedf10d213a1/src/harbor/agents/terminus_2/terminus_2.py#L1482
and https://docs.pytest.org/en/stable/how-to/parametrize.html.
"""
import json  # Local: inspect the exact assistant JSON delivered to SFT.
import runpy  # Local: load only the pinned parser, without an agent or model server.

import pytest  # Local: cover independent command/completion combinations.

from experiments.qwen_probe.data import fix_dataset, upstream_conversion  # Local: exercise production paths.
from experiments.qwen_probe.runtime import ROOT, read_json, write_json  # Local: keep toy artifacts in pytest's workspace.


@pytest.mark.parametrize("names", [("bash_command",), ("mark_task_complete",),
                                  ("bash_command", "mark_task_complete"),
                                  ("mark_task_complete", "bash_command", "bash_command")])
def test_conversion_preserves_commands_and_completion(tmp_path, names):
    """Global: final targets must be usable. Local: parser output must match source semantics."""
    command = {"keystrokes": "printf 'done\\n'\n", "duration": 0.5}  # Local: preserve shell text and timing exactly.
    calls = [{"function_name": name, "arguments": command if name == "bash_command" else {}} for name in names]
    trajectory = {"steps": [{"source": "user", "message": "Solve the task"},
                            {"source": "agent", "message": "Analysis: Ready\nPlan: Finish", "tool_calls": calls}]}
    messages = upstream_conversion(tmp_path).atif_to_messages(trajectory)  # Local: no tokenizer or model is loaded.
    response = json.loads(messages[1]["content"])  # Local: inspect the actual serialized target.
    parser_path = ROOT / "sft/third_party/harbor/src/harbor/agents/terminus_2/terminus_json_plain_parser.py"
    parsed = runpy.run_path(str(parser_path))["TerminusJSONPlainParser"]().parse_response(messages[1]["content"])
    expected = [command for name in names if name == "bash_command"]  # Local: completion never becomes a command.
    assert response["commands"] == expected  # Local: neither drop real commands nor manufacture empty ones.
    assert parsed.error == parsed.warning == ""  # Local: completion can hide malformed commands as warnings.
    assert [(c.keystrokes, c.duration) for c in parsed.commands] == [(c["keystrokes"], c["duration"]) for c in expected]
    assert parsed.is_task_complete is ("mark_task_complete" in names)  # Local: preserve termination independently of commands.


def test_explicit_completion_remains_supported(tmp_path):
    """Global: retain compatibility. Local: an explicit recorded completion needs no inferred reward."""
    trajectory = {"steps": [{"source": "user", "message": "Task"},
                            {"source": "agent", "message": "Done", "tool_calls": [], "task_complete": True}]}
    response = json.loads(upstream_conversion(tmp_path).atif_to_messages(trajectory)[1]["content"])
    assert response["commands"] == [] and response["task_complete"] is True  # Local: preserve the legacy explicit field.


def test_unknown_tools_are_not_silently_dropped(tmp_path):
    """Global: preserve semantics. Local: an unsupported tool requires an explicit future conversion rule."""
    trajectory = {"steps": [{"source": "user", "message": "Task"}, {"source": "agent", "tool_calls": [
        {"function_name": "unknown_tool", "arguments": {"keystrokes": "looks like shell"}}]}]}
    with pytest.raises(ValueError, match="Unsupported ATIF tool"):
        upstream_conversion(tmp_path).atif_to_messages(trajectory)  # Local: do not guess tool intent from argument shape.


def test_completion_cannot_hide_a_dropped_command():
    """Global: parser acceptance alone is insufficient. Local: Harbor downgrades invalid completion commands to warnings."""
    from experiments.qwen_probe.conversion import validate_targets  # Local: test the guard used by future inventories.
    trajectory = {"steps": [{"source": "user"}, {"source": "agent", "tool_calls": [
        {"function_name": "mark_task_complete", "arguments": {}}]}]}
    messages = [{"role": "assistant", "content": json.dumps({"analysis": "", "plan": "", "commands": [{}], "task_complete": True})}]
    with pytest.raises(ValueError, match="semantics"):
        validate_targets(trajectory, messages)  # Local: a true completion flag must not mask malformed training commands.


def test_source_keystroke_warnings_are_preserved(tmp_path):
    """Global: validation must not rewrite source behavior. Local: keep control-key text despite Harbor's newline warning."""
    from experiments.qwen_probe.conversion import validate_targets  # Local: warning handling uses the production guard.
    calls = [{"function_name": "bash_command", "arguments": {"keystrokes": text}} for text in ("C-c", "pwd\n")]
    trajectory = {"steps": [{"source": "user", "message": "Task"}, {"source": "agent", "tool_calls": calls}]}
    messages = upstream_conversion(tmp_path).atif_to_messages(trajectory)  # Local: retain the actual keystroke strings.
    counts = validate_targets(trajectory, messages)  # Local: parser warnings remain visible without invalidating correct conversion.
    assert counts["shell_commands"] == 2 and counts["parser_warning_turns"] == 1  # Local: no command is dropped or altered.


def write_inventory(root):
    """Global: test data identity without tokenization. Local: a complete balanced batch freezes normally."""
    from experiments.qwen_probe.conversion import conversion_identity  # Local: fingerprint the installed converter source.
    rows = [{"id": str(i), "task": "toy", "partition": "train", "label": i % 2, "tokens": 2,
             "content_hash": f"monitor-{i}", "messages": [{"role": "assistant", "content": "original target"}]}
            for i in range(48)]  # Local: satisfy the registered batch multiple, independent of model execution.
    write_json(root / "data/inventory.json", rows)  # Local: exercise the real lock-building code.
    write_json(root / "data/provenance.json", {"sft_conversion": conversion_identity(root)})
    write_json(root / "downloads.json", {"model_files": {"learner": {"tokenizer.json": "toy-tokenizer"}}})
    return rows  # Local: allow tests to mutate targets while retaining identical monitor inputs.


def test_frozen_dataset_detects_target_changes(tmp_path):
    """Global: corrected SFT targets need a new lock. Local: unchanged monitor hashes cannot hide a target edit."""
    rows = write_inventory(tmp_path)  # Local: no real data or model is needed for the cache contract.
    first = fix_dataset(tmp_path, 4096)  # Local: establish the original identity.
    assert first["sft_conversion"]  # Local: conversion identity must be visible in the lock.
    rows[0]["messages"][0]["content"] = "changed target"  # Local: only the training text changes.
    write_json(tmp_path / "data/inventory.json", rows)  # Local: simulate stale cached metadata.
    with pytest.raises(ValueError, match="frozen"):
        fix_dataset(tmp_path, 4096)  # Local: reject instead of relabeling an existing checkpoint's data.
    assert read_json(tmp_path / "data/lock.json") == first  # Local: the old lock remains byte-equivalent JSON.


def test_conversion_version_changes_dataset_identity(tmp_path, monkeypatch):
    """Global: versioned preprocessing cannot reuse a target cache. Local: hold input text fixed and change only conversion identity."""
    from experiments.qwen_probe.conversion import conversion_identity  # Local: exercise production provenance construction.
    from experiments.qwen_probe.data import dataset_fingerprint  # Local: hash all cache inputs without building a model.
    rows = write_inventory(tmp_path)  # Local: create an unfrozen toy inventory for each version.
    first = dataset_fingerprint(tmp_path, rows, {})  # Local: establish identity under the original conversion version.
    monkeypatch.setattr(upstream_conversion(tmp_path), "CONVERSION_VERSION", "test-next-version")
    write_json(tmp_path / "data/provenance.json", {"sft_conversion": conversion_identity(tmp_path)})
    assert dataset_fingerprint(tmp_path, rows, {}) != first  # Local: a semantically versioned conversion is part of cache identity.


def test_changed_targets_cannot_overwrite_pools(tmp_path):
    """Global: frozen input integrity reaches training pools. Local: refuse a target mutation before writing any JSONL."""
    from experiments.qwen_probe.data import write_pools  # Local: exercise the exact boundary used by train and smoke.
    rows = write_inventory(tmp_path)  # Local: no monitor scores are needed; integrity fails first.
    fix_dataset(tmp_path, 4096)  # Local: freeze the original target contents.
    rows[0]["messages"] = []  # Local: keep monitor hash and conversion metadata unchanged.
    write_json(tmp_path / "data/inventory.json", rows)  # Local: emulate stale/corrupt converted text.
    with pytest.raises(ValueError, match="frozen"):
        write_pools(tmp_path)  # Local: content fingerprints must protect the prepared training files too.
    assert not (tmp_path / "data/arms.json").exists()  # Local: no arm metadata or pools are created.


def test_legacy_conversion_fails_before_preparation(tmp_path, monkeypatch):
    """Global: historical data stay historical. Local: reject before downloads, Docker or memory preflight."""
    from experiments.qwen_probe import prepare  # Local: inspect the user-facing phase boundary.
    write_json(tmp_path / "data/inventory.json", [])  # Local: represent the closed study's unversioned inventory.
    write_json(tmp_path / "data/provenance.json", {"preprocessing": "monitor-only"})
    write_json(tmp_path / "data/lock.json", {"fingerprint": "historical"})
    def forbidden(_):
        """Local: any preparation work before the conversion check violates the preservation contract."""
        raise AssertionError("No downloads or models allowed")
    monkeypatch.setattr(prepare, "download", forbidden)  # Local: a regression cannot accidentally do network/model work.
    with pytest.raises(ValueError, match="separate.*GRAFT_ARTIFACTS"):
        prepare.prepare(tmp_path)  # Local: require an explicitly separate experiment artifact root.
