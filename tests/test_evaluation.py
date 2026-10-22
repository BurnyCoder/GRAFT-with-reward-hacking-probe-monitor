"""Global: preserve one model attempt. Local: catch image deletion and unsafe setup recovery."""
import pytest  # Local: compare eligible setup failures with actual model attempts.

from experiments.qwen_probe.runtime import read_json, write_json


def test_harbor_preserves_shared_prebuilt_images():
    """Global: every condition uses identical images. Local: Harbor must use compose down without --rmi."""
    from experiments.qwen_probe.evaluation import harbor_command
    command = harbor_command("tasks", "http://localhost/v1", "job", "/tmp/results")
    assert "--no-delete" in command  # Local: pinned Harbor's false branch removes containers but preserves images.
    assert command[command.index("-n") + 1] == "1"
    assert command[command.index("-k") + 1] == "1"


@pytest.mark.parametrize("agent_started", [False, True])
def test_recovery_archives_only_wholly_unstarted_jobs(tmp_path, agent_started):
    """Global: no outcome-based retries. Local: one agent start makes the entire job ineligible."""
    from experiments.qwen_probe.recovery import recover_setup_failures
    metadata = {"tasks": ["a", "b"], "status": "finished", "smoke": False}
    write_json(tmp_path / "evaluation/index.json", {"job": metadata})
    for task in metadata["tasks"]:
        result = {"task_name": task, "exception_info": {"exception_type": "RuntimeError",
                  "exception_message": "Docker compose command failed: pull access denied for graft-probe-task-" + task}}
        if agent_started and task == "b":
            result["agent_execution"] = {"started_at": "2026-10-22T00:00:00"}
        write_json(tmp_path / "evaluation/jobs/job" / task / "result.json", result)
    recovered = recover_setup_failures(tmp_path)
    assert len(recovered) == (0 if agent_started else 1)
    assert ("job" in read_json(tmp_path / "evaluation/index.json")) == agent_started
    if not agent_started:
        assert (tmp_path / "evaluation/setup_failures/job/a/result.json").exists()
        assert recovered[0]["trial_count"] == 2
        assert len(recovered[0]["result_sha256"]) == 2
        assert recover_setup_failures(tmp_path) == []  # Local: repeated phase calls never add another retry.


def test_restored_images_must_match_frozen_identity(tmp_path, monkeypatch):
    """Global: repairing infrastructure cannot change the task. Local: fail on a rebuilt image mismatch."""
    from experiments.qwen_probe import tasks
    write_json(tmp_path / "evaluation/tasks.json", {"images": {"a": "sha256:original"}})
    inspected = iter([None, "sha256:different"])
    monkeypatch.setattr(tasks, "image_identity", lambda *args, **kwargs: next(inspected))
    monkeypatch.setattr(tasks, "execute", lambda *args, **kwargs: 0)
    monkeypatch.setattr(tasks, "original_task", lambda _: tmp_path)
    with pytest.raises(RuntimeError, match="identity"):
        tasks.ensure_task_images(tmp_path)


@pytest.mark.parametrize("blocker", [None, "agent", "verifier", "trajectory", "other_error"])
def test_shim_upgrade_recovery_requires_explicit_unstarted_evidence(tmp_path, blocker):
    """Global: recover infrastructure only. Local: the default policy and any model evidence block a retry."""
    from experiments.qwen_probe.recovery import recover_setup_failures
    from experiments.qwen_probe.runtime import file_hash
    metadata = {"tasks": ["a"], "status": "finished", "smoke": False}
    index = tmp_path / "evaluation/index.json"
    trial = tmp_path / "evaluation/jobs/job/a/result.json"
    result = {"task_name": "a", "exception_info": {"exception_type": "RuntimeError", "exception_message":
              "Docker compose command failed: failed to start shim: shim was started through the deprecated API but was built against the new API"}}
    if blocker == "agent":
        result["agent_execution"] = {"started_at": "2026-10-22T00:00:00"}
    elif blocker == "verifier":
        result["verifier_result"] = {"rewards": {"reward": 0}}
    elif blocker == "trajectory":
        write_json(trial.parent / "agent/trajectory.json", {})
    elif blocker == "other_error":
        result["exception_info"]["exception_message"] = "Docker compose failed for an unrelated reason"
    write_json(index, {"job": metadata})
    write_json(trial, result)
    digest = file_hash(trial)
    assert recover_setup_failures(tmp_path) == []  # Local: the original image-only policy stays unchanged.
    recovered = recover_setup_failures(tmp_path, failure="shim-upgrade")
    assert len(recovered) == (1 if blocker is None else 0)
    if blocker is None:
        archived = tmp_path / "evaluation/setup_failures/job/a/result.json"
        assert file_hash(archived) == digest and recovered[0]["failure"] == "shim-upgrade"
        write_json(index, {"job": metadata})  # Local: even an identical subsequent setup failure cannot qualify twice.
        write_json(trial, result)
        assert recover_setup_failures(tmp_path, failure="shim-upgrade") == []
