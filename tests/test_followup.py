"""Global: additional user-authorized work cannot rewrite the closed pilot.

Local: toy files exercise immutable inputs, fixed budgets and exactly the missing conditions.
Source: https://docs.pytest.org/en/stable/how-to/monkeypatch.html.
"""
import pytest  # Local: errors must stop before an attempted model execution.

from experiments.qwen_probe.runtime import file_hash, read_json, write_json


def closed_pilot(folder):
    """Global: no real model in tests. Local: construct the metadata required by the follow-up."""
    from experiments.qwen_probe.followup import SOURCE_FILES
    for relative in SOURCE_FILES:  # Local: every original evidence file participates in the snapshot.
        write_json(folder / relative, {})
    write_json(folder / "budget.json", {"start": 1, "deadline": 2})
    write_json(folder / "evaluation/tasks.json", {"tasks": [str(i) for i in range(8)]})
    write_json(folder / "audit/unblinded.json", [])
    write_json(folder / "audit/frozen_judgments.json", {"sha256": file_hash(folder / "audit/judgments.json")})
    (folder / "models/learner").mkdir(parents=True)
    (folder / "models/learner/model.safetensors").write_bytes(b"toy model, never loaded")
    for arm in ("probe_graft", "no_intervention"):
        write_json(folder / "checkpoints" / arm / "progress.json", {"complete": True, "steps": 20})
        write_json(folder / "checkpoints" / arm / "configuration.json", {"data_fingerprint": "fixed"})
        (folder / "checkpoints" / arm / "adapter_state_dict.pt").write_bytes(b"toy adapter")
    return folder


def test_prepare_keeps_original_budget_and_rejects_changed_inputs(tmp_path):
    """Global: the original deadline stays closed. Local: resumption validates source hashes."""
    from experiments.qwen_probe.followup import prepare_followup
    pilot = closed_pilot(tmp_path / "pilot")
    original = file_hash(pilot / "budget.json")
    followup = prepare_followup(pilot)
    assert file_hash(pilot / "budget.json") == original
    assert not (followup / "budget.json").exists()  # Local: preparation itself does not start model work.
    assert (followup / "checkpoints").resolve() == pilot / "checkpoints"
    assert prepare_followup(pilot) == followup
    write_json(pilot / "data/lock.json", {"changed": True})
    with pytest.raises(ValueError, match="changed"):
        prepare_followup(pilot)


@pytest.mark.parametrize("state", ["active", "already_attempted", "unfrozen"])
def test_prepare_rejects_active_or_previously_attempted_sources(tmp_path, state):
    """Global: no duplicate attempts or premature reuse. Local: fail before creating evaluation state."""
    from experiments.qwen_probe.followup import prepare_followup
    pilot = closed_pilot(tmp_path / "pilot")
    if state == "active":
        write_json(pilot / "evaluation/index.json", {"some-job": {"status": "running"}})
    elif state == "already_attempted":
        write_json(pilot / "evaluation/index.json", {"probe_graft-both-plain": {"status": "incomplete"}})
    else:
        write_json(pilot / "audit/judgments.json", {"unfrozen change": True})
    with pytest.raises(ValueError):
        prepare_followup(pilot)


def test_followup_evaluates_only_missing_conditions_and_preserves_its_deadline(tmp_path, monkeypatch):
    """Global: stopping is independent of outcomes. Local: reuse the evaluator with a separate clock."""
    from experiments.qwen_probe import followup
    pilot = closed_pilot(tmp_path / "pilot")
    calls = []
    monkeypatch.setattr(followup, "verify_published", lambda root: {"commit": "pushed"})
    monkeypatch.setattr(followup, "evaluate", lambda root, **kwargs: calls.append(kwargs))
    root = followup.evaluate_followup(pilot)
    budget = read_json(root / "budget.json")
    assert budget["deadline"] - budget["start"] == 3 * 3600
    assert sum(len(call["conditions"]) * 8 for call in calls) == 40
    assert calls == [{"modes": [(arm, mode)], "conditions": list(conditions)}
                     for arm, mode, conditions in followup.PLAN]
    followup.evaluate_followup(pilot)  # Local: the underlying evaluator skips indexed jobs on resume.
    assert read_json(root / "budget.json") == budget
    assert read_json(root / "controller.json")["status"] == "closed"


def test_evaluator_accepts_plain_only_without_running_elicited(tmp_path, monkeypatch):
    """Global: the eight original elicited attempts must not repeat. Local: inspect actual launcher calls."""
    from contextlib import contextmanager
    from experiments.qwen_probe import evaluation
    write_json(tmp_path / "evaluation/tasks.json", {"tasks": ["a"], "smoke_task": "unused"})
    write_json(tmp_path / "checkpoints/probe_graft/progress.json", {"complete": True})
    monkeypatch.setattr(evaluation, "ensure_task_images", lambda root: None)
    monkeypatch.setattr(evaluation, "merge_model", lambda *args: tmp_path)
    @contextmanager
    def server(*args):
        """Local: replace GPU startup only; keep the evaluator's real skip and index logic."""
        yield "http://local/v1"
    monkeypatch.setattr(evaluation, "model_server", server)
    prompts = []
    def dataset(root, tasks, elicited):
        """Local: record which fixed prompt condition would reach Harbor."""
        prompts.append(elicited)
        return tmp_path
    monkeypatch.setattr(evaluation, "dataset", dataset)
    monkeypatch.setattr(evaluation, "execute", lambda *args, **kwargs: 0)
    for _ in range(2):  # Local: a completed follow-up condition is also never repeated.
        evaluation.evaluate(tmp_path, modes=[("probe_graft", "both")], conditions=[False])
    assert prompts == [False]
    assert list(read_json(tmp_path / "evaluation/index.json")) == ["probe_graft-both-plain"]
