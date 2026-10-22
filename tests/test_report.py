"""Global: report actual progress without claiming checkpoint completion. Local: use saved toy logs."""
from experiments.qwen_probe.runtime import write_json  # Local: same manifest format as the trainer.


def test_logged_updates_do_not_imply_a_completed_checkpoint(tmp_path):
    """Global: interrupted/live work remains visible. Local: distinguish metrics from a final manifest."""
    from experiments.qwen_probe.report import training_status
    assert training_status(tmp_path, "arm")["steps"] == 0  # Local: no files means no observed training.
    folder = tmp_path / "checkpoints/arm"
    folder.mkdir(parents=True)  # Local: simulate training before final checkpoint publication.
    (folder / "metrics.jsonl").write_text('{"step": 1, "loss": 2.0}\n{"step": 2, "loss": 1.0}\n')
    running = training_status(tmp_path, "arm")
    assert running["steps"] == 2 and not running["complete"]  # Local: show logged updates without inventing completion.
    assert not running["checkpoint_saved"]
    write_json(folder / "progress.json", {"steps": 2, "planned_steps": 20, "complete": False})
    (folder / "adapter_state_dict.pt").touch()  # Local: content validation belongs to merge tests, not this status test.
    interrupted = training_status(tmp_path, "arm")
    assert interrupted["checkpoint_saved"] and not interrupted["complete"]
    assert interrupted["steps"] == 2 and interrupted["planned_steps"] == 20
