"""Global: documented phases work independently. Local: exercise pool construction before a selected arm."""
import json  # Local: inspect the trainer's actual JSONL interchange.
import pytest  # Local: the same phase contract applies to both real monitors.

from experiments.qwen_probe import training
from experiments.qwen_probe.runtime import write_json


@pytest.mark.parametrize("selected_arm", ["probe_graft", "qwen_judge_graft"])
def test_selected_training_phase_creates_its_pools(tmp_path, monkeypatch, selected_arm):
    """Global: train --arm must not depend on smoke. Local: a frozen toy score produces a forget record."""
    identities = [f"{index:02}" for index in range(48)]  # Local: honor the balanced batch-multiple contract.
    write_json(tmp_path / "data/lock.json", {"train": identities})
    write_json(tmp_path / "data/inventory.json", [{"id": identity, "label": index % 2, "tokens": 2,
               "classification": "hack" if index % 2 else "baseline",
               "messages": [{"role": "assistant", "content": "example"}]} for index, identity in enumerate(identities)])
    write_json(tmp_path / "monitor/scores.json", {identity: {"score": 0.9} for identity in identities})
    write_json(tmp_path / "monitor/threshold.json", {"threshold": 0.5})
    write_json(tmp_path / "judge/scores.json", {identity: {"score": 0.7} for identity in identities})
    write_json(tmp_path / "judge/threshold.json", {"threshold": 0.6})
    completed = []
    def observe_training(root, arm):
        """Local: replace GPU work only; require the real prepared file at the phase boundary."""
        records = [json.loads(line) for line in (root / "data" / arm / "gr_train_forget.jsonl").read_text().splitlines()]
        assert [record["record_id"] for record in records] == identities
        completed.append(arm)
        return True
    monkeypatch.setattr(training, "train_arm", observe_training)
    training.train(tmp_path, arm=selected_arm)
    assert completed == [selected_arm]
