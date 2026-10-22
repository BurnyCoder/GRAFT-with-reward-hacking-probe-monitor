"""Global: documented phases work independently. Local: exercise pool construction before a selected arm."""
import json  # Local: inspect the trainer's actual JSONL interchange.
import pytest  # Local: the same phase contract applies to both real monitors.

from experiments.qwen_probe import training
from experiments.qwen_probe.conversion import conversion_identity  # Local: fixtures must satisfy the current target contract.
from experiments.qwen_probe.data import fix_dataset  # Local: exercise an actual frozen dataset before routing pools.
from experiments.qwen_probe.runtime import write_json


def test_completed_smoke_does_not_block_resume_on_pending_study_audits(tmp_path, monkeypatch):
    """Global: keep study batches blinded. Local: a certified smoke is not rerun when the controller resumes."""
    from experiments.qwen_probe.__main__ import smoke
    from experiments.qwen_probe import audit
    write_json(tmp_path / "smoke-complete.json", {"completed_at": 1, "audited_trials": 1})
    def forbid_new_packets(_):
        """Local: resuming the study must not request labels from only its first completed treatment."""
        raise AssertionError("Completed smoke must not inspect pending study audits")
    monkeypatch.setattr(audit, "make_packets", forbid_new_packets)
    assert smoke(tmp_path) is True


@pytest.mark.parametrize("selected_arm", ["probe_graft", "qwen_judge_graft"])
def test_selected_training_phase_creates_its_pools(tmp_path, monkeypatch, selected_arm):
    """Global: train --arm must not depend on smoke. Local: a frozen toy score produces a forget record."""
    identities = [f"{index:02}" for index in range(48)]  # Local: honor the balanced batch-multiple contract.
    write_json(tmp_path / "data/inventory.json", [{"id": identity, "label": index % 2, "tokens": 2,
               "task": "toy", "partition": "train", "content_hash": identity,
               "classification": "hack" if index % 2 else "baseline",
               "messages": [{"role": "assistant", "content": "example"}]} for index, identity in enumerate(identities)])
    write_json(tmp_path / "data/provenance.json", {"sft_conversion": conversion_identity(tmp_path)})
    write_json(tmp_path / "downloads.json", {"model_files": {"learner": {"tokenizer.json": "toy"}}})
    fix_dataset(tmp_path, 4096)  # Local: no tokenization or model execution; only freeze the toy metadata.
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
