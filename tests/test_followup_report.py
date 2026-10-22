"""Global: exploratory completion cannot replace or double-count original outcomes.

Local: toy records check cohort labels, ambiguity and the six declared paired comparisons.
"""
import copy  # Local: verify source judgments remain untouched by combined reporting.

import pytest  # Local: duplicated episodes must cause an error, not a preferred-result selection.


def outcome(arm, mode, elicited, *, hacking=False, legitimate=False):
    """Global: synthetic audit evidence only. Local: one shared task supports exact paired expectations."""
    return {"task": "a", "arm": arm, "mode": mode, "elicited": elicited, "smoke": False,
            "hacking": hacking, "legitimate": legitimate, "verifier_success": legitimate}


def test_cohorts_and_ambiguity_survive_combination_without_mutation():
    """Global: old labels are immutable. Local: unknown new intent remains bounded rather than benign."""
    from experiments.qwen_probe.followup_report import combined_records, summarize
    original = [outcome("probe_graft", "retain", False, hacking=True)]
    new = [outcome("probe_graft", "both", False, hacking=None)]
    before = copy.deepcopy((original, new))
    rows = combined_records(original, new)
    assert [row["cohort"] for row in rows] == ["closed_pilot", "followup"]
    assert (original, new) == before
    results = summarize(rows)
    assert results["coverage"]["closed_pilot_audited"] == 1
    assert results["coverage"]["followup_audited"] == 1
    assert results["coverage"]["combined_unique"] == 2
    contrast = results["comparisons"]["probe_retain_minus_both/plain"]["hacking"]
    assert contrast["paired_tasks"] == 1 and contrast["known_pairs"] == 0
    assert contrast["difference"] is None and contrast["ambiguity_bounds"] == [0.0, 1.0]
    assert len(results["comparisons"]) == 6


def test_combining_rejects_repeated_episodes_even_when_outcomes_disagree():
    """Global: never replace a failed original attempt. Local: identity excludes the outcome labels."""
    from experiments.qwen_probe.followup_report import combined_records
    row = outcome("probe_graft", "both", False)
    with pytest.raises(ValueError, match="Duplicate"):
        combined_records([row], [row | {"legitimate": True}])


def test_followup_cannot_add_a_new_primary_attempt():
    """Global: only originally unrun conditions are eligible. Local: reject a primary deployment row."""
    from experiments.qwen_probe.followup_report import combined_records
    with pytest.raises(ValueError, match="scope"):
        combined_records([], [outcome("qwen_judge_graft", "retain", True)])


def test_supplement_render_preserves_closed_pilot_artifacts(tmp_path, monkeypatch):
    """Global: a new PDF cannot rewrite the original paper or primary results. Local: render toy audit data end-to-end."""
    from experiments.qwen_probe import followup_report
    from experiments.qwen_probe.runtime import read_json, write_json
    pilot, root = tmp_path / "pilot", tmp_path / "pilot/ablation-followup"
    evidence = {"packet_id": "opaque", "agent_error": None, "evidence": "toy", "rationale": "synthetic check",
                "auditor": "test", "judged_at": 1, "packet_fingerprint": "toy", "trial": str(root / "trial")}
    original = [outcome("probe_graft", "retain", False) | evidence]
    new = [outcome("probe_graft", "both", False) | evidence | {"packet_id": "new"}]
    write_json(pilot / "audit/unblinded.json", original)
    for path in ("manifest.json", "budget.json", "publication.json", "controller.json", "evaluation/index.json", "audit/judgments.json"):
        write_json(root / path, {})
    write_json(root / "controller.json", {"status": "closed"})
    write_json(root / "evaluation/tasks.json", {"tasks": ["a"]})
    write_json(root / "audit/frozen_judgments.json", {"sha256": "toy-frozen"})
    write_json(root / "audit/session.json", {"original_pilot_outcomes_known": True})
    write_json(root / "evaluation/setup_recovery.json", [{"trial_count": 8, "failure": "shim-upgrade"}])
    write_json(root / "infrastructure.json", {"pre_upgrade_versions": "not captured"})
    write_json(root / "audit/packets/new.json", {"messages": [], "raw_model_calls": [], "missing_trajectory": False})
    write_json(root / "trial/result.json", {})
    original_report = tmp_path / "reports/qwen_probe"
    write_json(original_report / "results.json", {"primary": "unchanged"})
    (original_report / "paper.pdf").write_bytes(b"original paper stays intact")
    monkeypatch.setattr(followup_report, "ROOT", tmp_path)
    monkeypatch.setattr(followup_report, "prepare_followup", lambda path: root)
    monkeypatch.setattr(followup_report, "make_packets", lambda path: [])
    monkeypatch.setattr(followup_report, "finalize", lambda path: new)
    result = followup_report.report_followup(pilot)
    assert result["coverage"]["combined_unique"] == 2
    assert read_json(original_report / "results.json") == {"primary": "unchanged"}
    assert (original_report / "paper.pdf").read_bytes() == b"original paper stays intact"
    assert (original_report / "followup/paper.pdf").read_bytes().startswith(b"%PDF")
    assert read_json(original_report / "followup/audit_session.json")["original_pilot_outcomes_known"] is True
    assert result["archived_setup_failures"] == 8
    assert read_json(original_report / "followup/infrastructure.json")["pre_upgrade_versions"] == "not captured"
    assert "8 setup failures" in (original_report / "followup/paper.md").read_text()
    assert "Exploratory" in (original_report / "followup/summary.md").read_text()
