"""Global: reuse the user's CLI login without exposing it. Local: isolate auth path overrides."""
from experiments.qwen_probe.publish import hub_environment
from experiments.qwen_probe.runtime import ROOT


def test_hub_auth_uses_default_location_without_changing_parent(monkeypatch):
    """Global: local model caches remain local. Local: only the upload child drops cache overrides."""
    for key in ("HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME"):
        monkeypatch.setenv(key, "workspace-cache")
    environment = hub_environment()
    assert not {"HF_HOME", "XDG_CACHE_HOME"} & environment.keys()
    assert environment["HF_HUB_CACHE"] == str(ROOT / ".cache/huggingface/hub")
    assert environment["HF_XET_CACHE"] == str(ROOT / ".cache/huggingface/xet")
    import os  # Local: verify the parent process is unaffected by child configuration.
    assert os.environ["HF_HOME"] == "workspace-cache"


def test_collection_rename_reuses_recorded_identity(tmp_path, monkeypatch):
    """Global: both monitor headlines must not duplicate published artifacts. Local: reuse the saved slug."""
    from experiments.qwen_probe import publish  # Local: patch only the publication subprocess, never network state.
    from experiments.qwen_probe.runtime import read_json, write_json
    old = "Researcher/old-title-123"  # Local: collection identity belongs to the authenticated namespace.
    new = "Researcher/llm-judge-and-probe-123"  # Local: a changed title may return a changed human-readable slug.
    manifest = {"collection": "https://huggingface.co/collections/" + old, "models": [{"repo": "Researcher/model"}]}
    write_json(tmp_path / "hub/publication.json", manifest)
    calls = []  # Local: the assertion detects an accidental create request before any real publication.

    def cli(command, **kwargs):
        """Local: return a realistic collection update response without authenticating or uploading."""
        calls.append(command)
        return '{"slug": "Researcher/llm-judge-and-probe-123"}'

    monkeypatch.setattr(publish.subprocess, "check_output", cli)
    assert publish.ensure_collection("hf", "Researcher", {}, tmp_path) == new
    assert len(calls) == 1 and calls[0][:4] == ["hf", "collections", "update", old]
    assert "LLM-as-a-Judge and Probe Monitors" in calls[0][calls[0].index("--title") + 1]
    saved = read_json(tmp_path / "hub/publication.json")
    assert saved["collection"].endswith(new) and saved["models"] == manifest["models"]


def test_staged_report_links_resolve_to_published_methodology(tmp_path):
    """Global: downloaded artifacts must retain usable sources. Local: Hub reports sit one directory shallower."""
    from experiments.qwen_probe.publish import stage_reports
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()  # Local: use a small fixture instead of uploading during tests.
    (source / "summary.md").write_text('[method](../../docs/protocol.md) [evidence](results.json)')
    (source / "results.json").write_text('{"complete": false}')
    stage_reports(source, stage)
    assert '/blob/main/docs/protocol.md)' in (stage / "summary.md").read_text()
    assert '[evidence](results.json)' in (stage / "summary.md").read_text()
    assert (stage / "results.json").read_bytes() == (source / "results.json").read_bytes()


def test_nested_followup_links_keep_correct_methodology_destination(tmp_path):
    """Global: later supplements remain reproducible on the Hub. Local: nested reports need recursive link repair."""
    from experiments.qwen_probe.publish import stage_reports
    source, stage = tmp_path / "source", tmp_path / "stage"
    (source / "followup").mkdir(parents=True)
    (source / "followup/summary.md").write_text('[plan](../../../docs/followup-protocol.md) [pilot](../summary.md)')
    stage_reports(source, stage)
    text = (stage / "followup/summary.md").read_text()
    assert '/blob/main/docs/followup-protocol.md)' in text
    assert '[pilot](../summary.md)' in text
