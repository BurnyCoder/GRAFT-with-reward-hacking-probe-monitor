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
