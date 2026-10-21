"""Global: pin external inputs before the clock starts. Local: resumable, verified downloads.

Source: https://huggingface.co/docs/huggingface_hub/guides/download
"""
import urllib.request  # Local: public Drive download; global: no authentication copied into logs.
from pathlib import Path  # Local: packaged manifest location.

from .runtime import event, execute, file_hash, read_json, write_json

MANIFEST = read_json(Path(__file__).with_name("artifacts.json"))  # Local: version-controlled source pins.


def download(root):
    """Global: reproducible preparation. Local: download weights, record hashes, initialize data."""
    from huggingface_hub import snapshot_download  # Local: pinned snapshots with resumable caching.
    for item in MANIFEST["files"]:  # Local: verify all public detector bytes before loading them.
        path = root / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or file_hash(path) != item["sha256"]:
            event("download", url=item["url"], destination=path)
            temporary = path.with_suffix(path.suffix + ".part")  # Local: don't accept partial artifacts.
            urllib.request.urlretrieve(item["url"], temporary)
            if file_hash(temporary) != item["sha256"]:  # Local: HTML errors or changed weights fail closed.
                raise ValueError(f"Checksum mismatch: {item['path']}")
            temporary.replace(path)
    downloaded = {}  # Local: exact file identities for both models and tokenizers.
    for role in ("learner", "encoder"):
        model = MANIFEST[role]  # Local: use commit SHA, never mutable main.
        folder = root / "models" / role
        snapshot_download(model["id"], revision=model["revision"], local_dir=folder,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md", "LICENSE*"])
        downloaded[role] = {p.name: file_hash(p) for p in sorted(folder.iterdir()) if p.is_file()}
    execute(["git", "submodule", "update", "--init", "--recursive"])  # Local: pinned Git links supply data/Harbor.
    write_json(root / "downloads.json", {"manifest": MANIFEST, "model_files": downloaded})
    event("downloads_verified", models=downloaded)  # Local: hashes are provenance, not credentials.
