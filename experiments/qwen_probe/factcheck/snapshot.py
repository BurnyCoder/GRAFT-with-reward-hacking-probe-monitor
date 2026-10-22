"""Global: preserve the pre-review record and expose reproducibility gaps.

Local: hash evidence and archive existing tracked presentation files without modifying sources.
Sources: https://docs.python.org/3/library/hashlib.html; https://git-scm.com/docs/git-ls-files.
"""
import os  # Local: walk without following links into duplicate/shared study inputs.
import shutil  # Local: byte-preserving archival copies.
import subprocess  # Local: record the actual checkout and pinned submodule differences.
import time  # Local: distinguish the new review from historical execution.

from ..runtime import ROOT, file_hash, read_json, write_json  # Local: one hashing/JSON convention.


def snapshot(root, destination):
    """Global: a review cannot rewrite its starting evidence. Local: create the snapshot exactly once."""
    target = destination / "snapshot.json"
    if target.exists():  # Local: resume using the original snapshot rather than resetting the baseline.
        return read_json(target)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    inventory, archive = {}, {}  # Local: inventory includes code/data; archive stores presentation bytes.
    for name in filter(None, tracked):
        path = ROOT / name
        if not path.is_file():  # Local: submodule directories are identified separately below.
            continue
        inventory[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        if path.suffix in (".md", ".pdf", ".png", ".svg"):
            copy = destination / "archive" / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)  # Local: original relative paths are retained beneath archive/.
            archive[name] = file_hash(copy)
    directories = [root / name for name in ("data", "checkpoints", "audit", "evaluation", "logs", "monitor", "judge", "models", "sources")]
    directories += [root / "ablation-followup" / name for name in ("audit", "evaluation", "logs")]
    paths = set(root.glob("*.json")) | set((root / "ablation-followup").glob("*.json"))
    for directory in directories:
        for folder, _, files in os.walk(directory, followlinks=False):
            paths.update((directory.__class__(folder) / name) for name in files)
    evidence = {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(paths) if path.is_file()}
    submodules = {}  # Local: a pinned commit alone does not describe a modified runtime checkout.
    for name in ("harbor", "terminal-wrench"):
        path = ROOT / "sft/third_party" / name
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=path, text=True)
        submodules[name] = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip(),
                           "diff": diff, "status": subprocess.check_output(["git", "status", "--short"], cwd=path, text=True)}
    result = {"created_at": time.time(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "tracked_files": inventory, "archived_presentations": archive, "evidence": evidence, "submodules": submodules}
    write_json(target, result)
    return result


def verify_preservation(destination):
    """Global: detect changes instead of trusting a preservation claim. Local: compare exact file hashes."""
    baseline = read_json(destination / "snapshot.json")
    changed = [name for name, digest in baseline["evidence"].items() if not (ROOT / name).is_file() or file_hash(ROOT / name) != digest]
    changed += ["archive/" + name for name, digest in baseline["archived_presentations"].items()
                if file_hash(destination / "archive" / name) != digest]
    if changed:
        raise ValueError(f"Historical evidence changed: {changed}")
    return {"evidence_files": len(baseline["evidence"]), "archive_files": len(baseline["archived_presentations"]), "changed": []}
