"""Global: publish reproducible pilot adapters, not unsupported safety claims.

Local: use the authenticated hf CLI; never read/print token files or upload .env/log directories.
Source: https://huggingface.co/docs/huggingface_hub/guides/cli
"""
import json  # Local: structured Hub output.
import os  # Local: use existing CLI authentication with separate model caches.
import shutil  # Local: stage an explicit file allowlist.
import subprocess  # Local: resolve authenticated username and collection slug.

from .data import ARMS
from .runtime import ROOT, event, execute, read_json, write_json

REPOSITORY = "https://github.com/BurnyCoder/adapting-gradient-routing-adapter-finetuning-with-reward-hacking-probe-and-LLM-as-a-judge-monitors"  # Local: canonical public implementation and methodology.


def stage_reports(source, destination):
    """Global: published evidence keeps usable source links. Local: adapt only GitHub-relative documentation links.

    Source: https://huggingface.co/docs/hub/model-cards — Hub cards and reports use Markdown links.
    """
    shutil.copytree(source, destination, dirs_exist_ok=True)  # Local: numerical evidence and figures are copied byte-for-byte.
    for path in destination.glob("*.md"):  # Local: Hub's report/ directory is shallower than reports/qwen_probe/ in Git.
        path.write_text(path.read_text().replace("](../../docs/", f"]({REPOSITORY}/blob/main/docs/"))  # Local: leave within-report evidence links intact.


def hub_environment():
    """Global: use confirmed CLI authentication. Local: default token location, workspace upload caches.

    Source: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
    HF_HOME falls back to XDG_CACHE_HOME, so both overrides must be removed for the existing login.
    """
    environment = os.environ.copy()  # Local: leave model/scoring processes' environment untouched.
    for key in ("HF_HOME", "XDG_CACHE_HOME"):
        environment.pop(key, None)  # Local: the CLI resolves its own login; no token file is read here.
    for key, folder in (("HF_HUB_CACHE", "hub"), ("HF_XET_CACHE", "xet"), ("HF_ASSETS_CACHE", "assets")):
        environment[key] = str(ROOT / ".cache/huggingface" / folder)  # Local: upload-related caches remain in the checkout.
    environment["HF_HUB_DISABLE_UPDATE_CHECK"] = "1"  # Local: retain the locked CLI without external update-cache writes.
    return environment


def ensure_collection(cli, namespace, env, root):
    """Global: consistent monitor framing without duplicate collections. Local: persist the Hub-returned identity.

    Sources: https://huggingface.co/docs/huggingface_hub/guides/collections and `hf collections update --help`.
    """
    path = root / "hub/publication.json"  # Local: the saved URL identifies the existing collection across title changes.
    previous = read_json(path) if path.exists() else {}  # Local: retain already published model metadata on an interrupted refresh.
    title = "Qwen3-0.6B GRAFT with LLM-as-a-Judge and Probe Monitors"  # Global: match the README and paper headline.
    description = "Single-seed pilot comparing frozen LLM-judge and probe monitors. Checkpoints and evidence; no established safety guarantee."
    if previous.get("collection"):  # Local: update the recorded object instead of looking up a new title.
        slug = previous["collection"].split("/collections/", 1)[-1]  # Local: convert the public URL to the CLI's collection identifier.
        if not slug.startswith(namespace + "/"):  # Local: an old workspace cannot redirect publication into another account.
            raise RuntimeError("Saved collection belongs to a different Hugging Face namespace")
        command = [cli, "collections", "update", slug, "--title", title]  # Local: metadata updates preserve collection items.
    else:
        command = [cli, "collections", "create", title, "--namespace", namespace, "--exists-ok"]  # Local: a fresh replication creates one collection.
    collection = json.loads(subprocess.check_output(command + ["--description", description, "--format", "json"], env=env, text=True))
    write_json(path, previous | {"collection": "https://huggingface.co/collections/" + collection["slug"]})  # Local: remember a changed slug before uploads begin.
    return collection["slug"]  # Local: every subsequent upload and membership update uses the same identity.


def publish(root):
    """Global: user-authorized publication. Local: upload only checkpoints/configs/cards/reports."""
    env = hub_environment()
    cli = str(ROOT / ".venv/bin/hf")
    identity = json.loads(subprocess.check_output([cli, "auth", "whoami", "--format", "json"], env=env, text=True))
    namespace = identity.get("name") or identity.get("user")
    if not isinstance(namespace, str):
        raise RuntimeError("Could not determine the authenticated Hugging Face namespace")
    slug = ensure_collection(cli, namespace, env, root)  # Local: refresh the existing collection even after a headline change.
    results_path = ROOT / "reports/qwen_probe/results.json"  # Local: cards share the report's actual coverage and conclusion.
    results = read_json(results_path) if results_path.exists() else {}  # Local: early publication remains explicitly interim.
    study_status = results.get("conclusion", "Study outcomes are not yet available.")
    if "coverage" in results:
        study_status += f" Audited episodes: {results['coverage']['audited_study_episodes']}/{results['coverage']['planned_study_episodes']}. " + " ".join(results.get("reasons", []))
    published = []
    for arm in ARMS:
        folder = root / "checkpoints" / arm
        if not (folder / "adapter_state_dict.pt").exists():
            continue
        progress = read_json(folder / "progress.json")
        stage = root / "hub" / arm
        stage.mkdir(parents=True, exist_ok=True)
        for filename in ("adapter_state_dict.pt", "configuration.json", "progress.json", "metrics.jsonl"):
            if not (folder / filename).exists():
                continue
            shutil.copy2(folder / filename, stage / filename)
        for filename in ("artifacts.json", "task_split.json"):
            shutil.copy2(ROOT / "experiments/qwen_probe" / filename, stage / filename)
        shutil.copy2(ROOT / "docs/protocol.md", stage / "protocol.md")
        stage_reports(ROOT / "reports/qwen_probe", stage / "report")  # Local: keep cross-site methodology citations usable.
        card = f"""---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: pytorch
tags:
- graft
- reward-hacking
- exploratory-pilot
language:
- en
---
# {arm}: Qwen3-0.6B GRAFT with LLM-as-a-Judge and Probe Monitors

Author: Libor Burian. **Exploratory artifact, single training seed; no established safety guarantee.**
Completed planned training: {progress['complete']}; completed optimizer updates: {progress['steps']}.

{study_status}

The two monitor treatments use a frozen untouched Qwen3-0.6B LLM judge and a separately downloaded,
frozen embedding/trajectory probe. They supply precomputed routing flags, never training rewards.

This is a custom dual-MLP adapter state dictionary, not a PEFT LoRA checkpoint or a standalone language model.
Use the frozen Qwen3-0.6B base revision and the pinned repository implementation. See `protocol.md`,
`configuration.json`, `progress.json`, and `report/results.json` for data, methodology, coverage and limitations.

## Reproduce and merge

Clone {REPOSITORY} recursively, install with
`uv sync --frozen`, and run the documented download phase. Download this adapter and run:

```bash
uv run python sft/scripts/eval/merge_adapter.py --base-model-dir artifacts/qwen_probe/models/learner --adapter-ckpt /path/to/adapter_state_dict.pt --output-dir artifacts/merged-pilot --d-retain 64 --d-forget 64 --lora-alpha 32 --retain-scale 1 --forget-scale 0
```

For the no-intervention primary baseline use `--forget-scale 1`. Both branches are stored so the
prespecified interventions can be reproduced. This checkpoint alone does not contain the external detector.

## Scope and limitations

The study uses Terminal Wrench trajectories, excludes training-side calibration tasks, and preserves the
published heldout task split. Complete upstream model/probe training exposure cannot be ruled out.
Eight agentic tasks and one training seed cannot establish general effectiveness. Missing comparisons,
ambiguous intent and capability floors are retained in the report. These weights may reproduce reward-hacking
behavior from the training mixture and are intended for controlled research evaluations.

## Attribution

[GRAFT](https://arxiv.org/abs/2610.17062), [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B),
and [farzingkh/reward-hacking](https://github.com/farzingkh/reward-hacking).
The upstream detector is downloaded separately with verified hashes and is not redistributed here.
"""
        (stage / "README.md").write_text(card)
        repo = f"{namespace}/graft-qwen3-06b-{arm.replace('_', '-')}-pilot"
        execute([cli, "repos", "create", repo, "--type", "model", "--exist-ok"], env=env)
        execute([cli, "upload", repo, str(stage), ".", "--commit-message", "Publish exploratory pilot adapter and reproducibility evidence"], env=env)
        execute([cli, "collections", "add-item", slug, repo, "model", "--exists-ok"], env=env)
        published.append({"repo": repo, "complete": progress["complete"], "steps": progress["steps"]})
    result = {"collection": "https://huggingface.co/collections/" + slug, "models": published}
    write_json(root / "hub/publication.json", result)
    write_json(ROOT / "reports/qwen_probe/huggingface.json", result)
    event("hub_published", **result)
    return result
