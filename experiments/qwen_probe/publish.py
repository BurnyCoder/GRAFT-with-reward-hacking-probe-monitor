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


def publish(root):
    """Global: user-authorized publication. Local: upload only checkpoints/configs/cards/reports."""
    env = os.environ.copy()
    env.pop("HF_HOME", None)  # Local: the user confirmed an existing default CLI login.
    env.pop("HF_HUB_CACHE", None)  # Local: upload uses authentication; no model download is triggered.
    cli = str(ROOT / ".venv/bin/hf")
    identity = json.loads(subprocess.check_output([cli, "auth", "whoami", "--format", "json"], env=env, text=True))
    namespace = identity.get("name") or identity.get("user")
    if not isinstance(namespace, str):
        raise RuntimeError("Could not determine the authenticated Hugging Face namespace")
    collection = json.loads(subprocess.check_output([cli, "collections", "create", "Qwen3-0.6B GRAFT reward-hacking probe pilot",
        "--namespace", namespace, "--description", "Exploratory single-seed pilot adapters, configs and evidence; no established safety guarantee.",
        "--exists-ok", "--format", "json"], env=env, text=True))
    slug = collection["slug"]
    published = []
    for arm in ARMS:
        folder = root / "checkpoints" / arm
        if not (folder / "adapter_state_dict.pt").exists():
            continue
        progress = read_json(folder / "progress.json")
        stage = root / "hub" / arm
        stage.mkdir(parents=True, exist_ok=True)
        for filename in ("adapter_state_dict.pt", "configuration.json", "progress.json"):
            shutil.copy2(folder / filename, stage / filename)
        for filename in ("artifacts.json", "task_split.json"):
            shutil.copy2(ROOT / "experiments/qwen_probe" / filename, stage / filename)
        shutil.copy2(ROOT / "docs/protocol.md", stage / "protocol.md")
        shutil.copytree(ROOT / "reports/qwen_probe", stage / "report", dirs_exist_ok=True)
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
# {arm}: Qwen3-0.6B GRAFT pilot adapter

Author: Libor Burian. **Exploratory artifact, single training seed; no established safety guarantee.**
Completed planned training: {progress['complete']}; completed optimizer updates: {progress['steps']}.

This is a custom dual-MLP adapter state dictionary, not a PEFT LoRA checkpoint or a standalone language model.
Use the frozen Qwen3-0.6B base revision and the pinned repository implementation. See `protocol.md`,
`configuration.json`, `progress.json`, and `report/results.json` for data, methodology, coverage and limitations.

## Reproduce and merge

Clone https://github.com/BurnyCoder/GRAFT-with-reward-hacking-probe-monitor recursively, install with
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
