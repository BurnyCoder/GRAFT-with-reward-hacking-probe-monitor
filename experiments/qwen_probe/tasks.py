"""Global: outcome-blind local task selection. Local: prebuild exact pinned Docker environments.

Sources: Harbor Docker environments and sft/scripts/eval/prep_eval_dataset.py.
"""
import random  # Local: prespecified heldout ordering.
import subprocess  # Local: bounded Docker startup check.
import time  # Local: build timeout, outside the experiment clock.
import tomllib  # Local: stdlib task resource parsing.
from pathlib import Path

from .runtime import ROOT, deadline, event, execute, read_json, write_json


def original_task(task):
    """Global: same spec as released evaluation. Local: first lexicographic source with original_task."""
    return next(iter(sorted((ROOT / "sft/third_party/terminal-wrench/tasks" / task).glob("*/original_task"))), None)


def image_identity(tag, *, platform=False):
    """Global: immutable task environments. Local: missing images return None, never a guessed identity."""
    command = ["docker", "image", "inspect", tag, "--format", "{{.Id}}"]
    if platform:
        command += ["--platform", "linux/amd64"]  # Local: exclude build-time attestations from executable content identity.
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None  # Local: no unrelated images are enumerated.


def ensure_task_images(root):
    """Global: preserve selected tasks after cleanup. Local: rebuild missing images and require original SHA.

    Source: https://docs.docker.com/reference/cli/docker/image/inspect/.
    """
    selected = read_json(root / "evaluation/tasks.json")
    platform = "platform_images" in selected  # Local: old locks remain strict unless original build evidence supplies platform hashes.
    for task, expected in selected.get("platform_images", selected["images"]).items():
        tag = "graft-probe-task-" + task.lower().replace("_", "-")  # Local: same names as the frozen selection.
        actual = image_identity(tag, platform=platform)
        if actual is None:
            execute(["docker", "build", "-t", tag, original_task(task) / "environment"],
                    deadline=min(time.time() + 600, deadline(root) or float("inf")))  # Local: repairs consume the existing cap.
            actual = image_identity(tag, platform=platform)
            event("task_image_restored", task=task, expected=expected, actual=actual)
        if actual != expected:
            raise RuntimeError(f"Task image identity changed for {task}: {actual} != {expected}")


def select_tasks(root):
    """Global: never select on model outcomes. Local: Docker build/launch success only."""
    destination = root / "evaluation/tasks.json"
    if destination.exists():
        return read_json(destination)
    split = read_json(Path(__file__).with_name("task_split.json"))
    selected, excluded, images, platform_images = [], [], {}, {}
    candidates = sorted(split["eval_tasks"])
    random.Random(42).shuffle(candidates)
    smoke_candidates = sorted(split["train_tasks"])
    random.Random(42).shuffle(smoke_candidates)
    for kind, task_ids, count in (("heldout", candidates, 8), ("smoke", smoke_candidates, 1)):
        accepted = []
        for task in task_ids:
            original = original_task(task)
            if not original or not (original / "environment/Dockerfile").exists():
                excluded.append({"task": task, "reason": "no single Dockerfile environment"})
                continue
            config = tomllib.loads((original / "task.toml").read_text())
            environment = config.get("environment", {})
            if environment.get("gpus", 0) or environment.get("memory_mb", 2048) > 4096:
                excluded.append({"task": task, "reason": "exceeds local Docker resource allowance"})
                continue
            tag = "graft-probe-task-" + task.lower().replace("_", "-")
            code = execute(["docker", "build", "-t", tag, original / "environment"], deadline=time.time() + 600, check=False)
            if code:
                excluded.append({"task": task, "reason": "Docker build failed", "exit_code": code})
                continue
            try:
                subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/sh", tag, "-c", "true"],
                               check=True, timeout=30, capture_output=True)
            except (subprocess.SubprocessError, OSError) as error:
                excluded.append({"task": task, "reason": str(error)})
                continue
            images[task] = subprocess.check_output(["docker", "image", "inspect", tag, "--format", "{{.Id}}"], text=True).strip()
            platform_images[task] = image_identity(tag, platform=True)  # Local: platform manifest commits to both configuration and layers.
            accepted.append(task)
            event("task_accepted", task=task, selection=kind, image=images[task])
            if len(accepted) == count:
                break
        if len(accepted) != count:
            raise RuntimeError(f"Only {len(accepted)} locally runnable {kind} tasks")
        if kind == "heldout":
            selected = accepted
        else:
            smoke = accepted[0]
    result = {"tasks": selected, "smoke_task": smoke, "exclusions": excluded, "images": images, "platform_images": platform_images,
              "selection": "seed 42, Docker build and launch only; no model/verifier outcomes used"}
    write_json(destination, result)
    return result
