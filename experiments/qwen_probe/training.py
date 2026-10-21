"""Global: reuse GRAFT rather than reimplement it. Local: configure isolated trainer processes.

Source: sft/scripts/gr/train_gr.py; hyperparameters and deviations: docs/protocol.md.
"""
import os  # Local: child-only paths and deadline.
import subprocess  # Local: verify publication before parameter updates.
import sys  # Local: one locked environment.
import time  # Local: don't start work past the shared cap.

from .data import ARMS, write_pools
from .runtime import ROOT, deadline, event, execute, read_json, write_json


def verify_published(root):
    """Global: preregister before training. Local: HEAD must exist on origin with clean implementation."""
    paths = ["experiments", "tests", "docs/protocol.md", "pyproject.toml", "uv.lock", "sft/scripts"]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *paths], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branches = subprocess.check_output(["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True)
    if "origin/" not in branches:
        raise RuntimeError("Push implementation and protocol before real-model training")
    write_json(root / "publication.json", {"commit": commit, "remote_branches": branches.strip().splitlines()})


def train_arm(root, arm, *, smoke=False):
    """Global: matched recipe. Local: only monitor treatment and optimizer baseline differ."""
    end = deadline(root, start=True)
    if time.time() >= end:
        event("training_incomplete", arm=arm, reason="deadline")
        return False
    name = "smoke" if smoke else arm
    destination = root / "checkpoints" / name
    if (destination / "progress.json").exists():  # Local: never silently restart or extend an interrupted run.
        return read_json(destination / "progress.json")["complete"]
    if not (root / "data" / arm / "gr_train_retain.jsonl").exists():
        raise RuntimeError(f"Missing routing pools for {arm}; complete the corresponding monitor validation first")
    verify_published(root)
    lock = read_json(root / "data/lock.json")
    config = lock["configuration"]
    env = os.environ.copy()  # Local: inherited credentials are never logged or forwarded to model inputs.
    env.update({"PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "sft")]),
                "GRAFT_SFT_DATA_DIR": str(root / "data" / arm), "GRAFT_SFT_CKPT_DIR": str(root / "checkpoints"),
                "GRAFT_DEADLINE": str(end), "WANDB_DISABLED": "true", "OMP_NUM_THREADS": "4"})
    env.pop("WANDB_API_KEY", None)  # Local: no unrequested external experiment tracking.
    command = [sys.executable, "-u", "-m", "scripts.gr.train_gr", "--model-name", str(root / "models/learner"),
               "--run-name", name, "--epochs", "5", "--lr", "0.0005", "--d-retain", "64", "--d-forget", "64",
               "--step-size", "48", "--max-length", str(config["max_length"]), "--weight-decay", "0.0001",
               "--warmup-ratio", "0.05", "--seed", "42", "--classifier-seed", "42", "--classifier-retain-recall", "0.01",
               "--retain-from-unlabeled", "--match-rslora", "--lora-alpha", "32", "--filter-overlong", "--no-evals",
               "--attention", os.environ.get("GRAFT_ATTENTION", "sdpa"), "--liger-loss", "--correct-accumulation",
               "--stable-order", "--cache-key", lock["fingerprint"]]
    if arm == "probe_filtering":
        command += ["--gradient-ascent", "--gradient-ascent-magnitude", "0"]
    else:
        command += ["--split-moment", "--kappa", "2", "--unclassified-trains-both"]
    if arm == "no_intervention":
        command += ["--no-routing"]
    if smoke:
        command += ["--max-steps", "1"]
    write_json(destination / "configuration.json", {"arm": arm, "smoke": smoke, "argv": command,
                                                    "data_fingerprint": lock["fingerprint"], "deadline": end})
    code = execute(command, env=env, deadline=end, check=False)
    status = read_json(destination / "progress.json") if (destination / "progress.json").exists() else {"complete": False, "steps": 0}
    event("training_finished", arm=name, exit_code=code, **status)
    if code and time.time() < end:
        raise RuntimeError(f"Trainer failed for {name}; inspect timestamped logs")
    return status["complete"]


def train(root, arm=None):
    """Global: fixed priority independent of outcomes. Local: train each registered arm once."""
    if arm is not None and arm not in ARMS:
        raise ValueError(f"Unknown training arm: {arm}")
    write_pools(root)  # Local: also prepare the interchange for an independently requested arm.
    for selected in ((arm,) if arm else ARMS):
        if not train_arm(root, selected):
            break  # Local: preserve missing comparisons instead of changing their recipe.


def merge_model(root, arm, mode):
    """Global: valid deployment intervention. Local: reuse tested tensor-concatenation merger."""
    if arm == "base":
        return root / "models/learner"
    checkpoint = root / "checkpoints" / arm / "adapter_state_dict.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No trained checkpoint for {arm}")
    destination = root / "merged" / f"{arm}-{mode}"
    if (destination / "config.json").exists():  # Local: per-run checkpoints are immutable once finished.
        return destination
    sys.path.insert(0, str(ROOT / "sft"))
    from scripts.eval.merge_adapter import merge
    retain, forget = {"both": (1, 1), "retain": (1, 0), "forget": (0, 1)}[mode]
    merge(root / "models/learner", checkpoint, destination, d_retain=64, d_forget=64,
          lora_alpha=32, retain_scale=retain, forget_scale=forget)
    return destination
