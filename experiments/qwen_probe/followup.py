"""Global: complete missing ablations as a separate outcome-aware follow-up, never reopen the pilot.

Local: reuse the saved adapters and evaluator, with immutable source hashes and a new fixed clock.
Sources: https://www.cos.io/initiatives/prereg; https://docs.python.org/3/library/fcntl.html.
"""
import fcntl  # Local: an exclusive process lock prevents two follow-up controllers.
import os  # Local: record only our controller identity, never environment credentials.
import shutil  # Local: small immutable manifests are copied instead of editing pilot files.
import time  # Local: one persisted three-hour budget starts at evaluation.

from .evaluation import evaluate
from .runtime import event, file_hash, read_json, write_json
from .training import verify_published

NAME = "ablation-followup"  # Local: distinct artifact namespace, including audits and inference journals.
PLAN = [("probe_graft", "both", (False,)), ("probe_graft", "forget", (True, False)),
        ("no_intervention", "retain", (True, False))]  # Global: only the five previously unrun conditions, in original priority order.
SOURCE_FILES = ("budget.json", "evaluation/tasks.json", "evaluation/index.json", "audit/judgments.json",
                "audit/frozen_judgments.json", "audit/unblinded.json", "data/lock.json", "downloads.json",
                "monitor/threshold.json", "judge/threshold.json")  # Local: freeze the original clock, labels, settings and thresholds.


def source_snapshot(pilot):
    """Global: reuse exactly the closed study. Local: hash evidence and both reused checkpoints."""
    paths = [pilot / name for name in SOURCE_FILES]
    paths += sorted(path for path in (pilot / "models/learner").iterdir() if path.is_file())  # Local: freeze base weights, tokenizer and model configuration, excluding download-cache directories.
    paths += [pilot / "checkpoints" / arm / name for arm in ("probe_graft", "no_intervention")
              for name in ("adapter_state_dict.pt", "configuration.json", "progress.json")]
    return {str(path.relative_to(pilot)): file_hash(path) for path in paths}  # Local: stdlib streaming SHA-256 avoids loading weights.


def prepare_followup(pilot):
    """Global: reject overlap or changed inputs. Local: create a separate read-only-input workspace."""
    index, budget = (read_json(pilot / name) for name in ("evaluation/index.json", "budget.json"))
    if time.time() < budget["deadline"] or any(job.get("status") == "running" for job in index.values()):
        raise ValueError("The source pilot is still active")
    if read_json(pilot / "audit/frozen_judgments.json")["sha256"] != file_hash(pilot / "audit/judgments.json"):
        raise ValueError("Original pilot judgments are not frozen")
    jobs = [f"{arm}-{mode}-{'elicited' if condition else 'plain'}" for arm, mode, conditions in PLAN for condition in conditions]
    if any(job in index for job in jobs):  # Local: even a failed original attempt blocks repeating that condition here.
        raise ValueError("A follow-up condition was already attempted in the pilot")
    if len(read_json(pilot / "evaluation/tasks.json")["tasks"]) != 8:
        raise ValueError("This follow-up requires the original eight fixed tasks")
    for arm in ("probe_graft", "no_intervention"):
        if not read_json(pilot / "checkpoints" / arm / "progress.json")["complete"]:
            raise ValueError("Follow-up requires the already completed saved checkpoints")
    root = pilot / NAME
    manifest = {"study": NAME, "source_sha256": source_snapshot(pilot), "jobs": jobs,
                "planned_episodes": 40, "hours": 3, "analysis_status": "exploratory; original pilot outcomes already known"}
    path = root / "manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Frozen follow-up inputs or scope changed")
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("models", "checkpoints"):  # Local: evaluation only reads shared input directories; training is unavailable in this study CLI.
        link = root / directory
        if not link.exists():
            link.symlink_to(pilot / directory, target_is_directory=True)
        if not link.is_symlink() or link.resolve() != (pilot / directory).resolve():
            raise ValueError("Follow-up input link changed")
    for name in ("downloads.json", "evaluation/tasks.json"):
        target = root / name
        if target.exists() and file_hash(target) != file_hash(pilot / name):
            raise ValueError("Copied follow-up input changed")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pilot / name, target)
    if not path.exists():
        write_json(path, manifest)  # Local: write once; preparation does not start or reset the budget.
    return root


def evaluate_followup(pilot):
    """Global: no training or repeated attempts. Local: fixed conditions, persistent budget, scoped lock."""
    root = prepare_followup(pilot)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Local: fail immediately if another controller owns this study.
        publication = verify_published(root)  # Local: the exploratory protocol and implementation must already be pushed.
        budget_path = root / "budget.json"
        if not budget_path.exists():
            now = time.time()
            write_json(budget_path, {"start": now, "deadline": now + 3 * 3600, "hours": 3,
                                    "source_deadline": read_json(pilot / "budget.json")["deadline"]})
        write_json(root / "controller.json", {"status": "running", "pid": os.getpid(), "publication": publication})
        try:
            for arm, mode, conditions in PLAN:  # Local: the evaluator skips every condition already indexed, even a failed one.
                evaluate(root, modes=[(arm, mode)], conditions=list(conditions))
        finally:
            write_json(root / "controller.json", {"status": "closed", "finished_at": time.time(), "publication": publication})
        event("followup_evaluation_closed", planned_episodes=40)  # Local: do not disclose behavior before mixed audit packets freeze.
    return root


def require_closed(root):
    """Global: audit only after every deployment server closes. Local: an absent marker means not ready."""
    path = root / "controller.json"
    if not path.exists() or read_json(path)["status"] != "closed":
        raise ValueError("Wait for the complete follow-up controller to close before auditing")
    index = root / "evaluation/index.json"  # Local: exceptional shutdown can leave an unclosed job despite controller exit.
    if index.exists() and any(job["status"] == "running" for job in read_json(index).values()):
        raise ValueError("A follow-up evaluation job is still active")


def recover_followup(pilot):
    """Global: recover only the dated shim failure before any audit. Local: retain the fixed clock and model attempts."""
    from .recovery import recover_setup_failures
    from .tasks import ensure_task_images
    root = prepare_followup(pilot)  # Local: recovery still requires every original source hash to match.
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Local: do not race a running evaluator.
        require_closed(root)
        if any((root / "audit/packets").glob("*.json")):
            raise ValueError("Recovery must precede all follow-up audit packets")
        for name in ("publication", "controller"):
            target = root / f"{name}_initial.json"
            if not target.exists():
                shutil.copy2(root / f"{name}.json", target)  # Local: resumption cannot erase the first collection's provenance.
        verify_published(root)  # Local: the dated amendment and recovery implementation must be pushed first.
        ensure_task_images(root)  # Local: unchanged executable images are mandatory after the host upgrade.
        return recover_setup_failures(root, failure="shim-upgrade", allow_uninitialized=True)
