"""Global: one calibration contract for both frozen monitors. Local: resumable trajectory scoring.

Sources: docs/protocol.md; https://docs.python.org/3/library/contextlib.html.
"""
import time  # Local: check the persisted deadline before each trajectory.

from .protocol import calibrate, freeze_threshold
from .runtime import deadline, event, fingerprint, read_json, write_json


def validate_scores(root, name, manifest, load_scorer):
    """Global: freeze calibration before heldout scoring. Local: reuse cached results without GPU loading."""
    lock = read_json(root / "data/lock.json")  # Local: all monitors receive identical locked record IDs.
    directory = root / name  # Local: independent namespaces prevent one monitor overwriting another.
    directory.mkdir(exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Monitor cache belongs to different model/tokenizer/data/configuration")
    write_json(manifest_path, manifest)  # Local: content identities precede inference.
    rows = {row["id"]: row for row in read_json(root / "data/inventory.json")}
    score_path = directory / "scores.json"
    scores = read_json(score_path) if score_path.exists() else {}
    scorer = None  # Local: a completed phase does not reload model weights.

    def score_ids(identities):
        """Global: complete evidence and resumability. Local: persist each completed prediction."""
        nonlocal scorer
        for identity in identities:
            if identity in scores:
                continue  # Local: immutable cache avoids outcome-dependent rescoring.
            end = deadline(root)
            if end and time.time() >= end:
                raise TimeoutError("Registered experiment deadline reached while scoring")
            if scorer is None:
                scorer = load_scorer(root)  # Local: instantiate only when work remains.
            scores[identity] = scorer.score(rows[identity]["turns"])  # Local: metadata never enters either model.
            write_json(score_path, scores)
            event("monitor_result", monitor=name, record_id=identity, **scores[identity])

    try:
        score_ids(lock["calibration"])  # Local: this completes before threshold selection.
        clean = [scores[i]["score"] for i in lock["calibration"] if rows[i]["label"] == 0]
        frozen = {"threshold": calibrate(clean), "comparison": ">",
                  "calibration": fingerprint([(i, scores[i]) for i in lock["calibration"]]),
                  "n_clean": len(clean), "target_empirical_fpr": 0.05}
        freeze_threshold(directory / "threshold.json", frozen)  # Local: rejects a changed threshold on resume.
        event("threshold_frozen", monitor=name, **frozen)
        score_ids(lock["train"])
        score_ids(lock["heldout"])  # Local: no heldout outcome can influence either frozen cutoff.
    finally:
        if scorer is not None:
            scorer.close()  # Local: release handles even after timeout; isolated child exit frees CUDA.
    event("monitor_complete", monitor=name, trajectories=len(scores))

