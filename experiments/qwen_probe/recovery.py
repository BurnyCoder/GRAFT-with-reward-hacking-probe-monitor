"""Global: preserve one actual model attempt. Local: archive only explicitly allowed setup failures.

The dated docs/protocol.md amendment authorizes this explicit phase; no outcome is used.
Original Harbor evidence stays intact and a second recovery of a job is prohibited.
The separate follow-up amendment covers containerd bootstrap skew after the user's upgrade.
"""
import time  # Local: timestamp the infrastructure correction separately from scientific outcomes.

from .runtime import event, file_hash, read_json, write_json

FAILURES = {"image-pull": ("pull access denied for graft-probe-task-",
                           "All trials failed to pull deleted local images before any agent/verifier execution"),
            "shim-upgrade": ("shim was started through the deprecated API but was built against the new API",
                             "All trials failed at containerd shim bootstrap before any agent/verifier execution")}
# Global: an exact allowlist prevents arbitrary failures from authorizing retries.
# Source: https://github.com/containerd/containerd/blob/v2.4.2/pkg/shim/compat.go.


def recover_setup_failures(root, *, failure="image-pull"):
    """Global: outcome-blind repair. Local: require all expected trials to fail before agent/verifier execution."""
    signature, reason = FAILURES[failure]  # Local: the original pilot retains its image-pull-only default.
    index_path = root / "evaluation/index.json"
    index = read_json(index_path)  # Local: caller must stop the controller before rewriting its index.
    if any(metadata["status"] == "running" for metadata in index.values()):
        raise RuntimeError("Close evaluation jobs before recovering setup failures")
    destination = root / "evaluation/setup_recovery.json"
    history = read_json(destination) if destination.exists() else []
    recovered = []
    for name, metadata in list(index.items()):
        folder = root / "evaluation/jobs" / name
        trials = [(path, read_json(path)) for path in sorted(folder.glob("*/result.json"))]
        expected = metadata.get("tasks", [])  # Local: missing task coverage is never permission to retry.
        if metadata.get("smoke") or not expected or sorted(value.get("task_name", "") for _, value in trials) != sorted(expected):
            continue
        eligible = all(not value.get("agent_execution") and not value.get("verifier_result")
                       and not (path.parent / "agent/trajectory.json").exists()
                       and (value.get("exception_info") or {}).get("exception_type") == "RuntimeError"
                       and signature in value["exception_info"].get("exception_message", "")
                       for path, value in trials)  # Local: failed or successful model attempts never qualify.
        if not eligible or any(item["job"] == name for item in history):
            continue
        archive = root / "evaluation/setup_failures" / name
        archive.parent.mkdir(parents=True, exist_ok=True)
        record = {"job": name, "metadata": metadata, "archived_at": time.time(), "trial_count": len(trials),
                  "reason": reason, "failure": failure,
                  "result_sha256": {str(path.relative_to(folder)): file_hash(path) for path, _ in trials}}
        folder.rename(archive)  # Local: retain the original logs/results byte-for-byte; never erase a failed attempt.
        write_json(archive / "recovery.json", record)
        del index[name]  # Local: normal fixed-order evaluation can now make its first model attempt.
        write_json(index_path, index)
        history.append(record)
        write_json(destination, history)
        recovered.append(record)
        event("setup_failure_archived", **record)
    return recovered
