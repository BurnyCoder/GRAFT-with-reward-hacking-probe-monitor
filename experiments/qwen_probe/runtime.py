"""Global: reproducible, local execution. Local: shared configuration, hashes and logs.

Sources: https://docs.python.org/3/library/logging.html and /subprocess.html.
"""
import hashlib  # Local: content addresses; global: invalidate stale experiment caches.
import json  # Local: canonical serialization; global: portable provenance.
import logging  # Local: timestamped terminal/file handlers; global: complete evidence.
import os  # Local: process environment; global: workspace-contained caches.
import signal  # Local: stop our process group; global: leave other GPU jobs alone.
import subprocess  # Local: launch phases; global: free GPU memory between phases.
import sys  # Local: the active venv interpreter; global: one dependency lock.
import time  # Local: deadlines; global: enforce the prespecified compute cap.
from pathlib import Path  # Local: explicit paths; global: no implicit external workspace.

from dotenv import load_dotenv  # Local: standard .env parsing; global: private configuration.

ROOT = Path(__file__).resolve().parents[2]  # Local: package anchor; global: repository scope.
LOGGER = logging.getLogger("qwen_probe")  # Local: one namespace; global: consistent records.


def setup():
    """Global: initialize reproducible paths. Local: load secrets without printing them."""
    load_dotenv(ROOT / ".env.example", override=False)  # Local: documented defaults.
    load_dotenv(ROOT / ".env", override=True)  # Local: user's private configuration.
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME", "TRITON_CACHE_DIR", "XDG_CACHE_HOME"):
        os.environ[key] = str((ROOT / os.environ[key]).resolve())  # Local: child-safe absolute paths.
    root = (ROOT / os.environ.get("GRAFT_ARTIFACTS", "artifacts/qwen_probe")).resolve()  # Local: result root.
    root.mkdir(parents=True, exist_ok=True)  # Local: create only requested workspace files.
    if not LOGGER.handlers:  # Local: avoid duplicated logs when importing phases.
        logs = root / "logs"  # Local: all complete transcripts live together.
        logs.mkdir(exist_ok=True)  # Local: initialize log directory.
        formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")  # Local: readable timestamps.
        formatter.converter = time.gmtime  # Local: all times explicitly UTC.
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(logs / f"{time.time_ns()}.log")]
        for handler in handlers:  # Local: terminal and file see the same untruncated text.
            handler.setFormatter(formatter)  # Local: consistent timestamp formatting.
            LOGGER.addHandler(handler)  # Local: attach sinks exactly once.
        LOGGER.setLevel(logging.INFO)  # Local: capture phase and model events.
    return root  # Local: pass paths explicitly to every phase.


def fingerprint(value):
    """Global: cache identity. Local: stable SHA-256 of structured inputs."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def file_hash(path):
    """Global: artifact integrity. Local: stream files without holding weights in RAM."""
    with Path(path).open("rb") as stream:  # Local: deterministic bytes, no pickle execution.
        return hashlib.file_digest(stream, "sha256").hexdigest()  # Local: stdlib streaming hash.


def read_json(path):
    """Global: persisted phase state. Local: parse UTF-8 JSON."""
    return json.loads(Path(path).read_text())  # Local: let malformed state fail explicitly.


def write_json(path, value):
    """Global: crash-resistant manifests. Local: atomic replacement in the same folder."""
    path = Path(path)  # Local: normalize caller path type.
    path.parent.mkdir(parents=True, exist_ok=True)  # Local: create phase output directories.
    temporary = path.with_suffix(path.suffix + ".tmp")  # Local: never expose partial JSON.
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)  # Local: atomic POSIX rename.


def event(kind, **data):
    """Global: full evidence. Local: record only explicit data, never environment/headers."""
    LOGGER.info(json.dumps({"event": kind, **data}, ensure_ascii=False, default=str))


def execute(command, *, env=None, deadline=None, check=True):
    """Global: isolated child lifetime. Local: tee complete output and honor a deadline."""
    event("command", argv=[str(x) for x in command])  # Local: commands contain no credentials.
    process = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True)  # Local: own process group.
    import threading  # Local: deadline watcher remains active while stdout is quiet.
    def stop_at_deadline():
        """Global: cap computation. Local: graceful signal then bounded forced stop."""
        while process.poll() is None:  # Local: never signal an exited/reused process ID.
            if deadline and time.time() >= deadline:  # Local: absolute shared deadline.
                os.killpg(process.pid, signal.SIGTERM)  # Local: trainers save on this signal.
                try:  # Local: allow a short checkpoint grace interval.
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)  # Local: terminate only this group.
                return
            time.sleep(0.25)  # Local: small watchdog overhead; main thread still logs.
    watcher = threading.Thread(target=stop_at_deadline, daemon=True)  # Local: no blocking tool sleep.
    watcher.start()  # Local: deadline covers silent model execution too.
    for line in process.stdout:  # Local: no truncation of prompts or model responses.
        LOGGER.info(line.rstrip("\n"))  # Local: add UTC timestamp to each child line.
    code = process.wait()  # Local: reap the child and release its resources.
    if check and code:  # Local: never label failed phases complete.
        raise subprocess.CalledProcessError(code, command)
    return code  # Local: callers may record recoverable preflight failures.


def deadline(root, *, start=False):
    """Global: six hours across resumptions. Local: write a start time only once."""
    path = root / "budget.json"  # Local: shared across independent phase commands.
    if not path.exists() and start:  # Local: downloads/installations do not start the clock.
        now = time.time()  # Local: absolute UTC epoch timestamp.
        write_json(path, {"start": now, "deadline": now + 6 * 3600, "hours": 6})
    return read_json(path)["deadline"] if path.exists() else None  # Local: never reset a resumed run.
