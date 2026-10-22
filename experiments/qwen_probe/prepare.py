"""Global: complete prerequisites before treatment. Local: downloads, task builds, preflight, lock."""
import subprocess  # Local: idempotent pinned Harbor patch application.
import sys  # Local: isolated GPU memory preflight.

from .data import fix_dataset, inventory
from .conversion import require_current_conversion  # Local: fail on legacy caches before downloads or GPU preflight.
from .download import download
from .runtime import ROOT, deadline, execute, read_json
from .tasks import select_tasks


def prepare(root):
    """Global: data decisions precede monitor outcomes. Local: use outcome-free memory fallback."""
    require_current_conversion(root)  # Local: historical checkpoints require their historical inputs, never relabeled corrected ones.
    download(root)  # Local: install/download time remains outside the six-hour experiment clock.
    harbor = ROOT / "sft/third_party/harbor"
    patch = ROOT / "sft/scripts/eval/patches/harbor_verifier_on_ctx_overflow.patch"
    if subprocess.run(["git", "-C", str(harbor), "apply", "--reverse", "--check", str(patch)], capture_output=True).returncode:
        subprocess.run(["git", "-C", str(harbor), "apply", str(patch)], check=True)
    select_tasks(root)  # Local: pre-download/build Docker images without running a study model.
    if not (root / "data/inventory.json").exists():
        inventory(root)
    if (root / "data/lock.json").exists():
        return read_json(root / "data/lock.json")
    end = deadline(root, start=True)  # Local: persistent across every later phase and resume.
    for length in (16384, 8192, 4096):
        code = execute([sys.executable, "-u", "-m", "experiments.qwen_probe.preflight", str(length)], deadline=end, check=False)
        if code == 0:
            return fix_dataset(root, length)
        if code != 75:  # Local: only a confirmed CUDA OOM permits lowering the context limit.
            raise RuntimeError("Memory preflight failed for a reason other than CUDA OOM")
    raise RuntimeError("The 4096-token preflight also exhausted CUDA memory")
