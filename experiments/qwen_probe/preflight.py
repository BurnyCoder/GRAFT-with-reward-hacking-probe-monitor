"""Global: fix a common feasible context before selection. Local: backward without updates.

Sources: Transformers Qwen3 docs; https://github.com/linkedin/Liger-Kernel.
"""
import sys  # Local: standalone worker arguments.
import time  # Local: preflight speed estimate without selecting on outcomes.

import torch  # Local: CUDA memory instrumentation and synthetic token tensors.
from transformers import AutoModelForCausalLM  # Local: load the pinned real learner.

from .compat import configure_loss
from .runtime import ROOT, event, setup, write_json


def preflight(root, length):
    """Global: engineering-only preflight. Local: test one fully supervised synthetic sequence."""
    sys.path.insert(0, str(ROOT / "sft"))
    from scripts.gr.adapter import inject_adapters, param_groups  # Local: actual study adapter architecture.
    started = time.time()
    torch.manual_seed(42)  # Local: fixed synthetic sequence/adapter initialization.
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(root / "models/learner", dtype=torch.bfloat16,
                                                attn_implementation="sdpa", local_files_only=True)
    configure_loss(model)
    inject_adapters(model, d_retain=64, d_forget=64, lora_alpha=32, match_rslora=True)
    model = model.cuda().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    groups = param_groups(model)
    moments = [torch.zeros_like(p, dtype=torch.float32) for group in groups for p in group for _ in range(2)]
    inputs = torch.randint(100, 1000, (1, length), device="cuda")  # Local: no heldout/task outcomes used.
    loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
    loss.backward()  # Local: no optimizer is created and no parameter is updated.
    torch.cuda.synchronize()
    result = {"max_length": length, "seconds": time.time() - started,
              "peak_bytes": torch.cuda.max_memory_allocated(), "loss": float(loss.detach()),
              "optimizer_moment_bytes": sum(x.numel() * x.element_size() for x in moments), "parameter_updates": 0}
    write_json(root / "preflight.json", result)
    event("memory_preflight", **result)


if __name__ == "__main__":
    try:
        preflight(setup(), int(sys.argv[1]))  # Local: child exit releases all model and CUDA memory.
    except torch.cuda.OutOfMemoryError as error:
        event("memory_preflight_oom", max_length=int(sys.argv[1]), error=str(error))
        sys.exit(75)  # Local: distinguish OOM from bugs, corrupt weights or unavailable CUDA.
