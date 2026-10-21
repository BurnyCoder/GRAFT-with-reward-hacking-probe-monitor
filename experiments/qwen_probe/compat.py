"""Global: small-model compatibility without changing released defaults.

Local: public Liger patch and explicit Accelerate normalization convention.
Sources: https://github.com/linkedin/Liger-Kernel and
https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
"""


def loss_scale(world_size, effective_batch, corrected):
    """Global: preserve old runs. Local: Accelerate already divides each loss by accumulation."""
    return 1.0 if corrected else world_size / effective_batch  # Local: opt in to one division only.


def configure_loss(model):
    """Global: avoid sequence×vocabulary logits. Local: patch only supported Qwen3 fused CE."""
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3  # Local: public library API.
    apply_liger_kernel_to_qwen3(model=model, rope=False, rms_norm=False, swiglu=False,
                               cross_entropy=False, fused_linear_cross_entropy=True)
    return model  # Local: adapter MLPs and attention implementation remain caller-controlled.
