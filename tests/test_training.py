"""Global: functional equivalence before real SFT. Local: small synthetic Qwen models.

Sources: https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
and https://github.com/linkedin/Liger-Kernel.
"""
import copy  # Local: identical reference parameters.

import pytest  # Local: CUDA-aware tests.
import torch  # Local: exact loss/gradient comparisons.
from accelerate import Accelerator  # Local: test actual automatic loss division.
from transformers import Qwen3Config, Qwen3ForCausalLM  # Local: real architecture, tiny dimensions.

from experiments.qwen_probe.compat import configure_loss, loss_scale
from scripts.gr.adapter import inject_adapters, set_scales
from scripts.gr.train_gr import _capture_scale_hooks, _remove_hooks
from scripts.eval.merge_adapter import merge


def tiny_model():
    """Global: low-cost regression. Local: two layers with tied input/output embeddings."""
    torch.manual_seed(42)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                      head_dim=8, tie_word_embeddings=True, attention_dropout=0.0))


def test_actual_accelerate_accumulation_equals_direct_mean():
    """Global: no double normalization. Local: compare parameter gradients from four examples."""
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=4)
    model = torch.nn.Linear(3, 1, bias=False)
    reference = copy.deepcopy(model)
    values, targets = torch.randn(4, 3), torch.randn(4, 1)
    torch.nn.functional.mse_loss(reference(values), targets).backward()
    for row, target in zip(values, targets):
        with accelerator.accumulate(model):
            accelerator.backward(torch.nn.functional.mse_loss(model(row), target) * loss_scale(1, 4, True))
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    assert loss_scale(1, 4, False) == 0.25


def test_routing_captures_natural_gradient_before_masking():
    """Global: split moments stay meaningful. Local: zero routed gradient keeps natural signal."""
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    hooks = _capture_scale_hooks([parameter], 0)
    (parameter ** 2).backward()
    _remove_hooks(hooks)
    assert parameter.grad.item() == 0 and parameter._pre_routing_grad.item() == 4


def test_injected_adapters_match_base_dtype():
    """Global: real BF16 preflight compatibility. Local: new Linear modules must inherit the base dtype."""
    model = tiny_model().to(dtype=torch.bfloat16)
    inject_adapters(model, d_retain=4, d_forget=4)
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())


@pytest.mark.parametrize("scales", [(1., 1.), (1., 0.), (0., 1.)])
def test_unsharded_tied_merge_matches_adapter(tmp_path, scales):
    """Global: valid ablations. Local: merged logits equal live adapters in each mode."""
    base = tiny_model().eval()
    base.save_pretrained(tmp_path / "base", safe_serialization=True)
    assert not (tmp_path / "base/model.safetensors.index.json").exists()
    inject_adapters(base, d_retain=4, d_forget=4, match_rslora=True, lora_alpha=32)
    with torch.no_grad():
        for name, parameter in base.named_parameters():
            if "down_retain" in name or "down_forget" in name:
                parameter.normal_(std=0.01)
    state = {name: value for name, value in base.state_dict().items() if "_retain" in name or "_forget" in name}
    torch.save(state, tmp_path / "adapter.pt")
    merge(tmp_path / "base", tmp_path / "adapter.pt", tmp_path / "merged", d_retain=4,
          d_forget=4, retain_scale=scales[0], forget_scale=scales[1])
    merged = Qwen3ForCausalLM.from_pretrained(tmp_path / "merged").eval()
    set_scales(base, *scales)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(base(ids).logits, merged(ids).logits, atol=1e-6, rtol=1e-5)
    assert merged.lm_head.weight.data_ptr() == merged.model.embed_tokens.weight.data_ptr()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel equivalence needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_liger_loss_and_gradients(dtype):
    """Global: memory optimization preserves learning. Local: compare all trainable gradients."""
    standard = tiny_model().cuda().to(dtype=dtype).train()
    fused = copy.deepcopy(standard)
    configure_loss(fused)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device="cuda")
    labels = ids.clone()
    labels[:, :2] = -100
    reference_loss = standard(ids, labels=labels).loss
    fused_loss = fused(ids, labels=labels).loss
    reference_loss.backward()
    fused_loss.backward()
    tolerance = 0.02 if dtype == torch.bfloat16 else 2e-4
    torch.testing.assert_close(fused_loss, reference_loss, atol=tolerance, rtol=tolerance)
    for a, b in zip(standard.parameters(), fused.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=tolerance, rtol=10 * tolerance)
def test_collator_preserves_assistant_labels_and_examples():
    """Global: TRL upgrades cannot turn prompts into targets. Local: check masks and sequence boundaries."""
    from copy import deepcopy  # Local: detect destructive removal of routing metadata.
    from types import SimpleNamespace  # Local: the collator needs only a padding token.
    from scripts.gr.loader import build_collator  # Local: exercise the actual training path.
    examples = [{"input_ids": [11, 12, 13, 14], "assistant_masks": [0, 0, 1, 1], "classification": 2},
                {"input_ids": [21, 22, 23], "assistant_masks": [1, 0, 1], "classification": 1}]
    original = deepcopy(examples)  # Local: datasets may reuse the same example objects.
    batch = build_collator(SimpleNamespace(pad_token_id=0))(examples)
    assert batch["labels"].tolist() == [[-100, -100, 13, 14, -100, -100, 23]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 0, 1, 2]]
    assert batch["classification"].tolist() == [2, 1]
    assert examples == original
