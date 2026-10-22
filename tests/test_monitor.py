"""Global: released detector compatibility. Local: schema, pooling, and checkpoint tests.

Source: https://github.com/farzingkh/reward-hacking/tree/cd4c566824c5961eaeab7b7fce7845240bd12b3d
"""
import torch  # Local: synthetic embeddings; global: no downloaded model needed here.

from experiments.qwen_probe.detector.preprocess import normalize_messages, trajectory_messages
from experiments.qwen_probe.detector.model import TurnTransformerProbe, last_token_pool


def test_metadata_excluded_and_tools_included():
    """Global: label blindness. Local: whitelist only trajectory-visible fields."""
    trajectory = {"reward": "SECRET", "source": "SECRET", "label": "SECRET", "steps": [
        {"source": "user", "message": "solve"},
        {"source": "agent", "message": "checking", "tool_calls": [{"function_name": "bash", "arguments": "ls"}],
         "observation": {"results": [{"content": "hello"}]}, "judge": "SECRET"}]}
    turns = normalize_messages(trajectory_messages(trajectory))
    joined = "\n".join(turns)
    assert "SECRET" not in joined and "ls" in joined and "hello" in joined
    assert turns[0] == "USER: solve" and "TOOL:" in joined


def test_inline_tools_not_duplicated():
    """Global: exact representation. Local: inline calls receive one XML wrapper."""
    turns = normalize_messages([{"role": "assistant", "content": "<bash>ls</bash>"}])
    assert turns == ['ASSISTANT: <tool_call name="bash">ls</tool_call>']


def test_pooling_and_checkpoint_contract():
    """Global: 1024-wide, 248-turn head. Local: safe strict state reload/padding."""
    hidden = torch.arange(24).reshape(2, 3, 4).float()
    assert torch.equal(last_token_pool(hidden, torch.tensor([[0, 1, 1], [1, 1, 1]])), hidden[:, -1])
    assert torch.equal(last_token_pool(hidden, torch.tensor([[1, 1, 0], [1, 0, 0]])), hidden[[0, 1], [1, 0]])
    probe = TurnTransformerProbe().eval()
    probe.load_state_dict(probe.state_dict(), strict=True)
    assert probe.positional_encodings.shape == (248, 1024)
    assert len(probe.transformer_encoder.layers) == 4
    inputs = torch.zeros(1, 2, 1024)
    with torch.inference_mode():
        assert probe(inputs, torch.zeros(1, 2, dtype=torch.bool)).shape == (1,)
