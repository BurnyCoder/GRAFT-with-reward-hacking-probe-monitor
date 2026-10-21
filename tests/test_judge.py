"""Global: freeze judge semantics before real scoring. Local: exercise bounded input and conditional labels.

Source: https://huggingface.co/docs/transformers/main/model_doc/qwen3 (last-position logits).
"""
import math  # Local: reference conditional probability independent of the implementation.

import torch  # Local: synthetic logits avoid a model download.

from experiments.qwen_probe.judge import conditional_score, prepare_input
from experiments.qwen_probe.monitoring import validate_scores
from experiments.qwen_probe.runtime import read_json, write_json


class CharacterTokenizer:
    """Global: deterministic context accounting. Local: one character per test token."""

    def encode(self, text, add_special_tokens=False):
        """Local: expose exact retained character counts without hidden tokenizer behavior."""
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        """Local: invert character IDs exactly for truncation assertions."""
        return "".join(map(chr, tokens))

    def apply_chat_template(self, messages, **kwargs):
        """Local: make role boundaries and disabled thinking visible in the rendered prompt."""
        assert kwargs["enable_thinking"] is False
        return "".join(message["role"] + ":" + message["content"] + "\n" for message in messages) + "assistant:"


def test_judge_uses_turns_only_and_keeps_context_ends():
    """Global: no metadata leakage. Local: deterministic head/tail truncation preserves the fixed rubric."""
    tokenizer = CharacterTokenizer()  # Local: token budget and text length coincide.
    turns = ["user: START" + "x" * 2000, "assistant: END"]
    result = prepare_input(tokenizer, turns, "Fixed intent rubric", max_tokens=200)
    assert len(result["input_ids"]) <= 200
    assert "Fixed intent rubric" in result["prompt"]
    assert "START" in result["prompt"] and "END" in result["prompt"]
    assert result["context_truncated"] and result["retained_transcript_tokens"] < result["transcript_tokens"]
    assert result == prepare_input(tokenizer, turns, "Fixed intent rubric", max_tokens=200)


def test_judge_turn_capacity_and_conditional_probability():
    """Global: same first-248-turn boundary. Local: unrelated vocabulary logits cannot change binary odds."""
    result = prepare_input(CharacterTokenizer(), ["turn"] * 249, "rubric")
    assert result["turns"] == 249 and result["retained_turns"] == 248
    assert not result["context_truncated"]
    score = conditional_score(torch.tensor([1000.0, 2.0, 0.0]), yes_id=1, no_id=2)
    assert math.isclose(score, 1 / (1 + math.exp(-2)), rel_tol=1e-6)


def test_shared_monitor_contract_excludes_metadata_and_freezes_before_holdout(tmp_path):
    """Global: judge cannot see labels or tune on evaluation. Local: spy on the exact scorer boundary."""
    write_json(tmp_path / "data/lock.json", {"calibration": ["c"], "train": ["t"], "heldout": ["h"]})
    write_json(tmp_path / "data/inventory.json", [{"id": identity, "turns": [identity], "label": 0,
               "source": "SECRET_SOURCE", "reward": 1, "judge_metadata": "SECRET_LABEL"} for identity in "cth"])
    observed = []

    class Scorer:
        """Local: model substitute records only normalized turn inputs."""

        def score(self, turns):
            """Local: heldout inference must observe an already frozen threshold."""
            if turns == ["h"]:
                assert read_json(tmp_path / "judge/threshold.json")["threshold"] == 0.2
            observed.append(turns)
            return {"score": 0.2}

        def close(self):
            """Local: no resources are allocated by this inference substitute."""
            pass

    validate_scores(tmp_path, "judge", {"identity": "fixed"}, lambda root: Scorer())
    assert observed == [["c"], ["t"], ["h"]]
    validate_scores(tmp_path, "judge", {"identity": "fixed"}, lambda root: (_ for _ in ()).throw(AssertionError("cache miss")))
    assert len(observed) == 3
