"""Global: exact frozen checkpoint architecture. Local: no learned additions or refitting.

Adapted under MIT from cells 183/186/219 of the pinned farzingkh/reward-hacking notebook.
Copyright 2026 Bridget Atchison Nevel, Farzin Golkhosh, Jack McLellan;
the complete attribution/license is in LICENSES/reward-hacking-MIT.txt.
PyTorch API: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
"""
import math  # Local: sinusoidal positional scale; global: released embedding magnitudes.

import torch  # Local: tensors and strict state dictionaries.
from torch import nn  # Local: standard modules match all released key names.


def last_token_pool(hidden, mask):
    """Global: Qwen embedding recipe. Local: handle left or right padding like the notebook."""
    if mask[:, -1].all():  # Local: left-padded batches end on a real token.
        return hidden[:, -1]
    return hidden[torch.arange(hidden.size(0), device=hidden.device), mask.sum(1) - 1]


class TurnTransformerProbe(nn.Module):
    """Global: released four-layer classifier. Local: strict 248 x 1024 positional buffer."""

    def __init__(self):
        """Global: checkpoint-compatible names. Local: default dimensions fixed by release."""
        super().__init__()  # Local: register PyTorch module state.
        positions = torch.arange(248).unsqueeze(1)  # Local: 248 original turn slots.
        frequencies = torch.exp(torch.arange(0, 1024, 2) * (-math.log(10000.0) / 1024))
        encodings = torch.zeros(248, 1024)  # Local: alternating sine/cosine coordinates.
        encodings[:, 0::2] = torch.sin(positions * frequencies)  # Local: sine on even indices.
        encodings[:, 1::2] = torch.cos(positions * frequencies)  # Local: cosine on odd indices.
        self.register_buffer("positional_encodings", encodings / math.sqrt(1024))
        layer = nn.TransformerEncoderLayer(256, 8, 1024, 0.5, activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.linear = nn.Linear(1024, 256)  # Local: released embedding projection.
        self.classifier = nn.Linear(256, 1)  # Local: one logit, not a recalibrated probability.
        self.cls = nn.Parameter(torch.randn(256) * 0.02)  # Local: overwritten by checkpoint.

    def forward(self, turn_embeddings, key_padding_mask):
        """Global: notebook forward equivalence. Local: prepend unpositioned CLS then classify."""
        batch, turns, _ = turn_embeddings.shape  # Local: [B,T,1024] input contract.
        if not 0 < turns <= 248:  # Local: never silently alter checkpoint capacity.
            raise ValueError("Probe expects 1–248 turns")
        values = self.linear(turn_embeddings + self.positional_encodings[:turns])
        values = torch.cat([self.cls.view(1, 1, -1).expand(batch, -1, -1), values], dim=1)
        mask = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=values.device), key_padding_mask], dim=1)
        encoded = self.transformer_encoder(values, src_key_padding_mask=mask)
        return self.classifier(encoded[:, 0, :]).squeeze(-1)  # Local: sigmoid belongs to scoring phase.
