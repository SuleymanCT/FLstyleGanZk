"""Faz B'nin oyuncak modeli: küçük bir MLP (32 -> 32 -> 8).

StyleGAN'a bağımlı değildir; ezkl uçtan uca boru hattını (ONNX ->
derleme -> ispat -> Solidity verifier -> zincir) tek, ucuz bir modelle
kanıtlamak içindir. Gerçek mapping devresi Faz C'de ayrı yazılacak.
"""

from __future__ import annotations

import torch
import torch.nn as nn

IN_FEATURES = 32
HIDDEN_FEATURES = 32
OUT_FEATURES = 8


class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN_FEATURES)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_FEATURES, OUT_FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def make_example_input(batch_size: int = 1) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch_size, IN_FEATURES)
