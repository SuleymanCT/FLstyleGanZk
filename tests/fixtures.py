"""Faz A testleri için StyleGAN-XL'e bağımlı olmayan sahte modüller.

Bunlar birim testi fixture'larıdır (CLAUDE.md madde 6'daki mock/dummy
yasağı deney SONUÇLARINA ilişkindir, birim testi altyapısına değil).
Gerçek StyleGAN-XL mapping ağının isimlendirmesine yakın ama ondan
bağımsız, sadece Linear/Embedding katmanlarından oluşan küçük bir ağ.
"""

import torch
import torch.nn as nn


class FakeMappingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_dim = 8
        self.c_dim = 3
        self.w_dim = 8
        self.num_ws = 2

        self.mapping = nn.Sequential(
            nn.Linear(self.z_dim, self.w_dim),
            nn.ReLU(),
            nn.Linear(self.w_dim, self.w_dim),
        )
        self.embed = nn.Embedding(self.c_dim, self.w_dim)
        self.synthesis_stub = nn.Linear(self.w_dim, 4)

    def forward(self, z, c):
        w = self.mapping(z) + self.embed(c.argmax(dim=-1))
        return self.synthesis_stub(w)


def make_fake_state_dict(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    net = FakeMappingNet()
    return {k: v.clone() for k, v in net.state_dict().items()}
