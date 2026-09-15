"""Faz C1: StyleGAN-XL'e HİÇ bağımlı olmayan minimal mapping modülü.

`dnnlib`, `torch_utils`, `persistence` İÇERMEZ — sadece `torch.nn`.

Matematik, `autonomousvision/stylegan-xl`'in GERÇEK kaynak kodundan
(`training/networks_stylegan3.py`: `FullyConnectedLayer` + `MappingNetwork`,
`torch_utils/ops/bias_act.py`: `activation_funcs['lrelu']`) birebir
alındı (WebFetch ile doğrulandı, varsayım değil). Çıkan mimari sabitleri
(embed_proj 320→64: 20544 parametre, fc0 128→512: 66048, fc1 512→512:
262656, embed 1000×320: 320000) Faz A'nın gerçek `inventory.json`
sayılarıyla birebir örtüşüyor — bağımsız bir çapraz doğrulama.

Doğrulanmış matematik (activation='lrelu' olan HER katman için):
    w = weight * weight_gain          # weight_gain = lr_multiplier / sqrt(in_features)
    x = x @ w.T
    b = bias * bias_gain              # bias_gain = lr_multiplier
    x = leaky_relu(x + b, negative_slope=0.2) * sqrt(2)   # bias_act 'lrelu': def_alpha=0.2, def_gain=sqrt(2)

MappingNetwork.forward (truncation_psi=1.0, truncation_cutoff=None,
update_emas=False — Faz C0'ın gerçek çağrı kwargs'ı):
    x = normalize_2nd_moment(z)
    y = normalize_2nd_moment(embed_proj(embed(c.argmax(dim=1))))
    x = cat([x, y], dim=1)
    x = fc1(fc0(x))
    # broadcast (unsqueeze+repeat num_ws) ve truncation burada YOK —
    # bu modül (k, w_dim) döner, çağıran taraf gerekirse kendi broadcast
    # eder. update_emas sadece bir buffer'ı günceller, x'i etkilemez.

Mimari sabitler (şekilden türetilemez, doğrulanmış kaynaktan alınır):
    embed_proj: lr_multiplier = 1.0   (MappingNetwork.__init__'te BELİRTİLMEMİŞ
                                        → FullyConnectedLayer sınıf varsayılanı)
    fc0, fc1:   lr_multiplier = 0.01  (MappingNetwork.__init__'te AÇIKÇA verilir)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Doğrulanmış kaynaktan (bkz. modül docstring'i) — şekilden türetilmez.
FC_LR_MULTIPLIER = 0.01
EMBED_PROJ_LR_MULTIPLIER = 1.0
LRELU_NEGATIVE_SLOPE = 0.2
LRELU_GAIN = math.sqrt(2.0)
NORMALIZE_EPS = 1e-8

IGNORED_SHARD_KEYS = ("mapping.w_avg",)


def normalize_2nd_moment(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return x * (x.square().mean(dim=dim, keepdim=True) + NORMALIZE_EPS).rsqrt()


class EqualizedLinearLrelu(nn.Module):
    """StyleGAN-XL'in `FullyConnectedLayer(activation='lrelu')`'ının
    forward'ı ile BİREBİR aynı. Ağırlıklar rastgele başlatılmaz — sıfırla
    başlatılır, dışarıdan (shard'dan) yüklenmesi beklenir."""

    def __init__(self, in_features: int, out_features: int, lr_multiplier: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.weight_gain = lr_multiplier / math.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.weight_gain
        b = self.bias * self.bias_gain
        x = x.matmul(w.t())
        x = F.leaky_relu(x + b, negative_slope=LRELU_NEGATIVE_SLOPE) * LRELU_GAIN
        return x


class MinimalMappingNetwork(nn.Module):
    """StyleGAN-XL'e bağımlı olmayan, yeniden kurulmuş mapping ağı.

    `forward(z, c) -> (k, w_dim)` döner — `(k, num_ws, w_dim)` DEĞİL.
    num_ws boyunca broadcast salt bir tekrar olduğundan (bkz. modül
    docstring'i) ZK devresine dahil edilmiyor; referansla karşılaştırma
    yapan taraf referansın `w[:, 0, :]` dilimini almalı.
    """

    def __init__(self, z_dim: int, embed_num: int, embed_dim: int, w_dim: int):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.embed = nn.Embedding(embed_num, embed_dim)
        self.embed_proj = EqualizedLinearLrelu(embed_dim, z_dim, lr_multiplier=EMBED_PROJ_LR_MULTIPLIER)
        self.fc0 = EqualizedLinearLrelu(2 * z_dim, w_dim, lr_multiplier=FC_LR_MULTIPLIER)
        self.fc1 = EqualizedLinearLrelu(w_dim, w_dim, lr_multiplier=FC_LR_MULTIPLIER)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = normalize_2nd_moment(z.to(torch.float32))
        indices = c.argmax(dim=1)
        y = self.embed_proj(self.embed(indices))
        y = normalize_2nd_moment(y)
        x = torch.cat([x, y], dim=1)
        x = self.fc0(x)
        x = self.fc1(x)
        return x


def infer_dims_from_shard(shard: dict) -> dict:
    """Shard'daki tensör şekillerinden boyutları çıkarır, iç tutarlılığı
    doğrular (fc0'ın girdi boyutu 2*z_dim mi, embed_proj'un çıktısı
    z_dim mi) — tutmazsa net `ValueError`."""
    required = ("mapping.embed.weight", "mapping.embed_proj.weight", "mapping.fc0.weight", "mapping.fc1.weight")
    missing = [k for k in required if k not in shard]
    if missing:
        raise ValueError(f"Shard'da beklenen anahtarlar eksik: {missing}. Bulunan anahtarlar: {sorted(shard.keys())}")

    embed_num, embed_dim = shard["mapping.embed.weight"].shape
    embed_proj_out, embed_proj_in = shard["mapping.embed_proj.weight"].shape
    fc0_out, fc0_in = shard["mapping.fc0.weight"].shape
    fc1_out, fc1_in = shard["mapping.fc1.weight"].shape

    if embed_proj_in != embed_dim:
        raise ValueError(
            f"embed_proj girdi boyutu ({embed_proj_in}) embed_dim ({embed_dim}) ile uyuşmuyor."
        )
    z_dim = embed_proj_out
    if fc0_in != 2 * z_dim:
        raise ValueError(
            f"fc0 girdi boyutu ({fc0_in}) 2*z_dim ({2 * z_dim}) ile uyuşmuyor — "
            f"z_dim embed_proj'un çıktı boyutundan ({z_dim}) çıkarıldı."
        )
    w_dim = fc0_out
    if fc1_in != w_dim or fc1_out != w_dim:
        raise ValueError(f"fc1 şekli ({fc1_out},{fc1_in}) w_dim ({w_dim}) ile uyuşmuyor.")

    return {"z_dim": z_dim, "embed_num": embed_num, "embed_dim": embed_dim, "w_dim": w_dim}


def build_mapping_network_from_shard(shard: dict) -> MinimalMappingNetwork:
    dims = infer_dims_from_shard(shard)
    print(f"[rebuild_mapping] Çıkarılan boyutlar: {dims}")

    model = MinimalMappingNetwork(
        z_dim=dims["z_dim"], embed_num=dims["embed_num"], embed_dim=dims["embed_dim"], w_dim=dims["w_dim"]
    )

    key_map = {
        "mapping.embed.weight": (model.embed, "weight"),
        "mapping.embed_proj.weight": (model.embed_proj, "weight"),
        "mapping.embed_proj.bias": (model.embed_proj, "bias"),
        "mapping.fc0.weight": (model.fc0, "weight"),
        "mapping.fc0.bias": (model.fc0, "bias"),
        "mapping.fc1.weight": (model.fc1, "weight"),
        "mapping.fc1.bias": (model.fc1, "bias"),
    }

    loaded = []
    with torch.no_grad():
        for shard_key, (module, param_name) in key_map.items():
            if shard_key not in shard:
                raise ValueError(f"Shard'da '{shard_key}' yok — modül yüklenemiyor.")
            tensor = shard[shard_key]
            param = getattr(module, param_name)
            if tensor.shape != param.shape:
                raise ValueError(f"'{shard_key}' şekli {tuple(tensor.shape)}, beklenen {tuple(param.shape)}.")
            param.copy_(tensor.detach().to(torch.float32))
            loaded.append(shard_key)

    ignored = [k for k in IGNORED_SHARD_KEYS if k in shard]
    unrecognized = [k for k in shard.keys() if k not in loaded and k not in IGNORED_SHARD_KEYS]

    print(f"[rebuild_mapping] Yüklenen anahtarlar ({len(loaded)}): {loaded}")
    if ignored:
        print(f"[rebuild_mapping] Bilinçli olarak yoksayılan anahtarlar (truncation için, kullanılmıyor): {ignored}")
    if unrecognized:
        print(f"[rebuild_mapping] UYARI: shard'da tanınmayan/kullanılmayan anahtarlar var: {unrecognized}")

    return model
