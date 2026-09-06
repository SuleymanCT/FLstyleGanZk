"""StyleGAN'a bağımlı olmayan, saf `torch.nn.Module` gezinti yardımcıları.

Bu modül hiçbir StyleGAN-XL/dnnlib bağımlılığı içermez, bu yüzden
sentetik (Linear katmanlardan oluşan) modüllerle tam test edilebilir.
Amaç, `G_ema`'nın gerçek modül ağacı hakkında hiçbir varsayımda
bulunmadan, ne bulunduğunu olduğu gibi raporlamaktır.
"""

from __future__ import annotations

DEFAULT_KEYWORD_GROUPS = {
    "mapping": ["mapping", "style"],
    "embedding": ["embed", "class_embed", "label"],
}


def describe_module_tree(module) -> list[dict]:
    result = []
    for name, submodule in module.named_modules():
        num_params = sum(p.numel() for p in submodule.parameters(recurse=False))
        result.append(
            {
                "name": name if name else "<root>",
                "type": type(submodule).__name__,
                "num_params": num_params,
            }
        )
    return result


def detect_candidate_submodules(
    module, keyword_groups: dict[str, list[str]] | None = None
) -> dict[str, list[str]]:
    groups = keyword_groups or DEFAULT_KEYWORD_GROUPS
    matches: dict[str, list[str]] = {group: [] for group in groups}

    for name, _submodule in module.named_modules():
        if not name:
            continue
        lowered = name.lower()
        for group, keywords in groups.items():
            if any(keyword in lowered for keyword in keywords):
                matches[group].append(name)

    return matches


def extract_known_attrs(module, names: tuple[str, ...] = ("z_dim", "c_dim", "w_dim", "num_ws")) -> dict:
    found = {}
    missing = []
    for name in names:
        if hasattr(module, name):
            found[name] = getattr(module, name)
        else:
            missing.append(name)

    return {"found": found, "bulunamadi": missing}
