from fl.module_tree import (
    describe_module_tree,
    detect_candidate_submodules,
    extract_known_attrs,
)
from tests.fixtures import FakeMappingNet


def test_describe_module_tree_lists_all_submodules_with_param_counts():
    net = FakeMappingNet()
    info = describe_module_tree(net)
    names = {entry["name"] for entry in info}

    assert "<root>" in names
    assert "mapping" in names
    assert "mapping.0" in names
    assert "embed" in names

    embed_entry = next(e for e in info if e["name"] == "embed")
    assert embed_entry["type"] == "Embedding"
    assert embed_entry["num_params"] == net.embed.weight.numel()


def test_detect_candidate_submodules_finds_mapping_and_embedding():
    net = FakeMappingNet()
    matches = detect_candidate_submodules(net)

    assert "mapping" in matches["mapping"]
    assert "embed" in matches["embedding"]
    assert "synthesis_stub" not in matches["mapping"]
    assert "synthesis_stub" not in matches["embedding"]


def test_detect_candidate_submodules_returns_empty_when_no_match():
    net = FakeMappingNet()
    matches = detect_candidate_submodules(net, keyword_groups={"nope": ["doesnotexist"]})
    assert matches == {"nope": []}


def test_extract_known_attrs_reports_found_and_missing():
    net = FakeMappingNet()
    result = extract_known_attrs(net, names=("z_dim", "c_dim", "w_dim", "num_ws", "does_not_exist"))

    assert result["found"] == {"z_dim": 8, "c_dim": 3, "w_dim": 8, "num_ws": 2}
    assert result["bulunamadi"] == ["does_not_exist"]
