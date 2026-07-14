"""CPU-only unit tests for roto.genan.config_utils."""

import os
import sys

_GENAN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "genan")
sys.path.insert(0, _GENAN_DIR)

from config_utils import deep_update, load_config  # noqa: E402


def test_deep_update_merges_nested_dicts():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    overlay = {"nested": {"y": 20, "z": 30}}
    result = deep_update(base, overlay)
    assert result is base  # merges in place and returns it
    assert base == {"a": 1, "nested": {"x": 1, "y": 20, "z": 30}}


def test_deep_update_overwrites_non_dict_values_including_lists():
    base = {"paths": ["a", "b"], "n": 1}
    overlay = {"paths": ["c"], "n": {"now_a_dict": True}}
    deep_update(base, overlay)
    assert base["paths"] == ["c"]
    assert base["n"] == {"now_a_dict": True}


def test_load_config_without_overlay(tmp_path):
    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text("a: 1\nnested:\n  x: 1\n")
    cfg = load_config(str(cfg_path))
    assert cfg == {"a": 1, "nested": {"x": 1}}


def test_load_config_with_overlay(tmp_path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text("a: 1\nnested:\n  x: 1\n  y: 2\n")
    overlay_path = tmp_path / "overlay.yaml"
    overlay_path.write_text("nested:\n  y: 20\n")
    cfg = load_config(str(base_path), str(overlay_path))
    assert cfg == {"a": 1, "nested": {"x": 1, "y": 20}}
