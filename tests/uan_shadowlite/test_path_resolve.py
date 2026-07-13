"""CPU-only unit tests for roto.assets.path_resolve.

No Isaac Sim / isaaclab import anywhere in this file or in path_resolve.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "roto", "assets"))

import path_resolve  # noqa: E402
from path_resolve import resolve_path  # noqa: E402


def test_returns_path_unchanged_when_it_already_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(path_resolve, "KNOWN_ROTO_ROOTS", [str(tmp_path)])
    existing = tmp_path / "somefile.txt"
    existing.write_text("x")
    assert resolve_path(str(existing)) == str(existing)


def test_falls_back_to_the_other_known_root(tmp_path, monkeypatch):
    root_a = tmp_path / "machine_a" / "roto"
    root_b = tmp_path / "machine_b" / "roto"
    (root_b / "data").mkdir(parents=True)
    (root_b / "data" / "file.npz").write_text("x")
    monkeypatch.setattr(path_resolve, "KNOWN_ROTO_ROOTS", [str(root_a), str(root_b)])

    configured = str(root_a / "data" / "file.npz")  # written against root_a, only exists under root_b
    assert resolve_path(configured) == str(root_b / "data" / "file.npz")


def test_raises_with_every_candidate_listed_when_none_exist(tmp_path, monkeypatch):
    root_a = tmp_path / "machine_a" / "roto"
    root_b = tmp_path / "machine_b" / "roto"
    monkeypatch.setattr(path_resolve, "KNOWN_ROTO_ROOTS", [str(root_a), str(root_b)])

    configured = str(root_a / "data" / "missing.npz")
    with pytest.raises(FileNotFoundError) as exc:
        resolve_path(configured)
    assert str(root_a / "data" / "missing.npz") in str(exc.value)
    assert str(root_b / "data" / "missing.npz") in str(exc.value)


def test_path_not_under_any_known_root_raises_with_just_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(path_resolve, "KNOWN_ROTO_ROOTS", [str(tmp_path / "some_root")])
    unrelated = str(tmp_path / "elsewhere" / "file.txt")
    with pytest.raises(FileNotFoundError) as exc:
        resolve_path(unrelated)
    assert str(exc.value).count("elsewhere") == 1
