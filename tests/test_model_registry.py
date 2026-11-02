"""Tests for model_registry.py download guard, hash verification, and caching."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from dcp.analysis.model_registry import (
    MODEL_REGISTRY,
    _verify_hash,
    clear_cache,
    get_model_cache_dir,
    get_model_path,
    list_cached_models,
)


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="Unknown model: nonexistent"):
        get_model_path("nonexistent")


def test_download_guard_blocks_unauthorized_download(tmp_path, monkeypatch):
    """Verify that get_model_path raises RuntimeError if download is not explicitly authorized."""
    monkeypatch.setattr("dcp.analysis.model_registry.get_model_cache_dir", lambda: tmp_path)
    monkeypatch.delenv("DCP_ALLOW_MODEL_DOWNLOAD", raising=False)

    with pytest.raises(RuntimeError, match="Automatic network model downloads are disabled by policy"):
        get_model_path("beat_tracker_v1", allow_download=False)


def test_cached_model_returned_without_download(tmp_path, monkeypatch):
    """When a valid cached model file exists, it is returned without triggering download."""
    monkeypatch.setattr("dcp.analysis.model_registry.get_model_cache_dir", lambda: tmp_path)
    monkeypatch.delenv("DCP_ALLOW_MODEL_DOWNLOAD", raising=False)

    fake_model_file = tmp_path / "beat_tracker_v1.pt"
    fake_model_file.write_bytes(b"dummy_weights_content")

    # With placeholder hash, _verify_hash succeeds
    path = get_model_path("beat_tracker_v1", allow_download=False)
    assert path == fake_model_file
    assert path.exists()


def test_corrupted_cached_model_removed_and_blocked(tmp_path, monkeypatch):
    """When a cached file has an invalid hash, it is deleted and download guard fires."""
    monkeypatch.setattr("dcp.analysis.model_registry.get_model_cache_dir", lambda: tmp_path)
    monkeypatch.delenv("DCP_ALLOW_MODEL_DOWNLOAD", raising=False)

    fake_model_file = tmp_path / "beat_tracker_v1.pt"
    fake_model_file.write_bytes(b"corrupted_content")

    # Temporarily set an explicit sha256 that does not match
    orig_sha = MODEL_REGISTRY["beat_tracker_v1"]["sha256"]
    try:
        MODEL_REGISTRY["beat_tracker_v1"]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        with pytest.raises(RuntimeError, match="Automatic network model downloads are disabled by policy"):
            get_model_path("beat_tracker_v1", allow_download=False)

        # File was deleted due to corruption
        assert not fake_model_file.exists()
    finally:
        MODEL_REGISTRY["beat_tracker_v1"]["sha256"] = orig_sha


def test_allow_download_flag_triggers_download(tmp_path, monkeypatch):
    """When allow_download=True, download is invoked."""
    monkeypatch.setattr("dcp.analysis.model_registry.get_model_cache_dir", lambda: tmp_path)

    called = False

    def fake_download(name, dest, info):
        nonlocal called
        called = True
        dest.write_bytes(b"downloaded_model")

    monkeypatch.setattr("dcp.analysis.model_registry._download_model", fake_download)

    path = get_model_path("beat_tracker_v1", allow_download=True)
    assert called is True
    assert path.exists()
    assert path.read_bytes() == b"downloaded_model"


def test_env_var_triggers_download(tmp_path, monkeypatch):
    """When DCP_ALLOW_MODEL_DOWNLOAD=1, download is authorized."""
    monkeypatch.setattr("dcp.analysis.model_registry.get_model_cache_dir", lambda: tmp_path)
    monkeypatch.setenv("DCP_ALLOW_MODEL_DOWNLOAD", "1")

    called = False

    def fake_download(name, dest, info):
        nonlocal called
        called = True
        dest.write_bytes(b"downloaded_by_env")

    monkeypatch.setattr("dcp.analysis.model_registry._download_model", fake_download)

    path = get_model_path("beat_tracker_v1", allow_download=False)
    assert called is True
    assert path.exists()
    assert path.read_bytes() == b"downloaded_by_env"


def test_verify_hash_computes_sha256(tmp_path):
    """_verify_hash returns True for correct hash and False for incorrect."""
    test_file = tmp_path / "test.bin"
    content = b"hello dcp audio"
    test_file.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert _verify_hash(test_file, expected) is True
    assert _verify_hash(test_file, "wrong_hash") is False
    assert _verify_hash(test_file, "placeholder_hash_will_be_replaced") is True


def test_list_cached_models_and_clear_cache(tmp_path, monkeypatch):
    """list_cached_models accurately reflects presence and clear_cache purges it."""
    monkeypatch.setattr("dcp.analysis.model_registry.get_model_cache_dir", lambda: tmp_path)

    items = list_cached_models()
    assert len(items) == len(MODEL_REGISTRY)
    assert all(not item["cached"] for item in items)

    # Place a file for beat_tracker_v1
    (tmp_path / "beat_tracker_v1.pt").write_bytes(b"x" * 1024)
    items_after = list_cached_models()
    beat_info = next(i for i in items_after if i["name"] == "beat_tracker_v1")
    assert beat_info["cached"] is True
    assert beat_info["size_mb"] is not None

    clear_cache()
    items_cleared = list_cached_models()
    assert all(not item["cached"] for item in items_cleared)
