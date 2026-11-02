"""Model registry and downloader for neural analysis components.

Handles downloading, caching, and loading of pretrained model weights
for neural beat tracking, key detection, and structure segmentation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


# Model definitions with download URLs and expected SHA256 hashes
# These are placeholder URLs - in production, these would point to actual model artifacts
MODEL_REGISTRY = {
    "beat_tracker_v1": {
        "url": "https://github.com/dcp-project/models/releases/download/v0.1/beat_tracker_v1.pt",
        "sha256": "placeholder_hash_will_be_replaced",
        "filename": "beat_tracker_v1.pt",
        "description": "TCN beat & downbeat tracker (64 mel bands, 100 fps)",
    },
    "key_detector_v1": {
        "url": "https://github.com/dcp-project/models/releases/download/v0.1/key_detector_v1.pt",
        "sha256": "placeholder_hash_will_be_replaced",
        "filename": "key_detector_v1.pt",
        "description": "CNN key detector on CQT chromagram (24 classes)",
    },
    "structure_segmenter_v1": {
        "url": "https://github.com/dcp-project/models/releases/download/v0.1/structure_segmenter_v1.pt",
        "sha256": "placeholder_hash_will_be_replaced",
        "filename": "structure_segmenter_v1.pt",
        "description": "BiLSTM structure segmenter (intro/verse/chorus/bridge/outro/drop)",
    },
}


def get_model_cache_dir() -> Path:
    """Get the model cache directory, creating it if needed."""
    cache_dir = Path.home() / ".dcp" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_path(model_name: str, *, allow_download: bool = False) -> Path:
    """Get the local path for a model, downloading only if explicitly authorized."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    cache_dir = get_model_cache_dir()
    model_info = MODEL_REGISTRY[model_name]
    model_path = cache_dir / model_info["filename"]

    if model_path.exists():
        # Verify hash
        if _verify_hash(model_path, model_info["sha256"]):
            return model_path
        else:
            # Hash mismatch, remove corrupted file
            model_path.unlink()

    # Enforce authorization policy: no automatic network downloads without consent
    permitted = allow_download or os.environ.get("DCP_ALLOW_MODEL_DOWNLOAD", "").lower() in ("1", "true", "yes")
    if not permitted:
        raise RuntimeError(
            f"Pretrained weights for model '{model_name}' are not present locally at {model_path}.\n"
            "Automatic network model downloads are disabled by policy.\n"
            "To authorize downloading, set the environment variable DCP_ALLOW_MODEL_DOWNLOAD=1 "
            "or pass allow_download=True."
        )

    # Download model
    _download_model(model_name, model_path, model_info)

    # Verify after download
    if not _verify_hash(model_path, model_info["sha256"]):
        raise RuntimeError(f"Downloaded model {model_name} failed hash verification")

    return model_path


def _verify_hash(file_path: Path, expected_hash: str) -> bool:
    """Verify SHA256 hash of a file."""
    if expected_hash == "placeholder_hash_will_be_replaced":
        # Skip verification for placeholder hashes (development mode)
        return True

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash


def _download_model(model_name: str, dest_path: Path, model_info: dict) -> None:
    """Download a model from the registry URL."""
    url = model_info["url"]
    print(f"Downloading {model_info['description']} ({model_name})...", file=sys.stderr)
    print(f"From: {url}", file=sys.stderr)
    print(f"To: {dest_path}", file=sys.stderr)

    try:
        req = Request(url, headers={"User-Agent": "DCP/0.1"})
        with urlopen(req, timeout=300) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded * 100 / total_size
                        print(f"\r  Progress: {pct:.1f}% ({downloaded}/{total_size} bytes)", end="", file=sys.stderr)
            print(file=sys.stderr)  # New line after progress
    except (URLError, HTTPError) as exc:
        # Clean up partial download
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Failed to download model {model_name}: {exc}") from exc


def load_model(model_name: str, device: str = "cpu", *, allow_download: bool = False, **kwargs):
    """Load a model from cache, downloading if necessary.

    The model class must be imported separately and instantiated with the
    loaded state dict. This function handles the weight loading only.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for loading models. Install with: pip install torch")

    import torch

    model_path = get_model_path(model_name, allow_download=allow_download)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    return state_dict


def list_cached_models() -> list[dict]:
    """List all cached models with their status."""
    cache_dir = get_model_cache_dir()
    result = []
    for name, info in MODEL_REGISTRY.items():
        path = cache_dir / info["filename"]
        result.append({
            "name": name,
            "description": info["description"],
            "cached": path.exists(),
            "path": str(path) if path.exists() else None,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None,
        })
    return result


def clear_cache() -> None:
    """Remove all cached models."""
    cache_dir = get_model_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)


__all__ = [
    "MODEL_REGISTRY",
    "get_model_cache_dir",
    "get_model_path",
    "load_model",
    "list_cached_models",
    "clear_cache",
]