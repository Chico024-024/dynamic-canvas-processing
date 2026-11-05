"""Multi-factor deterministic cache invalidation for precomputed transitions.

This module implements the deterministic cache key specification from the
DCP Advanced architecture:

    Cache Key = SHA256(hash(A) + hash(B) + params + engine_version + config_hash + stem_version)

Stored artifacts:
- transition.wav (rendered audio buffer)
- plan.json (exact execution plan)
- validation.json (objective metrics: peak, rms, silence, discontinuity checks)
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ..audio_io import save_audio
from ..config import DCPConfig
from ..models import AudioBuffer, StemSet
from ..transitions import TransitionParams, TransitionPlan
from ..validation import validate_audio
from .context import TransitionContext


# Engine version for cache invalidation on engine updates
ENGINE_VERSION = "0.1.0"


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for the transition cache."""

    cache_root: Path = Path(".dcp_cache") / "transitions"
    max_entries: int = 1000
    max_size_mb: int = 2048  # 2 GB


@dataclass(frozen=True)
class CacheEntry:
    """A cached transition entry."""

    cache_key: str
    transition_wav: Path
    plan_json: Path
    validation_json: Path
    metadata: dict[str, Any]

    @property
    def size_bytes(self) -> int:
        """Total size of cached artifacts in bytes."""
        total = 0
        for path in [self.transition_wav, self.plan_json, self.validation_json]:
            if path.exists():
                total += path.stat().st_size
        return total


def _compute_audio_hash(buffer: AudioBuffer) -> str:
    """Compute SHA256 hash of audio buffer content."""
    # Use raw samples for content-based hashing
    samples = buffer.samples
    # Normalize for consistent hashing regardless of channel layout
    hasher = hashlib.sha256()
    hasher.update(samples.tobytes())
    return hasher.hexdigest()


def _compute_params_hash(params: Optional[TransitionParams]) -> str:
    """Compute hash of transition parameters."""
    if params is None:
        return "none"
    hasher = hashlib.sha256()
    hasher.update(json.dumps(params.to_dict(), sort_keys=True).encode())
    return hasher.hexdigest()


def _compute_config_hash(config: DCPConfig) -> str:
    """Compute hash of relevant DCP configuration."""
    # Only hash parameters that affect rendering output
    relevant = {
        "analysis": {
            "stft_window": config.analysis.stft_window,
            "stft_hop": config.analysis.stft_hop,
            "silence_dbfs": config.analysis.silence_dbfs,
            "silence_window_s": config.analysis.silence_window_s,
            "tempo_min_bpm": config.analysis.tempo_min_bpm,
            "tempo_max_bpm": config.analysis.tempo_max_bpm,
            "tempo_confidence_threshold": config.analysis.tempo_confidence_threshold,
            "tempo_smoothing_frames": config.analysis.tempo_smoothing_frames,
        },
        "transition": {
            "default_duration_s": config.transition.default_duration_s,
            "min_duration_s": config.transition.min_duration_s,
            "max_automation_gain_db": config.transition.max_automation_gain_db,
            "fade_floor_db": config.transition.fade_floor_db,
            "tempo_match_tolerance": config.transition.tempo_match_tolerance,
            "rhythmic_beats_per_bar": config.transition.rhythmic_beats_per_bar,
            "atmospheric_scale": config.transition.atmospheric_scale,
            "atmospheric_min_duration_s": config.transition.atmospheric_min_duration_s,
        },
        "render": {
            "prevent_clipping": config.render.prevent_clipping,
            "peak_ceiling": config.render.peak_ceiling,
            "wav_subtype": config.render.wav_subtype,
            "effect_chunk_samples": config.render.effect_chunk_samples,
        },
        "validation": {
            "max_gap_s": config.validation_max_gap_s,
            "click_threshold": config.validation_click_threshold,
            "silence_dbfs": config.validation_silence_dbfs,
        },
    }
    hasher = hashlib.sha256()
    hasher.update(json.dumps(relevant, sort_keys=True).encode())
    return hasher.hexdigest()


def _compute_stem_version_hash(
    stems_a: Optional[dict[str, AudioBuffer]],
    stems_b: Optional[dict[str, AudioBuffer]],
) -> str:
    """Compute hash of stem content and availability."""
    hasher = hashlib.sha256()

    if stems_a:
        for name in sorted(stems_a.keys()):
            hasher.update(name.encode())
            hasher.update(_compute_audio_hash(stems_a[name]).encode())
    else:
        hasher.update(b"no_stems_a")

    if stems_b:
        for name in sorted(stems_b.keys()):
            hasher.update(name.encode())
            hasher.update(_compute_audio_hash(stems_b[name]).encode())
    else:
        hasher.update(b"no_stems_b")

    return hasher.hexdigest()


def build_cache_key(
    song_a: AudioBuffer,
    song_b: AudioBuffer,
    stems_a: Optional[dict[str, AudioBuffer]],
    stems_b: Optional[dict[str, AudioBuffer]],
    params: Optional[TransitionParams],
    config: DCPConfig,
) -> str:
    """Build the complete cache key from all factors.

    Cache Key = SHA256(
        hash(A) + hash(B) + params + engine_version + config_hash + stem_version
    )
    """
    hash_a = _compute_audio_hash(song_a)
    hash_b = _compute_audio_hash(song_b)
    params_hash = _compute_params_hash(params)
    config_hash = _compute_config_hash(config)
    stem_version = _compute_stem_version_hash(stems_a, stems_b)

    parts = [
        hash_a,
        hash_b,
        params_hash,
        ENGINE_VERSION,
        config_hash,
        stem_version,
    ]

    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode())
    return hasher.hexdigest()


class TransitionCache:
    """Deterministic cache for precomputed transitions."""

    def __init__(self, config: CacheConfig = CacheConfig()):
        self.config = config
        self.config.cache_root.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, cache_key: str) -> Path:
        """Get the directory path for a cache entry."""
        return self.config.cache_root / cache_key[:2] / cache_key

    def _artifact_paths(self, cache_key: str) -> tuple[Path, Path, Path]:
        """Get paths for the three cached artifacts."""
        entry_dir = self._entry_path(cache_key)
        return (
            entry_dir / "transition.wav",
            entry_dir / "plan.json",
            entry_dir / "validation.json",
        )

    def get(self, cache_key: str) -> Optional[CacheEntry]:
        """Retrieve a cache entry if it exists and is valid."""
        wav_path, plan_path, val_path = self._artifact_paths(cache_key)

        if not (wav_path.exists() and plan_path.exists() and val_path.exists()):
            return None

        # Load metadata
        try:
            with open(plan_path, "r") as f:
                plan_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        # Verify the cached plan matches the cache key
        # (defense against corruption)
        cached_key = plan_data.get("metadata", {}).get("cache_key")
        if cached_key != cache_key:
            return None

        return CacheEntry(
            cache_key=cache_key,
            transition_wav=wav_path,
            plan_json=plan_path,
            validation_json=val_path,
            metadata=plan_data.get("metadata", {}),
        )

    @staticmethod
    def _dict_to_stemset(
        stems: Optional[dict[str, AudioBuffer]],
        source_label: str = "",
    ) -> Optional[StemSet]:
        """Convert a dictionary of stems to a StemSet object."""
        if stems is None:
            return None
        return StemSet(stems=stems, source_label=source_label)

    def put(
        self,
        cache_key: str,
        song_a: AudioBuffer,
        song_b: AudioBuffer,
        stems_a: Optional[dict[str, AudioBuffer]],
        stems_b: Optional[dict[str, AudioBuffer]],
        plan: TransitionPlan,
        plan_params: Optional[TransitionParams],
        config: DCPConfig,
    ) -> CacheEntry:
        """
        Render and cache a transition.

        Returns the CacheEntry with paths to the cached artifacts.
        """
        entry_dir = self._entry_path(cache_key)
        entry_dir.mkdir(parents=True, exist_ok=True)

        wav_path, plan_path, val_path = self._artifact_paths(cache_key)

        # Convert dict stems to StemSet objects
        stems_a_ss = self._dict_to_stemset(stems_a, "A")
        stems_b_ss = self._dict_to_stemset(stems_b, "B")

        # Build transition context for rendering
        context = TransitionContext(
            song_a=song_a,
            song_b=song_b,
            stems_a=stems_a_ss,
            stems_b=stems_b_ss,
            analysis_a=None,  # Will be filled by render_plan via analysis
            analysis_b=None,
            requested_duration_s=plan.duration_s,
            requested_strategy=plan.strategy,
            config=config,
        )

        # Render the transition
        from ..rendering.stem_mix import render_plan

        render_result = render_plan(
            song_a,
            song_b,
            plan,
            stems_a=stems_a_ss,
            stems_b=stems_b_ss,
            config=config,
        )

        # Save the rendered WAV
        save_audio(
            wav_path,
            render_result.audio,
            subtype=config.render.wav_subtype,
            allow_clipping=not config.render.prevent_clipping,
        )

        # Validate the rendered audio
        from ..rendering.stem_mix import rendered_duration_s

        expected_duration = rendered_duration_s(song_a, song_b, plan)
        validation_report = validate_audio(
            render_result.audio,
            expected_duration_s=expected_duration,
            expected_sample_rate=song_a.sample_rate,
            expected_channels=song_a.channels,
        )

        # Create metadata with cache key
        plan_dict = plan.to_dict()
        if "metadata" not in plan_dict:
            plan_dict["metadata"] = {}
        plan_dict["metadata"]["cache_key"] = cache_key

        # Write plan JSON
        with open(plan_path, "w") as f:
            json.dump(plan_dict, f, indent=2)

        # Write validation JSON
        validation_dict = validation_report.to_dict()
        validation_dict["cache_key"] = cache_key
        with open(val_path, "w") as f:
            json.dump(validation_dict, f, indent=2)

        return CacheEntry(
            cache_key=cache_key,
            transition_wav=wav_path,
            plan_json=plan_path,
            validation_json=val_path,
            metadata=plan_dict.get("metadata", {}),
        )

    def delete(self, cache_key: str) -> bool:
        """Delete a cache entry."""
        entry_dir = self._entry_path(cache_key)
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
            return True
        return False

    def clear(self) -> int:
        """Clear all cache entries. Returns number of entries deleted."""
        count = 0
        if self.config.cache_root.exists():
            for subdir in self.config.cache_root.iterdir():
                if subdir.is_dir():
                    for entry in subdir.iterdir():
                        if entry.is_dir():
                            shutil.rmtree(entry)
                            count += 1
        return count

    def list_entries(self) -> list[CacheEntry]:
        """List all cached entries."""
        entries = []
        if self.config.cache_root.exists():
            for subdir in self.config.cache_root.iterdir():
                if subdir.is_dir():
                    for entry in subdir.iterdir():
                        if entry.is_dir():
                            cache_key = entry.name
                            cached = self.get(cache_key)
                            if cached:
                                entries.append(cached)
        return entries

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        entries = self.list_entries()
        total_size = sum(e.size_bytes for e in entries)
        return {
            "entry_count": len(entries),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_root": str(self.config.cache_root),
        }


def get_cached_transition(
    cache: TransitionCache,
    song_a: AudioBuffer,
    song_b: AudioBuffer,
    stems_a: Optional[dict[str, AudioBuffer]],
    stems_b: Optional[dict[str, AudioBuffer]],
    plan: TransitionPlan,
    plan_params: Optional[TransitionParams],
    config: DCPConfig,
) -> Optional[str]:
    """
    Get cached transition path if available, otherwise return None.

    Returns the path to the cached transition.wav if cache hit,
    otherwise returns None (caller must render and cache).
    """
    cache_key = build_cache_key(
        song_a=song_a,
        song_b=song_b,
        stems_a=stems_a,
        stems_b=stems_b,
        params=plan_params,
        config=config,
    )

    entry = cache.get(cache_key)
    if entry and entry.transition_wav.exists():
        return str(entry.transition_wav)

    return None




__all__ = [
    "CacheConfig",
    "CacheEntry",
    "TransitionCache",
    "build_cache_key",
    "get_cached_transition",
    "ENGINE_VERSION",
]