"""Tests for the multi-factor transition cache."""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

from dcp.models import AudioBuffer
from dcp.transitions.cache import (
    CacheConfig,
    CacheEntry,
    TransitionCache,
    build_cache_key,
    get_cached_transition,
    ENGINE_VERSION,
)
from dcp.transitions.plan import TransitionPlan, StemAutomation, GainPoint
from dcp.config import DEFAULT_CONFIG


def _make_mock_audio(duration_s: float = 30.0, sample_rate: int = 44100) -> AudioBuffer:
    """Create a mock audio buffer for testing."""
    frames = int(duration_s * sample_rate)
    samples = np.zeros((2, frames), dtype=np.float64)
    return AudioBuffer(samples=samples, sample_rate=sample_rate)


def _make_mock_plan() -> TransitionPlan:
    """Create a mock transition plan for testing."""
    return TransitionPlan(
        strategy="stem_level",
        requested_strategy="stem_level",
        duration_s=4.0,
        start_time_s=4.0,
        end_time_s=8.0,
        automations=(
            StemAutomation(
                source="a",
                stem="__mix__",
                curve="smoothstep",
                points=(GainPoint(time_s=0.0, gain_db=0.0), GainPoint(time_s=4.0, gain_db=-60.0)),
            ),
            StemAutomation(
                source="b",
                stem="__mix__",
                curve="smoothstep",
                points=(GainPoint(time_s=0.0, gain_db=-60.0), GainPoint(time_s=4.0, gain_db=0.0)),
            ),
        ),
        metadata={"test": True},
    )


class TestBuildCacheKey:
    """Tests for the build_cache_key function."""

    def test_different_songs_different_keys(self):
        """Different audio content should produce different cache keys."""
        song_a1 = _make_mock_audio(10.0)
        song_a2 = _make_mock_audio(10.0)
        song_a2.samples[0, 0] = 0.5  # Different content
        song_b = _make_mock_audio(10.0)

        key1 = build_cache_key(song_a1, song_b, None, None, None, DEFAULT_CONFIG)
        key2 = build_cache_key(song_a2, song_b, None, None, None, DEFAULT_CONFIG)

        assert key1 != key2

    def test_same_songs_same_key(self):
        """Identical audio content should produce the same cache key."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)

        key1 = build_cache_key(song_a, song_b, None, None, None, DEFAULT_CONFIG)
        key2 = build_cache_key(song_a, song_b, None, None, None, DEFAULT_CONFIG)

        assert key1 == key2

    def test_params_affect_cache_key(self):
        """Different transition parameters should produce different cache keys."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)

        from dcp.transitions.plan import TransitionParams

        params1 = TransitionParams(strategy_type="stem_level", duration_bars=4)
        params2 = TransitionParams(strategy_type="rhythmic", duration_bars=8)

        key1 = build_cache_key(song_a, song_b, None, None, params1, DEFAULT_CONFIG)
        key2 = build_cache_key(song_a, song_b, None, None, params2, DEFAULT_CONFIG)

        assert key1 != key2

    def test_stems_affect_cache_key(self):
        """Different stems should produce different cache keys."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)

        stems_a1 = {"drums": _make_mock_audio(5.0)}
        stems_a2 = {"bass": _make_mock_audio(5.0)}

        key1 = build_cache_key(_make_mock_audio(10.0), _make_mock_audio(10.0), stems_a1, None, None, DEFAULT_CONFIG)
        key2 = build_cache_key(_make_mock_audio(10.0), _make_mock_audio(10.0), stems_a2, None, None, DEFAULT_CONFIG)

        assert key1 != key2

    def test_config_affects_cache_key(self):
        """Different config should produce different cache keys."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)

        from dcp.config import DCPConfig, TransitionConfig

        config1 = DCPConfig(transition=TransitionConfig(fade_floor_db=-120.0))
        config2 = DCPConfig(transition=TransitionConfig(fade_floor_db=-60.0))

        key1 = build_cache_key(song_a, song_b, None, None, None, config1)
        key2 = build_cache_key(song_a, song_b, None, None, None, config2)

        assert key1 != key2


class TestTransitionCache:
    """Tests for the TransitionCache class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a cache instance with a temporary directory."""
        config = CacheConfig(cache_root=temp_cache_dir)
        return TransitionCache(config)

    def test_put_and_get_full_mix(self, cache):
        """Test caching and retrieving a full-mix transition."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)
        plan = _make_mock_plan()

        # Cache the transition
        entry = cache.put(
            cache_key="test_key_1",
            song_a=song_a,
            song_b=song_b,
            stems_a=None,
            stems_b=None,
            plan=plan,
            plan_params=None,
            config=DEFAULT_CONFIG,
        )

        # Verify entry was created
        assert entry.cache_key == "test_key_1"
        assert entry.transition_wav.exists()
        assert entry.plan_json.exists()
        assert entry.validation_json.exists()

        # Retrieve from cache
        retrieved = cache.get("test_key_1")
        assert retrieved is not None
        assert retrieved.cache_key == "test_key_1"
        assert retrieved.transition_wav.exists()

    def test_put_and_get_with_stems(self, cache):
        """Test caching and retrieving a stem-level transition."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)
        plan = _make_mock_plan()

        stems_a = {"drums": _make_mock_audio(5.0), "bass": _make_mock_audio(5.0)}
        stems_b = {"drums": _make_mock_audio(5.0), "bass": _make_mock_audio(5.0)}

        entry = cache.put(
            cache_key="test_key_stems",
            song_a=song_a,
            song_b=song_b,
            stems_a=stems_a,
            stems_b=stems_b,
            plan=plan,
            plan_params=None,
            config=DEFAULT_CONFIG,
        )

        assert entry.transition_wav.exists()

        # Verify cache key includes stem content
        retrieved = cache.get(entry.cache_key)
        assert retrieved is not None

    def test_cache_miss(self, cache):
        """Test that cache miss returns None."""
        result = cache.get("nonexistent_key")
        assert result is None

    def test_cache_key_mismatch(self, cache):
        """Test that cache entry with wrong key is rejected."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)
        plan = _make_mock_plan()

        entry = cache.put(
            cache_key="correct_key",
            song_a=song_a,
            song_b=song_b,
            stems_a=None,
            stems_b=None,
            plan=plan,
            plan_params=None,
            config=DEFAULT_CONFIG,
        )

        # Try to get with wrong key
        result = cache.get("wrong_key")
        assert result is None

    def test_delete(self, cache):
        """Test deleting a cache entry."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)
        plan = _make_mock_plan()

        entry = cache.put(
            cache_key="delete_me",
            song_a=song_a,
            song_b=song_b,
            stems_a=None,
            stems_b=None,
            plan=plan,
            plan_params=None,
            config=DEFAULT_CONFIG,
        )

        assert cache.get("delete_me") is not None

        deleted = cache.delete("delete_me")
        assert deleted is True

        assert cache.get("delete_me") is None

    def test_clear(self, cache):
        """Test clearing all cache entries."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)
        plan = _make_mock_plan()

        cache.put("key1", song_a, song_b, None, None, plan, None, DEFAULT_CONFIG)
        cache.put("key2", song_a, song_b, None, None, plan, None, DEFAULT_CONFIG)

        count = cache.clear()
        assert count == 2

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_list_entries(self, cache):
        """Test listing all cache entries."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)
        plan = _make_mock_plan()

        cache.put("key1", song_a, song_b, None, None, plan, None, DEFAULT_CONFIG)
        cache.put("key2", song_a, song_b, None, None, plan, None, DEFAULT_CONFIG)

        entries = cache.list_entries()
        assert len(entries) == 2
        keys = {e.cache_key for e in entries}
        assert "key1" in keys
        assert "key2" in keys

    def test_get_stats(self, cache):
        """Test getting cache statistics."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)
        plan = _make_mock_plan()

        cache.put("key1", song_a, song_b, None, None, plan, None, DEFAULT_CONFIG)
        cache.put("key2", song_a, song_b, None, None, plan, None, DEFAULT_CONFIG)

        stats = cache.get_stats()
        assert stats["entry_count"] == 2
        assert stats["total_size_mb"] > 0
        assert "cache_root" in stats


class TestGetCachedTransition:
    """Tests for the get_cached_transition function."""

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        config = CacheConfig(cache_root=temp_cache_dir)
        return TransitionCache(config)

    def test_cache_hit(self, cache):
        """Test cache hit returns WAV path."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)
        plan = _make_mock_plan()

        # Pre-populate cache using computed key
        key = build_cache_key(song_a, song_b, None, None, None, DEFAULT_CONFIG)
        cache.put(
            cache_key=key,
            song_a=song_a,
            song_b=song_b,
            stems_a=None,
            stems_b=None,
            plan=plan,
            plan_params=None,
            config=DEFAULT_CONFIG,
        )

        # Try to get from cache
        wav_path = get_cached_transition(
            cache=TransitionCache(CacheConfig(cache_root=Path(cache.config.cache_root))),
            song_a=song_a,
            song_b=song_b,
            stems_a=None,
            stems_b=None,
            plan=plan,
            plan_params=None,
            config=DEFAULT_CONFIG,
        )

        assert wav_path is not None
        assert Path(wav_path).exists()

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        song_a = _make_mock_audio(10.0)
        song_b = _make_mock_audio(10.0)
        plan = _make_mock_plan()

        wav_path = get_cached_transition(
            cache=TransitionCache(CacheConfig(cache_root=Path(cache.config.cache_root))),
            song_a=song_a,
            song_b=song_b,
            stems_a=None,
            stems_b=None,
            plan=plan,
            plan_params=None,
            config=DEFAULT_CONFIG,
        )

        assert wav_path is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])