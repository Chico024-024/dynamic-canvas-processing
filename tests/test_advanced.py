"""The optional Advanced boundary must be failure-tolerant and isolated."""

from __future__ import annotations

from dataclasses import replace

import pytest

import dcp.advanced as advanced_module
from conftest import make_analysis, make_tone
from dcp.advanced import (
    ADVANCED_ENTRY_POINT_GROUP,
    build_planner_with_advanced,
    discover_advanced_strategies,
)
from dcp.transitions import TransitionPlanner
from dcp.transitions.strategies import CrossfadeStrategy


class _FakeEntryPoint:
    def __init__(self, name: str, loader):
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader()


class _FakeEntryPoints:
    def __init__(self, entries):
        self._entries = list(entries)

    def select(self, *, group):
        assert group == ADVANCED_ENTRY_POINT_GROUP
        return list(self._entries)


class _DemoAdvancedStrategy:
    name = "advanced_demo"
    description = "Test-only strategy plugged through the Advanced boundary."

    def plan(self, context):
        plan = CrossfadeStrategy().plan(context)
        return replace(plan, strategy=self.name)


def _install(monkeypatch, entries):
    monkeypatch.setattr(
        advanced_module.importlib.metadata,
        "entry_points",
        lambda: _FakeEntryPoints(entries),
    )


def _planner_plan(planner, song_a, song_b):
    return planner.plan(
        song_a,
        song_b,
        strategy="crossfade",
        duration_s=2.0,
        analysis_a=make_analysis(song_a),
        analysis_b=make_analysis(song_b),
    )


def test_absence_of_advanced_keeps_core_fully_operational(monkeypatch):
    _install(monkeypatch, [])
    registry, status = discover_advanced_strategies()

    assert status.available is False
    assert status.strategies == ()
    assert status.errors == ()
    assert "Core is fully operational" in status.detail
    assert registry.names() == (
        "atmospheric",
        "conservative",
        "crossfade",
        "optimized",
        "rhythmic",
        "stem_level",
    )

    song = make_tone(duration_s=3.0)
    plan = _planner_plan(TransitionPlanner(registry=registry), song, song)
    assert plan.strategy == "crossfade"


def test_broken_plugin_is_reported_without_breaking_core(monkeypatch):
    def explode():
        raise ImportError("simulated missing dependency")

    _install(monkeypatch, [_FakeEntryPoint("broken", explode)])
    registry, status = discover_advanced_strategies()

    assert status.available is False
    assert len(status.errors) == 1
    assert "ImportError" in status.errors[0]
    assert "Core remains fully operational" in status.detail
    assert len(registry) == 6


def test_discovery_failure_is_recorded(monkeypatch):
    def explode():
        raise RuntimeError("metadata store unavailable")

    monkeypatch.setattr(
        advanced_module.importlib.metadata, "entry_points", explode
    )
    registry, status = discover_advanced_strategies()

    assert status.available is False
    assert any("discovery failed" in error for error in status.errors)
    assert len(registry) == 6


def test_valid_plugin_is_registered_and_usable(monkeypatch):
    _install(
        monkeypatch,
        [_FakeEntryPoint("advanced_demo", lambda: _DemoAdvancedStrategy)],
    )
    registry, status = discover_advanced_strategies()

    assert status.available is True
    assert status.strategies == ("advanced_demo",)
    assert "advanced_demo" in registry

    song = make_tone(duration_s=3.0)
    planner = TransitionPlanner(registry=registry)
    plan = planner.plan(
        song,
        song,
        strategy="advanced_demo",
        duration_s=2.0,
        analysis_a=make_analysis(song),
        analysis_b=make_analysis(song),
    )
    assert plan.strategy == "advanced_demo"
    assert plan.requested_strategy == "advanced_demo"


def test_plugin_cannot_override_core_strategy_name(monkeypatch):
    class Impostor:
        name = "rhythmic"
        description = "Attempts to replace a Core strategy."

        def plan(self, context):
            raise AssertionError("must never be called")

    _install(monkeypatch, [_FakeEntryPoint("impostor", lambda: Impostor())])
    registry, status = discover_advanced_strategies()

    assert status.available is False
    assert any("already registered" in error for error in status.errors)
    assert registry.get("rhythmic").__class__.__name__ == "RhythmicStrategy"


def test_plugin_without_identity_is_rejected(monkeypatch):
    class Nameless:
        description = "No name."

        def plan(self, context):
            raise AssertionError("unused")

    _install(monkeypatch, [_FakeEntryPoint("nameless", lambda: Nameless())])
    registry, status = discover_advanced_strategies()

    assert status.available is False
    assert any("non-empty name" in error for error in status.errors)
    assert len(registry) == 6


def test_build_planner_with_advanced_returns_core_planner(monkeypatch):
    _install(monkeypatch, [])
    planner, status = build_planner_with_advanced()

    assert status.available is False
    assert isinstance(planner, TransitionPlanner)
    song = make_tone(duration_s=3.0)
    plan = _planner_plan(planner, song, song)
    assert plan.strategy == "crossfade"
