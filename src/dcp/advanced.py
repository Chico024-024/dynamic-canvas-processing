"""Optional DCP Advanced plugin discovery.

DCP Core owns this stable boundary but contains no proprietary implementation.
A separately distributed Advanced package can advertise strategy objects through
the ``dcp.advanced.strategies`` entry-point group. Discovery is local, silent,
and failure-tolerant: missing or broken plugins never prevent Core rendering.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

from .config import DCPConfig
from .models import DCPError
from .transitions import (
    StrategyRegistry,
    TransitionPlanner,
    build_core_registry,
)

ADVANCED_ENTRY_POINT_GROUP = "dcp.advanced.strategies"


@dataclass(frozen=True)
class AdvancedStatus:
    available: bool
    strategies: tuple[str, ...]
    errors: tuple[str, ...]
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "strategies": list(self.strategies),
            "errors": list(self.errors),
            "detail": self.detail,
        }


def discover_advanced_strategies(
    registry: StrategyRegistry | None = None,
) -> tuple[StrategyRegistry, AdvancedStatus]:
    """Register any installed Advanced strategies, failing gracefully."""

    target = registry or build_core_registry()
    loaded: list[str] = []
    errors: list[str] = []

    try:
        discovered = importlib.metadata.entry_points()
        if hasattr(discovered, "select"):
            candidates = list(discovered.select(group=ADVANCED_ENTRY_POINT_GROUP))
        else:  # pragma: no cover - compatibility with older importlib metadata
            candidates = list(discovered.get(ADVANCED_ENTRY_POINT_GROUP, ()))
    except Exception as exc:  # metadata problems must not break Core
        candidates = []
        errors.append(f"entry-point discovery failed: {type(exc).__name__}: {exc}")

    for entry_point in candidates:
        try:
            loaded_object = entry_point.load()
            strategy = loaded_object() if isinstance(loaded_object, type) else loaded_object
            target.register(strategy)
            loaded.append(str(strategy.name))
        except (DCPError, TypeError, ValueError, ImportError, AttributeError) as exc:
            errors.append(
                f"{entry_point.name}: {type(exc).__name__}: {exc}"
            )
        except Exception as exc:  # plugin code is untrusted at this boundary
            errors.append(
                f"{entry_point.name}: unexpected {type(exc).__name__}: {exc}"
            )

    if loaded and not errors:
        detail = (
            "DCP Advanced strategies loaded: " + ", ".join(loaded) + "."
        )
    elif loaded and errors:
        detail = (
            "Some DCP Advanced strategies loaded, but one or more plugins "
            "failed. Core strategies remain available."
        )
    elif errors:
        detail = (
            "No DCP Advanced strategies were loaded; plugin errors were "
            "recorded. Core remains fully operational."
        )
    else:
        detail = (
            "No DCP Advanced plugin was discovered. DCP Core is fully "
            "operational; proprietary strategies are absent by design."
        )

    return target, AdvancedStatus(
        available=bool(loaded),
        strategies=tuple(loaded),
        errors=tuple(errors),
        detail=detail,
    )


def build_planner_with_advanced(
    config: DCPConfig | None = None,
) -> tuple[TransitionPlanner, AdvancedStatus]:
    """Create a planner with Core strategies plus any installed Advanced ones."""

    registry, status = discover_advanced_strategies()
    return TransitionPlanner(config=config, registry=registry), status
