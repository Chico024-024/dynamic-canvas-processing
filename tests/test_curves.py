"""Gain conversion and automation-curve tests."""

from __future__ import annotations

import numpy as np
import pytest

from dcp.transitions import (
    GainPoint,
    db_to_gain,
    equal_power_fades,
    gain_to_db,
    interpolate_gain,
)


def test_db_gain_round_trip_and_floor():
    assert db_to_gain(0.0) == pytest.approx(1.0)
    assert db_to_gain(-6.0206) == pytest.approx(0.5, rel=1e-4)
    assert db_to_gain(-120.0) == 0.0
    assert db_to_gain(-200.0) == 0.0
    assert gain_to_db(0.5) == pytest.approx(-6.0206, abs=1e-3)
    assert gain_to_db(0.0) == -120.0


def test_db_to_gain_supports_arrays():
    gains = db_to_gain(np.array([0.0, -6.0206, -120.0, -200.0]))
    assert gains[0] == pytest.approx(1.0)
    assert gains[1] == pytest.approx(0.5, rel=1e-4)
    assert gains[2] == 0.0
    assert gains[3] == 0.0


def test_equal_power_fades_sum_to_unit_power():
    fade_out, fade_in = equal_power_fades(1000)
    assert fade_out[0] == pytest.approx(1.0)
    assert fade_in[0] == pytest.approx(0.0, abs=1e-12)
    assert fade_out[-1] == pytest.approx(0.0, abs=1e-12)
    assert fade_in[-1] == pytest.approx(1.0)
    assert np.allclose(fade_out**2 + fade_in**2, 1.0, atol=1e-12)


def test_equal_power_fades_reject_invalid_length():
    with pytest.raises(ValueError, match="at least 1"):
        equal_power_fades(0)


def test_interpolate_gain_holds_endpoints():
    points = (GainPoint(1.0, 0.0), GainPoint(3.0, -12.0))
    times = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    gains = interpolate_gain(points, times)
    assert gains[0] == pytest.approx(1.0)
    assert gains[1] == pytest.approx(1.0)
    assert gains[3] == pytest.approx(db_to_gain(-12.0))
    assert gains[4] == pytest.approx(db_to_gain(-12.0))
    assert gains[2] < gains[1]


def test_smoothstep_and_linear_curves_differ_midway():
    points = (GainPoint(0.0, 0.0), GainPoint(1.0, -24.0))
    times = np.array([0.25, 0.5, 0.75])
    smooth = interpolate_gain(points, times, curve="smoothstep")
    linear = interpolate_gain(points, times, curve="linear")
    assert smooth[1] == pytest.approx(db_to_gain(-12.0), rel=1e-6)
    assert linear[1] == pytest.approx(db_to_gain(-12.0), rel=1e-6)
    assert smooth[0] > linear[0]
    assert smooth[2] < linear[2]


def test_interpolate_gain_floor_produces_exact_zero():
    points = (GainPoint(0.0, -120.0), GainPoint(1.0, 0.0))
    gains = interpolate_gain(points, np.array([0.0, 0.01]))
    assert gains[0] == 0.0
    assert gains[1] > 0.0


def test_interpolate_gain_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="at least one"):
        interpolate_gain((), np.array([0.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_gain(
            (GainPoint(1.0, 0.0), GainPoint(1.0, -6.0)), np.array([1.0])
        )
    with pytest.raises(ValueError, match="unsupported curve"):
        interpolate_gain(
            (GainPoint(0.0, 0.0), GainPoint(1.0, -6.0)),
            np.array([0.5]),
            curve="bogus",
        )
