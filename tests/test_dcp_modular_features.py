"""Tests for DCP Modular Features, Effect Boundary Smoothing, and HTTP 206 Streaming."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_tone
from dcp.analysis import analyze_track_with_backend
from dcp.config import DCPConfig, DCPFeatureConfig, DEFAULT_CONFIG
from dcp.models import AudioBuffer, StemSet
from dcp.rendering.stem_mix import _blend_effect_edges, render_plan
from dcp.transitions.curves import crossfade_envelopes


# ---------------------------------------------------------------------------
# Transition Envelopes (Internal Curves)
# ---------------------------------------------------------------------------

def test_crossfade_envelopes_equal_power_power_conservation():
    fade_out, fade_in = crossfade_envelopes(1000, curve="equal_power")
    # Equal power sum: cos^2 + sin^2 == 1.0
    power_sum = fade_out ** 2 + fade_in ** 2
    np.testing.assert_allclose(power_sum, 1.0, atol=1e-5)
    assert fade_out[0] == pytest.approx(1.0, abs=1e-5)
    assert fade_out[-1] == pytest.approx(0.0, abs=1e-5)
    assert fade_in[0] == pytest.approx(0.0, abs=1e-5)
    assert fade_in[-1] == pytest.approx(1.0, abs=1e-5)


def test_crossfade_envelopes_s_curve_smooth_derivatives():
    fade_out, fade_in = crossfade_envelopes(1000, curve="s_curve")
    power_sum = fade_out ** 2 + fade_in ** 2
    np.testing.assert_allclose(power_sum, 1.0, atol=1e-5)
    # At t=0 and t=T, smoothstep derivative is 0, so rate of change is tiny
    d_out_start = abs(fade_out[1] - fade_out[0])
    d_out_mid = abs(fade_out[501] - fade_out[500])
    assert d_out_start < d_out_mid * 0.1


def test_crossfade_envelopes_linear():
    fade_out, fade_in = crossfade_envelopes(500, curve="linear")
    amplitude_sum = fade_out + fade_in
    np.testing.assert_allclose(amplitude_sum, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Modular DCP Features (DCPFeatureConfig) & Click-Free Effect Boundaries
# ---------------------------------------------------------------------------

def test_blend_effect_edges_smooths_boundaries():
    sr = 22050
    n = 2000
    orig = np.ones((2, n), dtype=np.float32)
    proc = np.full((2, n), 2.0, dtype=np.float32)

    blended = _blend_effect_edges(orig, proc, sr, at_start=True, at_end=True, crossfade_ms=10.0)
    # At sample 0, matches orig (1.0); in middle, matches proc (2.0)
    assert blended[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert blended[0, n // 2] == pytest.approx(2.0, abs=1e-3)
    assert blended[0, -1] == pytest.approx(1.0, abs=1e-3)


def test_dcp_render_plan_respects_modular_toggles():
    from dcp.transitions.strategies import build_core_registry
    from dcp.transitions.context import TransitionContext

    sr = 22050
    dur = 6.0
    a = make_tone(frequency=440.0, duration_s=dur, sample_rate=sr)
    b = make_tone(frequency=554.37, duration_s=dur, sample_rate=sr)

    # StemSet decomposition
    from dcp.stems.dsp_splitter import decompose_audio_to_stemset
    stems_a = decompose_audio_to_stemset(a, "A")
    stems_b = decompose_audio_to_stemset(b, "B")

    analysis_a = analyze_track_with_backend(a, DEFAULT_CONFIG.analysis)
    analysis_b = analyze_track_with_backend(b, DEFAULT_CONFIG.analysis)

    strat = build_core_registry().get("studio_remix")
    ctx = TransitionContext(
        song_a=a,
        song_b=b,
        stems_a=stems_a,
        stems_b=stems_b,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
        requested_duration_s=4.0,
        requested_strategy="studio_remix",
        config=DEFAULT_CONFIG,
    )
    plan = strat.plan(ctx)

    # 1. Render with all DCP generative features disabled
    dcp_cfg_minimal = DCPFeatureConfig(
        enable_generative_assets=False,
        enable_vocal_chops=False,
        enable_drum_regroove=False,
        enable_fdn_freeze=False,
        enable_zdf_sweeps=False,
        enable_master_glue=False,
    )
    cfg_min = DCPConfig(dcp_features=dcp_cfg_minimal)
    res_min = render_plan(a, b, plan, stems_a=stems_a, stems_b=stems_b, config=cfg_min)
    assert res_min.audio.sample_rate == sr

    # 2. Render with all DCP generative features enabled
    dcp_cfg_full = DCPFeatureConfig(
        enable_generative_assets=True,
        enable_vocal_chops=True,
        enable_drum_regroove=True,
        enable_fdn_freeze=True,
        enable_zdf_sweeps=True,
        enable_master_glue=True,
    )
    cfg_full = DCPConfig(dcp_features=dcp_cfg_full)
    res_full = render_plan(a, b, plan, stems_a=stems_a, stems_b=stems_b, config=cfg_full)
    assert res_full.audio.sample_rate == sr


# ---------------------------------------------------------------------------
# UI Server HTTP 206 Partial Content Byte-Range Streaming
# ---------------------------------------------------------------------------

def test_ui_server_http_206_byte_range_streaming(tmp_path):
    import urllib.request
    import threading
    import time
    from dcp.ui_server import create_server
    from dcp.audio_io import save_audio

    # Create dummy audio in temp output dir
    audio_file = tmp_path / "stream_test.wav"
    tone = make_tone(duration_s=2.0, sample_rate=22050)
    save_audio(audio_file, tone)

    server = create_server(host="127.0.0.1", port=0, output_dir=tmp_path)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)

    try:
        # Request with Range header
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/audio/stream_test.wav")
        req.add_header("Range", "bytes=0-1023")

        with urllib.request.urlopen(req) as resp:
            assert resp.status == 206
            assert resp.headers.get("Content-Range", "").startswith("bytes 0-1023/")
            chunk = resp.read()
            assert len(chunk) == 1024
    finally:
        server.shutdown()
        thread.join(timeout=1.0)

