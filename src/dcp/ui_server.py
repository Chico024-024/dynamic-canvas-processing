"""DCP Web Studio - Zero-Dependency Local Server for Drag-and-Drop Song Mixer.

This module provides a zero-dependency local HTTP server for the DCP Web Studio,
allowing users to drag & drop songs, choose transition strategies, and visualize
transitions with a Spotify-style animated visualizer.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import threading
import webbrowser
from dataclasses import replace
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# Add src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from dcp.audio_io import load_audio, save_audio
from dcp.analysis import analyze_track_with_backend
from dcp.config import DCPConfig, DCPFeatureConfig, DEFAULT_CONFIG
from dcp.models import AudioBuffer
from dcp.transitions import TransitionContext
from dcp.transitions.strategies import build_core_registry
from dcp.rendering import render_plan
from dcp.validation import validate_audio


class DCPWebStudioHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for DCP Web Studio."""

    def __init__(
        self,
        *args,
        config: Optional[DCPConfig] = None,
        static_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        **kwargs,
    ):
        self.config = config or DEFAULT_CONFIG
        self.static_dir = static_dir or Path(__file__).parent / "static"
        self.output_dir = output_dir or Path.cwd() / "output" / "web_mixes"
        if args:
            super().__init__(*args, directory=str(self.static_dir), **kwargs)

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_index()
        elif path.startswith("/static/"):
            self._serve_static(path[8:])  # Remove "/static/" prefix
        elif path.startswith("/api/audio/"):
            self._serve_audio(path[11:])  # Remove "/api/audio/" prefix
        elif path.startswith("/api/"):
            self._send_json_response({"error": "Not found"}, status=404)
        else:
            self._send_json_response({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/analyze":
            self._handle_analyze()
        elif path == "/api/mix":
            self._handle_mix()
        else:
            self._send_json_response({"error": "Not found"}, status=404)

    def _serve_index(self) -> None:
        """Serve the main index.html page."""
        index_path = self.static_dir / "index.html"
        if index_path.exists():
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(index_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self._send_json_response({"error": "index.html not found"}, status=404)

    def _serve_static(self, path: str) -> None:
        """Serve static files (CSS, JS, SVG, etc.)."""
        file_path = self.static_dir / path
        if file_path.exists() and file_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type is None:
                mime_type = "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self._send_json_response({"error": "Not found"}, status=404)

    def _serve_audio(self, filename: str) -> None:
        """Serve audio files from the output directory with RFC 7233 HTTP 206 Byte-Range support."""
        safe_name = Path(filename).name
        file_path = self.output_dir / safe_name
        if not (file_path.exists() and file_path.is_file()):
            self._send_json_response({"error": "Audio file not found"}, status=404)
            return

        file_size = file_path.stat().st_size
        range_header = self.headers.get("Range")
        if range_header and range_header.strip().lower().startswith("bytes="):
            try:
                range_spec = range_header.strip()[6:].split(",")[0].strip()
                start_str, end_str = range_spec.split("-", 1)
                if start_str == "":
                    suffix_len = int(end_str)
                    start = max(0, file_size - suffix_len)
                    end = file_size - 1
                else:
                    start = int(start_str)
                    end = int(end_str) if end_str else (file_size - 1)
                start = max(0, min(start, file_size - 1))
                end = max(start, min(end, file_size - 1))
                chunk_len = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(chunk_len))
                self.end_headers()
                with open(file_path, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(chunk_len))
                return
            except Exception:
                pass

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        with open(file_path, "rb") as f:
            self.wfile.write(f.read())

    def _handle_analyze(self) -> None:
        """Handle POST /api/analyze - analyze uploaded audio file."""
        try:
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            audio_data = None
            if content_type.startswith("multipart/form-data"):
                boundary = self._get_boundary(content_type)
                if boundary:
                    parts = self._parse_multipart(body, boundary)
                    for part in parts:
                        if part.get("name") in ("audio", "file", "song"):
                            audio_data = part.get("data")
                            break
                    if not audio_data and parts:
                        audio_data = parts[0].get("data")
                else:
                    audio_data = body
            else:
                audio_data = body

            if not audio_data:
                self._send_json_response({"error": "No audio file provided"}, status=400)
                return

            # Save to temporary file and analyze
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            try:
                audio = load_audio(tmp_path)
                analysis = analyze_track_with_backend(
                    audio, config=self.config.analysis, backend="auto"
                )
                waveform = self._generate_waveform_envelope(audio)

                result = {
                    "tempo_bpm": analysis.tempo_bpm,
                    "tempo_confidence": analysis.tempo_confidence,
                    "camelot_code": analysis.camelot_code,
                    "key": analysis.key,
                    "key_confidence": analysis.key_confidence,
                    "duration_s": analysis.duration_s,
                    "peak_dbfs": analysis.peak_dbfs,
                    "rms_dbfs": analysis.rms_dbfs,
                    "waveform": waveform,
                }
                self._send_json_response(result)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            self._send_json_response({"error": str(e)}, status=500)

    def _handle_mix(self) -> None:
        """Handle POST /api/mix - create transition between two songs."""
        try:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self._send_json_response({"error": "Expected multipart/form-data"}, status=400)
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            boundary = self._get_boundary(self.headers.get("Content-Type", ""))
            if not boundary:
                self._send_json_response({"error": "Invalid boundary"}, status=400)
                return

            parts = self._parse_multipart(body, boundary)

            song_a_data = None
            song_b_data = None
            strategy = "optimized"
            transition_duration = 8.0
            user_specified_duration = False
            crossfade_curve = "equal_power"
            feature_flags: dict[str, bool] = {}

            def _decode_str(raw: Any) -> str:
                return (
                    raw.decode("utf-8", errors="ignore").strip()
                    if isinstance(raw, bytes)
                    else str(raw).strip()
                )

            def _decode_bool(raw: Any, default: bool = True) -> bool:
                val = _decode_str(raw).lower()
                if val in ("true", "1", "yes", "on"):
                    return True
                if val in ("false", "0", "no", "off"):
                    return False
                return default

            for part in parts:
                name = part.get("name", "")
                if name == "song_a":
                    song_a_data = part.get("data")
                elif name == "song_b":
                    song_b_data = part.get("data")
                elif name == "strategy":
                    strategy = _decode_str(part.get("data", ""))
                    if strategy == "stem-level":
                        strategy = "stem_level"
                elif name == "transition_duration":
                    dur_str = _decode_str(part.get("data", "8.0"))
                    try:
                        transition_duration = float(dur_str)
                        user_specified_duration = True
                    except (ValueError, TypeError):
                        transition_duration = 8.0
                elif name == "crossfade_curve":
                    c_val = _decode_str(part.get("data", "equal_power")).lower()
                    if c_val in ("equal_power", "s_curve", "linear"):
                        crossfade_curve = c_val
                elif name and name.startswith(("dcp_", "automix_")):
                    feature_flags[name] = _decode_bool(part.get("data", "true"))

            if strategy == "studio_remix":
                # Studio Remix computes exact bar-aligned duration autonomously from musical analysis
                # Use a generous fallback for trimming/baseline; the strategy ignores this value
                transition_duration = 32.0
                user_specified_duration = False

            if not song_a_data and not song_b_data:
                self._send_json_response({"error": "Both song_a and song_b required"}, status=400)
                return
            if not song_a_data:
                self._send_json_response({"error": "Song A is required (missing song_a payload)"}, status=400)
                return
            if not song_b_data:
                self._send_json_response({"error": "Song B is required (missing song_b payload)"}, status=400)
                return

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_a:
                tmp_a.write(song_a_data)
                tmp_a_path = tmp_a.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_b:
                tmp_b.write(song_b_data)
                tmp_b_path = tmp_b.name

            try:
                song_a = load_audio(tmp_a_path)
                song_b = load_audio(tmp_b_path)

                # Trim trailing silence from Song A so the transition blends while Song A is actively audible
                _, end_a_s = self._detect_effective_boundaries(song_a)
                if end_a_s < song_a.duration_s - 1.0:
                    end_a_frame = int(round(end_a_s * song_a.sample_rate))
                    min_frames = int(round((transition_duration + 1.0) * song_a.sample_rate))
                    if end_a_frame > min_frames:
                        song_a = AudioBuffer(samples=song_a.samples[:, :end_a_frame], sample_rate=song_a.sample_rate)

                # Trim leading silence from Song B so Song B enters with audible music
                start_b_s, _ = self._detect_effective_boundaries(song_b)
                if start_b_s > 0.3:
                    start_b_frame = int(round(start_b_s * song_b.sample_rate))
                    if start_b_frame < song_b.frames - int(round(transition_duration * song_b.sample_rate)):
                        song_b = AudioBuffer(samples=song_b.samples[:, start_b_frame:], sample_rate=song_b.sample_rate)

                analysis_a = analyze_track_with_backend(song_a, self.config.analysis, backend="auto")
                analysis_b = analyze_track_with_backend(song_b, self.config.analysis, backend="auto")

                # Build modular DCP feature config from request toggles
                dcp_feat_cfg = replace(
                    self.config.dcp_features,
                    enable_zdf_sweeps=feature_flags.get("dcp_zdf_sweeps", True),
                    enable_fdn_freeze=feature_flags.get("dcp_fdn_freeze", True),
                    enable_sidechain_duck=feature_flags.get("dcp_sidechain_duck", True),
                    enable_generative_assets=feature_flags.get("dcp_generative_assets", True),
                    enable_vocal_chops=feature_flags.get("dcp_vocal_chops", True),
                    enable_drum_regroove=feature_flags.get("dcp_drum_regroove", True),
                    enable_master_glue=feature_flags.get("dcp_master_glue", True),
                )
                request_config = replace(
                    self.config,
                    dcp_features=dcp_feat_cfg,
                )

                # If a stem-based DCP strategy is chosen, decompose into 4 stems via 3-tier hierarchy
                stems_a = None
                stems_b = None
                if strategy in ("studio_remix", "stem_level", "atmospheric", "optimized"):
                    from .stems import separate_hierarchical
                    stems_a, backend_a = separate_hierarchical(song_a, label="A")
                    stems_b, backend_b = separate_hierarchical(song_b, label="B")

                    # Stem Quality Guard: When falling back to DSP filterbank, auto-disable features
                    # that corrupt un-isolated pads/reverb tails into watery chops or noisy regrooving
                    if backend_a == "dsp" or backend_b == "dsp":
                        force_dsp = feature_flags.get("dcp_force_dsp_features", False)
                        if not force_dsp:
                            dcp_feat_cfg = replace(
                                dcp_feat_cfg,
                                enable_vocal_chops=False,
                                enable_drum_regroove=False,
                                enable_fdn_freeze=False,
                                enable_sidechain_duck=False,
                            )
                            request_config = replace(request_config, dcp_features=dcp_feat_cfg)

                # Build transition context for DCP
                context = TransitionContext(
                    song_a=song_a,
                    song_b=song_b,
                    stems_a=stems_a,
                    stems_b=stems_b,
                    analysis_a=analysis_a,
                    analysis_b=analysis_b,
                    requested_duration_s=transition_duration,
                    requested_strategy=strategy,
                    config=request_config,
                )

                # Build planner and create DCP plan
                registry = build_core_registry()
                if strategy in registry:
                    strategy_obj = registry.get(strategy)
                elif "optimized" in registry:
                    strategy_obj = registry.get("optimized")
                else:
                    strategy_obj = list(registry.items())[0][1]
                plan = strategy_obj.plan(context)

                # Render DCP transition
                dcp_result = render_plan(
                    song_a, song_b, plan, stems_a=stems_a, stems_b=stems_b, config=request_config
                )

                # Save output
                import uuid
                mix_id = str(uuid.uuid4())[:8]

                self.output_dir.mkdir(parents=True, exist_ok=True)
                dcp_path = self.output_dir / f"mix_{mix_id}.wav"
                save_audio(dcp_path, dcp_result.audio, subtype="PCM_24")

                # Generate waveforms for visualization
                waveform_a = self._generate_waveform_envelope(song_a)
                waveform_b = self._generate_waveform_envelope(song_b)

                from .rendering.telemetry import evaluate_bridge_telemetry
                telemetry_report = evaluate_bridge_telemetry(
                    dcp_result.audio.samples, sr=dcp_result.audio.sample_rate
                )

                plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else {}
                t_impact_val = float(plan.metadata.get("t_impact_s", 0.0) or 0.0)
                t_breath_val = float(plan.metadata.get("t_breath_s", 0.0) or 0.0)

                result = {
                    "mix_id": mix_id,
                    "mix_url": f"/api/audio/mix_{mix_id}.wav",
                    "baseline_url": None,
                    "duration_s": dcp_result.audio.duration_s,
                    "song_a_duration_s": song_a.duration_s,
                    "song_b_duration_s": song_b.duration_s,
                    "transition_start_s": plan.start_time_s,
                    "transition_duration_s": plan.duration_s,
                    "impact_timeline_s": plan.start_time_s + t_impact_val,
                    "breath_timeline_s": plan.start_time_s + t_breath_val,
                    "downbeat_a_s": float(plan.metadata.get("downbeat_a_s", 0.0) or 0.0),
                    "downbeat_b_s": float(plan.metadata.get("downbeat_b_s", 0.0) or 0.0),
                    "automations": plan_dict.get("automations", []),
                    "reasoning": list(plan.reasoning),
                    "warnings": list(plan.warnings),
                    "waveform_a": waveform_a,
                    "waveform_b": waveform_b,
                    "metadata": plan.metadata,
                    "telemetry": telemetry_report.to_dict(),
                }
                self._send_json_response(result)

            finally:
                try:
                    os.unlink(tmp_a_path)
                    os.unlink(tmp_b_path)
                except Exception:
                    pass

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json_response({"error": str(e)}, status=500)

    def _detect_effective_boundaries(self, audio: AudioBuffer, threshold_dbfs: float = -42.0) -> Tuple[float, float]:
        """Detect non-silent start and end boundaries (in seconds) of an audio track."""
        mono = audio.mono()
        if len(mono) == 0:
            return 0.0, audio.duration_s
        sr = audio.sample_rate
        hop = int(0.05 * sr)
        if hop == 0:
            return 0.0, audio.duration_s
        threshold_linear = 10.0 ** (threshold_dbfs / 20.0)
        n_hops = len(mono) // hop
        if n_hops == 0:
            return 0.0, audio.duration_s

        # Find where audio first becomes audible
        start_s = 0.0
        for i in range(n_hops):
            frame = mono[i * hop : (i + 1) * hop]
            if np.sqrt(np.mean(frame ** 2)) >= threshold_linear:
                start_s = max(0.0, i * 0.05 - 0.05)
                break

        # Find where audible audio ends before trailing silence/fadeout
        end_s = audio.duration_s
        for i in range(n_hops - 1, -1, -1):
            frame = mono[i * hop : (i + 1) * hop]
            if np.sqrt(np.mean(frame ** 2)) >= threshold_linear:
                end_s = min(audio.duration_s, (i + 1) * 0.05 + 0.5)
                break

        return start_s, end_s

    def _generate_waveform_envelope(self, audio_or_analysis: Any, num_points: int = 100) -> List[float]:
        """Generate a simplified waveform envelope for visualization."""
        if isinstance(audio_or_analysis, AudioBuffer):
            mono = audio_or_analysis.mono()
            if len(mono) == 0:
                return [0.0] * num_points
            chunk_size = max(1, len(mono) // num_points)
            envelope = []
            for i in range(num_points):
                chunk = mono[i * chunk_size : (i + 1) * chunk_size]
                if len(chunk) > 0:
                    peak = float(np.max(np.abs(chunk)))
                    envelope.append(round(min(1.0, peak), 3))
                else:
                    envelope.append(0.0)
            return envelope
        return [0.0] * num_points

    def _get_boundary(self, content_type: str) -> Optional[bytes]:
        """Extract boundary from Content-Type header."""
        if "boundary=" in content_type:
            boundary = content_type.split("boundary=")[1].strip()
            if ";" in boundary:
                boundary = boundary.split(";")[0].strip()
            if boundary.startswith('"') and boundary.endswith('"'):
                boundary = boundary[1:-1]
            return boundary.encode("utf-8")
        return None

    def _parse_multipart(self, body: bytes, boundary: bytes) -> List[Dict[str, Any]]:
        """Parse multipart form data."""
        parts = []
        delimiter = b"--" + boundary
        raw_chunks = body.split(delimiter)
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not chunk or chunk == b"--" or chunk.startswith(b"--"):
                continue

            # Locate header end (\r\n\r\n or \n\n)
            header_end = chunk.find(b"\r\n\r\n")
            header_sep_len = 4
            if header_end == -1:
                header_end = chunk.find(b"\n\n")
                header_sep_len = 2
            if header_end == -1:
                continue

            headers_raw = chunk[:header_end].decode("utf-8", errors="ignore")
            data = chunk[header_end + header_sep_len:]
            if data.endswith(b"\r\n"):
                data = data[:-2]
            elif data.endswith(b"\n"):
                data = data[:-1]

            headers = {}
            for line in re.split(r"\r?\n", headers_raw):
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            name = None
            if "content-disposition" in headers:
                cd = headers["content-disposition"]
                m = re.search(r'name=["\']?([^"\';\r\n]+)["\']?', cd, re.IGNORECASE)
                if m:
                    name = m.group(1).strip('"\' ')

            parts.append({
                "name": name,
                "data": data,
                "headers": headers,
            })
        return parts

    def _send_json_response(self, data: Dict[str, Any], status: int = 200) -> None:
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, allow_nan=False).encode("utf-8"))


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    config: Optional[DCPConfig] = None,
    static_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> HTTPServer:
    """Create and configure the DCP Web Studio HTTP server."""
    if config is None:
        config = DEFAULT_CONFIG
    if static_dir is None:
        static_dir = Path(__file__).parent / "static"
    if output_dir is None:
        output_dir = Path.cwd() / "output" / "web_mixes"

    def handler(*args, **kwargs):
        return DCPWebStudioHandler(*args, config=config, static_dir=static_dir, output_dir=output_dir, **kwargs)

    server = ThreadingHTTPServer((host, port), handler)
    return server


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    config: Optional[DCPConfig] = None,
    static_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    open_browser: bool = True,
) -> None:
    """Run the DCP Web Studio server."""
    server = create_server(host, port, config, static_dir, output_dir)
    print(f"DCP Web Studio running at http://{host}:{port}")
    if open_browser:
        webbrowser.open(f"http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DCP Web Studio Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )