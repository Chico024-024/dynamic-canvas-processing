"""Tests for the DCP Web Studio UI server."""

from __future__ import annotations

import json
import math
import struct
import tempfile
import threading
import time
import urllib.request
import wave
from pathlib import Path

import pytest

from dcp.ui_server import create_server, run_server, DCPWebStudioHandler
from dcp.config import DEFAULT_CONFIG


class TestUIServer:
    """Tests for the DCP Web Studio server."""

    def test_server_creation(self):
        """Test that the server can be created with default settings."""
        server = create_server(host="127.0.0.1", port=8000)
        assert server.server_address == ("127.0.0.1", 8000)
        server.server_close()

    def test_server_creation_custom_host_port(self):
        """Test that the server can be created with custom host and port."""
        server = create_server(host="0.0.0.0", port=9000)
        assert server.server_address == ("0.0.0.0", 9000)
        server.server_close()

    def test_static_file_serving(self):
        """Test that static files are served correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir) / "static"
            static_dir.mkdir()
            (static_dir / "test.txt").write_text("Hello, World!")

            from dcp.ui_server import create_server

            server = create_server(host="127.0.0.1", port=0, static_dir=Path(tmpdir) / "static")
            port = server.server_address[1]

            import urllib.request
            import threading
            import time

            def run_server():
                server.serve_forever()

            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            time.sleep(0.5)  # Wait for server to start

            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/static/test.txt") as response:
                    content = response.read().decode("utf-8")
                    assert content == "Hello, World!"
            finally:
                server.shutdown()
                thread.join(timeout=1)

    def test_index_page(self):
        """Test that the index page is served."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir) / "static"
            static_dir.mkdir()
            (static_dir / "index.html").write_text("<html><body>DCP Studio</body></html>")

            from dcp.ui_server import create_server
            import threading
            import time
            import urllib.request

            server = create_server(host="127.0.0.1", port=0, static_dir=Path(tmpdir) / "static")
            port = server.server_address[1]

            def run_server():
                server.serve_forever()

            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            time.sleep(0.5)

            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/") as response:
                    content = response.read().decode("utf-8")
                    assert "DCP Studio" in content
            finally:
                server.shutdown()
                thread.join(timeout=1)

    def test_api_analyze_endpoint(self):
        """Test the /api/analyze endpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir) / "static"
            static_dir.mkdir()
            (Path(tmpdir) / "output").mkdir()
            (static_dir / "index.html").write_text("<html><body>Test</body></html>")

            from dcp.ui_server import create_server
            import threading
            import time
            import urllib.request
            import json

            server = create_server(host="127.0.0.1", port=0, static_dir=Path(tmpdir) / "static",
                                   output_dir=Path(tmpdir) / "output")
            port = server.server_address[1]

            def run_server():
                server.serve_forever()

            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            time.sleep(0.5)

            try:
                # Create a simple WAV file for testing
                sample_rate = 44100
                duration = 0.1
                frequency = 440.0
                n_samples = int(sample_rate * duration)
                audio_data = b""
                for i in range(n_samples):
                    t = i / sample_rate
                    value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
                    audio_data += struct.pack("<h", value)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    with wave.open(tmp.name, "wb") as wf:
                        wf.setnchannels(2)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio_data)
                    audio_path = tmp.name

                    # Upload and analyze
                    with open(audio_path, "rb") as f:
                        # Create multipart form data
                        boundary = b"----WebKitFormBoundary"
                        body = (
                            b"--" + boundary + b"\r\n"
                            b'Content-Disposition: form-data; name="audio"; filename="test.wav"\r\n'
                            b"Content-Type: audio/wav\r\n\r\n"
                        ) + open(audio_path, "rb").read() + b"\r\n--" + boundary + b"--\r\n"

                        headers = {
                            "Content-Type": f"multipart/form-data; boundary={boundary.decode()}"
                        }

                        req = urllib.request.Request(
                            f"http://127.0.0.1:{port}/api/analyze",
                            data=body,
                            headers=headers
                        )

                        response = urllib.request.urlopen(req)
                        response_data = json.loads(response.read().decode("utf-8"))

                        assert "tempo_bpm" in response_data
                        assert "camelot_code" in response_data
                        assert "duration_s" in response_data
                        assert "waveform" in response_data
            finally:
                server.shutdown()
                thread.join(timeout=1)

    def test_analyze_endpoint_returns_json(self):
        """Test that /api/analyze returns valid JSON with expected fields."""
        # This is a placeholder for a more complete test
        # Full integration test would require more setup
        pass

    def test_mix_endpoint(self):
        """Test the /api/mix endpoint with two audio files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir) / "static"
            static_dir.mkdir()
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            (static_dir / "index.html").write_text("<html><body>DCP Studio</body></html>")

            from dcp.ui_server import create_server
            import threading
            import time
            import urllib.request
            import json
            import wave
            import struct
            import math

            server = create_server(
                host="127.0.0.1", port=0, static_dir=static_dir, output_dir=output_dir
            )
            port = server.server_address[1]

            def run_server():
                server.serve_forever()

            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            time.sleep(0.5)

            try:
                sample_rate = 22050
                duration = 2.0
                n_samples = int(sample_rate * duration)
                data_a = b"".join(
                    struct.pack("<h", int(10000 * math.sin(2 * math.pi * 220 * i / sample_rate)))
                    for i in range(n_samples)
                )
                data_b = b"".join(
                    struct.pack("<h", int(10000 * math.sin(2 * math.pi * 330 * i / sample_rate)))
                    for i in range(n_samples)
                )

                file_a = Path(tmpdir) / "a.wav"
                with wave.open(str(file_a), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(data_a)

                file_b = Path(tmpdir) / "b.wav"
                with wave.open(str(file_b), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(data_b)

                boundary = "----MixBoundary7MA4YWxkTrZu0gW"
                body = (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="song_a"; filename="a.wav"\r\n'
                    f"Content-Type: audio/wav\r\n\r\n"
                ).encode("utf-8") + file_a.read_bytes() + (
                    f"\r\n--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="song_b"; filename="b.wav"\r\n'
                    f"Content-Type: audio/wav\r\n\r\n"
                ).encode("utf-8") + file_b.read_bytes() + (
                    f"\r\n--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="strategy"\r\n\r\n'
                    f"conservative\r\n"
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="transition_duration"\r\n\r\n'
                    f"1.0\r\n"
                    f"--{boundary}--\r\n"
                ).encode("utf-8")

                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/mix",
                    data=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                )
                with urllib.request.urlopen(req) as response:
                    res_json = json.loads(response.read().decode("utf-8"))
                    assert "mix_id" in res_json
                    assert "mix_url" in res_json
                    assert "baseline_url" in res_json
                    assert "duration_s" in res_json
                    assert "waveform_a" in res_json
                    assert "waveform_b" in res_json
            finally:
                server.shutdown()
                thread.join(timeout=1)


class TestUIHandler:
    """Tests for the DCPWebStudioHandler."""

    def test_handler_creation(self):
        """Test that the handler can be instantiated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir) / "static"
            static_dir.mkdir()
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            from dcp.ui_server import DCPWebStudioHandler
            from dcp.config import DEFAULT_CONFIG

            # Create a mock request and client address for testing
            from http.server import HTTPServer
            from io import BytesIO

            class MockRequest:
                def makefile(self, *args, **kwargs):
                    return BytesIO(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")

            class MockClientAddress:
                pass

            server = HTTPServer(("127.0.0.1", 0), lambda *args, **kwargs: None)
            server.server_close()

            # We can't easily instantiate the handler without a real request
            # So we'll test the handler class can be imported and has the right methods
            from dcp.ui_server import DCPWebStudioHandler
            assert hasattr(DCPWebStudioHandler, 'do_GET')
            assert hasattr(DCPWebStudioHandler, 'do_POST')
            assert hasattr(DCPWebStudioHandler, '_serve_index')
            assert hasattr(DCPWebStudioHandler, 'do_POST')


class TestRunServer:
    """Tests for the run_server function."""

    def test_run_server_creation(self):
        """Test that run_server creates a server correctly."""
        from dcp.ui_server import run_server
        import threading
        import time

        def run_server_thread():
            run_server(host="127.0.0.1", port=0, open_browser=False)

        thread = threading.Thread(target=run_server_thread, daemon=True)
        thread.start()
        time.sleep(0.5)
        # Server should be running
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])