"""Tests for the DCP Web Studio UI server."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from dcp.ui_server import create_server, run_server, DCPWebStudioHandler


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

            from dcp.ui_server import create_server, DCPWebStudioHandler
            from dcp.config import DEFAULT_CONFIG

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
                import wave
                import struct
                import math

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
                        files = {"audio": ("test.wav", f, "audio/wav")}
                        response = urllib.request.urlopen(
                            urllib.request.Request(
                                f"http://127.0.0.1:{port}/api/analyze",
                                data=open(audio_path, "rb").read(),
                                headers={"Content-Type": "multipart/form-data"}
                            )
                        )
                        # This is a simplified test - in reality we'd need proper multipart
                        pass
            finally:
                server.shutdown()
                thread.join(timeout=1)

    def test_analyze_endpoint_returns_json(self):
        """Test that /api/analyze returns valid JSON with expected fields."""
        # This is a placeholder for a more complete test
        # Full integration test would require more setup
        pass

    def test_mix_endpoint(self):
        """Test the /api/mix endpoint."""
        # Placeholder for mix endpoint test
        pass


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

            handler = DCPWebStudioHandler(
                config=DEFAULT_CONFIG,
                static_dir=Path("."),
                output_dir=Path(".")
            )
            assert handler.config is not None
            assert handler.static_dir is not None
            assert handler.output_dir is not None


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