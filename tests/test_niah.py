"""Tests for scripts/niah_test.py — NIAH (Needle-In-A-Haystack) test runner.

Mocks ALL subprocess and HTTP calls so no llama-server is needed.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Import niah_test.py as a module
# ---------------------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "niah", str(Path(__file__).parent.parent / "scripts" / "niah_test.py")
)
niah = importlib.util.module_from_spec(spec)
# Register in sys.modules so dataclasses can resolve the module's __dict__
sys.modules["niah"] = niah

# Prevent atexit/signal registration from firing during import
with patch("atexit.register"), patch("signal.signal"):
    spec.loader.exec_module(niah)


# ===================================================================
# Needle generation
# ===================================================================


class TestGenerateNeedles:
    """Tests for generate_needles()."""

    def test_determinism_seed42(self):
        """Same seed always produces identical needles."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        n1 = niah.generate_needles(5, rng1)
        n2 = niah.generate_needles(5, rng2)
        for a, b in zip(n1, n2):
            assert a.city == b.city
            assert a.code == b.code
            assert a.position == b.position

    def test_single_needle_at_midpoint(self):
        """Single needle should be placed at position 0.5."""
        rng = random.Random(42)
        needles = niah.generate_needles(1, rng)
        assert len(needles) == 1
        assert needles[0].position == 0.5
        assert needles[0].city == "Paris"

    def test_count_matches_request(self):
        rng = random.Random(42)
        for count in [1, 3, 5, 10]:
            needles = niah.generate_needles(count, rng)
            assert len(needles) == count

    def test_codes_in_valid_range(self):
        rng = random.Random(42)
        needles = niah.generate_needles(10, rng)
        for n in needles:
            assert 100000 <= n.code <= 999999

    def test_positions_distributed_for_multiple(self):
        """Multiple needles should span 5% to 95% of context."""
        rng = random.Random(42)
        needles = niah.generate_needles(5, rng)
        # 5 needles: 0.05, 0.275, 0.5, 0.725, 0.95
        expected_positions = [0.05 + (0.90 * i / 4) for i in range(5)]
        for n, ep in zip(needles, expected_positions):
            assert abs(n.position - ep) < 1e-9

    def test_sentence_format(self):
        rng = random.Random(42)
        needles = niah.generate_needles(1, rng)
        n = needles[0]
        assert n.sentence == f"The secret code for {n.city} is {n.code}."


# ===================================================================
# Haystack generation
# ===================================================================


class TestGenerateHaystack:
    """Tests for generate_haystack()."""

    def test_filler_reaches_target_chars(self):
        """Haystack should be at least target_chars long."""
        rng = random.Random(42)
        needles = niah.generate_needles(1, rng)
        target = 4000
        haystack = niah.generate_haystack(needles, target)
        # The haystack includes needles + filler, should meet or exceed target
        assert len(haystack) >= target

    def test_needle_text_present(self):
        """Every needle sentence must appear in the haystack."""
        rng = random.Random(42)
        needles = niah.generate_needles(3, rng)
        haystack = niah.generate_haystack(needles, 8000)
        for n in needles:
            assert n.sentence in haystack

    def test_multiple_needles_all_present(self):
        rng = random.Random(42)
        needles = niah.generate_needles(10, rng)
        haystack = niah.generate_haystack(needles, 20000)
        for n in needles:
            assert n.sentence in haystack

    def test_empty_needles(self):
        """Haystack with no needles should just be filler."""
        haystack = niah.generate_haystack([], 2000)
        assert len(haystack) >= 2000

    def test_small_target(self):
        """Even a very small target should produce valid output."""
        rng = random.Random(42)
        needles = niah.generate_needles(1, rng)
        haystack = niah.generate_haystack(needles, 100)
        assert needles[0].sentence in haystack


# ===================================================================
# Scoring
# ===================================================================


class TestScoreResponse:
    """Tests for score_response()."""

    def test_exact_match(self):
        assert niah.score_response("123456", 123456) is True

    def test_code_in_sentence(self):
        """Code embedded in a sentence should still match."""
        assert niah.score_response("The code is 654321, I think.", 654321) is True

    def test_no_match(self):
        assert niah.score_response("I don't know", 123456) is False

    def test_partial_match_fails(self):
        """A partial digit overlap shouldn't count (unless substring matches)."""
        # 12345 is a substring of 123456, but we're looking for 123456
        assert niah.score_response("12345", 123456) is False

    def test_empty_response(self):
        assert niah.score_response("", 123456) is False

    def test_garbage_response(self):
        assert niah.score_response("asdfghjkl!@#$%^", 999999) is False

    def test_number_with_extra_text(self):
        assert niah.score_response("Sure! The answer is 555123.", 555123) is True

    def test_repeated_wrong_numbers(self):
        assert niah.score_response("111111 222222 333333", 444444) is False


# ===================================================================
# Data classes
# ===================================================================


class TestDataClasses:
    """Tests for Needle, TrialResult, ConfigResult."""

    def test_needle_post_init(self):
        n = niah.Needle(city="Berlin", code=111111, position=0.3)
        assert n.sentence == "The secret code for Berlin is 111111."

    def test_config_result_accuracy(self):
        cr = niah.ConfigResult(depth=4096, needle_count=3, cache_type="q8_0")
        cr.trials = [
            niah.TrialResult("Paris", 100000, "100000", True),
            niah.TrialResult("Tokyo", 200000, "nope", False),
            niah.TrialResult("Mumbai", 300000, "300000", True),
        ]
        assert cr.accuracy == "2/3"
        assert abs(cr.accuracy_pct - 66.666) < 1.0

    def test_config_result_empty_trials(self):
        cr = niah.ConfigResult(depth=4096, needle_count=0, cache_type="q8_0")
        assert cr.accuracy == "0/0"
        assert cr.accuracy_pct == 0.0

    def test_config_result_all_hits(self):
        cr = niah.ConfigResult(depth=4096, needle_count=2, cache_type="q8_0")
        cr.trials = [
            niah.TrialResult("Paris", 100000, "100000", True),
            niah.TrialResult("Tokyo", 200000, "200000", True),
        ]
        assert cr.accuracy == "2/2"
        assert cr.accuracy_pct == 100.0

    def test_config_result_all_misses(self):
        cr = niah.ConfigResult(depth=4096, needle_count=2, cache_type="q8_0")
        cr.trials = [
            niah.TrialResult("Paris", 100000, "wrong", False),
            niah.TrialResult("Tokyo", 200000, "wrong", False),
        ]
        assert cr.accuracy == "0/2"
        assert cr.accuracy_pct == 0.0


# ===================================================================
# Port finding
# ===================================================================


class TestFindFreePort:
    """Tests for _find_free_port()."""

    def test_finds_port(self):
        """Should find a free port without error."""
        port = niah._find_free_port(18000)
        assert 18000 <= port < 18100

    def test_port_conflict_skips_busy(self):
        """If a port is occupied, it should skip to the next one."""
        # Occupy a port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 19000))
            s.listen(1)
            port = niah._find_free_port(19000)
            assert port != 19000
            assert 19001 <= port < 19100

    def test_all_ports_busy_raises(self):
        """If all ports in range are busy, should raise RuntimeError."""
        def always_busy(*args, **kwargs):
            s = MagicMock()
            s.__enter__ = MagicMock(return_value=s)
            s.__exit__ = MagicMock(return_value=False)
            s.bind = MagicMock(side_effect=OSError("Address in use"))
            return s

        with patch("socket.socket", side_effect=always_busy):
            with pytest.raises(RuntimeError, match="No free port"):
                niah._find_free_port(8090)


# ===================================================================
# Server management
# ===================================================================


class TestStartServer:
    """Tests for start_server()."""

    def _make_health_response(self, status="ok"):
        """Create a mock HTTP response for /health."""
        body = json.dumps({"status": status}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def _make_proc(self, poll_return=None, returncode=0):
        proc = MagicMock()
        proc.poll.return_value = poll_return
        proc.returncode = returncode
        return proc

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_successful_startup(self, mock_popen, mock_urlopen, mock_sleep):
        """Server starts and becomes healthy."""
        proc = self._make_proc()
        mock_popen.return_value = proc
        mock_urlopen.return_value = self._make_health_response("ok")

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()
            server_bin.chmod(0o755)

            result = niah.start_server(Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090)
            assert result is proc

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_server_binary_not_found(self, mock_popen, mock_urlopen, mock_sleep):
        """Should raise FileNotFoundError if llama-server binary missing."""
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError, match="llama-server not found"):
                niah.start_server(Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_server_exits_prematurely(self, mock_popen, mock_urlopen, mock_sleep):
        """Should raise RuntimeError if server process exits during health check."""
        proc = self._make_proc(poll_return=1, returncode=1)
        mock_popen.return_value = proc

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            with pytest.raises(RuntimeError, match="exited prematurely"):
                niah.start_server(Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090)

    @patch("time.monotonic")
    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_server_health_timeout(self, mock_popen, mock_urlopen, mock_sleep, mock_monotonic):
        """Should raise TimeoutError if health never returns ok."""
        proc = self._make_proc()
        mock_popen.return_value = proc

        # Simulate time passing beyond the 120s deadline
        mock_monotonic.side_effect = [0, 0, 121]  # start, first check, past deadline
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            with pytest.raises(TimeoutError, match="did not become healthy"):
                niah.start_server(Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_health_check_retries_on_url_error(self, mock_popen, mock_urlopen, mock_sleep):
        """Server should retry health checks when connection is refused."""
        proc = self._make_proc()
        mock_popen.return_value = proc

        # Fail twice, succeed third time
        mock_urlopen.side_effect = [
            urllib.error.URLError("refused"),
            urllib.error.URLError("refused"),
            self._make_health_response("ok"),
        ]

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            result = niah.start_server(Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090)
            assert result is proc


class TestStopServer:
    """Tests for stop_server()."""

    def test_graceful_stop(self):
        proc = MagicMock()
        niah.stop_server(proc)
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=10)

    def test_force_kill_on_terminate_failure(self):
        proc = MagicMock()
        proc.terminate.side_effect = Exception("won't stop")
        niah.stop_server(proc)
        proc.kill.assert_called_once()


class TestCleanupServer:
    """Tests for _cleanup_server()."""

    def test_cleanup_terminates(self):
        proc = MagicMock()
        niah._active_server = proc
        niah._cleanup_server()
        proc.terminate.assert_called_once()
        assert niah._active_server is None

    def test_cleanup_kills_on_terminate_timeout(self):
        proc = MagicMock()
        proc.wait.side_effect = Exception("timeout")
        niah._active_server = proc
        niah._cleanup_server()
        proc.kill.assert_called_once()
        assert niah._active_server is None

    def test_cleanup_noop_when_none(self):
        niah._active_server = None
        niah._cleanup_server()  # Should not raise


# ===================================================================
# Query logic
# ===================================================================


class TestQueryNeedle:
    """Tests for query_needle()."""

    def _make_chat_response(self, content: str):
        body = json.dumps({
            "choices": [{"message": {"content": content}}]
        }).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_successful_query(self, mock_urlopen, mock_sleep):
        mock_urlopen.return_value = self._make_chat_response("123456")
        result = niah.query_needle(8090, "some haystack", "Paris")
        assert result == "123456"

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_retries_on_network_error(self, mock_urlopen, mock_sleep):
        """Should retry on URLError and succeed on final attempt."""
        mock_urlopen.side_effect = [
            urllib.error.URLError("timeout"),
            urllib.error.URLError("timeout"),
            self._make_chat_response("654321"),
        ]
        result = niah.query_needle(8090, "haystack", "Tokyo", max_retries=3)
        assert result == "654321"

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_all_retries_exhausted(self, mock_urlopen, mock_sleep):
        """Should raise RuntimeError after exhausting all retries."""
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        with pytest.raises(RuntimeError, match="Failed to query server"):
            niah.query_needle(8090, "haystack", "Paris", max_retries=3)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_garbage_json_response(self, mock_urlopen, mock_sleep):
        """Malformed JSON should raise RuntimeError."""
        resp = MagicMock()
        resp.read.return_value = b"not json at all"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp
        with pytest.raises(RuntimeError, match="Unexpected response format"):
            niah.query_needle(8090, "haystack", "Paris")

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_missing_choices_key(self, mock_urlopen, mock_sleep):
        """Response missing 'choices' should raise RuntimeError."""
        body = json.dumps({"error": "something"}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp
        with pytest.raises(RuntimeError, match="Unexpected response format"):
            niah.query_needle(8090, "haystack", "Paris")

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_empty_choices_array(self, mock_urlopen, mock_sleep):
        """Empty choices array should raise RuntimeError."""
        body = json.dumps({"choices": []}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp
        with pytest.raises(RuntimeError, match="Unexpected response format"):
            niah.query_needle(8090, "haystack", "Paris")

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_response_stripped(self, mock_urlopen, mock_sleep):
        """Whitespace should be stripped from response."""
        mock_urlopen.return_value = self._make_chat_response("  123456  \n")
        result = niah.query_needle(8090, "haystack", "Paris")
        assert result == "123456"


# ===================================================================
# Table output formatting
# ===================================================================


class TestBuildTable:
    """Tests for build_table()."""

    def test_single_cache_type(self):
        results = [
            niah.ConfigResult(depth=4096, needle_count=1, cache_type="q8_0"),
        ]
        results[0].trials = [niah.TrialResult("Paris", 100000, "100000", True)]
        table = niah.build_table(results, "test-model")
        assert "test-model" in table
        assert "4K" in table  # 4096 -> 4K
        assert "q8_0" in table

    def test_two_cache_types_with_delta(self):
        r1 = niah.ConfigResult(depth=4096, needle_count=1, cache_type="q8_0")
        r1.trials = [niah.TrialResult("Paris", 100000, "100000", True)]
        r2 = niah.ConfigResult(depth=4096, needle_count=1, cache_type="turbo3")
        r2.trials = [niah.TrialResult("Paris", 100000, "wrong", False)]
        table = niah.build_table([r1, r2], "test-model")
        assert "Delta" in table

    def test_error_result_shows_err(self):
        """Config with no trials should show ERR."""
        r = niah.ConfigResult(depth=8192, needle_count=5, cache_type="q8_0")
        # No trials added
        table = niah.build_table([r], "test-model")
        assert "ERR" in table

    def test_sub_1024_depth_label(self):
        """Depths < 1024 should show raw number, not K suffix."""
        r = niah.ConfigResult(depth=512, needle_count=1, cache_type="q8_0")
        r.trials = [niah.TrialResult("Paris", 100000, "100000", True)]
        table = niah.build_table([r], "model")
        assert "512" in table

    def test_two_cache_types_delta_na_on_error(self):
        """Delta should be N/A when one config has no trials."""
        r1 = niah.ConfigResult(depth=4096, needle_count=1, cache_type="q8_0")
        r1.trials = [niah.TrialResult("Paris", 100000, "100000", True)]
        r2 = niah.ConfigResult(depth=4096, needle_count=1, cache_type="turbo3")
        # r2 has no trials (error)
        table = niah.build_table([r1, r2], "model")
        assert "N/A" in table


# ===================================================================
# Save results
# ===================================================================


class TestSaveResults:
    """Tests for save_results()."""

    def test_creates_json_and_md(self):
        r = niah.ConfigResult(depth=4096, needle_count=1, cache_type="q8_0")
        r.trials = [niah.TrialResult("Paris", 100000, "100000", True)]

        with tempfile.TemporaryDirectory() as td:
            json_path, md_path = niah.save_results([r], "test-model", Path(td))
            assert json_path.exists()
            assert md_path.exists()
            assert json_path.suffix == ".json"
            assert md_path.suffix == ".md"

            # Verify JSON is valid
            with open(json_path) as f:
                data = json.load(f)
            assert data["model"] == "test-model"
            assert data["seed"] == 42
            assert len(data["results"]) == 1

    def test_creates_output_dir(self):
        """Should create output directory if it doesn't exist."""
        r = niah.ConfigResult(depth=4096, needle_count=1, cache_type="q8_0")
        r.trials = []

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "nested" / "output"
            niah.save_results([r], "model", out)
            assert out.exists()


# ===================================================================
# Argparse
# ===================================================================


class TestParseArgs:
    """Tests for parse_args()."""

    def test_required_arg(self):
        args = niah.parse_args(["/path/to/llama"])
        assert args.llama_dir == "/path/to/llama"

    def test_both_positional_args(self):
        args = niah.parse_args(["/llama", "/model.gguf"])
        assert args.llama_dir == "/llama"
        assert args.model_path == "/model.gguf"

    def test_defaults(self):
        args = niah.parse_args(["/llama"])
        assert args.depths == "4096,8192,16384,32768"
        assert args.needles == "1,5,10"
        assert args.port == "8090"
        assert args.cache_types == "q8_0,turbo3"
        assert args.verbose is False
        assert args.output_dir is None

    def test_custom_depths(self):
        args = niah.parse_args(["/llama", "--depths", "2048,4096"])
        assert args.depths == "2048,4096"

    def test_verbose_flag(self):
        args = niah.parse_args(["/llama", "-v"])
        assert args.verbose is True

    def test_all_options(self):
        args = niah.parse_args([
            "/llama", "/model.gguf",
            "--depths", "1024",
            "--needles", "1,2",
            "--port", "9090",
            "--cache-types", "f16",
            "--output-dir", "/tmp/out",
            "--verbose",
        ])
        assert args.depths == "1024"
        assert args.needles == "1,2"
        assert args.port == "9090"
        assert args.cache_types == "f16"
        assert args.output_dir == "/tmp/out"
        assert args.verbose is True

    def test_missing_required_arg(self):
        with pytest.raises(SystemExit):
            niah.parse_args([])


# ===================================================================
# run_config (integration-ish, fully mocked)
# ===================================================================


class TestRunConfig:
    """Tests for run_config() with fully mocked server + queries."""

    def _make_chat_response(self, content: str):
        body = json.dumps({
            "choices": [{"message": {"content": content}}]
        }).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def _make_health_response(self):
        body = json.dumps({"status": "ok"}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_run_config_single_needle_hit(self, mock_popen, mock_urlopen, mock_sleep):
        """Full run_config with 1 needle that matches."""
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc

        # We need to know what code seed=42 generates for Paris
        rng = random.Random(42)
        expected_code = rng.randint(100000, 999999)

        # First call is health check, subsequent are query calls
        mock_urlopen.side_effect = [
            self._make_health_response(),
            self._make_chat_response(str(expected_code)),
        ]

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            result = niah.run_config(
                Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 1, 8090
            )
            assert result.accuracy == "1/1"
            assert result.accuracy_pct == 100.0

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_run_config_needle_miss(self, mock_popen, mock_urlopen, mock_sleep):
        """Full run_config with 1 needle that doesn't match."""
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc

        mock_urlopen.side_effect = [
            self._make_health_response(),
            self._make_chat_response("I have no idea"),
        ]

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            result = niah.run_config(
                Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 1, 8090
            )
            assert result.accuracy == "0/1"
            assert result.accuracy_pct == 0.0

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_run_config_stops_server_on_error(self, mock_popen, mock_urlopen, mock_sleep):
        """Server should be stopped even if query raises."""
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc

        mock_urlopen.side_effect = [
            self._make_health_response(),
            urllib.error.URLError("timeout"),
            urllib.error.URLError("timeout"),
            urllib.error.URLError("timeout"),
        ]

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            with pytest.raises(RuntimeError):
                niah.run_config(
                    Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 1, 8090
                )

        # Server should have been stopped in the finally block
        proc.terminate.assert_called()


# ===================================================================
# run_all (integration, fully mocked)
# ===================================================================


class TestRunAll:
    """Tests for run_all() with fully mocked internals."""

    @patch.object(niah, "run_config")
    @patch.object(niah, "_find_free_port", return_value=8090)
    def test_run_all_iterates_configs(self, mock_port, mock_run_config):
        """run_all should call run_config for each (depth, needle, cache) combo."""
        mock_run_config.return_value = niah.ConfigResult(
            depth=4096, needle_count=1, cache_type="q8_0",
            trials=[niah.TrialResult("Paris", 123456, "123456", True)],
        )

        with tempfile.TemporaryDirectory() as td:
            llama_dir = Path(td)
            model_path = llama_dir / "model.gguf"
            model_path.touch()
            server_bin = llama_dir / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            args = niah.parse_args([
                str(llama_dir), str(model_path),
                "--depths", "4096",
                "--needles", "1",
                "--cache-types", "q8_0",
            ])
            results = niah.run_all(args)
            assert len(results) == 1
            mock_run_config.assert_called_once()

    @patch.object(niah, "run_config")
    @patch.object(niah, "_find_free_port", return_value=8090)
    def test_run_all_handles_config_error(self, mock_port, mock_run_config):
        """Errors in run_config should be caught and recorded as failed."""
        mock_run_config.side_effect = RuntimeError("server exploded")

        with tempfile.TemporaryDirectory() as td:
            llama_dir = Path(td)
            model_path = llama_dir / "model.gguf"
            model_path.touch()
            server_bin = llama_dir / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            args = niah.parse_args([
                str(llama_dir), str(model_path),
                "--depths", "4096",
                "--needles", "1",
                "--cache-types", "q8_0",
            ])
            results = niah.run_all(args)
            assert len(results) == 1
            assert results[0].trials == []
            assert results[0].accuracy_pct == 0.0

    def test_run_all_missing_llama_dir(self):
        """Should sys.exit if llama dir doesn't exist."""
        args = niah.parse_args(["/nonexistent/llama", "/nonexistent/model.gguf"])
        with pytest.raises(SystemExit):
            niah.run_all(args)

    @patch.object(niah, "run_config")
    @patch.object(niah, "_find_free_port", return_value=8090)
    def test_run_all_multiple_configs(self, mock_port, mock_run_config):
        """Should produce N results for N config combos."""
        mock_run_config.return_value = niah.ConfigResult(
            depth=4096, needle_count=1, cache_type="q8_0",
            trials=[niah.TrialResult("Paris", 123456, "123456", True)],
        )

        with tempfile.TemporaryDirectory() as td:
            llama_dir = Path(td)
            model_path = llama_dir / "model.gguf"
            model_path.touch()
            server_bin = llama_dir / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            args = niah.parse_args([
                str(llama_dir), str(model_path),
                "--depths", "4096,8192",
                "--needles", "1,5",
                "--cache-types", "q8_0,turbo3",
            ])
            results = niah.run_all(args)
            # 2 depths * 2 needle_counts * 2 cache_types = 8
            assert len(results) == 8


# ===================================================================
# Signal handler
# ===================================================================


class TestSignalHandler:
    """Tests for _signal_handler()."""

    @patch.object(niah, "_cleanup_server")
    def test_signal_handler_calls_cleanup_and_exits(self, mock_cleanup):
        with pytest.raises(SystemExit) as exc_info:
            niah._signal_handler(2, None)
        mock_cleanup.assert_called_once()
        assert exc_info.value.code == 1


# ===================================================================
# main() end-to-end (fully mocked)
# ===================================================================


class TestMain:
    """Tests for main()."""

    @patch.object(niah, "save_results", return_value=(Path("/fake/results.json"), Path("/fake/results.md")))
    @patch.object(niah, "build_table", return_value="| table |")
    @patch.object(niah, "run_all")
    def test_main_happy_path(self, mock_run_all, mock_build_table, mock_save):
        """main() should parse args, run all, build table, save results."""
        mock_run_all.return_value = [
            niah.ConfigResult(depth=4096, needle_count=1, cache_type="q8_0",
                              trials=[niah.TrialResult("Paris", 123456, "123456", True)])
        ]
        # Should not raise
        niah.main(["/fake/llama", "/fake/model.gguf", "--depths", "4096", "--needles", "1"])
        mock_run_all.assert_called_once()
        mock_build_table.assert_called_once()
        mock_save.assert_called_once()

    @patch.object(niah, "save_results", return_value=(Path("/fake/results.json"), Path("/fake/results.md")))
    @patch.object(niah, "build_table", return_value="| table |")
    @patch.object(niah, "run_all", return_value=[])
    def test_main_custom_output_dir(self, mock_run_all, mock_build_table, mock_save):
        """main() should respect --output-dir."""
        niah.main(["/fake/llama", "/fake/model.gguf", "--output-dir", "/tmp/custom_out"])
        # Verify save_results was called with the custom output dir
        call_args = mock_save.call_args
        assert str(call_args[0][2]) == "/tmp/custom_out"

    @patch.object(niah, "save_results", return_value=(Path("/fake/results.json"), Path("/fake/results.md")))
    @patch.object(niah, "build_table", return_value="| table |")
    @patch.object(niah, "run_all", return_value=[])
    def test_main_default_output_dir(self, mock_run_all, mock_build_table, mock_save):
        """main() with no --output-dir should default to 'niah_results'."""
        niah.main(["/fake/llama", "/fake/model.gguf"])
        call_args = mock_save.call_args
        assert str(call_args[0][2]) == "niah_results"


# ===================================================================
# Verbose mode in start_server
# ===================================================================


class TestStartServerVerbose:
    """Test verbose output paths in start_server()."""

    def _make_health_response(self):
        body = json.dumps({"status": "ok"}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_verbose_prints_cmd(self, mock_popen, mock_urlopen, mock_sleep, capsys):
        """Verbose mode should print the server command."""
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        mock_urlopen.return_value = self._make_health_response()

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            niah.start_server(Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090, verbose=True)

        captured = capsys.readouterr()
        assert "[CMD]" in captured.out

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_verbose_passes_stdout_stderr(self, mock_popen, mock_urlopen, mock_sleep):
        """Verbose mode should pass stdout/stderr=None (not DEVNULL)."""
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        mock_urlopen.return_value = self._make_health_response()

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            niah.start_server(Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090, verbose=True)

        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["stdout"] is None
        assert call_kwargs["stderr"] is None

    def _make_health_response(self):
        body = json.dumps({"status": "ok"}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp


# ===================================================================
# run_config verbose mode
# ===================================================================


class TestRunConfigVerbose:
    """Test verbose output in run_config()."""

    def _make_health_response(self):
        body = json.dumps({"status": "ok"}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def _make_chat_response(self, content: str):
        body = json.dumps({
            "choices": [{"message": {"content": content}}]
        }).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_verbose_prints_haystack_info(self, mock_popen, mock_urlopen, mock_sleep, capsys):
        """Verbose run_config should print haystack length and needle positions."""
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc

        rng = random.Random(42)
        expected_code = rng.randint(100000, 999999)

        mock_urlopen.side_effect = [
            self._make_health_response(),
            self._make_chat_response(str(expected_code)),
        ]

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            niah.run_config(
                Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 1, 8090, verbose=True
            )

        captured = capsys.readouterr()
        assert "Haystack length" in captured.out
        assert "Needle:" in captured.out


# ===================================================================
# run_all missing model file
# ===================================================================


class TestRunAllEdgeCases:
    """Additional edge cases for run_all()."""

    def test_run_all_missing_model_file(self):
        """Should sys.exit if model file doesn't exist."""
        with tempfile.TemporaryDirectory() as td:
            llama_dir = Path(td)
            server_bin = llama_dir / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            args = niah.parse_args([str(llama_dir), "/nonexistent/model.gguf"])
            with pytest.raises(SystemExit):
                niah.run_all(args)

    def test_run_all_missing_server_binary(self):
        """Should sys.exit if llama-server binary doesn't exist."""
        with tempfile.TemporaryDirectory() as td:
            llama_dir = Path(td)
            model_path = llama_dir / "model.gguf"
            model_path.touch()

            args = niah.parse_args([str(llama_dir), str(model_path)])
            with pytest.raises(SystemExit):
                niah.run_all(args)
