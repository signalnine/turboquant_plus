#!/usr/bin/env python3
"""TurboQuant Needle-In-A-Haystack (NIAH) Test.

Measures retrieval accuracy across cache types, context depths, and needle counts.
Spins up llama-server for each configuration, inserts unique "needle" facts into
filler text, then queries each needle and scores exact-match retrieval.

Usage:
    python3 scripts/niah_test.py /path/to/llama.cpp /path/to/model.gguf
    python3 scripts/niah_test.py --help

Requirements: Python 3.10+ stdlib only (no pip deps).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42

CITIES = [
    "Paris", "Tokyo", "Mumbai", "Cairo", "Lima",
    "Oslo", "Seoul", "Rome", "Dublin", "Nairobi",
    "Berlin", "Sydney", "Toronto", "Bangkok", "Lisbon",
    "Stockholm", "Athens", "Jakarta", "Santiago", "Helsinki",
]

# Filler paragraphs (~200 tokens / ~800 chars each). Repeating these fills
# arbitrary context sizes without hardcoding 128 KB of text.
FILLER_PARAGRAPHS = [
    (
        "The Amazon River, flowing through South America, is the largest river by "
        "discharge volume of water in the world. It stretches approximately 6,400 "
        "kilometers from the Andes Mountains in Peru to the Atlantic Ocean in Brazil. "
        "The river basin covers about 7 million square kilometers and is home to the "
        "Amazon Rainforest, which contains roughly 390 billion individual trees divided "
        "into 16,000 species. The rainforest produces about 20 percent of the world's "
        "oxygen and plays a crucial role in regulating the global climate. Scientists "
        "have identified over 2,200 species of fish in the Amazon River system, making "
        "it the most biodiverse river system on Earth."
    ),
    (
        "The periodic table of elements organizes all known chemical elements by their "
        "atomic number, electron configuration, and recurring chemical properties. "
        "Dmitri Mendeleev published the first widely recognized periodic table in 1869, "
        "predicting the existence and properties of elements not yet discovered. The "
        "table currently contains 118 confirmed elements, with oganesson being the most "
        "recently named. Elements are arranged in rows called periods and columns called "
        "groups. Elements in the same group share similar chemical properties because "
        "they have the same number of electrons in their outer shell. The lanthanides "
        "and actinides are placed separately at the bottom of the table."
    ),
    (
        "Mount Everest, located on the border between Nepal and Tibet, stands at 8,849 "
        "meters above sea level, making it the tallest mountain on Earth. The first "
        "confirmed ascent was made on May 29, 1953, by Sir Edmund Hillary and Tenzing "
        "Norgay. The mountain was formed approximately 50 million years ago when the "
        "Indian tectonic plate collided with the Eurasian plate. Climbers typically "
        "attempt the summit during a narrow window in May when weather conditions are "
        "most favorable. The mountain continues to grow at a rate of about 4 millimeters "
        "per year due to ongoing geological forces."
    ),
    (
        "The human brain contains approximately 86 billion neurons, each connected to "
        "thousands of other neurons through synapses. The brain consumes about 20 "
        "percent of the body's total energy despite accounting for only 2 percent of "
        "body weight. Different regions of the brain are responsible for different "
        "functions: the frontal lobe handles decision-making and planning, the temporal "
        "lobe processes auditory information and memory, the parietal lobe integrates "
        "sensory information, and the occipital lobe processes visual information. The "
        "brain can process images in as little as 13 milliseconds and generates enough "
        "electrical power to light a small bulb."
    ),
    (
        "The Great Barrier Reef, located off the coast of Queensland, Australia, is the "
        "world's largest coral reef system. It stretches over 2,300 kilometers and is "
        "composed of over 2,900 individual reef systems and 900 islands. The reef is "
        "home to 1,500 species of fish, 411 types of hard coral, and dozens of species "
        "of sharks and rays. It can be seen from outer space and is the world's biggest "
        "single structure made by living organisms. The reef has experienced significant "
        "bleaching events due to rising ocean temperatures, with major events recorded "
        "in 1998, 2002, 2016, 2017, 2020, and 2022."
    ),
    (
        "The speed of light in a vacuum is exactly 299,792,458 meters per second, a "
        "fundamental constant of nature denoted by the letter c. Albert Einstein's "
        "special theory of relativity, published in 1905, established that nothing can "
        "travel faster than light and that the speed of light is the same for all "
        "observers regardless of their relative motion. Light from the Sun takes about "
        "8 minutes and 20 seconds to reach Earth. The nearest star system, Alpha "
        "Centauri, is approximately 4.37 light-years away. A light-year is the distance "
        "light travels in one year, approximately 9.461 trillion kilometers."
    ),
    (
        "The Sahara Desert in North Africa is the largest hot desert in the world, "
        "covering approximately 9.2 million square kilometers. It spans 11 countries "
        "including Algeria, Chad, Egypt, Libya, Mali, Mauritania, Morocco, Niger, "
        "Sudan, Tunisia, and Western Sahara. Despite its reputation as an endless sea "
        "of sand dunes, only about 25 percent of the Sahara is actually sandy. The "
        "rest consists of rocky plateaus, gravel plains, dry valleys, and mountains. "
        "The highest point in the Sahara is Emi Koussi, a shield volcano in Chad, "
        "reaching 3,445 meters above sea level."
    ),
    (
        "Jupiter is the largest planet in our solar system, with a mass more than twice "
        "that of all other planets combined. Its Great Red Spot is a persistent "
        "anticyclonic storm that has been observed since at least 1831 and is large "
        "enough to contain two or three Earths. Jupiter has at least 95 known moons, "
        "with the four largest known as the Galilean moons: Io, Europa, Ganymede, and "
        "Callisto. Ganymede is the largest moon in the solar system, even bigger than "
        "the planet Mercury. Jupiter's magnetic field is 14 times stronger than Earth's "
        "and extends millions of kilometers into space."
    ),
    (
        "The Pacific Ocean is the largest and deepest ocean on Earth, covering "
        "approximately 165.25 million square kilometers. It contains the Mariana Trench, "
        "the deepest point on Earth at approximately 10,994 meters below sea level. The "
        "Pacific Ocean contains more than half of the free water on Earth and could fit "
        "all the world's landmasses within its boundaries with room to spare. The Ring "
        "of Fire, a horseshoe-shaped zone of intense seismic and volcanic activity, "
        "encircles the Pacific and is home to about 75 percent of the world's active "
        "volcanoes and 90 percent of the world's earthquakes."
    ),
    (
        "The invention of the printing press by Johannes Gutenberg around 1440 is "
        "considered one of the most important events in human history. Before the "
        "printing press, books were copied by hand, a process that was slow, expensive, "
        "and prone to errors. Gutenberg's movable type system allowed for the rapid "
        "production of books and pamphlets, dramatically reducing their cost and "
        "increasing literacy rates across Europe. The first major book printed using "
        "this technology was the Gutenberg Bible, completed around 1455. Within 50 "
        "years of Gutenberg's invention, an estimated 20 million volumes had been "
        "printed across Europe."
    ),
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Needle:
    """A single needle (fact) embedded in the haystack."""
    city: str
    code: int
    position: float  # 0.0-1.0, where in the haystack it's inserted
    sentence: str = ""

    def __post_init__(self) -> None:
        self.sentence = f"The secret code for {self.city} is {self.code}."


@dataclass
class TrialResult:
    """Result of querying one needle."""
    city: str
    expected_code: int
    response: str
    found: bool


@dataclass
class ConfigResult:
    """Result for one (depth, needle_count, cache_type) configuration."""
    depth: int
    needle_count: int
    cache_type: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def accuracy(self) -> str:
        hits = sum(1 for t in self.trials if t.found)
        return f"{hits}/{len(self.trials)}"

    @property
    def accuracy_pct(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.found) / len(self.trials) * 100


# ---------------------------------------------------------------------------
# Haystack generation
# ---------------------------------------------------------------------------

def generate_needles(count: int, rng: random.Random) -> list[Needle]:
    """Generate N needles with deterministic random codes."""
    if count > len(CITIES):
        raise ValueError(
            f"--needles={count} exceeds available cities ({len(CITIES)}). "
            f"Add more cities to CITIES list or reduce needle count."
        )

    cities = CITIES[:count]
    if count == 1:
        positions = [0.5]
    else:
        # Spread needles from 5% to 95% of the context to cover the FULL range
        positions = [0.05 + (0.90 * i / (count - 1)) for i in range(count)]

    needles = []
    for city, pos in zip(cities, positions):
        code = rng.randint(100000, 999999)
        needles.append(Needle(city=city, code=code, position=pos))
    return needles


def generate_haystack(needles: list[Needle], target_chars: int) -> str:
    """Build filler text with needles inserted at specified positions.

    Args:
        needles: List of Needle objects with position (0.0-1.0).
        target_chars: Approximate total character count for the haystack.
    """
    # Build filler text by repeating paragraphs
    filler_parts: list[str] = []
    total_chars = 0
    idx = 0
    while total_chars < target_chars:
        para = FILLER_PARAGRAPHS[idx % len(FILLER_PARAGRAPHS)]
        filler_parts.append(para)
        total_chars += len(para) + 2  # +2 for paragraph breaks
        idx += 1

    filler_text = "\n\n".join(filler_parts)

    # Pre-compute all insertion positions based on original text length to avoid
    # position drift from earlier insertions shifting byte offsets.
    original_len = len(filler_text)
    insertion_plan: list[tuple[Needle, int]] = []
    for needle in needles:
        insert_pos = int(original_len * needle.position)
        # Find a paragraph boundary near the insert position
        newline_pos = filler_text.rfind("\n\n", 0, insert_pos)
        if newline_pos == -1:
            newline_pos = insert_pos
        insertion_plan.append((needle, newline_pos))

    # Insert in reverse order (by position) so earlier insertions don't shift
    # the byte offsets of later ones.
    insertion_plan.sort(key=lambda x: x[1], reverse=True)
    for needle, pos in insertion_plan:
        filler_text = (
            filler_text[:pos]
            + f"\n\n{needle.sentence}\n\n"
            + filler_text[pos:]
        )

    return filler_text


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

_active_server: Optional[subprocess.Popen] = None
_server_stderr_file: Optional[str] = None


def _cleanup_server() -> None:
    """Kill the server process if still running (atexit / signal handler)."""
    global _active_server, _server_stderr_file
    if _active_server is not None:
        try:
            _active_server.terminate()
            _active_server.wait(timeout=5)
        except Exception:
            try:
                _active_server.kill()
            except Exception:
                pass
        _active_server = None
    # Clean up stderr temp file if it exists
    if _server_stderr_file is not None:
        try:
            os.unlink(_server_stderr_file)
        except OSError:
            pass
        _server_stderr_file = None


atexit.register(_cleanup_server)


def _signal_handler(signum: int, frame) -> None:
    _cleanup_server()
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _find_free_port(start: int = 8090) -> int:
    """Find a free port starting from `start`.

    NOTE: There is an inherent TOCTOU race here -- the port could be grabbed by
    another process between our check and llama-server's bind.  Fixing this
    properly would require llama-server to support port 0 (OS-assigned) and
    report the actual port back, which it currently does not.
    """
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{start + 99}")


def start_server(
    llama_dir: Path,
    model_path: Path,
    cache_type: str,
    context_size: int,
    port: int,
    verbose: bool = False,
    server_timeout: int = 120,
    server_bin_override: Optional[Path] = None,
) -> subprocess.Popen:
    """Start llama-server and wait until it's healthy."""
    global _active_server, _server_stderr_file

    if server_bin_override is not None:
        server_bin = server_bin_override
    else:
        server_bin = llama_dir / "build" / "bin" / "llama-server"
    if not server_bin.exists():
        raise FileNotFoundError(f"llama-server not found at {server_bin}")

    cmd = [
        str(server_bin),
        "-m", str(model_path),
        "--cache-type-k", cache_type,
        "--cache-type-v", cache_type,
        "-c", str(context_size),
        "-ngl", "99",
        "-fa", "on",
        "--port", str(port),
        "-np", "1",
        "--jinja",
    ]

    if verbose:
        print(f"  [CMD] {' '.join(cmd)}")

    # Always capture stderr to a temp file for debugging, even in non-verbose mode.
    # In verbose mode we also tee to the console via stdout.
    if verbose:
        stderr_dest = None  # inherit terminal
        stdout_dest = None
        _server_stderr_file = None
    else:
        stderr_tmp = tempfile.NamedTemporaryFile(
            prefix="niah_server_stderr_", suffix=".log", delete=False, mode="w"
        )
        _server_stderr_file = stderr_tmp.name
        stderr_dest = stderr_tmp
        stdout_dest = subprocess.DEVNULL

    proc = subprocess.Popen(
        cmd,
        stdout=stdout_dest,
        stderr=stderr_dest,
    )
    _active_server = proc

    # Poll /health until ready
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + server_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr_snippet = _read_server_stderr()
            raise RuntimeError(
                f"llama-server exited prematurely with code {proc.returncode}\n"
                f"Server stderr:\n{stderr_snippet}"
            )
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok":
                    return proc
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError):
            pass
        time.sleep(0.5)

    stderr_snippet = _read_server_stderr()
    proc.terminate()
    proc.wait(timeout=5)
    _active_server = None
    raise TimeoutError(
        f"llama-server did not become healthy within {server_timeout} seconds\n"
        f"Server stderr:\n{stderr_snippet}"
    )


def _read_server_stderr(max_bytes: int = 4096) -> str:
    """Read the captured server stderr log, if available."""
    if _server_stderr_file is None:
        return "(stderr not captured -- run with --verbose to see server output)"
    try:
        with open(_server_stderr_file, "r") as f:
            content = f.read()
        # Return the last max_bytes chars (tail) which are most useful
        if len(content) > max_bytes:
            return f"... (truncated)\n{content[-max_bytes:]}"
        return content if content else "(empty)"
    except OSError:
        return "(could not read stderr log)"


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server."""
    global _active_server, _server_stderr_file
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)
    _active_server = None
    # Clean up stderr temp file
    if _server_stderr_file is not None:
        try:
            os.unlink(_server_stderr_file)
        except OSError:
            pass
        _server_stderr_file = None


# ---------------------------------------------------------------------------
# Query logic
# ---------------------------------------------------------------------------

def query_needle(
    port: int,
    haystack: str,
    city: str,
    timeout: int = 300,
    max_retries: int = 3,
) -> str:
    """Send a chat completion request querying a specific needle.

    Returns the model's response text.
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    user_content = (
        f"{haystack}\n\n"
        f"What is the secret code for {city}? Reply with only the number."
    )

    payload = json.dumps({
        "model": "model",
        "messages": [
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": 32,
    }).encode()

    headers = {
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Failed to query server after {max_retries} attempts: {e}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Unexpected response format: {e}")

    return ""  # unreachable, but keeps mypy happy


def score_response(response: str, expected_code: int) -> bool:
    """Check if the expected 6-digit code is the first 6-digit number in the response.

    Uses regex word-boundary match to avoid false positives from partial matches
    (e.g., matching '123456' inside '1234567').
    """
    match = re.search(r'\b(\d{6})\b', response)
    return match is not None and int(match.group(1)) == expected_code


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_config(
    llama_dir: Path,
    model_path: Path,
    cache_type: str,
    depth: int,
    needle_count: int,
    port: int,
    verbose: bool = False,
    server_timeout: int = 120,
    query_timeout: int = 300,
    server_bin: Optional[Path] = None,
    chars_per_token: float = 4.0,
) -> ConfigResult:
    """Run NIAH test for one (depth, needle_count, cache_type) configuration."""
    rng = random.Random(SEED)

    needles = generate_needles(needle_count, rng)
    # Token estimation: ~chars_per_token chars per token (approximate; override
    # with --chars-per-token if your model's tokenizer is significantly different).
    target_chars = int(depth * chars_per_token)
    if verbose:
        print(f"  Token estimation: {depth} tokens * {chars_per_token} chars/token = {target_chars:,} chars")
    haystack = generate_haystack(needles, target_chars)

    if verbose:
        print(f"\n  Haystack length: {len(haystack):,} chars (~{len(haystack) // int(chars_per_token):,} tokens)")
        for n in needles:
            print(f"  Needle: {n.city} = {n.code} @ {n.position:.0%}")

    result = ConfigResult(depth=depth, needle_count=needle_count, cache_type=cache_type)

    print(f"\n  Starting server: cache={cache_type}, ctx={depth}, needles={needle_count}")
    # Use context size with some headroom for the query + response
    server_ctx = depth + 512
    proc = start_server(
        llama_dir, model_path, cache_type, server_ctx, port, verbose,
        server_timeout=server_timeout, server_bin_override=server_bin,
    )

    try:
        for needle in needles:
            print(f"    Querying {needle.city}...", end=" ", flush=True)
            response = query_needle(port, haystack, needle.city, timeout=query_timeout)
            found = score_response(response, needle.code)
            result.trials.append(TrialResult(
                city=needle.city,
                expected_code=needle.code,
                response=response,
                found=found,
            ))
            status = "HIT" if found else "MISS"
            print(f"{status} (expected={needle.code}, got={response!r})")
    finally:
        stop_server(proc)

    return result


def run_all(args: argparse.Namespace) -> list[ConfigResult]:
    """Run all configurations and return results."""
    llama_dir = Path(args.llama_dir)
    model_path = Path(args.model_path)
    depths = [int(d) for d in args.depths.split(",")]
    needle_counts = [int(n) for n in args.needles.split(",")]
    cache_types = [c.strip() for c in args.cache_types.split(",")]
    port = int(args.port)

    # Validate needle counts against available cities
    max_needles = max(needle_counts)
    if max_needles > len(CITIES):
        print(
            f"Error: --needles includes {max_needles} but only {len(CITIES)} cities "
            f"are available. Reduce needle count or add more cities.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve server binary
    server_bin: Optional[Path] = None
    if args.server_bin:
        server_bin = Path(args.server_bin)
        if not server_bin.exists():
            print(f"Error: server binary not found at {server_bin}", file=sys.stderr)
            sys.exit(1)
    else:
        # Validate paths
        if not llama_dir.exists():
            print(f"Error: llama.cpp directory not found: {llama_dir}", file=sys.stderr)
            sys.exit(1)
        default_bin = llama_dir / "build" / "bin" / "llama-server"
        if not default_bin.exists():
            print(f"Error: llama-server not found at {default_bin}", file=sys.stderr)
            sys.exit(1)

    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    total_configs = len(depths) * len(needle_counts) * len(cache_types)
    print(f"NIAH Test: {total_configs} configurations")
    print(f"  Model: {model_path.name}")
    print(f"  Depths: {depths}")
    print(f"  Needle counts: {needle_counts}")
    print(f"  Cache types: {cache_types}")
    print(f"  Port: {port}")
    print(f"  Seed: {SEED}")
    print(f"  Chars/token: {args.chars_per_token}")

    results: list[ConfigResult] = []
    config_num = 0

    for depth in depths:
        for needle_count in needle_counts:
            for cache_type in cache_types:
                config_num += 1
                print(f"\n{'='*60}")
                print(f"Config {config_num}/{total_configs}: "
                      f"depth={depth}, needles={needle_count}, cache={cache_type}")
                print(f"{'='*60}")

                try:
                    # Find a free port in case the default is occupied
                    actual_port = _find_free_port(port)
                    result = run_config(
                        llama_dir, model_path, cache_type, depth,
                        needle_count, actual_port, args.verbose,
                        server_timeout=args.server_timeout,
                        query_timeout=args.query_timeout,
                        server_bin=server_bin,
                        chars_per_token=args.chars_per_token,
                    )
                    results.append(result)
                    print(f"  Result: {result.accuracy} ({result.accuracy_pct:.0f}%)")
                except Exception as e:
                    print(f"  ERROR: {e}", file=sys.stderr)
                    # Record a failed config with zero hits
                    failed = ConfigResult(
                        depth=depth,
                        needle_count=needle_count,
                        cache_type=cache_type,
                    )
                    results.append(failed)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def build_table(results: list[ConfigResult], model_name: str) -> str:
    """Build a markdown comparison table from results."""
    # Group results by (depth, needle_count)
    cache_types = sorted(set(r.cache_type for r in results))
    lookup: dict[tuple[int, int, str], ConfigResult] = {}
    for r in results:
        lookup[(r.depth, r.needle_count, r.cache_type)] = r

    depths = sorted(set(r.depth for r in results))
    needle_counts = sorted(set(r.needle_count for r in results))

    lines = [
        f"## NIAH Results: {model_name}",
        "",
        "| Context | Needles |",
    ]

    # Build header dynamically based on cache types
    header = "| Context | Needles |"
    separator = "|---------|---------|"
    for ct in cache_types:
        header += f" {ct} |"
        separator += f"{'─' * (max(len(ct), 6) + 2)}|"

    # Delta column only makes sense when comparing exactly 2 cache types
    if len(cache_types) == 2:
        header += " Delta |"
        separator += "-------|"

    lines = [
        f"## NIAH Results: {model_name}",
        "",
        header,
        separator,
    ]

    for depth in depths:
        depth_label = f"{depth // 1024}K" if depth >= 1024 else str(depth)
        for nc in needle_counts:
            row = f"| {depth_label:<7} | {nc:<7} |"
            pcts = []
            for ct in cache_types:
                r = lookup.get((depth, nc, ct))
                if r and r.trials:
                    row += f" {r.accuracy:<{max(len(ct), 6)}} |"
                    pcts.append(r.accuracy_pct)
                else:
                    row += f" {'ERR':<{max(len(ct), 6)}} |"
                    pcts.append(None)

            if len(cache_types) == 2 and all(p is not None for p in pcts):
                delta = pcts[1] - pcts[0]  # type: ignore[operator]
                sign = "+" if delta > 0 else ""
                row += f" {sign}{delta:.0f}%   |"
            elif len(cache_types) == 2:
                row += " N/A   |"

            lines.append(row)

    return "\n".join(lines)


def save_results(
    results: list[ConfigResult],
    model_name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save results as JSON and markdown table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # JSON results
    json_path = output_dir / f"niah_results_{timestamp}.json"
    json_data = {
        "model": model_name,
        "seed": SEED,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [
            {
                "depth": r.depth,
                "needle_count": r.needle_count,
                "cache_type": r.cache_type,
                "accuracy": r.accuracy,
                "accuracy_pct": r.accuracy_pct,
                "trials": [
                    {
                        "city": t.city,
                        "expected_code": t.expected_code,
                        "response": t.response,
                        "found": t.found,
                    }
                    for t in r.trials
                ],
            }
            for r in results
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Markdown table
    md_path = output_dir / f"niah_results_{timestamp}.md"
    table = build_table(results, model_name)
    with open(md_path, "w") as f:
        f.write(table + "\n")

    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TurboQuant Needle-In-A-Haystack (NIAH) test. "
        "Measures retrieval accuracy across cache types, context depths, and needle counts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python3 scripts/niah_test.py /path/to/llama.cpp /path/to/model.gguf
              python3 scripts/niah_test.py /path/to/llama.cpp /path/to/model.gguf --depths 4096,8192 --needles 1,5
              python3 scripts/niah_test.py /path/to/llama.cpp /path/to/model.gguf --cache-types q8_0 --verbose
        """),
    )

    parser.add_argument(
        "llama_dir",
        help="Path to the llama.cpp directory (must contain build/bin/llama-server)",
    )
    parser.add_argument(
        "model_path",
        default=None,
        nargs="?",
        help="Path to the GGUF model file (required)",
    )
    parser.add_argument(
        "--depths",
        default="4096,8192,16384,32768",
        help="Comma-separated context depths to test (default: %(default)s)",
    )
    parser.add_argument(
        "--needles",
        default="1,5,10",
        help="Comma-separated needle counts to test (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        default="8090",
        help="Base port for llama-server (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for results files (default: ./niah_results/)",
    )
    parser.add_argument(
        "--cache-types",
        default="q8_0,turbo3",
        help="Comma-separated cache types to test (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show server output and detailed needle info",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for llama-server to become healthy (default: %(default)s)",
    )
    parser.add_argument(
        "--query-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each query request (default: %(default)s). "
             "32K prefill on M1 can take 130s+, so the default is generous.",
    )
    parser.add_argument(
        "--server-bin",
        default=None,
        help="Path to a specific llama-server binary (e.g., ROCm build, Windows .exe). "
             "Overrides the default <llama_dir>/build/bin/llama-server.",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="Approximate characters per token for haystack sizing (default: %(default)s). "
             "Override if your model's tokenizer differs significantly (e.g., 3.5 for CJK-heavy models).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Validate model_path is provided
    if args.model_path is None:
        print(
            "Error: model_path is required. Pass the path to a GGUF model file as the second argument.\n"
            "Usage: python3 scripts/niah_test.py /path/to/llama.cpp /path/to/model.gguf",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve output dir
    if args.output_dir is None:
        args.output_dir = "niah_results"
    output_dir = Path(args.output_dir)

    model_name = Path(args.model_path).stem

    print(f"{'='*60}")
    print(f"  TurboQuant NIAH Test")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}")

    results = run_all(args)

    # Print table
    table = build_table(results, model_name)
    print(f"\n\n{table}\n")

    # Save results
    json_path, md_path = save_results(results, model_name, output_dir)
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Table: {md_path}")


if __name__ == "__main__":
    main()
