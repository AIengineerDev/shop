"""Test helpers."""

from __future__ import annotations

import subprocess
import sys


def assert_modules_do_not_load(modules: list[str], *, forbidden_prefix: str) -> None:
    """Import `modules` in a fresh interpreter and assert nothing under `forbidden_prefix` got loaded."""
    code = (
        "import sys, importlib\n"
        f"for m in {modules!r}:\n    importlib.import_module(m)\n"
        f"bad = sorted(m for m in sys.modules if m.startswith({forbidden_prefix!r}))\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"forbidden modules loaded: {out.stdout.strip()}"


# ---- e2e / load helpers: real subprocesses and a live HTTP server ---------------------------

import contextlib  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def subprocess_env(**extra: str) -> dict[str, str]:
    """Environment for CLI subprocesses: fake LLM, no API key, no repo .env (spec §8: no network)."""
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "LLM_PROVIDER")}
    env.update({"LLM_PROVIDER": "fake", "ENV_FILE": str(REPO / "tests" / "no-such.env")})
    env.update(extra)
    return env


def run_cli(*args: str, env: dict[str, str] | None = None, timeout: float = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *args], cwd=REPO, env=env or subprocess_env(), capture_output=True,
                          text=True, timeout=timeout)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def live_server(db: Path, *, workers: int = 1, startup_s: float = 30) -> Iterator[tuple[str, dict[str, str]]]:
    """Run scripts/serve.py as a real process. Yields (base_url, logs) where logs["stderr"] is
    filled in after shutdown. Stops with SIGINT so coverage data and logs are flushed."""
    import tempfile

    import httpx

    port = free_port()
    # Logs go to files, not pipes: an undrained PIPE fills after ~64 KB of access-log lines and
    # then blocks the server's event loop on its next log write (a harness hang, not a server bug).
    out_f = tempfile.TemporaryFile(mode="w+")
    err_f = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(
        [sys.executable, "scripts/serve.py", "--db", str(db), "--port", str(port), "--workers", str(workers)],
        cwd=REPO, env=subprocess_env(), stdout=out_f, stderr=err_f, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    logs: dict[str, str] = {}
    deadline = time.monotonic() + startup_s
    while True:
        if proc.poll() is not None:
            err_f.seek(0)
            raise RuntimeError(f"server exited early: {err_f.read()}")
        try:
            if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            proc.kill()
            raise RuntimeError("server did not start")
        time.sleep(0.1)
    try:
        yield base, logs
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        out_f.seek(0)
        err_f.seek(0)
        logs["stdout"], logs["stderr"] = out_f.read(), err_f.read()
        out_f.close()
        err_f.close()
