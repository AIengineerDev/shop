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
