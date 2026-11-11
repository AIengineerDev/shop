"""AC-R5: no float arithmetic on the money path."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from repricer import config

ENGINE_DIR = config.PACKAGE_DIR / "engine"
FILES = sorted(ENGINE_DIR.glob("*.py")) + [config.PACKAGE_DIR / "money.py"]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_float_on_money_path(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "float":
            problems.append(f"{path.name}:{node.lineno} uses float")
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            problems.append(f"{path.name}:{node.lineno} float literal {node.value!r}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            src = ast.unparse(node)
            if "Decimal(" not in src:
                problems.append(f"{path.name}:{node.lineno} true division without Decimal: {src}")
    assert not problems, "\n".join(problems)
