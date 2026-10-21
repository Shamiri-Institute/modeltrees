"""Text-mode file I/O names its encoding, so reads and writes do not depend on
the platform default (cp1252 on many Windows installs)."""

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIRS = ("src", "tests", "scripts")


def _mode(call: ast.Call, position: int) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value
    return call.args[position] if len(call.args) > position else None


def _needs_encoding(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
        return True
    if isinstance(func, ast.Name) and func.id == "open":
        mode = _mode(call, 1)
    elif isinstance(func, ast.Attribute) and func.attr == "open":
        # os.open takes flags and returns a file descriptor.
        if isinstance(func.value, ast.Name) and func.value.id == "os":
            return False
        mode = _mode(call, 0)
    else:
        return False
    binary = (
        isinstance(mode, ast.Constant)
        and isinstance(mode.value, str)
        and "b" in mode.value
    )
    return not binary


def _missing_encoding(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        f"{path.relative_to(ROOT)}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _needs_encoding(node)
        and not any(kw.arg == "encoding" for kw in node.keywords)
    ]


def test_text_io_names_its_encoding() -> None:
    files = [p for d in DIRS for p in sorted((ROOT / d).rglob("*.py"))]
    assert files
    missing = [hit for p in files for hit in _missing_encoding(p)]
    assert missing == []


def test_scan_flags_calls_without_encoding() -> None:
    source = (
        "open(p)\n"
        "open(p, 'w')\n"
        "p.open()\n"
        "p.read_text()\n"
        "p.write_text(s)\n"
        "open(p, 'rb')\n"
        "p.open(mode='wb')\n"
        "os.open(p, flags)\n"
        "open(p, encoding='utf-8')\n"
    )
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)]
    flagged = sorted(
        c.lineno
        for c in calls
        if _needs_encoding(c) and not any(kw.arg == "encoding" for kw in c.keywords)
    )
    assert flagged == [1, 2, 3, 4, 5]
