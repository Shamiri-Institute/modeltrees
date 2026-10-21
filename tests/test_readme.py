"""The README's Python code blocks run as written."""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


def python_blocks() -> list[str]:
    text = README.read_text(encoding="utf-8")
    return re.findall(r"^```python\n(.*?)^```$", text, flags=re.DOTALL | re.MULTILINE)


def test_readme_has_python_blocks() -> None:
    assert python_blocks()


def printed_tree() -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(
        r"^The printed tree:\n\n```\n(.*?)^```$", text, flags=re.DOTALL | re.MULTILINE
    )
    assert match is not None
    return match.group(1)


@pytest.mark.parametrize("index", range(len(python_blocks())))
def test_readme_block_runs(index: int, capsys: pytest.CaptureFixture[str]) -> None:
    code = python_blocks()[index]
    # The README is part of this repository, so running its code is intended.
    exec(compile(code, f"README.md block {index}", "exec"), {"__name__": "readme"})  # noqa: S102
    out = capsys.readouterr().out
    assert out
    if index == 0:
        assert printed_tree() in out
