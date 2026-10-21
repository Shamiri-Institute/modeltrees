"""Import rules of CLAUDE.md: no pandas, polars or statsmodels in the
library, and narwhals only through the dataframe boundary modeltrees._frame."""

import ast
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "modeltrees"
BOUNDARY = SRC / "_frame.py"

# Records the modeltrees module whose import first loads narwhals, by walking
# the stack from the import hook to the nearest modeltrees frame.
_SCRIPT = """
import importlib, json, pkgutil, sys

narwhals_importers = []

class _Watch:
    def find_spec(self, name, path=None, target=None):
        if name == "narwhals":
            frame = sys._getframe(1)
            while frame is not None:
                module = frame.f_globals.get("__name__", "")
                if module.startswith("modeltrees"):
                    narwhals_importers.append(module)
                    break
                frame = frame.f_back
            else:
                narwhals_importers.append(None)
        return None

sys.meta_path.insert(0, _Watch())

import modeltrees
names = ["modeltrees"]
for info in pkgutil.walk_packages(modeltrees.__path__, "modeltrees."):
    importlib.import_module(info.name)
    names.append(info.name)
print(json.dumps({
    "imported": names,
    "forbidden": sorted(
        m for m in ("pandas", "polars", "statsmodels") if m in sys.modules
    ),
    "narwhals_importers": narwhals_importers,
}))
"""


def _run() -> dict[str, list[str | None]]:
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, check=True
    )
    result: dict[str, list[str | None]] = json.loads(out.stdout)
    return result


def test_library_does_not_import_pandas_polars_or_statsmodels() -> None:
    result = _run()
    assert result["forbidden"] == []
    # The walk found the subpackages, so the check covers them.
    assert {
        "modeltrees.instability",
        "modeltrees.pvalues",
        "modeltrees.node_model",
        "modeltrees.models.glm",
        "modeltrees.models.glm_fit",
        "modeltrees.models.lm",
    } <= set(result["imported"])


def test_narwhals_loads_only_through_the_boundary() -> None:
    importers = _run()["narwhals_importers"]
    assert set(importers) <= {"modeltrees._frame"}


def _imports_narwhals(path: Path) -> bool:
    for node in ast.walk(
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    ):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "narwhals" for a in node.names):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and (node.module or "").split(".")[0] == "narwhals"
        ):
            return True
    return False


def test_no_module_but_the_boundary_imports_narwhals() -> None:
    sources = sorted(SRC.rglob("*.py"))
    assert sources
    offenders = [p for p in sources if p != BOUNDARY and _imports_narwhals(p)]
    assert offenders == []
