"""The library must not import pandas or statsmodels (CLAUDE.md)."""

import json
import subprocess
import sys

_SCRIPT = """
import importlib, json, pkgutil, sys
import modeltrees
names = ["modeltrees"]
for info in pkgutil.walk_packages(modeltrees.__path__, "modeltrees."):
    importlib.import_module(info.name)
    names.append(info.name)
print(json.dumps({
    "imported": names,
    "forbidden": sorted(m for m in ("pandas", "statsmodels") if m in sys.modules),
}))
"""


def test_library_does_not_import_pandas_or_statsmodels() -> None:
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, check=True
    )
    result = json.loads(out.stdout)
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
