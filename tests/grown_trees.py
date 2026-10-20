"""Every tree and extension tree fixture grown by mob(), cached per session.

The model, inputs and control are rebuilt from the fixture as the per-model
test modules do. Shared by the tests that need all fixture trees at once.
"""

import math
import warnings
from functools import cache
from typing import Any

import numpy as np
from fixtures_io import FIXTURES, load_json
from tree_fixtures import cluster, data, inputs, variables

from modeltrees.errors import GlmFitWarning, TooFewClustersWarning
from modeltrees.models.glm import GlmModel
from modeltrees.models.lm import LmModel
from modeltrees.models.multinom import MultinomModel
from modeltrees.models.negbin import NegBinModel
from modeltrees.models.safelogit import SafeLogitModel
from modeltrees.node_model import NodeModel
from modeltrees.tree import MobControl, ModelTree, mob

TREES = sorted(p.stem for p in (FIXTURES / "trees").glob("*.json"))
EXTENSION_TREES = sorted(p.stem for p in (FIXTURES / "extension_trees").glob("*.json"))
# (directory, name) of every tree fixture.
ALL_TREES = [("trees", n) for n in TREES] + [
    ("extension_trees", n) for n in EXTENSION_TREES
]


@cache
def fixture(directory: str, name: str) -> dict[str, Any]:
    f: dict[str, Any] = load_json(f"{directory}/{name}.json")
    return f


def model(f: dict[str, Any]) -> NodeModel:
    names = f["coefficient_names"]
    if f["function_name"] == "lmtree":
        return LmModel(coef_names=names)
    if f["function_name"] == "glmtree":
        return GlmModel(f["family"]["family"], coef_names=names, **f["glm_control"])
    fit_function = f["fit_function"]
    if fit_function == "negbinfit":
        return NegBinModel(coef_names=names, **f["fit_args"])
    if fit_function == "safelogitfit":
        return SafeLogitModel(coef_names=names, **f["fit_args"])
    assert fit_function == "multinomfit"
    return MultinomModel(
        f["response_levels"], coef_names=["(Intercept)", *f["regressors"]]
    )


def tree_inputs(
    f: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Response, regressors, weights and offset; multinomial y as codes."""
    if f.get("fit_function") != "multinomfit":
        return inputs(f)
    d, _ = data(f["data_csv"])
    y = d[f["response"]]
    x = np.column_stack([np.ones(y.shape[0])] + [d[r] for r in f["regressors"]])
    w = d[f["weights"]] if f["weights"] else np.ones(y.shape[0])
    return y, x, w, None


def control(f: dict[str, Any]) -> MobControl:
    c = f["control"]
    maxdepth = math.inf if c["maxdepth"] == "Inf" else float(c["maxdepth"])
    return MobControl(
        alpha=c["alpha"],
        bonferroni=c["bonferroni"],
        minsize=c["minsize"],
        maxdepth=maxdepth,
        trim=c["trim"],
        dfsplit=c["dfsplit"],
        prune=c["prune"],
        restart=c["restart"],
        caseweights=c["caseweights"],
        numsplit=c["numsplit"],
        catsplit=c["catsplit"],
    )


@cache
def grow(directory: str, name: str) -> ModelTree:
    f = fixture(directory, name)
    y, x, w, off = tree_inputs(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GlmFitWarning)
        warnings.simplefilter("ignore", TooFewClustersWarning)
        return mob(
            model(f),
            y,
            x,
            variables(f),
            weights=w,
            offset=off,
            cluster=cluster(f),
            control=control(f),
        )
