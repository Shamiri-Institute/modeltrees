"""Model, inputs and partitioning variables rebuilt from the tree fixtures."""

from functools import cache
from typing import Any

import numpy as np
from fixtures_io import FIXTURES, load_dataset, load_json

from modeltrees.instability import VariableKind
from modeltrees.models.glm import GlmModel, binomial_response
from modeltrees.models.lm import LmModel
from modeltrees.node_model import NodeModel
from modeltrees.tree import Variable

TREE_FILES = sorted(p.stem for p in (FIXTURES / "trees").glob("*.json"))

_KIND = {
    "factor": VariableKind.NOMINAL,
    "ordered": VariableKind.ORDERED,
    "double": VariableKind.NUMERIC,
    "integer": VariableKind.NUMERIC,
}


@cache
def tree(name: str) -> dict[str, Any]:
    fixture: dict[str, Any] = load_json(f"trees/{name}.json")
    return fixture


@cache
def data(csv: str) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    return load_dataset(csv)


def model(fixture: dict[str, Any]) -> NodeModel:
    names = fixture["coefficient_names"]
    if fixture["function_name"] == "lmtree":
        return LmModel(coef_names=names)
    assert fixture["function_name"] == "glmtree"
    return GlmModel(
        fixture["family"]["family"], coef_names=names, **fixture["glm_control"]
    )


def offset(fixture: dict[str, Any], d: dict[str, np.ndarray]) -> np.ndarray | None:
    expr = fixture["offset"]
    if expr is None:
        return None
    if expr.startswith("log(") and expr.endswith(")"):
        return np.log(d[expr[4:-1]])
    return d[expr]


def inputs(
    fixture: dict[str, Any], csv: str | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Response, regressors (None for intercept only), weights and offset."""
    d, columns = data(csv or fixture["data_csv"])
    response = fixture["response"]
    if columns[response]["type"] == "factor":
        # The factor arrives as codes in R's level order; binomial_response
        # applies R's rule: first level failure, the rest success.
        levels = columns[response]["levels"]
        assert levels == fixture["response_levels"]
        y = binomial_response(np.array(levels)[d[response]], levels=levels)
    else:
        y = d[response]
    regressors = fixture["regressors"]
    x = (
        np.column_stack([np.ones(y.shape[0])] + [d[r] for r in regressors])
        if regressors
        else None
    )
    w = d[fixture["weights"]] if fixture["weights"] else np.ones(y.shape[0])
    return y, x, w, offset(fixture, d)


def variables(fixture: dict[str, Any], csv: str | None = None) -> dict[str, Variable]:
    """The partitioning variables, factors as codes with R's levels."""
    d, columns = data(csv or fixture["data_csv"])
    out = {}
    for name in fixture["partitioning"]:
        col = columns[name]
        levels = tuple(col["levels"]) if col.get("levels") else None
        out[name] = Variable(name, _KIND[col["type"]], d[name], levels)
    return out


def cluster(fixture: dict[str, Any], csv: str | None = None) -> np.ndarray | None:
    name = fixture.get("cluster")
    return None if name is None else data(csv or fixture["data_csv"])[0][name]
