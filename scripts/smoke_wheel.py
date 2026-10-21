"""Smoke test for an installed wheel.

Run it with the interpreter of a fresh virtual environment that has only
the wheel and its runtime dependencies installed, from a directory outside
the repository so that ``import modeltrees`` finds the installed package:

    python scripts/smoke_wheel.py

It grows a tree with each of the five entry points on small simulated
data given as a dict of numpy arrays, calls ``predict`` and ``vcov``, and
checks that no dataframe library or statsmodels was imported.
"""

import sys

import numpy as np

import modeltrees as mt


def simulate(n: int = 400, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z1 = rng.uniform(size=n)
    z2 = rng.choice(np.array(["a", "b", "c"]), size=n)
    school = rng.integers(0, 20, size=n)
    eta = np.where(z1 <= 0.5, -0.5 + 1.0 * x, 0.5 - 1.0 * x)
    p = 1.0 / (1.0 + np.exp(-eta))
    mu = np.exp(eta / 2.0)
    theta = 2.0
    probs = np.column_stack([np.ones(n), np.exp(eta), np.exp(-eta)])
    probs /= probs.sum(axis=1, keepdims=True)
    classes = np.array(["lo", "mid", "hi"])
    return {
        "y_gauss": eta + rng.normal(size=n),
        "y_bin": (rng.uniform(size=n) < p).astype(np.float64),
        "y_count": rng.negative_binomial(theta, theta / (theta + mu)).astype(
            np.float64
        ),
        "y_class": classes[[rng.choice(3, p=row) for row in probs]],
        "x": x,
        "z1": z1,
        "z2": z2,
        "school": school,
    }


def check(tree: mt.ModelTree, label: str) -> None:
    if label == "multinomtree":
        pred = tree.predict(type="prob")
    else:
        pred = tree.predict(type="response")
    nodes = tree.predict(type="node")
    assert pred.shape[0] == tree.n_rows, label
    assert set(np.unique(nodes)) <= set(tree.leaves), label
    covs = tree.vcov()
    assert set(covs) == set(tree.leaves), label
    for cov in covs.values():
        assert cov.matrix.shape == (len(cov.names), len(cov.names)), label
    print(
        f"{label}: {len(tree.leaves)} leaves, vcov status "
        f"{sorted({c.status for c in covs.values()})}"
    )


def main() -> None:
    data = simulate()
    trees = {
        "lmtree": mt.lmtree("y_gauss ~ x | z1 + z2", data, cluster="school"),
        "glmtree": mt.glmtree(
            "y_bin ~ x | z1 + z2", data, family="binomial", cluster="school"
        ),
        "negbintree": mt.negbintree("y_count ~ x | z1 + z2", data),
        "safelogittree": mt.safelogittree("y_bin ~ x | z1 + z2", data),
        "multinomtree": mt.multinomtree(
            "y_class ~ x | z1 + z2", data, levels={"y_class": ["lo", "mid", "hi"]}
        ),
    }
    for label, tree in trees.items():
        check(tree, label)
    for name in ("pandas", "polars", "statsmodels"):
        assert name not in sys.modules, f"{name} was imported"
    print(f"modeltrees {mt.__version__} from {mt.__file__}: ok")


if __name__ == "__main__":
    main()
