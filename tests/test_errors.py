"""The public exception and warning classes (docs/api-design.md, section 7)."""

import pytest
from grown_trees import ALL_TREES, fixture, grow

import modeltrees
from modeltrees import errors
from modeltrees.errors import (
    DataError,
    FitError,
    FormulaError,
    GlmFitWarning,
    MissingDataWarning,
    ModelTreesError,
    ModelTreesWarning,
    NodeFitFailedWarning,
    TooFewClustersWarning,
    UnseenLevelError,
)
from modeltrees.instability import InstabilityTestError, TooFewClustersError
from modeltrees.models.glm_fit import GlmFitError
from modeltrees.models.negbin import NegBinError


def test_exception_hierarchy() -> None:
    for cls in (FormulaError, DataError, FitError):
        assert issubclass(cls, ModelTreesError)
        assert issubclass(cls, ValueError)
    assert issubclass(UnseenLevelError, DataError)
    assert issubclass(GlmFitError, FitError)
    assert issubclass(NegBinError, FitError)
    # Everything that was a ValueError stays one.
    assert issubclass(GlmFitError, ValueError)
    assert issubclass(NegBinError, ValueError)
    assert FormulaError("bad", position=3).position == 3
    assert str(GlmFitError("no coefficients")) == "no coefficients"


def test_warning_hierarchy() -> None:
    for cls in (
        NodeFitFailedWarning,
        TooFewClustersWarning,
        MissingDataWarning,
        GlmFitWarning,
    ):
        assert issubclass(cls, ModelTreesWarning)
    assert issubclass(ModelTreesWarning, UserWarning)


def test_old_import_paths_are_the_same_classes() -> None:
    from modeltrees import instability, tree
    from modeltrees.models import families

    assert instability.TooFewClustersWarning is errors.TooFewClustersWarning
    assert tree.NodeFitFailedWarning is errors.NodeFitFailedWarning
    assert families.GlmFitWarning is errors.GlmFitWarning


def test_internal_exceptions_are_not_public() -> None:
    assert not issubclass(InstabilityTestError, ModelTreesError)
    assert not issubclass(TooFewClustersError, ModelTreesError)
    assert "InstabilityTestError" not in modeltrees.__all__
    assert "TooFewClustersError" not in modeltrees.__all__


def test_public_names_are_exported() -> None:
    for name in (
        "ModelTreesError",
        "FormulaError",
        "DataError",
        "UnseenLevelError",
        "FitError",
        "ModelTreesWarning",
        "NodeFitFailedWarning",
        "TooFewClustersWarning",
        "MissingDataWarning",
        "GlmFitWarning",
    ):
        assert name in modeltrees.__all__
        assert getattr(modeltrees, name) is getattr(errors, name)


@pytest.mark.parametrize(
    ("directory", "name"), ALL_TREES, ids=[f"{d}/{n}" for d, n in ALL_TREES]
)
def test_internal_exceptions_never_escape_mob(directory: str, name: str) -> None:
    """Growing every fixture tree raises no instability-test exception.

    The core catches InstabilityTestError (and TooFewClustersError, its
    subclass) at every node, as partykit's try() does. The fixtures include
    nodes where the tests fail (sim_cluster_few_lmtree_cluster).
    """
    try:
        grown = grow(directory, name)
    except InstabilityTestError as exc:  # pragma: no cover - the failure case
        pytest.fail(f"{type(exc).__name__} escaped mob(): {exc}")
    assert len(grown.nodes) == fixture(directory, name)["summary"]["n_nodes"] or (
        name == "sim_gaussian_lmtree_zero_weights"
    )
