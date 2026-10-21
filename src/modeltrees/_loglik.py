# Ported from partykit 1.3-0 (GPL-2 | GPL-3), R/modelparty.R (logLik.modelparty).
# See THIRD_PARTY_NOTICES.md.
"""The log-likelihood of a grown tree, as ``logLik.modelparty``.

Port of partykit 1.3-0 ``logLik.modelparty`` and ``nobs.modelparty``
(reference/partykit/R/modelparty.R:907-926): the sum of the leaf
log-likelihoods, their df plus ``dfsplit`` per inner node, and the sum of
the leaf ``nobs`` (docs/api-design.md, section 3.6).
"""

from dataclasses import dataclass

from modeltrees.node_model import LogLikModel
from modeltrees.tree import GrownTree


@dataclass(frozen=True)
class LogLik:
    value: float  # NaN if a leaf fit failed
    df: int
    nobs: float


def tree_loglik(tree: GrownTree, model: LogLikModel) -> LogLik:
    """Sum the leaf log-likelihoods of ``tree`` fitted by ``model``."""
    nodes = tree.nodes
    leaves = [node for node in nodes if node.terminal]
    value = 0.0
    df = 0
    for leaf in leaves:
        v, d = model.loglik(leaf.fit)
        value += v
        df += d
    # dfsplit counts once per inner node (length(object) - length(ll)).
    df += int(tree.control.dfsplit) * (len(nodes) - len(leaves))
    nobs = sum(leaf.nobs for leaf in leaves)
    return LogLik(value, df, float(nobs))
