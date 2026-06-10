"""
``ops.strategy`` namespace — ADR 0057 Phase A.

Constructs :class:`~apeGmsh.opensees.analysis.strategy.Ladder`
declarations and the established :func:`profile` presets.  Unlike the
other bridge namespaces, strategy declarations are NOT registered
primitives — a ladder never emits standalone; it parameterizes an
analyze loop (``s.run(..., strategy=)`` / ``apeSees.analyze(...,
strategy=)``), so there is no tag to allocate and nothing for the
topological emit pass to order.  H5 persistence is ADR 0057 Phase C.
"""
from __future__ import annotations

from typing import Sequence

from ...analysis.strategy import PROFILE_NAMES, profile as _profile
from ...analysis.strategy import Ladder as _Ladder
from ..types import SolutionAlgorithm
from ._base import _BridgeNamespace

__all__ = ["_StrategyNS"]


class _StrategyNS(_BridgeNamespace):
    """``ops.strategy.<verb>(...)`` — ADR 0057 solution-strategy ladders."""

    def Ladder(
        self,
        *,
        rungs: Sequence[SolutionAlgorithm],
        name: str = "custom",
    ) -> _Ladder:
        """Declare a bespoke escalation ladder.

        ``rungs`` lists solution-algorithm primitives
        (``ops.algorithm.*``) in escalation order; the analysis
        chain's own algorithm is implicitly rung 0.  See ADR 0057 §2
        for the per-increment walk semantics.
        """
        return _Ladder(rungs=tuple(rungs), name=name)

    def profile(self, name: str) -> _Ladder:
        """Return an established profile ladder by name.

        Canonical names: ``{}`` (aliases: ``geotech`` /
        ``mohr-coulomb`` → ``non-smooth``; ``metal`` →
        ``smooth-hardening``).  See ADR 0057 §3 for each profile's
        rationale — the orderings are evidence-based, notably
        ``"non-smooth"`` carries NO line-search rung.
        """
        return _profile(name)

    # Render the canonical names into the docstring once at import time
    # so help(ops.strategy.profile) lists them without drift.
    profile.__doc__ = (profile.__doc__ or "").format(", ".join(PROFILE_NAMES))
