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

from typing import Any, Sequence

from ...analysis.strategy import PROFILE_NAMES, profile as _profile
from ...analysis.strategy import Ladder as _Ladder
from ...analysis.strategy import OpenSeesPyDriver as _OpenSeesPyDriver
from ...analysis.strategy import Substep as _Substep
from ..types import SolutionAlgorithm
from ._base import _BridgeNamespace

__all__ = ["_StrategyNS"]


class _StrategyNS(_BridgeNamespace):
    """``ops.strategy.<verb>(...)`` — ADR 0057 solution-strategy ladders."""

    def Ladder(
        self,
        *,
        rungs: Sequence[SolutionAlgorithm | _Substep],
        name: str = "custom",
    ) -> _Ladder:
        """Declare a bespoke escalation ladder.

        ``rungs`` lists solution-algorithm primitives
        (``ops.algorithm.*``) in escalation order, optionally with one
        :meth:`Substep`; the analysis chain's own algorithm is
        implicitly rung 0.  See ADR 0057 §2 for the per-increment walk
        semantics.
        """
        return _Ladder(rungs=tuple(rungs), name=name)

    def Substep(
        self,
        *,
        node: int,
        dof: int,
        target: float,
        ds: float,
        ds_min: float,
        ds_max: float | None = None,
        regrow: float = 2.0,
        regrow_after: int = 2,
        budget: int = 8,
        wall_budget: float | None = None,
        plateau: float | None = None,
        plateau_window: float | None = None,
    ) -> _Substep:
        """Declare the adaptive step-size rung (ADR 0104).

        Runs to criterion at ``node`` / ``dof`` — either the ``target``
        advance, or (with ``plateau`` + ``plateau_window``) a tail
        slope that has fallen to ``plateau`` × the initial tangent.
        The step starts at ``ds``, halves on a failed increment,
        regrows by ``regrow`` after ``regrow_after`` consecutive good
        steps up to ``ds_max`` (default: ``ds``), and stops on the
        ``ds_min`` floor, the ``budget`` of consecutive halvings spent
        on one increment, or the ``wall_budget`` in seconds — each of
        those three a FAILURE verdict.  Drive it with
        :meth:`~apeGmsh.opensees.analysis.strategy.Substep.drive`.
        """
        return _Substep(
            node=node, dof=dof, target=target, ds=ds, ds_min=ds_min,
            ds_max=ds_max, regrow=regrow, regrow_after=regrow_after,
            budget=budget, wall_budget=wall_budget, plateau=plateau,
            plateau_window=plateau_window,
        )

    def OpenSeesPyDriver(
        self,
        ops_module: Any,
        *,
        node: int,
        dof: int,
        sign: float = -1.0,
    ) -> _OpenSeesPyDriver:
        """The stock ``DisplacementControl`` driver for :meth:`Substep`.

        ``ops_module`` is a live openseespy module (or anything with
        ``analyze`` / ``nodeDisp`` / ``getTime``).  The analysis chain
        and the unit reference load at the control node are ALREADY
        the caller's; each attempt re-issues ``integrator
        DisplacementControl $node $dof $sign*ds`` and takes one step.
        ``sign=-1.0`` (the default) drives the DOF negative.  See ADR
        0104 D1 — the sp-platen path needs its own three-method
        driver, because there ``getTime()`` is a settlement, not a
        force.
        """
        return _OpenSeesPyDriver(
            ops_module, node=node, dof=dof, sign=sign,
        )

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
