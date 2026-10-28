"""
``GravityModalResult`` — the return type of
:meth:`apeGmsh.opensees.apeSees.modal_after_gravity` (ADR 0124).

The driver runs the gravity preload, ``loadConst -time 0``, ``eigen`` and
``modalProperties`` on **one** live domain, so the eigenproblem sees the
tangent stiffness the gravity state leaves behind (``K - Kg``, the P-Delta
"iterative based on loads" route). The result carries the modal properties
(the existing :class:`~apeGmsh.opensees.analysis.modal.ModalPropertiesResult`)
and a snapshot of the gravity state (:class:`GravityState`: every node's
displacement and reaction right after ``loadConst``), so a caller can prove
the eigen solve, or anything run after it, left that state alone
(:meth:`GravityModalResult.gravity_drift`).

Same staleness contract as :class:`~apeGmsh.opensees.analysis.eigen.EigenResult`:
the mode shapes and the live re-read go through the retained live emitter,
and a later driver call or ``wipe`` invalidates them without detection.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .modal import ModalPropertiesResult

if TYPE_CHECKING:
    from ..emitter.live import LiveOpsEmitter
    from ..node import Node


__all__ = ["GravityModalResult", "GravityState"]

#: A reference state whose every displacement is within this fraction of the
#: model's extent (:attr:`GravityState.extent`) is undeformed, at
#: round-off: a Corotational column at rest reads 6.7e-16 (one ulp of its
#: 3.0 coordinates), 1e-16 of the extent, while the live test frames'
#: gravity states read 1e-5. A quotient by such a reference reads a
#: round-off difference as a drift of order one.
_UNLOADED_REL = 1e-12


@dataclass(frozen=True, slots=True)
class GravityState:
    """Every node's displacement and reaction at one committed state.

    Attributes
    ----------
    node_tags
        1-D ``int64`` array of the live domain's node tags, in
        ``getNodeTags`` order.
    disp
        Node tag -> ``nodeDisp`` vector (one entry per node DOF).
    reactions
        Node tag -> ``nodeReaction`` vector, read after ``reactions``.
    time
        The domain pseudo-time when the snapshot was taken (``0.0`` after
        ``loadConst -time 0``).
    extent
        The model's largest per-axis span of the node coordinates, the
        length :func:`state_drift` tells a round-off displacement from a
        real one by. It is the span, not the coordinates' magnitude: the
        elements work on coordinate differences, and a model moved 1e6
        from the origin reads the same round-off at rest. ``0.0`` (the
        default, for a hand-built snapshot) treats only an exactly zero
        displacement field as undeformed.
    """

    node_tags: np.ndarray
    disp: Mapping[int, np.ndarray]
    reactions: Mapping[int, np.ndarray]
    time: float
    extent: float = 0.0

    def node_disp(self, node: "int | Node") -> np.ndarray:
        """The snapshot displacement vector of ``node``."""
        return self.disp[_node_tag(node)]

    def node_reaction(self, node: "int | Node") -> np.ndarray:
        """The snapshot reaction vector of ``node``."""
        return self.reactions[_node_tag(node)]

    def total_reaction(self, dof: int) -> float:
        """Sum of every node's reaction at 1-based ``dof`` (nodes with
        fewer DOFs are skipped) — the gravity base reaction for a
        translational ``dof``."""
        i = int(dof) - 1
        return float(sum(
            float(r[i]) for r in self.reactions.values() if r.size > i
        ))


def snapshot_state(ops: Any) -> GravityState:
    """Read every node's displacement and reaction from the live domain.

    ``ops`` is the module the live emitter drives (its ``.ops`` proxy).
    ``reactions`` is issued first, so the reactions are those of the
    current committed state.
    """
    tags = [int(t) for t in ops.getNodeTags()]
    ops.reactions()
    disp = {t: np.asarray(ops.nodeDisp(t), dtype=np.float64) for t in tags}
    reac = {
        t: np.asarray(ops.nodeReaction(t), dtype=np.float64) for t in tags
    }
    xyz = np.asarray([ops.nodeCoord(t) for t in tags], dtype=np.float64)
    return GravityState(
        node_tags=np.asarray(tags, dtype=np.int64),
        disp=disp,
        reactions=reac,
        time=float(ops.getTime()),
        extent=float(np.max(np.ptp(xyz, axis=0))) if xyz.size else 0.0,
    )


def _peak(x: Mapping[int, np.ndarray]) -> float:
    return max((float(np.max(np.abs(v), initial=0.0)) for v in x.values()),
               default=0.0)


def state_drift(a: GravityState, b: GravityState) -> tuple[float, float]:
    """``(disp, reaction)`` drift between two snapshots of one domain.

    Each is the largest absolute difference over every node DOF, divided
    by the largest absolute value in ``a`` (absolute where that is zero).

    An undeformed reference, one whose every displacement is within
    ``1e-12`` of ``a.extent`` (round-off, or exactly zero), changes that:

    * its displacement drift is absolute, since a quotient by round-off
      reads a round-off difference as a drift of order one;
    * its reaction drift is the smaller of the absolute and the relative
      one. An undeformed reference's reactions are either the loads on its
      restrained nodes, which the relative drift measures, or round-off of
      a zero field (a Corotational column at rest reads ``3.5e-20``
      beside ``1e-10`` after a return), which only the absolute one
      measures. The domain cannot tell them apart, and a drift that moved
      the domain makes both large. No snapshot holds a force scale to
      tell them by: the ratio of the two snapshots' round-off reactions
      ranges over ``8e-11`` to ``8e-6`` on the live test frames.

    A deformed reference keeps both drifts relative.
    """
    for x, y in ((a.disp, b.disp), (a.reactions, b.reactions)):
        if set(x) != set(y):
            raise ValueError(
                "state_drift: the two snapshots hold different node sets."
            )

    def _diff(x: Mapping[int, np.ndarray], y: Mapping[int, np.ndarray]) -> float:
        return max((float(np.max(np.abs(x[t] - y[t]), initial=0.0))
                    for t in x), default=0.0)

    def _rel(diff: float, scale: float) -> float:
        return diff / scale if scale > 0.0 else diff

    d_diff, r_diff = _diff(a.disp, b.disp), _diff(a.reactions, b.reactions)
    r_rel = _rel(r_diff, _peak(a.reactions))
    if _peak(a.disp) <= _UNLOADED_REL * a.extent:
        return d_diff, min(r_diff, r_rel)
    return _rel(d_diff, _peak(a.disp)), r_rel


@dataclass(frozen=True, slots=True)
class GravityModalResult:
    """Modal properties on the gravity-preloaded tangent, plus the gravity
    state they were taken at.

    Attributes
    ----------
    modal
        The ``eigen`` + ``modalProperties -return`` pair, as
        :class:`~apeGmsh.opensees.analysis.modal.ModalPropertiesResult`
        (periods, participation factors, effective mass ratios and their
        cumulative sums, total mass).
    gravity
        The domain state right after ``loadConst -time 0``. Under
        :meth:`~apeGmsh.opensees.apeSees.spectrum_after_gravity` with
        ``p_delta=True`` it is retaken after the per-mode solves, at the
        settled gravity state the driver leaves the domain at (a few
        unloaded ``Linear`` steps from the state ``loadConst`` froze, nearer
        equilibrium under a loose declared test). Either way it is the
        state the driver returned with, so :meth:`gravity_drift` reads
        ``0.0`` right after the call.
    p_delta
        ``True``: the eigenproblem was solved after the gravity preload, on
        the tangent ``K - Kg`` of the gravity state. ``False``: it was
        solved before the preload, on the unstressed stiffness ``K`` (no
        initial axial force), and the gravity analysis ran afterwards.
    gravity_steps
        The number of ``LoadControl`` steps the gravity analysis took.
    """

    modal: ModalPropertiesResult
    gravity: GravityState
    p_delta: bool
    gravity_steps: int

    _live: "LiveOpsEmitter"

    @property
    def eigenvalues(self) -> np.ndarray:
        """``λ_i = ω_i²`` of the solved eigenproblem."""
        return self.modal.eigenvalues

    @property
    def periods(self) -> np.ndarray:
        """Natural periods ``T_i`` (s)."""
        return self.modal.periods

    def gravity_drift(self) -> tuple[float, float]:
        """Re-read the live domain and compare it with :attr:`gravity`.

        Returns the ``(disp, reaction)`` relative drift of
        :func:`state_drift`; both are ``0.0`` while nothing has moved the
        domain off the gravity state.
        """
        return state_drift(self.gravity, snapshot_state(self._live.ops))


def _node_tag(node: "int | Node") -> int:
    from ..node import Node as _Node  # local import — avoid cycle

    if isinstance(node, _Node):
        return int(node.tag)
    return int(node)
