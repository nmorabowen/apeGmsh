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
    """

    node_tags: np.ndarray
    disp: Mapping[int, np.ndarray]
    reactions: Mapping[int, np.ndarray]
    time: float

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
    return GravityState(
        node_tags=np.asarray(tags, dtype=np.int64),
        disp=disp,
        reactions=reac,
        time=float(ops.getTime()),
    )


def state_drift(a: GravityState, b: GravityState) -> tuple[float, float]:
    """``(disp, reaction)`` drift between two snapshots of one domain.

    Each is the largest absolute difference over every node DOF, divided
    by the largest absolute value in ``a`` (or ``1.0`` where ``a`` is all
    zero, so an unloaded reference reports an absolute drift).
    """
    def _rel(x: Mapping[int, np.ndarray], y: Mapping[int, np.ndarray]) -> float:
        if set(x) != set(y):
            raise ValueError(
                "state_drift: the two snapshots hold different node sets."
            )
        diff = max((float(np.max(np.abs(x[t] - y[t]), initial=0.0))
                    for t in x), default=0.0)
        scale = max((float(np.max(np.abs(x[t]), initial=0.0))
                     for t in x), default=0.0)
        return diff / scale if scale > 0.0 else diff

    return _rel(a.disp, b.disp), _rel(a.reactions, b.reactions)


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
        The domain state right after ``loadConst -time 0``.
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
