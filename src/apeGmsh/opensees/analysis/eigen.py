"""
``EigenResult`` — the return type of :meth:`apeGmsh.opensees.apeSees.eigen`.

Eigen is the one OpenSees analysis call that is **not** an Analysis
primitive: it does not emit ``analysis <Type>``, has no analysis chain,
takes no stepping, and returns values directly. Modelled as a bridge
method (``apeSees.eigen``) that drives a :class:`LiveOpsEmitter` end-
to-end and wraps the returned eigenvalues in this dataclass.

The dataclass carries the eigenvalue array plus a back-reference to the
live emitter so :meth:`mode_shape` can query ``ops.nodeEigenvector``
without re-running the eigen solve. It is intentionally minimal — no
modal-mass / participation calculations, no Results-broker integration,
no persistence to ``model.h5`` / ``results.h5``.  Add those on demand
when a second use case appears.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..emitter.live import LiveOpsEmitter
    from ..node import Node


__all__ = ["EigenResult"]


@dataclass(frozen=True, slots=True)
class EigenResult:
    """Eigenvalues + lazy mode-shape access from one ``eigen`` call.

    Attributes
    ----------
    eigenvalues
        1-D ``np.ndarray`` of ``λ_i = ω_i²`` in the order returned by
        OpenSees (ascending magnitude for the default solver).

    Notes
    -----
    The eigenvectors are NOT eagerly fetched — they live in openseespy's
    domain state until :meth:`mode_shape` queries them via
    ``ops.nodeEigenvector``. Calling :meth:`mode_shape` after another
    ``apeSees.eigen(...)`` or ``ops.wipe()`` will either return the new
    solve's vectors or raise from openseespy. The :class:`LiveOpsEmitter`
    back-reference is kept so the call can still be made; no attempt is
    made to detect staleness.
    """

    eigenvalues: np.ndarray

    # Implementation handles for lazy mode-shape access. Underscore-
    # prefixed; not part of the user-facing surface.
    _live: "LiveOpsEmitter"

    @property
    def omega(self) -> np.ndarray:
        """Natural circular frequencies ``ω_i = √λ_i`` (rad/s)."""
        return np.asarray(np.sqrt(self.eigenvalues))

    @property
    def freq(self) -> np.ndarray:
        """Natural frequencies ``f_i = ω_i / (2π)`` (Hz)."""
        return self.omega / (2.0 * np.pi)

    @property
    def periods(self) -> np.ndarray:
        """Natural periods ``T_i = 2π / ω_i = 1 / f_i``, one per mode.

        A 1-D ``np.ndarray`` in the order of :attr:`eigenvalues` (mode 1
        first, so the longest period first for the default solver), in
        seconds when the model's time unit is the second: ``λ_i`` comes
        from the model's own mass and stiffness, with no unit conversion.
        A zero eigenvalue (a mechanism, or a DOF with no mass) gives
        ``inf``; a negative one gives ``nan``.

        This is the eigen read-back ADR 0115 D1 documents: ``eigen``
        keeps returning :class:`EigenResult`, and the periods are read
        here rather than through ``Node.read`` (which reads the static
        or transient state an ``analyze`` left).
        """
        return 1.0 / self.freq

    def mode_shape(self, node: "int | Node", mode: int) -> np.ndarray:
        """Return the mode shape for ``node`` in ``mode`` (1-indexed).

        Parameters
        ----------
        node
            Either a plain integer node tag or a :class:`Node` instance
            from ``ops.nodes.get(...)``.
        mode
            1-indexed mode number, matching OpenSees' ``$modeNumber``
            convention.

        Returns
        -------
        np.ndarray
            Length-``ndf`` vector of the eigenvector's DOF entries at
            ``node``, fetched live via ``ops.nodeEigenvector``.
        """
        from ..node import Node as _Node  # local import — avoid cycle

        if isinstance(node, _Node):
            tag = int(node.tag)
        else:
            tag = int(node)
        values: Any = self._live.ops.nodeEigenvector(tag, int(mode))
        return np.asarray(values, dtype=np.float64)
