"""
``_GeomTransfNS`` — backs ``ops.geomTransf.<Type>(...)``.

Phase 1D populates the three concrete transforms (Linear, PDelta,
Corotational). Each method registers a typed primitive on the bridge
and returns the typed instance for downstream use as
``transf=trans`` on element creation calls.

Per ADR 0010, the bridge accepts an orientation (``orientation=``) OR
an explicit ``vecxz=`` vector — not both. The orientation path defers
per-element vecxz derivation to the build pipeline (Phase 4); the
vecxz path emits one ``geomTransf`` line as-is.

When the user supplies **neither** ``orientation`` nor ``vecxz``, the
bridge's :attr:`apeSees._default_orientation` is substituted (typical
3D use: implicit Z-up Cartesian). The substitution is skipped for 2D
models, where vecxz is omitted at emit time and the orientation field
is meaningless. Users opt out of the auto-default at construction:
``apeSees(fem, default_orientation=None)``.
"""
from __future__ import annotations

from ...transform import Corotational, Linear, Orientation, PDelta
from ._base import _BridgeNamespace


__all__ = ["_GeomTransfNS"]


class _GeomTransfNS(_BridgeNamespace):
    """``ops.geomTransf.<Type>(...)`` — typed geomTransf primitives."""

    def _resolve_orientation(
        self,
        orientation: Orientation | None,
        vecxz: tuple[float, float, float] | None,
    ) -> Orientation | None:
        """Fill ``orientation`` from the bridge default when appropriate.

        Substitutes the bridge's ``_default_orientation`` only when:
          1. The user supplied neither ``orientation`` nor ``vecxz``.
          2. The model is 3D (``ndm == 3``). The 2D path omits vecxz
             at emit time, so substituting an orientation would be
             unused metadata.
          3. ``ndm`` has been set (i.e. ``model()`` was called before
             ``geomTransf.X()``). If ``ndm`` is unknown we leave both
             fields ``None`` so the existing 2D-tolerant path keeps
             working in legacy tests that construct transforms before
             setting ``ndm``.
        """
        if orientation is not None or vecxz is not None:
            return orientation
        if self._bridge._ndm != 3:
            return None
        return self._bridge._default_orientation

    def Linear(
        self,
        *,
        orientation: Orientation | None = None,
        vecxz: tuple[float, float, float] | None = None,
        roll_deg: float = 0.0,
    ) -> Linear:
        """Register a ``geomTransf Linear``. See :class:`Linear`."""
        orientation = self._resolve_orientation(orientation, vecxz)
        return self._bridge._register(
            Linear(orientation=orientation, vecxz=vecxz, roll_deg=roll_deg)
        )

    def PDelta(
        self,
        *,
        orientation: Orientation | None = None,
        vecxz: tuple[float, float, float] | None = None,
        roll_deg: float = 0.0,
    ) -> PDelta:
        """Register a ``geomTransf PDelta``. See :class:`PDelta`."""
        orientation = self._resolve_orientation(orientation, vecxz)
        return self._bridge._register(
            PDelta(orientation=orientation, vecxz=vecxz, roll_deg=roll_deg)
        )

    def Corotational(
        self,
        *,
        orientation: Orientation | None = None,
        vecxz: tuple[float, float, float] | None = None,
        roll_deg: float = 0.0,
    ) -> Corotational:
        """Register a ``geomTransf Corotational``. See :class:`Corotational`."""
        orientation = self._resolve_orientation(orientation, vecxz)
        return self._bridge._register(
            Corotational(orientation=orientation, vecxz=vecxz, roll_deg=roll_deg)
        )
