"""
``_RecorderNS`` — backs ``ops.recorder.<Type>(...)``.

Phase 3B populates the three core recorder kinds:

  * :meth:`_RecorderNS.Node`    — :class:`apeGmsh.opensees.recorder.Node`
  * :meth:`_RecorderNS.Element` — :class:`apeGmsh.opensees.recorder.Element`
  * :meth:`_RecorderNS.MPCO`    — :class:`apeGmsh.opensees.recorder.MPCO`

Each method's signature mirrors the matching dataclass exactly and
constructs + registers a typed primitive on the bridge. No ``**kwargs``
at the user-facing surface (charter P12).
"""
from __future__ import annotations

from ...recorder import MPCO, Element, Node
from ._base import _BridgeNamespace


__all__ = ["_RecorderNS"]


class _RecorderNS(_BridgeNamespace):
    """``ops.recorder.<Type>(...)`` — typed methods for Phase 3B."""

    # -- Node -----------------------------------------------------------
    def Node(
        self,
        *,
        file: str,
        response: str,
        nodes: tuple[int, ...] | None = None,
        pg: str | None = None,
        dofs: tuple[int, ...],
        dT: float | None = None,
        time_format: str = "step",
    ) -> Node:
        """Construct + register a ``recorder Node``.

        Exactly one of ``nodes`` or ``pg`` must be supplied. See
        :class:`apeGmsh.opensees.recorder.Node` for the full parameter
        contract.
        """
        return self._bridge._register(
            Node(
                file=file,
                response=response,
                nodes=nodes,
                pg=pg,
                dofs=dofs,
                dT=dT,
                time_format=time_format,
            )
        )

    # -- Element --------------------------------------------------------
    def Element(
        self,
        *,
        file: str,
        response: tuple[str, ...],
        elements: tuple[int, ...] | None = None,
        pg: str | None = None,
        dT: float | None = None,
        time_format: str = "step",
    ) -> Element:
        """Construct + register a ``recorder Element``.

        Exactly one of ``elements`` or ``pg`` must be supplied. See
        :class:`apeGmsh.opensees.recorder.Element` for the full
        parameter contract.
        """
        return self._bridge._register(
            Element(
                file=file,
                response=response,
                elements=elements,
                pg=pg,
                dT=dT,
                time_format=time_format,
            )
        )

    # -- MPCO -----------------------------------------------------------
    def MPCO(
        self,
        *,
        file: str,
        nodal_responses: tuple[str, ...] = (),
        elem_responses: tuple[str, ...] = (),
        dT: float | None = None,
        nsteps: int | None = None,
    ) -> MPCO:
        """Construct + register a ``recorder mpco``.

        At least one of ``nodal_responses`` or ``elem_responses`` must
        be non-empty; supplying both ``dT`` and ``nsteps`` raises. See
        :class:`apeGmsh.opensees.recorder.MPCO` for the full parameter
        contract.
        """
        return self._bridge._register(
            MPCO(
                file=file,
                nodal_responses=nodal_responses,
                elem_responses=elem_responses,
                dT=dT,
                nsteps=nsteps,
            )
        )
