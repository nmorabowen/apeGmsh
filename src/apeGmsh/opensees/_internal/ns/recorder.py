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

from typing import Iterable

from ...recorder import (
    MPCO,
    Element,
    Ladruno,
    Node,
    RecorderDeclaration,
    build_recorder_declaration,
)
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

    # -- declare (Phase 9 unified) --------------------------------------
    def declare(
        self,
        *,
        nodes: Iterable[str] | str = (),
        elements: Iterable[str] | str = (),
        line_stations: Iterable[str] | str = (),
        gauss: Iterable[str] | str = (),
        raw_nodes: Iterable[str] | str | None = None,
        raw_elements: Iterable[str] | str | None = None,
        raw_line_stations: Iterable[str] | str | None = None,
        raw_gauss: Iterable[str] | str | None = None,
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        name: str = "default",
        record_name: str | None = None,
        element_class_name: str | None = None,
        file_root: str = ".",
    ) -> RecorderDeclaration:
        """Declare a unified recorder spec; register on the bridge.

        Phase 9 commits 3a–3c: supports the file-emit-able categories
        ``nodes`` / ``elements`` / ``line_stations`` / ``gauss``
        with ``pg=`` / ``label=`` / ``selection=`` / ``ids=`` selectors.
        The other categories (``fibers`` / ``layers`` / ``modal``)
        raise :class:`NotImplementedError` from emit time — they belong
        on the DomainCapture side (Phase 9 commit 5 surfaces an
        explicit bridge entry point for that).

        Parameters
        ----------
        nodes
            Tuple of canonical component names or a single shorthand
            string for nodal kinematics / reactions. Shorthand
            expansion uses the bridge's ``ndm``/``ndf`` (per D8).
        elements
            Tuple of canonical names for per-element-node forces (the
            ``nodal_resisting_force_*`` family — global or local frame).
        line_stations
            Tuple of canonical names for beam-column section forces
            (``axial_force``, ``bending_moment_y``, etc.).
        gauss
            Tuple of canonical names for continuum stress/strain at
            Gauss points (``stress_xx``, ``strain_yy``,
            ``von_mises_stress``, etc.).
        raw_nodes, raw_elements, raw_line_stations, raw_gauss
            Per-category escape hatch for non-catalogued OpenSees
            response tokens. Each entry produces its own ``recorder``
            command at emit time (canonical-vocabulary validation is
            bypassed; the token reaches OpenSees verbatim).
        pg, label, selection, ids
            Selectors. ``ids=`` is mutually exclusive with the named
            selectors. ``pg=`` / ``label=`` / ``selection=`` may be
            combined — at emit time the resolver unions and
            deduplicates their target IDs (mirroring the legacy
            ``Recorders`` helper semantics).
        dt, n_steps
            Recording cadence; at most one may be set.
        name
            Declaration identifier; multiple coexist on one bridge.
        record_name
            Optional per-record name. Auto-generated when ``None``.
        element_class_name
            Optional OpenSees C++ class name override for element-
            level records (lifted from the legacy
            ``Recorders.elements`` contract for .out transcoder
            disambiguation — see :class:`RecorderRecord`).
        file_root
            Directory prefix for emitted ``.out`` files. Defaults to
            ``"."`` (current working directory).

        Returns
        -------
        The registered :class:`RecorderDeclaration`.

        Raises
        ------
        RuntimeError
            If ``ops.model(ndm=, ndf=)`` has not been called yet.
        """
        ndm = self._bridge._ndm
        ndf = self._bridge._ndf
        if ndm is None or ndf is None:
            raise RuntimeError(
                "ops.recorder.declare: ops.model(ndm=, ndf=) must be "
                "called before declaring recorders (Phase 9 D8 binds "
                "ndm/ndf at declaration time)."
            )

        decl = build_recorder_declaration(
            ndm=ndm,
            ndf=ndf,
            nodes=nodes,
            elements=elements,
            line_stations=line_stations,
            gauss=gauss,
            raw_nodes=raw_nodes,
            raw_elements=raw_elements,
            raw_line_stations=raw_line_stations,
            raw_gauss=raw_gauss,
            pg=pg,
            label=label,
            selection=selection,
            ids=ids,
            dt=dt,
            n_steps=n_steps,
            name=name,
            record_name=record_name,
            element_class_name=element_class_name,
            file_root=file_root,
        )
        return self._bridge._register(decl)

    # -- MPCO -----------------------------------------------------------
    def MPCO(
        self,
        *,
        file: str,
        nodal_responses: tuple[str, ...] = (),
        elem_responses: tuple[str, ...] = (),
        dT: float | None = None,
        nsteps: int | None = None,
        nodes: tuple[int, ...] | None = None,
        nodes_pg: str | None = None,
        elements: tuple[int, ...] | None = None,
        elements_pg: str | None = None,
    ) -> MPCO:
        """Construct + register a ``recorder mpco``.

        At least one of ``nodal_responses`` or ``elem_responses`` must
        be non-empty; supplying both ``dT`` and ``nsteps`` raises.
        ``nodes=`` / ``nodes_pg=`` and ``elements=`` / ``elements_pg=``
        are pairwise mutex; supplying any of the four triggers
        auto-emission of an OpenSees ``region`` plus ``-R $tag`` on
        the MPCO line at build time. See
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
                nodes=nodes,
                nodes_pg=nodes_pg,
                elements=elements,
                elements_pg=elements_pg,
            )
        )

    # -- Ladruno (fork-only canonical recorder) -------------------------
    def Ladruno(
        self,
        *,
        file: str,
        nodal_responses: tuple[str, ...] = (),
        elem_responses: tuple[str, ...] = (),
        dT: float | None = None,
        nsteps: int | None = None,
    ) -> Ladruno:
        """Construct + register a ``recorder ladruno`` (fork-only).

        Whole-model value channels (``-N``/``-E``/``-T``), mirroring
        :meth:`MPCO`. At least one of ``nodal_responses`` or
        ``elem_responses`` must be non-empty; supplying both ``dT`` and
        ``nsteps`` raises. Emission works on any build; the Ladruno fork
        is required only to *run* the deck. See
        :class:`apeGmsh.opensees.recorder.Ladruno` for the full contract
        (and the deferred ``-R`` filter / ``-G energy`` channels).
        """
        return self._bridge._register(
            Ladruno(
                file=file,
                nodal_responses=nodal_responses,
                elem_responses=elem_responses,
                dT=dT,
                nsteps=nsteps,
            )
        )
