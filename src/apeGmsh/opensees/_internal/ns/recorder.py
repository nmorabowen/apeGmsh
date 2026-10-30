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

from typing import TYPE_CHECKING, Iterable, Literal, Sequence

from ...recorder import (
    MPCO,
    Element,
    Ladruno,
    Monitor,
    Node,
    RecorderDeclaration,
    build_recorder_declaration,
)
from ...node import _label_names, _resolve_label
from ..._readback import resolve_element_label
from ._base import _BridgeNamespace

if TYPE_CHECKING:
    from ...apesees import apeSees


__all__ = ["_RecorderNS"]


def _element_label_ids(bridge: "apeSees", label: str | Sequence[str]) -> tuple[int, ...]:
    """The FEM element ids ``label`` names, for ``recorder.Element(label=)``,
    in snapshot order.

    The label resolves at the call to FEM ids under either tag mode (ADR
    0115 D4); the recorder keeps the label beside them, which marks them
    as FEM ids, and the build translates them to the emitted OpenSees
    tags exactly as it does for ``pg=`` (ADR 0115 amendment R2-f). Under
    ``element_tags="fem"`` that translation is the identity.
    """
    ids = resolve_element_label(bridge, label, where="ops.recorder.Element")
    index = bridge.fem.elements.index
    return tuple(sorted(ids, key=index))


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
        label: str | Sequence[str] | None = None,
        dofs: tuple[int, ...],
        dT: float | None = None,
        time_format: str = "step",
        precision: int | None = None,
        name: str | None = None,
    ) -> Node:
        """Construct + register a ``recorder Node``.

        Exactly one of ``nodes``, ``pg`` or ``label`` must be supplied.
        See :class:`apeGmsh.opensees.recorder.Node` for the full
        parameter contract.

        ``label`` (one geometry label or a list) resolves on the
        snapshot here, before anything registers, to the recorder's
        ``nodes``: node tags are FEM node ids, so the deck is the one
        ``pg=`` over the same nodes gives (ADR 0115 D4). An unknown
        label raises, listing the labels that exist; it never falls
        back to a physical group. The recorder keeps the label names
        as provenance (``/opensees/decl_params``); the ids stay
        authoritative and nothing re-resolves the label.

        ``precision`` (a positive ``int``) emits ``-precision N``, the
        significant digits OpenSees writes per value (its default is 6);
        ``None`` emits no flag.

        ``name`` labels the recorder's declaration (its
        ``opensees/recorder/<name>`` key, unique among the recorders,
        read back from ``/opensees/decls``); a label only,
        the deck and ``model_hash`` do not change with it.
        """
        if label is not None:
            if nodes is not None or pg is not None:
                raise ValueError(
                    "ops.recorder.Node: supply exactly one of nodes=, pg= "
                    f"or label= (got nodes={nodes!r}, pg={pg!r}, "
                    f"label={label!r})."
                )
            nodes = _resolve_label(
                self._bridge.fem, label, where="ops.recorder.Node")
        return self._bridge._register(
            Node(
                file=file,
                response=response,
                nodes=nodes,
                pg=pg,
                label=None if label is None else _label_names(label),
                dofs=dofs,
                dT=dT,
                time_format=time_format,
                precision=precision,
            ),
            name=name, alias=False,
        )

    # -- Element --------------------------------------------------------
    def Element(
        self,
        *,
        file: str,
        response: tuple[str, ...],
        elements: tuple[int, ...] | None = None,
        pg: str | None = None,
        label: str | Sequence[str] | None = None,
        dT: float | None = None,
        time_format: str = "step",
        precision: int | None = None,
        name: str | None = None,
    ) -> Element:
        """Construct + register a ``recorder Element``.

        Exactly one of ``elements``, ``pg`` or ``label`` must be
        supplied. See :class:`apeGmsh.opensees.recorder.Element` for the
        full parameter contract.

        ``label`` (one geometry label or a list) resolves on the
        snapshot here, before anything registers, to FEM element ids
        (ADR 0115 D4); the build translates them to the emitted
        OpenSees tags as it does for ``pg=``, under either
        ``element_tags`` mode. The recorder keeps the label names as
        provenance (``/opensees/decl_params``). An unknown label
        raises, listing the labels that exist; it never falls back to
        a physical group.

        ``precision`` (a positive ``int``) emits ``-precision N``, the
        significant digits OpenSees writes per value (its default is 6);
        ``None`` emits no flag.

        ``name`` labels the recorder's declaration (its
        ``opensees/recorder/<name>`` key, unique among the recorders,
        read back from ``/opensees/decls``); a label only,
        the deck and ``model_hash`` do not change with it.
        """
        if label is not None:
            if elements is not None or pg is not None:
                raise ValueError(
                    "ops.recorder.Element: supply exactly one of elements=, "
                    f"pg= or label= (got elements={elements!r}, pg={pg!r}, "
                    f"label={label!r})."
                )
            elements = _element_label_ids(self._bridge, label)
        return self._bridge._register(
            Element(
                file=file,
                response=response,
                elements=elements,
                pg=pg,
                label=None if label is None else _label_names(label),
                dT=dT,
                time_format=time_format,
                precision=precision,
            ),
            name=name, alias=False,
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
        name: str | None = None,
        record_name: str | None = None,
        element_class_name: str | None = None,
        file_root: str = ".",
    ) -> RecorderDeclaration:
        """Declare a unified recorder spec; register on the bridge.

        Phase 9 commits 3a–3c: supports the file-emit-able categories
        ``nodes`` / ``elements`` / ``line_stations`` / ``gauss``
        with ``pg=`` / ``label=`` / ``selection=`` / ``ids=`` selectors.
        The other categories (``fibers`` / ``layers`` / ``modal``)
        raise :class:`NotImplementedError` from emit time — record them
        with the MPCO or ``.ladruno`` recorder, and modes with
        ``ops.eigen(...)`` (``EigenResult.to_native``).

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
            The declaration's name (K1-6): the
            ``.out`` file stem and ``declaration_name`` of every record
            it fans into, and the ``opensees/recorder/<name>``
            declaration key in ``/opensees/decls``.
            A given name is unique among the recorders.
            Omitted, the declaration is unnamed on the bridge (``#k``)
            and its records keep the stem ``"default"``, so several
            unnamed declarations coexist and their decks are unchanged.
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
        ValueError
            If ``name`` is already registered on the bridge.
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
            name="default" if name is None else name,
            record_name=record_name,
            element_class_name=element_class_name,
            file_root=file_root,
        )
        # The declaration's own ``name`` keys it when given;
        # the ``"default"`` stem of an unnamed one is not a name, so two
        # unnamed declarations do not collide (K1-6).
        return self._bridge._register(decl, name=name, alias=False)

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
        name: str | None = None,
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

        ``name`` labels the recorder's declaration (its
        ``opensees/recorder/<name>`` key, unique among the recorders,
        read back from ``/opensees/decls``); a label only,
        the deck and ``model_hash`` do not change with it.
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
            ),
            name=name, alias=False,
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
        energy: bool = False,
        energy_pg: str | None = None,
        precision: Literal["f64", "f32"] = "f64",
        nodes: tuple[int, ...] | None = None,
        nodes_pg: str | None = None,
        elements: tuple[int, ...] | None = None,
        elements_pg: str | None = None,
        name: str | None = None,
    ) -> Ladruno:
        """Construct + register a ``recorder ladruno`` (fork-only).

        Value channels (``-N``/``-E``/``-T``) mirroring :meth:`MPCO`,
        plus the energy balance (``-G energy``, read back via
        ``Results.energy()``). At least one of ``nodal_responses``,
        ``elem_responses``, ``energy`` or ``energy_pg`` must be supplied;
        both ``dT`` and ``nsteps`` raises.

        ``precision="f32"`` records the per-step data as float32
        (``-precision f32``, about half the file); the default ``"f64"``
        emits no flag. Any other value raises.

        ``nodes=`` / ``nodes_pg=`` and ``elements=`` / ``elements_pg=``
        are pairwise mutex; supplying any of the four triggers
        auto-emission of an OpenSees ``region`` plus ``-R $tag`` on the
        ladruno line at build time (flat and partitioned). Combining a
        filter with ``energy=True`` records the energy balance over that
        same region (``-G energy $tag``); ``energy=True`` alone is
        whole-model. ``energy_pg=`` records energy over an **independent**
        region (decoupled from the ``-R`` value filter) — ``-G energy
        $tag`` over that PG; the fork also writes the whole-model balance
        alongside any per-region one. Emission works on any build; the
        Ladruno fork is required only to *run* the deck. See
        :class:`apeGmsh.opensees.recorder.Ladruno` for the full contract.

        ``name`` labels the recorder's declaration (its
        ``opensees/recorder/<name>`` key, unique among the recorders,
        read back from ``/opensees/decls``); a label only,
        the deck and ``model_hash`` do not change with it.
        """
        return self._bridge._register(
            Ladruno(
                file=file,
                nodal_responses=nodal_responses,
                elem_responses=elem_responses,
                dT=dT,
                nsteps=nsteps,
                energy=energy,
                energy_pg=energy_pg,
                precision=precision,
                nodes=nodes,
                nodes_pg=nodes_pg,
                elements=elements,
                elements_pg=elements_pg,
            ),
            name=name, alias=False,
        )

    # -- Monitor (fork-only live-telemetry SWMR sink) -------------------
    def Monitor(
        self,
        *,
        sink: str,
        dofs: tuple[int, ...],
        nodes: tuple[int, ...] | None = None,
        pg: str | None = None,
        resp: str = "disp",
        every: int | None = None,
        hz: float | None = None,
        name: str | None = None,
    ) -> Monitor:
        """Construct + register a ``recorder Monitor`` (fork-only).

        A lightweight SWMR-HDF5 sink streaming nodes × dofs nodal scalars
        for live tailing (read back via
        :func:`apeGmsh.results.read_monitor` / ``tail_monitor``). Exactly
        one of ``nodes`` or ``pg`` must be supplied. Emission works on any
        build; the Ladruno fork is required only to *run* the deck. See
        :class:`apeGmsh.opensees.recorder.Monitor` for the full contract.

        ``name`` labels the recorder's declaration (its
        ``opensees/recorder/<name>`` key, unique among the recorders,
        read back from ``/opensees/decls``); a label only,
        the deck and ``model_hash`` do not change with it.
        """
        return self._bridge._register(
            Monitor(
                sink=sink,
                dofs=dofs,
                nodes=nodes,
                pg=pg,
                resp=resp,
                every=every,
                hz=hz,
            ),
            name=name, alias=False,
        )
