"""
``ModelData`` — declarative ``model.h5`` orientation enrichment for
hand-written OpenSees decks.

Ratified by ADR 0018.  Use when you build your OpenSees model BY HAND
in vanilla openseespy (not through :class:`apeSees`) and you want the
results viewer to orient beam / line diagrams correctly.

.. code-block:: python

    from apeGmsh import FEMData
    from apeGmsh.opensees import ModelData

    fem = FEMData.from_h5("frame.h5")
    md  = ModelData(fem, ndm=3, ndf=6, model_name="frame")
    md.oriented_elements(pg="columns",     ele_type="forceBeamColumn",
                         vecxz=(0.0, 1.0, 0.0))
    md.oriented_elements(pg="floor_beams", ele_type="forceBeamColumn",
                         vecxz=(0.0, 0.0, 1.0))
    md.write("frame.h5")

The class owns no HDF5 bytes and no schema knowledge — it delegates to
:class:`H5Emitter` (one new public method, ``add_oriented_elements``)
and to the shared ``_compose_model_h5`` composer.  Output is
byte-equivalent (modulo ``created_iso``) to ``apeSees(fem).h5()`` for
the orientation zone, so the viewer / future P2 read path needs zero
``ModelData`` awareness.

Scope is orientation **plus recorders**: no materials, sections,
patterns, analysis, constraints, loads, or masses — those remain the
bridge's domain (ADR 0018 INV-5).

Recorders (ADR 0018 amendment).  Hand-written decks need a way to
declare recorders in canonical vocabulary without dropping down to raw
``ops.recorder(...)`` lines.  :meth:`recorders` declares them (same
grammar as ``ops.recorder.declare``); :meth:`attach_recorders` forwards
them into a live ``openseespy`` session in-process; and
:meth:`recorder_commands` renders them as Tcl / py script lines for the
file / subprocess workflow.  Recorders are admitted (unlike materials /
analysis) because they only *observe* the domain — they declare no
model state — so they don't reopen the "ModelData owns the model"
boundary INV-5 draws.

.. warning::

    Recorder selectors resolve to **FEM element / node ids**, so the
    emitted ``recorder`` commands target *your* ``ops.element`` /
    ``ops.node`` tags only if those equal the broker fem ids — the same
    ``tag == fem_eid`` invariant :meth:`oriented_elements` already
    relies on (see the tag-correspondence caveat above).  A mismatch
    silently records the wrong entities.

.. note::

    **Tag-correspondence caveat.**  The viewer's orientation join is
    keyed by **FEM element id** (``fem_eid``).  ``ModelData`` writes
    the broker's ``fem_eid`` for every element (resolved from your
    ``pg=`` against the bound :class:`FEMData`).  Your hand-written
    OpenSees recorder, however, writes its output keyed by the
    **OpenSees element tag** you typed into ``ops.element(...)``.
    The two are compared only when results bind the model; if your
    OpenSees tags do **not** equal the broker fem eids, the result
    fields will land on the wrong elements.  The safe pattern is to
    drive your ``ops.element`` calls from ``fem.elements`` so the
    tags are equal by construction:

    .. code-block:: python

        ids, conn = fem.elements.resolve(pg="columns", element_type="line2")
        for eid, c in zip(ids, conn):
            ops.element("forceBeamColumn", int(eid), *map(int, c), tt, intg)

    ``ModelData`` cannot police the tags your live deck used; it
    makes ``model.h5`` internally correct.
"""
from __future__ import annotations

from typing import Any, Iterable, Literal, cast

from ._internal.build import (
    BridgeError,
    _emit_recorder_declaration,
    expand_pg_to_elements,
)
from ._internal.compose import _compose_model_h5, _path_stem
from .emitter.base import Emitter
from .emitter.h5 import H5Emitter
from .recorder import RecorderDeclaration, build_recorder_declaration


__all__ = ["ModelData"]


class ModelData:
    """Declarative orientation-enrichment authoring façade for ``model.h5``.

    See module docstring for the design rationale (ADR 0018) and the
    tag-correspondence caveat.
    """

    def __init__(
        self,
        fem: Any,
        *,
        ndm: int,
        ndf: int,
        model_name: str | None = None,
    ) -> None:
        """Bind to a broker snapshot and prepare an internal emitter.

        Parameters
        ----------
        fem
            The :class:`FEMData` broker.  Mandatory: the neutral zone
            (nodes, elements, physical groups, labels) is what the
            viewer's orientation overlay keys against.  Resolved at
            inject time to produce real positive ``fem_eid``s.
        ndm
            2 or 3.  Required, kwarg-only — the reader's transf-arg
            slot index is ``ndm``-dependent (ADR 0018 INV-9).  Must
            match the geometry stored in ``fem``: ``write()`` checks
            ``_derive_ndm(fem)`` against this value and raises on
            mismatch.
        ndf
            DOFs per node.  Required, kwarg-only — stamped into
            ``/meta`` so consumers know the DOF space.  Typical:
            ``ndf=6`` for 3-D frames, ``ndf=3`` for 2-D frames.
        model_name
            Optional human-readable label for ``/meta/model_name``.
            Defaults to ``path``'s file-name stem at write time.

        Raises
        ------
        TypeError
            ``fem`` does not expose a ``.elements`` surface.
        ValueError
            ``ndm`` is not in ``{2, 3}`` or ``ndf <= 0``.
        """
        if not hasattr(fem, "elements"):
            raise TypeError(
                "ModelData(fem, ...): fem must be a FEMData-like object "
                "with a .elements composite (got "
                f"{type(fem).__name__})."
            )
        if ndm not in (2, 3):
            raise ValueError(
                f"ModelData: ndm must be 2 or 3, got {ndm!r}."
            )
        if int(ndf) <= 0:
            raise ValueError(
                f"ModelData: ndf must be > 0, got {ndf!r}."
            )
        self._fem = fem
        self._ndm = int(ndm)
        self._ndf = int(ndf)
        self._model_name = model_name
        self._em = H5Emitter(
            model_name=model_name or "model",
            snapshot_id=str(getattr(fem, "snapshot_id", "") or ""),
        )
        self._em.model(ndm=self._ndm, ndf=self._ndf)
        # Set by :meth:`from_h5` to the exact byte string read from
        # the source file's ``/meta/snapshot_id``; carried opaque
        # through :meth:`write` (ADR 0018 INV-8 — never recomputed
        # from the loaded fem on a re-write).  ``None`` on a fresh
        # ``ModelData(fem, ...)`` — broker stamps ``/meta`` as usual.
        self._loaded_snapshot_id: str | None = None
        # Recorders declared via :meth:`recorders` (ADR 0018
        # amendment).  Consumed by :meth:`attach_recorders` (live) and
        # :meth:`recorder_commands` (script lines).  Never written to
        # ``model.h5`` by :meth:`write` — they describe runtime
        # observation, not model definition.
        self._recorder_decls: list[RecorderDeclaration] = []

    # -- Read-only views -------------------------------------------------

    @property
    def fem(self) -> Any:
        """The bound :class:`FEMData` broker (read-only)."""
        return self._fem

    @property
    def ndm(self) -> int:
        return self._ndm

    @property
    def ndf(self) -> int:
        return self._ndf

    # -- Inject ----------------------------------------------------------

    def oriented_elements(
        self,
        *,
        pg: str,
        ele_type: str,
        vecxz: "tuple[float, float, float]",
    ) -> None:
        """Record per-element ``vecxz`` orientation for one beam PG.

        Resolves ``pg`` against the bound :class:`FEMData` into
        ``(fem_eid, connectivity)`` pairs and delegates to
        :meth:`H5Emitter.add_oriented_elements`.  The user never types
        a tag; ``fem_eid`` comes from the broker, so it is correct by
        construction (ADR 0018 — eliminates the sentinel-``-1`` /
        ops-tag-mistyped-as-fem-eid failure modes).

        Call once per (physical group, vecxz) combination.  Multiple
        calls within the same ``ModelData`` are accumulative; each
        produces one ``/opensees/transforms/{type}_{tag}/`` group and
        appends to ``/opensees/element_meta/{type_token}/``.

        Parameters
        ----------
        pg
            Physical-group name in the bound :class:`FEMData`.
        ele_type
            OpenSees element type token (e.g. ``"forceBeamColumn"``,
            ``"elasticBeamColumn"``).  Validated against
            ``_ELEM_REGISTRY`` by the writer; an unknown token / one
            without a transf slot raises ``ValueError`` (ADR 0018
            INV-7).
        vecxz
            Three-tuple ``(vx, vy, vz)`` — the reference axis from
            which the local y-axis is built (``x × vecxz``).

        Raises
        ------
        ValueError
            ``pg`` resolves to no elements; ``ele_type`` is unknown /
            has no transf slot; ``vecxz`` is not three components.
        """
        # selection-unification v2: delegate PG → (eid, conn) expansion
        # to the bridge's canonical helper, which uses
        # ``fem.elements.select(pg=pg).groups()`` (the only resolution
        # path the v2-migrated real ``FEMData`` exposes — ``.get(pg=)``
        # was removed in P3-R).  Single source of truth for the
        # expansion; the BridgeError → ValueError wrap keeps
        # ModelData's public exception API stable.
        try:
            items = expand_pg_to_elements(self._fem, pg)
        except BridgeError as e:
            raise ValueError(
                f"ModelData.oriented_elements: pg={pg!r} not found in "
                f"the bound FEMData."
            ) from e
        if not items:
            raise ValueError(
                f"ModelData.oriented_elements: pg={pg!r} resolves to "
                f"no elements (broker has no matching mesh entries)."
            )

        # The writer raises on unknown ele_type / vecxz shape — keep
        # the error path single-sourced.
        self._em.add_oriented_elements(
            type_token=ele_type,
            vecxz=vecxz,
            elements=items,
            ndm=self._ndm,
        )

    # -- Recorders (ADR 0018 amendment) ----------------------------------

    def recorders(
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
        """Declare recorders for a hand-written deck; store on this
        ``ModelData``.

        Mirrors ``ops.recorder.declare`` exactly (same canonical
        vocabulary, same selectors), but binds the recorder to *this*
        ``ModelData``'s ``ndm`` / ``ndf`` rather than a bridge's.  The
        declaration is materialized into concrete OpenSees ``recorder``
        commands by :meth:`attach_recorders` (live session) or
        :meth:`recorder_commands` (script lines).

        Call once per recorder spec; calls accumulate.  See the
        canonical-component contract on
        :meth:`apeGmsh.opensees.apeSees.recorder` /
        ``ops.recorder.declare`` for the full parameter semantics.

        Returns
        -------
        The stored :class:`RecorderDeclaration`.
        """
        decl = build_recorder_declaration(
            ndm=self._ndm,
            ndf=self._ndf,
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
        self._recorder_decls.append(decl)
        return decl

    def attach_recorders(self, ops: Any) -> None:
        """Forward every declared recorder into a live ``openseespy``
        session.

        ``ops`` is the caller's already-imported ``openseespy.opensees``
        module, holding the model the user hand-built.  Each stored
        :class:`RecorderDeclaration` fans out into one or more
        ``ops.recorder(...)`` calls, resolving ``pg=`` / ``label=`` /
        ``selection=`` selectors against the bound :class:`FEMData`.

        Unlike the bridge's whole-model ``LiveOpsEmitter``, this does
        **not** wipe the domain or re-declare the model — it only issues
        ``recorder`` calls, so it is safe to call mid-session against an
        already-populated ``ops`` (the failure mode the bridge's live
        emitter has, where ``__init__`` calls ``ops.wipe()``).

        Recorders must be declared after the user's ``ops.node`` /
        ``ops.element`` calls and before ``ops.analyze`` — the usual
        OpenSees ordering.  See the class ``tag == fem_eid`` warning:
        selectors resolve to fem ids, so they target the caller's tags
        only if those equal the fem ids.
        """
        sink = _LiveRecorderSink(ops)
        for decl in self._recorder_decls:
            _emit_recorder_declaration(
                decl,
                cast(Emitter, sink),  # _LiveRecorderSink implements only the three recorder methods
                self._fem,
                fem_eid_to_ops_tag=None,
            )

    def recorder_commands(
        self, *, target: Literal["py", "tcl"] = "py",
    ) -> list[str]:
        """Render every declared recorder as script lines.

        For the file / subprocess workflow (no live session to attach
        to): the returned lines are pasted into a hand-written ``.py``
        (``ops.recorder(...)``) or ``.tcl`` (``recorder ...``) deck.
        Selectors resolve against the bound :class:`FEMData`.

        A leading comment banner records the ``tag == fem_eid``
        assumption (see the class warning), since the rendered lines
        target fem ids and the script's own tags must match.

        Parameters
        ----------
        target
            ``"py"`` (default) for openseespy ``ops.recorder(...)``
            lines, or ``"tcl"`` for ``recorder ...`` lines.
        """
        if target == "py":
            from .emitter.py import PyEmitter

            emitter: Any = PyEmitter()
            banner = "# recorders assume ops.element/ops.node tags == fem eids"
        elif target == "tcl":
            from .emitter.tcl import TclEmitter

            emitter = TclEmitter()
            banner = "# recorders assume element/node tags == fem eids"
        else:
            raise ValueError(
                f"ModelData.recorder_commands: target must be 'py' or "
                f"'tcl', got {target!r}."
            )

        # Drop whatever deck preamble the emitter seeds (PyEmitter
        # pre-seeds an ``import`` + ``ops.wipe()`` header — pasting that
        # into the user's live deck would erase their model).  Capture
        # the pre-fan-out length and return only the recorder lines.
        prefix_len = len(emitter.lines())
        for decl in self._recorder_decls:
            _emit_recorder_declaration(
                decl, emitter, self._fem, fem_eid_to_ops_tag=None,
            )
        recorder_lines = emitter.lines()[prefix_len:]
        return [banner, *recorder_lines] if recorder_lines else []

    # -- Write -----------------------------------------------------------

    def write(self, path: str) -> None:
        """Compose ``model.h5`` at ``path`` via the shared composer.

        Validates that the bound ``fem``'s derived spatial dimension
        matches ``self._ndm`` before writing — a mismatch would stamp
        ``/meta.ndm`` from the fem while the orientation slot was
        placed for the caller's ``ndm``, causing a silent reader
        mis-read (ADR 0018 INV-9).
        """
        from ..mesh._femdata_h5_io import _derive_ndm

        fem_ndm: int | None
        try:
            fem_ndm = int(_derive_ndm(self._fem))
        except Exception:
            # Stub FEM that doesn't expose .info.types — accept the
            # caller's ndm and rely on the writer's own ndm parameter
            # for slot placement (still correct internally).
            fem_ndm = None
        if fem_ndm is not None and fem_ndm != self._ndm:
            raise ValueError(
                "ModelData.write: ndm mismatch — caller passed "
                f"ndm={self._ndm} but the bound fem encodes "
                f"ndm={fem_ndm}.  /meta.ndm would be stamped from "
                "fem while the orientation slot was placed for the "
                "caller's ndm, causing a silent reader mis-read "
                "(ADR 0018 INV-9).  Pass the matching ndm to "
                "ModelData(...)."
            )

        name = self._model_name or _path_stem(path)
        _compose_model_h5(
            self._fem,
            self._em,
            path,
            model_name=name,
            ndf=self._ndf,
            # ``self._loaded_snapshot_id`` is set by ``from_h5`` to
            # the exact ``/meta/snapshot_id`` byte string read from
            # the source file; ``None`` on a fresh ``ModelData(fem,
            # ...)`` lets the broker / bridge stamp ``/meta`` as
            # usual (ADR 0018 INV-8 — opaque carry-through, never
            # recompute).
            snapshot_id=self._loaded_snapshot_id,
        )

    # -- Load + enrich (round-trip) --------------------------------------

    @classmethod
    def from_h5(cls, path: str) -> "ModelData":
        """Load a ``model.h5`` written by :meth:`write` (or by
        :meth:`apeSees.h5`) and return an enrichable ``ModelData``.

        Symmetric with :meth:`apeGmsh.mesh.FEMData.from_h5`.  Rehydrates
        the broker neutral zone via ``FEMData.from_h5`` and the two
        orientation record lists (``/opensees/transforms`` +
        ``/opensees/element_meta``) onto a fresh internal emitter.
        Optional ``/opensees`` children are probed with ``in`` /
        ``H5Lexists`` rather than ``Group.get()`` (ADR 0018 INV-15;
        the ``project_h5py_optional_child_get_hazard`` PR #261
        pattern).

        ``snapshot_id`` is preserved opaque: the value at
        ``/meta/snapshot_id`` is stamped back unchanged by
        :meth:`write`, even if the loaded ``FEMData``'s own
        ``snapshot_id`` would recompute differently (ADR 0018 INV-8).

        ``fem_eids ↔ per_element_emitted_tag ↔ args`` row
        correspondence is preserved by appending records in disk-read
        order (ADR 0018 INV-9).

        Notes
        -----
        Files written by :meth:`apeSees.h5` may carry string-valued
        positional args (section / material name refs) in
        ``args_str``.  These are preserved on round-trip so the
        re-written file is shape-equivalent in the orientation zone.

        Raises
        ------
        FileNotFoundError
            ``path`` does not exist.
        OSError
            ``path`` is not a valid HDF5 file (h5py.File raises).
        """
        import h5py
        import numpy as np

        from ..mesh.FEMData import FEMData

        # Broker — its ``snapshot_id`` already round-trips through
        # ``FEMData.from_h5`` (read from /meta, not recomputed), but
        # we still carry the source value opaque below.
        fem = FEMData.from_h5(path)

        with h5py.File(path, "r") as f:
            meta = f["meta"]
            ndm = int(meta.attrs.get("ndm", 3))
            ndf = int(meta.attrs.get("ndf", 0))
            model_name_raw = meta.attrs.get("model_name", "")
            snapshot_id_raw = meta.attrs.get("snapshot_id", "")

            model_name: str | None
            if isinstance(model_name_raw, bytes):
                model_name = model_name_raw.decode("utf-8", "replace") or None
            else:
                model_name = str(model_name_raw) or None
            if isinstance(snapshot_id_raw, bytes):
                snapshot_id = snapshot_id_raw.decode("utf-8", "replace")
            else:
                snapshot_id = str(snapshot_id_raw)

            md = cls(fem, ndm=ndm, ndf=ndf, model_name=model_name)
            md._loaded_snapshot_id = snapshot_id

            # ADR 0018 INV-15 — probe optional children with ``in``
            # (H5Lexists), never ``Group.get()``.
            if "opensees" not in f:
                return md
            ops_grp = f["opensees"]

            # /opensees/transforms — one group per geomTransf call.
            if "transforms" in ops_grp:
                t_grp = ops_grp["transforms"]
                for tname in t_grp:
                    g = t_grp[tname]
                    if "per_element_vecxz" not in g:
                        continue
                    if "per_element_emitted_tag" not in g:
                        continue
                    vec_arr = np.asarray(g["per_element_vecxz"][...]).reshape(-1)
                    tag_arr = np.asarray(g["per_element_emitted_tag"][...]).reshape(-1)
                    if vec_arr.size < 3 or tag_arr.size < 1:
                        continue
                    ttype_attr = g.attrs.get("type", "Linear")
                    if isinstance(ttype_attr, bytes):
                        ttype = ttype_attr.decode("utf-8", "replace")
                    else:
                        ttype = str(ttype_attr)
                    ttag = int(tag_arr[0])
                    vec = (
                        float(vec_arr[0]),
                        float(vec_arr[1]),
                        float(vec_arr[2]),
                    )
                    # Append the same record-shape the writer produces.
                    from ._internal.typed_records import TransformRecord as _TransformRecord  # local
                    md._em._transforms.append(
                        _TransformRecord(
                            type_token=ttype, tag=ttag, vec=vec,
                        )
                    )
                    # Keep the orientation tag counter ahead of any
                    # tag we observed, so a subsequent enrich
                    # (md.oriented_elements(...)) cannot collide.
                    if ttag > md._em._orientation_tag_counter:
                        md._em._orientation_tag_counter = ttag

            # /opensees/element_meta — one group per OpenSees type.
            if "element_meta" in ops_grp:
                em_grp = ops_grp["element_meta"]
                for type_group_name in em_grp:
                    g = em_grp[type_group_name]
                    if "ids" not in g or "fem_eids" not in g:
                        continue
                    ids = np.asarray(g["ids"][...]).reshape(-1)
                    fem_eids = np.asarray(g["fem_eids"][...]).reshape(-1)
                    args = (
                        np.asarray(g["args"][...])
                        if "args" in g else np.zeros((len(ids), 0))
                    )
                    args_str = g["args_str"][...] if "args_str" in g else None
                    inline_conn = (
                        g["inline_connectivity"][...]
                        if "inline_connectivity" in g else None
                    )

                    type_attr = g.attrs.get("type", type_group_name)
                    if isinstance(type_attr, bytes):
                        type_token = type_attr.decode("utf-8", "replace")
                    else:
                        type_token = str(type_attr)

                    from ._internal.typed_records import ElementRecord as _ElementRecord  # local

                    n_rows = min(len(ids), len(fem_eids))
                    for i in range(n_rows):
                        if args.ndim == 2 and args.shape[1] > 0:
                            row_args: list[float | str] = []
                            for j in range(args.shape[1]):
                                if args_str is not None:
                                    s = args_str[i, j]
                                    if isinstance(s, bytes):
                                        s = s.decode("utf-8", "replace")
                                    if s != "":
                                        row_args.append(str(s))
                                        continue
                                row_args.append(float(args[i, j]))
                            # Trim TRAILING ragged-pad cells (NaN + empty
                            # string): rows narrower than the group's widest
                            # sibling are end-padded by the writer; replaying
                            # the pads would emit literal ``nan`` args.
                            # Interior NaNs (orientation-inject placeholders)
                            # are real slots and stay.
                            while (
                                row_args
                                and isinstance(row_args[-1], float)
                                and np.isnan(row_args[-1])
                            ):
                                row_args.pop()
                        else:
                            row_args = []
                        # ADR 0049: node-pair row (fem_eid<0) restores its
                        # connectivity from inline_connectivity and
                        # re-prepends it to args, so the H5→H5 rewrite
                        # re-slices args[len(connectivity):] back to the same
                        # tail and re-stores the same inline_connectivity —
                        # byte-stable.  PG rows keep connectivity=() (their
                        # prefix lives in the neutral /elements zone).
                        conn: tuple[int, ...] = ()
                        if (
                            inline_conn is not None
                            and int(fem_eids[i]) < 0
                            and i < len(inline_conn)
                            and len(inline_conn[i]) > 0
                        ):
                            conn = tuple(int(c) for c in inline_conn[i])
                        md._em._elements.append(
                            _ElementRecord(
                                type_token=type_token,
                                tag=int(ids[i]),
                                args=(*conn, *row_args),
                                connectivity=conn,
                                fem_eid=int(fem_eids[i]),
                            )
                        )
                        if int(ids[i]) > md._em._orientation_tag_counter:
                            md._em._orientation_tag_counter = int(ids[i])

            return md


class _LiveRecorderSink:
    """Recorder-only emitter forwarding into a live ``openseespy`` module.

    The recorder fan-out (:func:`_emit_recorder_declaration`) touches
    exactly three emitter methods: :meth:`recorder`,
    :meth:`recorder_declaration_begin`, :meth:`recorder_declaration_end`.
    This sink implements only those, so it cannot wipe the domain or
    re-declare the model the way the bridge's whole-model
    ``LiveOpsEmitter`` can.  Used by :meth:`ModelData.attach_recorders`.
    """

    __slots__ = ("_ops",)

    def __init__(self, ops: Any) -> None:
        self._ops = ops

    def recorder(self, kind: str, *args: "int | float | str") -> None:
        self._ops.recorder(kind, *args)

    def recorder_declaration_begin(
        self,
        *,
        declaration_name: str,
        record_name: str | None,
        category: str,
        components: tuple[str, ...],
        raw: tuple[str, ...] = (),
        pg: tuple[str, ...] = (),
        label: tuple[str, ...] = (),
        selection: tuple[str, ...] = (),
        ids: tuple[int, ...] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        file_root: str = ".",
    ) -> None:
        """Declaration metadata is archival-only — no-op for a live sink."""

    def recorder_declaration_end(self) -> None:
        """Declaration metadata is archival-only — no-op for a live sink."""
