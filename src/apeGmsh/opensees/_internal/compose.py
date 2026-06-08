"""
Shared ``model.h5`` composition (ADR 0018 / modeldata-enrichment-scope C1).

The single composer both authoring front doors use:

* ``apeSees.h5`` — bridge typed primitives → ``BuiltModel`` → ``H5Emitter``.
* ``ModelData.write`` — declarative orientation inject → ``H5Emitter``.

This module owns the broker-zone / bridge-zone / cuts composition
order, the stub-FEM fallback, the schema-version stamp, and the
partial-write teardown — exactly once.  Neither front door
reimplements any of it (ADR 0018 INV-1/3, scope C1).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import h5py


__all__ = [
    "_compose_model_h5",
    "_replay_into",
    "_try_write_broker_zone",
    "_override_schema_version",
    "_path_stem",
]


def _write_opensees_nodes_ndf(
    f: "h5py.File", fem: Any, envelope_ndf: int,
    effective: "dict[int, int] | None" = None,
) -> None:
    """ADR 0048 — persist the *effective* per-node ndf the deck emits into the
    opensees zone at ``/opensees/nodes_ndf`` (``tags`` int64 + ``ndf`` int8,
    aligned to the broker node order).

    *effective* is the precomputed ``{tag: ndf}`` map the deck emitted — the
    element-class inference result (:func:`infer_node_ndf`) on a fresh bridge
    write, or the map read back from ``/opensees/nodes_ndf`` on a
    ``from_h5 → to_h5`` re-emit. Both feed the SAME map here, so the automatic
    ``model_hash`` fold stays stable across round-trips. Nodes absent from the
    map — element-less / decoupled, or adaptive-only — take the ``ops.model``
    *envelope*. A no-op when the broker carries no nodes.
    """
    import numpy as np

    nodes = getattr(fem, "nodes", None)
    if nodes is None:
        return
    try:
        ids = [int(t) for t in nodes.ids]
    except Exception:
        return
    if not ids:
        return

    eff_map = effective or {}
    eff = np.empty(len(ids), dtype=np.int8)
    for i, tag in enumerate(ids):
        eff[i] = int(eff_map.get(int(tag), envelope_ndf))

    grp = f.require_group("opensees").create_group("nodes_ndf")
    grp.create_dataset("tags", data=np.asarray(ids, dtype=np.int64))
    grp.create_dataset("ndf", data=eff)


def _compose_model_h5(
    fem: object,
    emitter: Any,
    path: str,
    *,
    model_name: str,
    ndf: int,
    cuts: "Sequence[Any]" = (),
    sweeps: "Sequence[Any]" = (),
    names: "Sequence[tuple[str, str, int]]" = (),
    snapshot_id: str | None = None,
    nodes_ndf: "dict[int, int] | None" = None,
) -> None:
    """Compose a ``model.h5`` from a broker ``fem`` + a populated ``emitter``.

    The one composition path.  Order: broker ``/meta`` + neutral zone
    (with a stub-FEM fallback to the bridge's own ``/meta`` +
    schema-version override), then the ``emitter``'s ``/opensees/...``
    enrichment, then apeGmsh.cuts v4 ``/opensees/cuts`` / ``/sweeps``.

    Parameters
    ----------
    fem
        The broker snapshot.  A hand-rolled stub lacking the FEMData
        surface triggers the bridge-only fallback (no neutral zone).
    emitter
        An already-populated :class:`H5Emitter`.
    path
        Destination HDF5 path (opened ``"w"``).
    model_name, ndf
        Written into ``/meta`` by the broker writer.
    cuts, sweeps
        apeGmsh.cuts v4 sequences; empty ⇒ no cuts/sweeps groups.
    names
        Bridge-side ``(name, kind, tag)`` alias records; empty ⇒ no
        ``/opensees/names`` group (byte-equivalent to the pre-sidecar
        layout).  Excluded from ``model_hash`` (INV-4 carve-out).
    snapshot_id
        When not ``None``, overwrite ``/meta/snapshot_id`` with this
        exact string after meta is written (ADR 0018 INV-8 — opaque
        carry-through for ``ModelData.from_h5``).  ``None`` ⇒ leave
        whatever the broker / bridge wrote: the pre-extraction
        behaviour, so ``apeSees.h5`` is byte-invariant under C1.
    """
    import h5py

    from ...cuts._h5_io import write_cuts_into
    from ...mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION
    from ..emitter.h5 import SCHEMA_VERSION
    from ._names_h5 import write_names_into
    from .lineage import (
        compute_fem_hash,
        compute_model_hash,
        write_lineage_attrs,
    )

    with h5py.File(path, "w") as f:
        broker_used = _try_write_broker_zone(
            fem, f,
            schema_version=NEUTRAL_SCHEMA_VERSION,
            model_name=model_name,
            ndf=ndf,
        )
        if not broker_used:
            # Stub FEM or otherwise missing broker surface — fall back
            # to bridge-only /meta with the bridge's own SCHEMA_VERSION
            # (the file still validates; absent neutral zone is the
            # right "no broker" signal).  The bridge's ``_write_meta``
            # already stamps both ``schema_version`` and
            # ``opensees_schema_version`` per ADR 0023.
            emitter._write_meta(f)
            _override_schema_version(f, SCHEMA_VERSION)
        if snapshot_id is not None and "meta" in f:
            # INV-8: opaque carry-through. A ModelData has no FEMData
            # to legitimately recompute the hash from; preserve the
            # exact /meta/snapshot_id byte string read off the source.
            f["meta"].attrs["snapshot_id"] = snapshot_id
        emitter.write_opensees_into(f)
        if broker_used:
            # ADR 0048 — persist the effective per-node ndf into the opensees
            # zone (one bridge-owned ndf store). Only when the broker zone
            # exists (a stub/bridge-only file has no per-node ndf to derive).
            # Folds into model_hash, not fem_hash.
            _write_opensees_nodes_ndf(f, fem, ndf, nodes_ndf)
        # ADR 0023 §"Three per-zone version stamps" — when the broker
        # wrote /meta it only stamped the neutral per-zone key; the
        # bridge now contributes /opensees/... and so we add the
        # opensees per-zone key alongside the existing neutral one.
        # Bridge-fallback files already have ``opensees_schema_version``
        # stamped by ``emitter._write_meta`` so this is a no-op there.
        if broker_used and "meta" in f:
            f["meta"].attrs["opensees_schema_version"] = SCHEMA_VERSION
        # Empty sequences are a no-op inside write_cuts_into — neither
        # /opensees/cuts/ nor /opensees/sweeps/ is created when nothing
        # was supplied.
        write_cuts_into(f, cuts=cuts, sweeps=sweeps)
        # Bridge-side name aliases (also a no-op when empty); excluded
        # from model_hash so relabelling never drifts lineage.
        write_names_into(f, names)

        # ADR 0021 — stamp the lineage triple ``/meta/lineage/...``
        # after every zone is written.  ``fem_hash`` is recomputed
        # from the neutral zone (INV-1 byte-identical to
        # ``FEMData.snapshot_id``); ``model_hash`` chains it with the
        # canonical bytes of ``/opensees/...`` minus cuts and sweeps
        # (INV-4).  Standalone ``model.h5`` files have no run zone
        # ⇒ ``results_hash`` is unwritten here (NativeWriter stamps
        # it at close time for results files).
        if "meta" in f:
            fem_hash = compute_fem_hash(f) if broker_used else ""
            model_hash = None
            if "opensees" in f:
                model_hash = compute_model_hash(fem_hash, f["opensees"])
            write_lineage_attrs(
                f["meta"],
                fem_hash=fem_hash if fem_hash else None,
                model_hash=model_hash,
            )


def _try_write_broker_zone(
    fem: object,
    f: "h5py.File",
    *,
    schema_version: str,
    model_name: str,
    ndf: int,
) -> bool:
    """Attempt to write the broker's ``/meta`` + neutral zone.

    Returns ``True`` if the broker writer ran end-to-end, ``False`` if
    the FEM lacks the surface the writer needs (typically a hand-rolled
    test stub).  On ``False`` the file is rewound to a fresh state — no
    half-populated groups linger.
    """
    from ...mesh._femdata_h5_io import write_meta, write_neutral_zone

    if not hasattr(fem, "snapshot_id"):
        return False
    try:
        write_meta(
            fem, f,  # type: ignore[arg-type]
            schema_version=schema_version,
            model_name=model_name,
            ndf=ndf,
        )
        write_neutral_zone(fem, f)  # type: ignore[arg-type]
    except (AttributeError, TypeError):
        # Stub FEM didn't expose enough surface.  Tear down any
        # partial groups so the bridge's fallback `/meta` write
        # doesn't collide.
        for key in list(f.keys()):
            del f[key]
        return False
    return True


def _override_schema_version(f: "h5py.File", schema_version: str) -> None:
    """Overwrite ``/meta/schema_version`` after the bridge wrote it.

    The bridge stamps :data:`SCHEMA_VERSION` so even bridge-only files
    declare the post-Phase-8.5 schema; this is a no-op when the bridge
    already wrote that exact version, but guards against future drift
    between the constants.
    """
    if "meta" in f:
        f["meta"].attrs["schema_version"] = schema_version


def _path_stem(path: str) -> str:
    """Return ``path``'s file-name stem (no extension). Used as the
    default H5 ``/meta/model_name``."""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem or "model"


def _replay_into(
    emitter: Any,
    *,
    ndm: int,
    ndf: int,
    nodes: "Sequence[tuple[int, tuple[float, ...]] | tuple[int, tuple[float, ...], int | None]]" = (),
    uniaxial_materials: "Sequence[Any]" = (),
    nd_materials: "Sequence[Any]" = (),
    simple_sections: "Sequence[Any]" = (),
    complex_sections: "Sequence[Any]" = (),
    transforms: "Sequence[Any]" = (),
    beam_integrations: "Sequence[Any]" = (),
    time_series: "Sequence[Any]" = (),
    dampings: "Sequence[Any]" = (),
    elements: "Sequence[Any]" = (),
    fixes: "Sequence[Any]" = (),
    masses: "Sequence[Any]" = (),
    patterns: "Sequence[Any]" = (),
    recorders: "Sequence[Any]" = (),
    fem: Any = None,
    initial_stress: "Sequence[Any]" = (),
    analysis_attrs: "dict[str, Any] | None" = None,
    analyze_call: "tuple[int, float | None] | None" = None,
) -> None:
    """Walk a typed-record graph and re-emit it through ``emitter``.

    Used by :meth:`OpenSeesModel.build` (ADR 0019) to produce
    ``tcl`` / ``py`` / ``live`` / ``h5`` emissions from a rehydrated
    record graph.  The single helper centralises the protocol-call
    order — materials before sections, sections before elements,
    time-series before patterns — so the four build targets agree on
    a single deck shape.

    The order mirrors :meth:`BuiltModel.emit` for the categories
    :class:`OpenSeesModel` carries:

      1. ``emitter.model(ndm=, ndf=)``  — model directive
      2. ``emitter.node(tag, x, y, z[, ndf=K])``  — every FEM node;
         per-node ``ndf=K`` token sourced from the broker (S2 /
         ADR 0033) when ``node_ndf`` is non-None in the input tuple,
         omitted otherwise (envelope wins).
      3. ``emitter.uniaxialMaterial`` / ``emitter.nDMaterial``
      4. ``emitter.section`` (simple) and the open/patch/fiber/layer/close
         sequence (complex)
      5. ``emitter.geomTransf``
      6. ``emitter.beamIntegration``
      7. ``emitter.timeSeries`` (+ ``emitter.damping`` for tagged damping
         objects — ADR 0053 D3b — after the series a ``-factor`` references,
         before the elements an element-flag ``-damp`` references)
      8. ``emitter.element``  (with ``set_element_nodes`` /
         ``set_current_fem_element_id`` side channels for the H5 path)
      9. ``emitter.fix`` / ``emitter.mass``
      10. ``emitter.pattern_open`` (+ load / sp / eleLoad +
          pattern_close)
      11. ``emitter.recorder`` (wrapped in declaration-begin/end when
          ``decl_context`` is present)
      12. ``emitter.constraints`` / ``numberer`` / ``system`` /
          ``test`` / ``algorithm`` / ``integrator`` / ``analysis``
          + ``emitter.analyze`` if present

    .. note::

        **Tag identity may diverge from a fresh ``apeSees(fem).run()``**
        (ADR 0019 INV-5).  The bridge's :class:`TagAllocator`
        allocations are lost across H5 round-trip; this helper
        replays exactly the tags stored in the record graph, which
        the bridge picked deterministically — but reordering at the
        bridge level (a future PR) could break that.  Callers who
        need bridge-time tag stability must capture the
        :class:`BuiltModel` from :meth:`apeSees.build` directly and
        not round-trip through H5.

    Parameters mirror :class:`apeGmsh.opensees._internal.typed_records`
    field names; see :mod:`apeGmsh.opensees.opensees_model` for the
    canonical instantiation pattern.
    """
    from .tag_resolution import set_current_fem_element_id, set_element_nodes

    # 1. Model directive.
    emitter.model(ndm=int(ndm), ndf=int(ndf))

    # 2. Nodes.  S2 (ADR 0033): the OpenSeesModel build path widens
    # the per-node tuple to ``(tag, coords, ndf|None)`` so per-node
    # ``-ndf K`` declarations survive the H5 round-trip.  Legacy
    # 2-tuples ``(tag, coords)`` are tolerated for callers that
    # haven't rebound to the wider shape.
    for entry in nodes:
        if len(entry) == 3:
            tag, coords, node_ndf = entry
        else:
            tag, coords = entry
            node_ndf = None
        if node_ndf is None:
            emitter.node(int(tag), *(float(c) for c in coords))
        else:
            emitter.node(
                int(tag), *(float(c) for c in coords),
                ndf=int(node_ndf),
            )

    # 3. Materials — uniaxial first, then nD.  ADR 0011 schema mirrors
    # this group nesting; the emitter doesn't enforce order, but the
    # OpenSees domain does (a section_open referencing matTag=1 needs
    # the uniaxialMaterial 1 already present).
    for rec in uniaxial_materials:
        emitter.uniaxialMaterial(rec.type_token, int(rec.tag), *rec.params)
    for rec in nd_materials:
        emitter.nDMaterial(rec.type_token, int(rec.tag), *rec.params)

    # 4. Sections — simple (Elastic / Aggregator) then complex
    # (Fiber).  Same ordering rule: a section_open referencing a
    # matTag needs the material registered first (already done).
    for rec in simple_sections:
        emitter.section(rec.type_token, int(rec.tag), *rec.params)
    for rec in complex_sections:
        emitter.section_open(rec.type_token, int(rec.tag), *rec.params)
        for patch in rec.patches:
            emitter.patch(patch.kind, *patch.args)
        for fiber in rec.fibers:
            emitter.fiber(fiber.y, fiber.z, fiber.area, int(fiber.mat_tag))
        for layer in rec.layers:
            emitter.layer(layer.kind, *layer.args)
        emitter.section_close()

    # 5. Transforms.  Schema deviation (TransformRecord docstring):
    # one record per ``geomTransf`` call, not per spec.  Replay
    # produces the same call shape: one ``geomTransf`` line per
    # record's stored ``vec``.
    for rec in transforms:
        emitter.geomTransf(rec.type_token, int(rec.tag), *rec.vec)

    # 6. Beam integration.
    for rec in beam_integrations:
        emitter.beamIntegration(rec.type_token, int(rec.tag), *rec.args)

    # 7. Time series.
    for rec in time_series:
        emitter.timeSeries(rec.type_token, int(rec.tag), *rec.args)

    # 7b. Damping objects (ADR 0053 D3b).  After time_series (a ``-factor``
    # tail may reference a series tag) and before elements (an element's
    # ``-damp $tag`` rides in its own arg tail and resolves the object by
    # tag).  Region ``-damp`` attaches are NOT replayed (they live in the
    # archival-only ``/opensees/regions`` zone — same limitation as all
    # region / rayleigh state).
    for rec in dampings:
        emitter.damping(rec.type_token, int(rec.tag), *rec.args)

    # 8. Elements.  The H5 emitter consults two side channels for
    # connectivity (``set_element_nodes``) and FEM element id
    # (``set_current_fem_element_id``); driving them here keeps the
    # H5 round-trip byte-stable AND lets Tcl / Py / Live emitters
    # ignore the calls (their ``set_*`` helpers no-op when the attr
    # is absent).
    for rec in elements:
        if rec.connectivity:
            set_element_nodes(emitter, tuple(int(c) for c in rec.connectivity))
        set_current_fem_element_id(emitter, int(rec.fem_eid))
        emitter.element(rec.type_token, int(rec.tag), *rec.args)

    # 9. Fix / mass (model-level).
    for rec in fixes:
        emitter.fix(int(rec.tag), *(int(d) for d in rec.dofs))
    for rec in masses:
        emitter.mass(int(rec.tag), *(float(v) for v in rec.values))

    # 9b. Global initial stress (ADR 0054 Phase 1).  Emitted BEFORE
    # patterns / the analysis chain so ``step_hook_ramp`` registers and
    # the trailing ``analyze`` re-wraps into the hook-driven loop — without
    # this ordering the ramp procs declare but never fire (the emitter's
    # ``analyze`` emits the bare form when ``_step_hooks_registered`` is
    # False; tcl.py:402).  Mirrors the bridge's 7d-before-8 order.  Re-runs
    # the bridge emit helpers against the rehydrated declarative records:
    # parameter tags are freshly allocated (INV-5 — tags diverge across
    # round-trip) but deterministically, so the deck regenerates the same
    # bytes on every replay.  The H5 target never reaches here (its
    # initial-stress persists via the side-channel, not _replay_into).
    if initial_stress:
        from .build import (
            emit_initial_stress_addtoparameter,
            emit_initial_stress_global,
        )
        from .tag_allocator import TagAllocator

        _is_tags = TagAllocator()
        fem_eid_to_ops_tag = {
            int(e.fem_eid): int(e.tag)
            for e in elements
            if int(e.fem_eid) >= 0
        }
        name_to_param_tags = emit_initial_stress_global(
            initial_stress, emitter, _is_tags,
        )
        emit_initial_stress_addtoparameter(
            initial_stress, emitter, fem,
            name_to_param_tags=name_to_param_tags,
            fem_eid_to_ops_tag=fem_eid_to_ops_tag,
        )

    # 10. Patterns.  ``args`` are the original ``pattern_open`` args
    # (e.g. ``(ts_tag,)`` for Plain).  Inside each pattern: loads,
    # sps, ele_loads in the order the bridge emitted them.
    for rec in patterns:
        emitter.pattern_open(rec.type_token, int(rec.tag), *rec.args)
        for load in rec.loads:
            emitter.load(int(load.target), *load.forces)
        for sp in rec.sps:
            emitter.sp(int(sp.target), int(sp.dof), float(sp.value))
        for ele_load in rec.ele_loads:
            emitter.eleLoad(*ele_load.args)
        emitter.pattern_close()

    # 11. Recorders.  Schema 2.3.0 wraps declared recorders in a
    # begin/end context; the helper replays the wrapping so the
    # downstream emitter sees the same calls the bridge originally
    # produced.  Typed primitives (``decl_context is None``) emit
    # bare.
    for rec in recorders:
        ctx = rec.decl_context
        if ctx is not None:
            emitter.recorder_declaration_begin(
                declaration_name=ctx.declaration_name,
                record_name=ctx.record_name,
                category=ctx.category,
                components=ctx.components,
                raw=ctx.raw,
                pg=ctx.pg,
                label=ctx.label,
                selection=ctx.selection,
                ids=ctx.ids,
                dt=ctx.dt,
                n_steps=ctx.n_steps,
                file_root=ctx.file_root,
            )
            try:
                emitter.recorder(rec.kind, *rec.args)
            finally:
                emitter.recorder_declaration_end()
        else:
            emitter.recorder(rec.kind, *rec.args)

    # 12. Analysis chain.  ``analysis_attrs`` is the same flat dict
    # the H5 emitter accumulates via its constraints / numberer /
    # system / test / algorithm / integrator / analysis methods.
    if analysis_attrs:
        _replay_analysis_chain(emitter, analysis_attrs)
    if analyze_call is not None:
        steps, dt = analyze_call
        if dt is None:
            emitter.analyze(steps=int(steps))
        else:
            emitter.analyze(steps=int(steps), dt=float(dt))


def _replay_analysis_chain(
    emitter: Any, attrs: "dict[str, Any]",
) -> None:
    """Replay an analysis-chain flat dict onto ``emitter``.

    Mirrors how :class:`H5Emitter` accumulates analysis-chain calls
    into ``self._analysis_attrs`` — each key maps to one Protocol
    method; ``<key>_args`` carries the trailing positional args when
    present.
    """
    if "handler" in attrs:
        args = attrs.get("handler_args", ())
        emitter.constraints(attrs["handler"], *args)
    if "numberer" in attrs:
        emitter.numberer(attrs["numberer"])
    if "system" in attrs:
        emitter.system(attrs["system"], *attrs.get("system_args", ()))
    if "test" in attrs:
        emitter.test(attrs["test"], *attrs.get("test_args", ()))
    if "algorithm" in attrs:
        emitter.algorithm(attrs["algorithm"], *attrs.get("algorithm_args", ()))
    if "integrator" in attrs:
        emitter.integrator(
            attrs["integrator"], *attrs.get("integrator_args", ()),
        )
    if "analysis" in attrs:
        emitter.analysis(attrs["analysis"])
