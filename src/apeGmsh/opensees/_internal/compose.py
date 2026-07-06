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
    skip_node_tags: "frozenset[int]" = frozenset(),
    skip_element_tags: "frozenset[int]" = frozenset(),
    initial_stress_tags: Any = None,
    reinforce_name_to_tag: "dict[str, int] | None" = None,
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
      8b. ``emit_reinforce_ties`` — g.reinforce ``LadrunoEmbeddedRebar``
          couplings, re-emitted from the neutral-zone ``fem`` (no dedicated
          deck record; tags allocated past the max element tag). Skipped when
          ``fem`` is ``None`` (the H5 re-emit path; ties persist via the
          neutral zone there). Scoped/best-effort — see the inline note.
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
        # ADR 0055 P2.3: stage-owned nodes emit INSIDE their stage block
        # (_replay_staged_into), not in the global prefix — skip here.
        if int(tag) in skip_node_tags:
            continue
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
        # ADR 0055 P2.3: stage-owned elements emit inside their stage
        # block (_replay_staged_into); skip them in the global prefix.
        if int(rec.tag) in skip_element_tags:
            continue
        if rec.connectivity:
            set_element_nodes(emitter, tuple(int(c) for c in rec.connectivity))
        set_current_fem_element_id(emitter, int(rec.fem_eid))
        emitter.element(rec.type_token, int(rec.tag), *rec.args)

    # 8b. Embedded-reinforcement ties (g.reinforce → LadrunoEmbeddedRebar).
    # The deck-replay path re-emits these from the NEUTRAL-zone ``fem`` (which
    # carries ``fem.elements.reinforce_ties`` from the /reinforce_ties group,
    # ADR 0067 P5.1) rather than a dedicated /opensees deck record — the deck
    # zone never stored them (the H5 emitter no-ops the tie deck record;
    # persistence is the neutral zone's job). This mirrors how _replay_into
    # already leans on ``fem`` for element-connectivity rehydration. Tie element
    # tags are freshly allocated PAST the max replayed element tag (the ties
    # share the element namespace, so a fresh 1-based counter would collide with
    # the directly-replayed element tags); only intra-deck uniqueness matters
    # (ADR 0019 INV-5 — tags may diverge across round-trip). The bond material
    # name→tag map is threaded from OpenSeesModel._names.
    #
    # SCOPED / best-effort: the broader MP-constraint family (equalDOF /
    # rigidLink / rigidDiaphragm / embeddedNode / contact / equation ties) is
    # still NOT replayed by _replay_into — reinforce ties are the first ("A4
    # full", plan_rebar_p5.md). The canonical recovery for ALL of these remains
    # FEMData.from_h5 → forward re-emit; deck-replay is a secondary, partial
    # path. The H5 re-emit caller (OpenSeesModel._replay_into_h5) passes no
    # ``fem``, so this is skipped there (the H5 target persists ties via the
    # neutral zone).
    reinforce_ties = (
        getattr(getattr(fem, "elements", None), "reinforce_ties", None)
        if fem is not None else None
    )
    if reinforce_ties:
        from .build import emit_reinforce_ties
        from .tag_allocator import TagAllocator

        _rt_tags = TagAllocator()
        # Seed the element counter past the max replayed element tag so the tie
        # element tags don't collide with the directly-replayed elements.
        max_elem_tag = max((int(e.tag) for e in elements), default=0)
        _rt_tags._counters["element"] = max_elem_tag
        emit_reinforce_ties(
            emitter, fem, _rt_tags,
            name_to_tag=dict(reinforce_name_to_tag or {}),
        )

    # 9. Fix / mass (model-level).
    for rec in fixes:
        emitter.fix(int(rec.tag), *(int(d) for d in rec.dofs))
    for rec in masses:
        emitter.mass(int(rec.tag), *(float(v) for v in rec.values))

    # 9b. Global initial stress (ADR 0055 Phase 1).  Emitted BEFORE
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
            FemToOpsTagMap,
            emit_initial_stress_addtoparameter,
            emit_initial_stress_global,
        )
        from .tag_allocator import TagAllocator

        # ADR 0055 P2.3: the staged caller threads its SHARED allocator
        # so global + per-stage parameter tags accumulate on one counter
        # (the bridge reuses one ``tags`` across everything).  Flat
        # callers pass None → fresh allocator (unchanged behaviour).
        _is_tags = initial_stress_tags or TagAllocator()
        # ADR 0065 v2 B3: the emit helpers now take a FemToOpsTagMap.
        fem_eid_to_ops_tag = FemToOpsTagMap.from_pairs(
            (int(e.fem_eid), int(e.tag))
            for e in elements
            if int(e.fem_eid) >= 0
        )
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
        _replay_recorder(emitter, rec)

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


def _int_recover(args: "Sequence[Any]") -> "tuple[Any, ...]":
    """Coerce integral floats back to ``int`` in a chain ``*_args`` tuple.

    ADR 0055 P2.3: an analysis-chain arg tuple with mixed numeric types
    (``NormDispIncr(tol=1e-4, max_iter=50, p_flag=0, n_type=2)``) is
    stored by ``_set_attr`` as a single ``float64`` array and read back
    all-float (``(1e-4, 50.0, 0.0, 2.0)``).  ``OPS_GetIntInput`` rejects
    a float where it wants an int, and the Tcl deck bytes drift
    (``50.0`` vs ``50``).  Recover any float that is exactly integral to
    ``int`` — a genuine float tol like ``1e-4`` is untouched.  Applies
    to BOTH the flat and staged replay so neither drifts.
    """
    out: list[Any] = []
    for a in args:
        if isinstance(a, float) and a.is_integer():
            out.append(int(a))
        else:
            out.append(a)
    return tuple(out)


def _replay_analysis_chain(
    emitter: Any, attrs: "dict[str, Any]",
) -> None:
    """Replay an analysis-chain flat dict onto ``emitter``.

    Mirrors how :class:`H5Emitter` accumulates analysis-chain calls
    into ``self._analysis_attrs`` — each key maps to one Protocol
    method; ``<key>_args`` carries the trailing positional args when
    present.  ``*_args`` tuples are int-recovered (see
    :func:`_int_recover`) so a round-tripped ``NormDispIncr`` etc.
    re-renders byte-identically to the bridge.
    """
    if "handler" in attrs:
        args = _int_recover(attrs.get("handler_args", ()))
        emitter.constraints(attrs["handler"], *args)
    # ADR 0027 INV-5 (P5.0b): a partitioned build auto-emits the
    # runtime-conditional numberer / system (parallel primary with a
    # serial fallback).  The fallback attrs persist alongside the
    # primary; replay must re-drive the dedicated Protocol methods or
    # the conditional silently degrades to a bare primary — the H5
    # re-write drops the ``*_runtime_fallback`` attrs (model_hash
    # drift) and a tcl / py re-emit loses single-process portability.
    if "numberer" in attrs:
        numberer_fb = attrs.get("numberer_runtime_fallback")
        if numberer_fb is not None:
            emitter.parallel_runtime_fallback_numberer(
                attrs["numberer"], str(numberer_fb),
            )
        else:
            emitter.numberer(attrs["numberer"])
    if "system" in attrs:
        system_fb = attrs.get("system_runtime_fallback")
        if system_fb is not None:
            # The INV-5 auto-emit is the only fallback producer and
            # never carries system args.
            emitter.parallel_runtime_fallback_system(
                attrs["system"], str(system_fb),
            )
        else:
            emitter.system(
                attrs["system"],
                *_int_recover(attrs.get("system_args", ())),
            )
    if "test" in attrs:
        emitter.test(attrs["test"], *_int_recover(attrs.get("test_args", ())))
    if "algorithm" in attrs:
        emitter.algorithm(
            attrs["algorithm"], *_int_recover(attrs.get("algorithm_args", ())),
        )
    if "integrator" in attrs:
        emitter.integrator(
            attrs["integrator"], *_int_recover(attrs.get("integrator_args", ())),
        )
    if "analysis" in attrs:
        emitter.analysis(attrs["analysis"])


def _replay_recorder(emitter: Any, rec: Any) -> None:
    """Replay one recorder record, wrapping a declared recorder in its
    declaration begin/end context (shared by flat + staged replay)."""
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


def _region_is_scoped(args: "Sequence[Any]") -> bool:
    """True iff a stage region carries a ``-rayleigh`` / ``-damp`` tail.

    Those region forms emit at slot 11 (after ``domainChange``,
    interleaved with the global-form rayleighs); a plain ``s.region``
    (``-node`` / ``-ele`` / ``-eleOnly``) emits at slot 7. Re-derives
    the kind from the arg tokens exactly as the writer's ``kind`` attr
    derivation does (ADR 0055 P2.1)."""
    toks = {a for a in args if isinstance(a, str)}
    return "-rayleigh" in toks or "-damp" in toks


def _replay_staged_into(
    emitter: Any,
    *,
    stages: "Sequence[Any]",
    **replay_kwargs: Any,
) -> None:
    """Re-emit a STAGED archive's deck (ADR 0055 P2.3) onto ``emitter``.

    Sibling of :func:`_replay_into` for tcl / py targets only.  Emits
    the global prefix (with stage-owned nodes/elements filtered out),
    then re-drives each stage's emit block in the exact order the
    bridge's ``_emit_stages_flat`` uses.  The H5 target never reaches
    here (it round-trips via ``restore_stage_blocks`` + the writer);
    the Live target raises upfront (``LiveOpsEmitter.stage_open``
    raises — fail clean, not deep in replay).

    ``replay_kwargs`` are the same keyword arguments :func:`_replay_into`
    accepts (the global record graph); ``elements`` MUST already be
    connectivity-rehydrated (the owned-element lookup keys into it).
    """
    from .build import (
        ActivateAbsorbingRecord,
        emit_activate_absorbing,
        emit_initial_stress_addtoparameter,
        emit_initial_stress_global,
    )
    from .tag_allocator import TagAllocator
    from .tag_resolution import set_current_fem_element_id, set_element_nodes

    # 0. Live guard — fail clean before any emit (the live emitter's
    # stage_open raises; a deep mid-replay crash would be opaque).
    from ..emitter.live import LiveOpsEmitter
    if isinstance(emitter, LiveOpsEmitter):
        raise NotImplementedError(
            "OpenSeesModel.build('live'): live re-emit of a staged "
            "archive is not supported (LiveOpsEmitter.stage_open "
            "raises). Use build('tcl') / build('py') for staged decks."
        )

    nodes = replay_kwargs.get("nodes", ())
    elements = replay_kwargs.get("elements", ())
    # Always supplied by the caller (OpenSeesModel._populate_emitter);
    # typed Any so the bridge emit helpers (FEMData param) accept it,
    # exactly as the flat ``_replay_into(fem=...)`` path does.
    fem: Any = replay_kwargs.get("fem")

    owned_node_tags = frozenset(
        int(t) for s in stages for t in s.owned_node_ids
    )
    owned_element_tags = frozenset(
        int(t) for s in stages for t in s.owned_element_ids
    )

    # ONE allocator threaded across the global prefix AND every stage
    # (the bridge reuses a single ``tags``; a per-stage allocator would
    # restart parameter counters at stage boundaries — gate-1 FATAL).
    tags = TagAllocator()

    # 1. Global prefix — _replay_into with stage-owned topology filtered
    # out and the shared allocator threaded for any GLOBAL initial_stress.
    _replay_into(
        emitter,
        skip_node_tags=owned_node_tags,
        skip_element_tags=owned_element_tags,
        initial_stress_tags=tags,
        **replay_kwargs,
    )

    # Lookups for owned-topology re-emit inside each stage block.
    node_map: "dict[int, tuple[tuple[float, ...], int | None]]" = {}
    for entry in nodes:
        if len(entry) == 3:
            t, coords, nndf = entry
        else:
            t, coords = entry
            nndf = None
        node_map[int(t)] = (coords, nndf)
    elem_map = {int(r.tag): r for r in elements}
    # ADR 0065 v2 B3: the stage emit helpers (initial_stress /
    # activate_absorbing) now take a FemToOpsTagMap.
    from .build import FemToOpsTagMap

    fem_eid_to_ops_tag = FemToOpsTagMap.from_pairs(
        (int(r.fem_eid), int(r.tag)) for r in elements if int(r.fem_eid) >= 0
    )

    # 2. Per-stage blocks — exact _emit_stages_flat order.
    for st in stages:
        emitter.stage_open(st.name)
        if st.set_time is not None:
            emitter.set_time(float(st.set_time))
        if st.set_creep_on is not None:
            emitter.set_creep(bool(st.set_creep_on))

        # owned nodes (verbatim, ndf elide-on-equal already baked into
        # the stored node_ndf via _ndf_or_none on the caller side).
        for nid in st.owned_node_ids:
            ent = node_map.get(int(nid))
            if ent is None:
                continue
            coords, nndf = ent
            if nndf is None:
                emitter.node(int(nid), *(float(c) for c in coords))
            else:
                emitter.node(
                    int(nid), *(float(c) for c in coords), ndf=int(nndf),
                )
        # owned elements (look up the rehydrated record by ops tag).
        for etag in st.owned_element_ids:
            rec = elem_map.get(int(etag))
            if rec is None:
                continue
            if rec.connectivity:
                set_element_nodes(
                    emitter, tuple(int(c) for c in rec.connectivity),
                )
            set_current_fem_element_id(emitter, int(rec.fem_eid))
            emitter.element(rec.type_token, int(rec.tag), *rec.args)

        # SSI-2.E removals (before new BCs).
        for n_tag, dof in st.remove_sps:
            emitter.remove_sp(int(n_tag), int(dof))
        for e_tag in st.remove_elements:
            emitter.remove_element(int(e_tag))

        # stage fix / mass.
        for r in st.fixes:
            emitter.fix(int(r.tag), *(int(d) for d in r.dofs))
        for r in st.masses:
            emitter.mass(int(r.tag), *(float(v) for v in r.values))
        # Stage regions split by kind (bridge emits them at TWO slots):
        # plain ``s.region`` (node_or_filter) here at slot 7; the
        # region-scoped ``-rayleigh`` / ``-damp`` forms emit at slot 11
        # (after domain_change), interleaved with the global-form
        # rayleighs by their captured emit_index.  Kind is re-derived
        # from the arg tokens exactly as the writer derived it.
        scoped_regions = [
            (seq, r) for seq, r in zip(st.region_seq, st.regions)
            if _region_is_scoped(r.args)
        ]
        for seq, r in zip(st.region_seq, st.regions):
            if not _region_is_scoped(r.args):
                emitter.region(int(r.tag), *r.args)

        # stage MP constraints — reconstructed from the POST-fan-out RO
        # records (the build-side pool is gone) in the bridge's emit
        # order.  The bridge interleaves the four kinds across one pass
        # (rigid_links → equal_dofs[genuine] → rigid_diaphragms →
        # equal_dofs[kinematic] → embedded_nodes), so a kinematic
        # equalDOF straddles rigidDiaphragm.  Merge-sort by the captured
        # emit_index to reproduce that exactly; fall back to the fixed
        # kind order for pre-P2.3 archives that carry no seq.
        def _emit_rigid_link(r: Any) -> None:
            if r.name:
                emitter.mp_constraint_comment(r.name)
            emitter.rigidLink(r.kind, int(r.master), int(r.slave))

        def _emit_equal_dof(r: Any) -> None:
            if r.name:
                emitter.mp_constraint_comment(r.name)
            emitter.equalDOF(
                int(r.master), int(r.slave), *(int(d) for d in r.dofs),
            )

        def _emit_rigid_diaphragm(r: Any) -> None:
            if r.name:
                emitter.mp_constraint_comment(r.name)
            emitter.rigidDiaphragm(
                int(r.perp_dir), int(r.master),
                *(int(s2) for s2 in r.slaves),
            )

        def _emit_embedded(r: Any) -> None:
            if r.name:
                emitter.mp_constraint_comment(r.name)
            emitter.embeddedNode(
                int(r.ele_tag), int(r.cnode), *(int(a) for a in r.args),
                stiffness=r.stiffness, stiffness_p=r.stiffness_p,
                rotational=r.rotational, pressure=r.pressure,
            )

        mp_groups = (
            (st.rigid_links, st.rigid_link_seq, _emit_rigid_link),
            (st.equal_dofs, st.equal_dof_seq, _emit_equal_dof),
            (st.rigid_diaphragms, st.rigid_diaphragm_seq, _emit_rigid_diaphragm),
            (st.embedded_nodes, st.embedded_node_seq, _emit_embedded),
        )
        have_seq = all(
            len(seq) == len(recs) for recs, seq, _ in mp_groups
        ) and any(seq for _, seq, _ in mp_groups)
        if have_seq:
            mp_items: "list[tuple[int, Any, Any]]" = []
            for recs, seq, fn in mp_groups:
                for s_idx, rec in zip(seq, recs):
                    mp_items.append((int(s_idx), fn, rec))
            mp_items.sort(key=lambda it: it[0])
            for _s, fn, rec in mp_items:
                fn(rec)
        else:
            # Pre-P2.3 fallback: fixed kind order (correct unless a
            # stage mixes rigid_diaphragm + kinematic_coupling).
            for recs, _seq, fn in mp_groups:
                for rec in recs:
                    fn(rec)

        # HOLD support patterns (slot 10, BEFORE domain_change) — split
        # by sp_holds presence (the role attr is not read back).
        hold_patterns = [p for p in st.patterns if p.sp_holds]
        load_patterns = [p for p in st.patterns if not p.sp_holds]
        for p in hold_patterns:
            emitter.pattern_open(p.type_token, int(p.tag), *p.args)
            for n_tag, dof in p.sp_holds:
                emitter.sp_hold(int(n_tag), int(dof))
            emitter.pattern_close()

        # domain_change — replay the captured bool, do NOT recompute.
        if st.domain_changed:
            emitter.domain_change()

        # Slot 11 — rayleigh + region-scoped damping, in the bridge's
        # interleaved order.  The bridge runs _emit_rayleigh then
        # _emit_damping_attach after domain_change; the global-form
        # rayleigh (bare ``rayleigh`` line) and the region-scoped
        # ``-rayleigh`` / ``-damp`` lines were captured with a shared
        # per-stage emit_index, so merge-sort by it to reproduce the
        # exact sequence.
        slot11: "list[tuple[int, int, Any]]" = []
        for seq, coeffs in zip(st.rayleigh_seq, st.rayleighs):
            slot11.append((int(seq), 0, coeffs))
        for seq, r in scoped_regions:
            slot11.append((int(seq), 1, r))
        slot11.sort(key=lambda item: item[0])
        for _seq, kind, payload in slot11:
            if kind == 0:
                emitter.rayleigh(*(float(c) for c in payload))
            else:
                emitter.region(int(payload.tag), *payload.args)

        # stage initial_stress — re-run the bridge helpers with the
        # SHARED allocator (parameter tags accumulate across stages).
        if st.initial_stress:
            name_to_param_tags = emit_initial_stress_global(
                st.initial_stress, emitter, tags,
            )
            emit_initial_stress_addtoparameter(
                st.initial_stress, emitter, fem,
                name_to_param_tags=name_to_param_tags,
                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            )

        # activate_absorbing — declarative (pg/elements); re-run the
        # helper (allocates a fresh parameter tag from the shared pool).
        if st.activate_absorbing:
            ab_records = tuple(
                ActivateAbsorbingRecord(pg=pg, elements=els)
                for pg, els in st.activate_absorbing
            )
            emit_activate_absorbing(
                ab_records, emitter, fem,
                fem_eid_to_ops_tag=fem_eid_to_ops_tag, tags=tags,
            )

        # stage analysis chain.
        if st.chain_attrs:
            _replay_analysis_chain(emitter, dict(st.chain_attrs))

        # load patterns (slot 16, AFTER the chain).
        for p in load_patterns:
            emitter.pattern_open(p.type_token, int(p.tag), *p.args)
            for load in p.loads:
                emitter.load(int(load.target), *load.forces)
            for sp in p.sps:
                emitter.sp(int(sp.target), int(sp.dof), float(sp.value))
            for ele_load in p.ele_loads:
                emitter.eleLoad(*ele_load.args)
            emitter.pattern_close()

        # stage recorders.
        for rec in st.recorders:
            _replay_recorder(emitter, rec)

        if st.pre_analyze_reset:
            emitter.reset()

        # label= mirrors the bridge's _emit_stages_flat call so the
        # fail-loud analyze banner names the stage (deck equality).
        if st.analyze_dt is None:
            emitter.analyze(steps=int(st.analyze_steps), label=st.name)
        else:
            emitter.analyze(
                steps=int(st.analyze_steps), dt=float(st.analyze_dt),
                label=st.name,
            )
        emitter.stage_close()
