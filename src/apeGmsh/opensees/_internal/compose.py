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
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import h5py

    from ..._internal.provenance import ProvenanceTable


__all__ = [
    "ReplaySkippedStreamWarning",
    "_compose_model_h5",
    "_merge_provenance",
    "_replay_into",
    "_try_write_broker_zone",
    "_override_schema_version",
    "_path_stem",
]


class ReplaySkippedStreamWarning(UserWarning):
    """Deck replay left out a stream the source model carries (D8, #1412).

    :func:`_replay_into` re-emits the ``/opensees`` deck records plus the
    ``g.reinforce`` ties.  It has no deck record for the neutral-zone
    streams that the forward bridge fans out at emit time: equalDOF /
    rigidLink / rigidDiaphragm / node-to-surface couplings, penalty and
    equation ties, ``g.embed`` ties, contacts, and phantom-bridged
    interfaces.  A replayed deck that carries any of them differs from the
    forward ``apeSees`` deck, so the replay warns and names each one.
    """


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
    ndm: int,
    ndf: int,
    cuts: "Sequence[Any]" = (),
    sweeps: "Sequence[Any]" = (),
    names: "Sequence[tuple[str, str, int]]" = (),
    computed_sections: "Sequence[tuple[int, str, str]]" = (),
    snapshot_id: str | None = None,
    nodes_ndf: "dict[int, int] | None" = None,
    provenance: "ProvenanceTable | None" = None,
) -> None:
    """Compose a ``model.h5`` from a broker ``fem`` + a populated ``emitter``.

    The one composition path.  Order: broker ``/meta`` + neutral zone
    (with a stub-FEM fallback to the bridge's own ``/meta`` +
    schema-version override), then the ``emitter``'s ``/opensees/...``
    enrichment, then apeGmsh.cuts v4 ``/opensees/cuts`` / ``/sweeps``,
    then ``/provenance`` (ADR 0112 D3).

    Parameters
    ----------
    fem
        The broker snapshot.  A hand-rolled stub lacking the FEMData
        surface triggers the bridge-only fallback (no neutral zone).
    emitter
        An already-populated :class:`H5Emitter`.
    path
        Destination HDF5 path (opened ``"w"``).
    model_name, ndm, ndf
        Written into ``/meta`` by the broker writer.  ``ndm`` / ``ndf``
        are the ``ops.model`` declaration (the spatial dimension and
        DOFs per node); the writer never derives them from the mesh,
        so a line-only 2-D frame stamps ``ndm=2`` (#1291).  An
        undeclared ``ndm`` (``< 1``) is refused: every reader takes
        ``/meta/ndm`` as the model's dimension.
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
    provenance
        The bridge's declaration provenance (``apeSees._provenance``),
        appended to the snapshot's own table (``fem.provenance``, the
        session's records, which ``FEMData.from_h5`` carries forward
        from a source file).  The merged table is written as
        ``/provenance`` when the snapshot is a real :class:`FEMData`
        and either side has records; a replay writer passes ``None``
        and so copies the snapshot's table forward (V0 Q7).  A stub
        snapshot (the bridge-only fallback) writes none: it belongs to
        no run, and its records would hold test paths.  No hash reads
        it.
    """
    import h5py

    from ..._internal.provenance import base_dir_for, encode_columns
    from ...cuts._h5_io import write_cuts_into
    from ...mesh._femdata_h5_io import (
        NEUTRAL_SCHEMA_VERSION,
        _write_provenance,
    )
    from ..emitter.h5 import SCHEMA_VERSION
    from ._computed_sections_h5 import write_computed_sections_into
    from ._names_h5 import write_names_into
    from .lineage import (
        compute_fem_hash,
        compute_model_hash,
        write_lineage_attrs,
    )

    if int(ndm) < 1:
        raise ValueError(
            f"_compose_model_h5: ndm={ndm!r} is not a declared spatial "
            "dimension; call ops.model(ndm=, ndf=) before writing model.h5 "
            "(/meta/ndm is read as the model's dimension, #1291)."
        )

    # ADR 0112 D3: merge and encode before the file is opened, so an
    # int32 overflow refuses with nothing written (h5-schema.md,
    # "Integer policy").  Only a real FEMData snapshot carries the zone;
    # the stub fallback below writes none.
    from ...mesh.FEMData import FEMData

    prov_table = None
    if isinstance(fem, FEMData):
        base = fem.provenance
        if provenance is not None and base is not None:
            # The bridge owns the ``opensees/`` zone: a snapshot loaded
            # from a bridge-written file carries that file's bridge
            # records, which this bridge declares afresh (the "script 2,
            # analyse" flow, FEMData.from_h5 -> apeSees -> h5).  Keeping
            # them would collide on a repeated name or leave a stale
            # record for a declaration this bridge did not make.
            base = _drop_zone(base, "opensees")
        prov_table = _merge_provenance(base, provenance)
    prov_base_dir = base_dir_for(path)
    prov_columns = (
        encode_columns(prov_table, prov_base_dir)
        if prov_table is not None else None
    )

    with h5py.File(path, "w") as f:
        broker_used = _try_write_broker_zone(
            fem, f,
            schema_version=NEUTRAL_SCHEMA_VERSION,
            model_name=model_name,
            ndm=int(ndm),
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
        # ComputedSection provenance sidecar (ADR 0078 Amendment A1) —
        # no-op when empty; hash-excluded like names (provenance
        # metadata, not authored model state).
        write_computed_sections_into(f, computed_sections)
        # ADR 0112 D3 — the declaration provenance of the snapshot and
        # the bridge, after /opensees.  ``model_hash`` below reads
        # /opensees only and ``fem_hash`` the neutral zone, so the zone
        # changes neither (V2a invariance).
        if prov_columns is not None and "meta" in f:
            _write_provenance(f, prov_columns, prov_base_dir)

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


def _merge_provenance(
    base: "ProvenanceTable | None",
    extra: "ProvenanceTable | None",
) -> "ProvenanceTable | None":
    """Append ``extra``'s records to ``base``, re-indexing its rows.

    The tables come from two stores (the session's and the bridge's),
    each with its own ``files`` / ``sites`` rows and 0-based ``seq``.
    ``extra``'s files and sites are deduplicated into ``base``'s, its
    records keep their order with ``seq`` continued after ``base``'s
    last record, and a declaration path present on both sides is
    refused: the zones differ (``neutral/`` / ``geometry/`` against
    ``opensees/``), so a collision is a programming error, not a
    merge.  ``None`` when neither side has a table; an empty ``extra``
    leaves ``base`` as it is.
    """
    from ..._internal.provenance import ProvenanceTable, RecordRow, SiteRow

    if extra is None or not extra.records:
        return base
    if base is None:
        return extra

    files = list(base.files)
    file_index = {row: i for i, row in enumerate(files)}
    sites = list(base.sites)
    site_index = {row: i for i, row in enumerate(sites)}
    records = list(base.records)
    paths = {r.path for r in records}

    def file_row(i: int) -> int:
        row = extra.files[i]
        j = file_index.get(row)
        if j is None:
            j = file_index[row] = len(files)
            files.append(row)
        return j

    def site_row(i: int) -> int:
        if i < 0:
            return -1
        s = extra.sites[i]
        row = SiteRow(file_row(s.file), s.line, s.function)
        j = site_index.get(row)
        if j is None:
            j = site_index[row] = len(sites)
            sites.append(row)
        return j

    offset = len(records)
    for r in extra.records:
        if r.path in paths:
            raise ValueError(
                f"_merge_provenance: declaration path {r.path!r} is "
                "recorded by both the snapshot and the bridge; the two "
                "stores must not share a zone")
        paths.add(r.path)
        records.append(RecordRow(
            r.path, site_row(r.site), site_row(r.script), offset + r.seq,
            r.origin))
    return ProvenanceTable(tuple(files), tuple(sites), tuple(records))


def _drop_zone(table: "ProvenanceTable", zone: str) -> "ProvenanceTable":
    """``table`` without the records of ``zone``, compacted: ``seq``
    renumbered from 0 in the surviving order and the ``files`` / ``sites``
    rows nothing references any more dropped."""
    from ..._internal.provenance import ProvenanceTable, RecordRow

    kept = [r for r in table.records if not r.path.startswith(f"{zone}/")]
    if len(kept) == len(table.records):
        return table
    renumbered = ProvenanceTable(
        table.files, table.sites,
        tuple(RecordRow(r.path, r.site, r.script, i, r.origin)
              for i, r in enumerate(kept)))
    # Merging onto an empty table re-indexes files and sites through the
    # dedupe path, so only the rows the kept records reach survive.
    merged = _merge_provenance(ProvenanceTable(), renumbered)
    assert merged is not None
    return merged


def _try_write_broker_zone(
    fem: object,
    f: "h5py.File",
    *,
    schema_version: str,
    model_name: str,
    ndm: int,
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
            ndm=ndm,
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


def _constraint_replays_as_element(rec: Any, kind_cls: Any) -> bool:
    """True iff the forward emit writes ``rec`` as an ``element`` line.

    Those lines are archived in ``/opensees/element_meta``, and replay
    re-emits them with the other elements, so the record is not skipped:
    an RBE2 ``kinematic_coupling`` (``LadrunoKinematicCoupling``), a
    ``rigid_body`` with ``as_element`` (``LadrunoRigidBody``), an
    interpolation on the ``penalty_al`` route (``LadrunoEmbeddedNode``)
    and a ``distributing`` RBE3 off the equation route
    (``LadrunoDistributingCoupling``).  This mirrors the branches of
    ``build.emit_mp_constraints`` and ``build._emit_one_interpolation``.
    Every other kind goes out through a verb the deck zone has no replay
    for, so it counts as skipped; an unknown kind counts as skipped too.
    """
    kind = rec.kind
    if kind == kind_cls.KINEMATIC_COUPLING:
        return True
    if kind == kind_cls.RIGID_BODY:
        return bool(rec.as_element)
    return False


def _interpolation_replays_as_element(rec: Any, kind_cls: Any) -> bool:
    """:func:`_constraint_replays_as_element` for one interpolation row."""
    if rec.enforce == "penalty_al":
        return True
    return bool(
        rec.kind == kind_cls.DISTRIBUTING and rec.enforce != "equation"
    )


def _stage_mp_keys(stages: "Sequence[Any]") -> "frozenset[tuple[Any, ...]]":
    """The keys of every MP line the stage blocks of ``stages`` re-emit.

    A key is ``(bucket, *what it ties)`` over the four ``StageRecordRO``
    MP buckets; :func:`_constraint_stage_keys` builds the matching side.
    Keys carry the bucket and the tied nodes, not the declaration name.
    The archived name is no identity: a ``tied_contact`` slave row has
    ``name=None`` (only its parent is named), and a ``rigid_body`` group
    writes its name on its first ``rigidLink`` only.
    """
    keys: "set[tuple[Any, ...]]" = set()
    for s in stages:
        for r in s.equal_dofs:
            keys.add(("equal_dof", int(r.master), int(r.slave),
                      tuple(int(d) for d in r.dofs)))
        for r in s.rigid_links:
            keys.add(("rigid_link", str(r.kind), int(r.master), int(r.slave)))
        for r in s.rigid_diaphragms:
            keys.add(("rigid_diaphragm", int(r.master),
                      tuple(sorted(int(n) for n in r.slaves))))
        for r in s.embedded_nodes:
            keys.add(("embedded_node", int(r.cnode),
                      tuple(int(n) for n in r.args)))
    return frozenset(keys)


def _pair_stage_keys(rec: Any, kind_cls: Any) -> "list[tuple[Any, ...]]":
    """Stage-bucket keys of one ``NodePairRecord`` (empty: no bucket)."""
    if rec.kind == kind_cls.EQUAL_DOF:
        return [("equal_dof", int(rec.master_node), int(rec.slave_node),
                 tuple(int(d) for d in rec.dofs))]
    if rec.kind in (kind_cls.RIGID_BEAM, kind_cls.RIGID_ROD):
        link = "beam" if rec.kind == kind_cls.RIGID_BEAM else "bar"
        return [("rigid_link", link, int(rec.master_node),
                 int(rec.slave_node))]
    return []


def _constraint_stage_keys(
    rec: Any, kind_cls: Any,
) -> "list[tuple[Any, ...]]":
    """The stage-bucket keys a stage block writes for one node constraint.

    Mirrors ``build._emit_rigid_links`` / ``_emit_equal_dofs`` /
    ``_emit_rigid_diaphragms``.  An empty list means no stage bucket can
    hold the record, so it never counts as stage-replayed.
    """
    kind = rec.kind
    if kind == kind_cls.RIGID_BODY:
        return [("rigid_link", "beam", int(rec.master_node), int(sn))
                for sn in rec.slave_nodes]
    if kind == kind_cls.RIGID_DIAPHRAGM:
        return [("rigid_diaphragm", int(rec.master_node),
                 tuple(sorted(int(n) for n in rec.slave_nodes)))]
    if kind in (kind_cls.NODE_TO_SURFACE, kind_cls.NODE_TO_SURFACE_SPRING):
        # The nested pairs, as the two emitters walk them: the rigid-kind
        # links, then every phantom -> slave equalDOF.
        keys: "list[tuple[Any, ...]]" = []
        for pair in rec.rigid_link_records:
            keys += _pair_stage_keys(pair, kind_cls)
        for pair in rec.equal_dof_records:
            keys.append(("equal_dof", int(pair.master_node),
                         int(pair.slave_node),
                         tuple(int(d) for d in pair.dofs)))
        return keys
    return _pair_stage_keys(rec, kind_cls)


def _interpolation_stage_keys(rec: Any) -> "list[tuple[Any, ...]]":
    """Stage-bucket key of one non-element interpolation row: its
    ``embeddedNode`` line (none on the equation route, a ledger verb)."""
    if rec.enforce == "equation":
        return []
    return [("embedded_node", int(rec.slave_node),
             tuple(int(n) for n in rec.master_nodes))]


def _skipped_replay_streams(
    fem: Any, *, stage_mp_keys: "frozenset[tuple[Any, ...]]",
) -> "dict[str, Counter[str]]":
    """Return ``{stream: Counter(kind -> count)}`` for every non-empty
    neutral-zone stream that :func:`_replay_into` does not re-emit.

    ``stage_mp_keys`` (:func:`_stage_mp_keys`) holds the MP lines that a
    staged archive's stage blocks re-emit.  A record counts as replayed
    there only when every line its fan-out writes is among them, matched
    on bucket and nodes, so a stage that claims the equalDOF named ``x``
    does not also excuse a rigidDiaphragm named ``x``.

    The streams are read without defaults: a ``fem`` that lacks one fails
    here rather than passing for empty.
    """
    out: "dict[str, Counter[str]]" = {}

    def _staged(keys: "list[tuple[Any, ...]]") -> bool:
        return bool(keys) and all(k in stage_mp_keys for k in keys)

    node_set = fem.nodes.constraints
    kind_cls = node_set.Kind
    nodes_skipped: "Counter[str]" = Counter(
        str(rec.kind) for rec in node_set
        if not _constraint_replays_as_element(rec, kind_cls)
        and not _staged(_constraint_stage_keys(rec, kind_cls))
    )
    if nodes_skipped:
        out["fem.nodes.constraints"] = nodes_skipped

    # interpolations() expands tied_contact / mortar into their slave rows,
    # as the forward emit does, so each slave is matched on its own.
    surface_skipped: "Counter[str]" = Counter(
        str(rec.kind) for rec in fem.elements.constraints.interpolations()
        if not _interpolation_replays_as_element(rec, kind_cls)
        and not _staged(_interpolation_stage_keys(rec))
    )
    if surface_skipped:
        out["fem.elements.constraints"] = surface_skipped

    # An interface on an equal-ndf pair is a zeroLength plus its materials,
    # all replayed; a mixed-ndf pair also needs a phantom node and its
    # equalDOF, which replay has no record for.  No stage match is tried:
    # the writer refuses a stage phantom node (``set_stage_records``), so a
    # stage-claimed phantom interface never reaches an archive.
    interfaces_skipped: "Counter[str]" = Counter(
        str(rec.kind) for rec in fem.elements.interfaces
        if rec.phantom_node is not None
    )
    if interfaces_skipped:
        out["fem.elements.interfaces"] = interfaces_skipped

    # The H5 emitter keeps no deck record for these at all (ledger verbs).
    for stream, recs in (
        ("fem.elements.embed_ties", fem.elements.embed_ties),
        ("fem.elements.contacts", fem.elements.contacts),
        ("fem.elements.contact_planes", fem.elements.contact_planes),
    ):
        if recs:
            out[stream] = Counter({"record": len(recs)})
    return out


def _warn_skipped_replay_streams(
    fem: Any, *, stage_mp_keys: "frozenset[tuple[Any, ...]]",
) -> None:
    """Warn :class:`ReplaySkippedStreamWarning` iff ``fem`` carries a
    stream that replay does not re-emit, naming each stream and count."""
    skipped = _skipped_replay_streams(fem, stage_mp_keys=stage_mp_keys)
    if not skipped:
        return
    parts = []
    for stream, kinds in skipped.items():
        detail = ", ".join(f"{k}: {n}" for k, n in sorted(kinds.items()))
        parts.append(f"{stream} ({detail})")
    warnings.warn(
        "Deck replay does not re-emit "
        f"{len(skipped)} stream(s) the model carries: "
        + "; ".join(parts)
        + ". The replayed deck leaves these constraints and ties out, so "
        "it differs from the forward apeSees deck. For a faithful deck, "
        "load FEMData.from_h5(path) and emit it through apeSees(fem).",
        ReplaySkippedStreamWarning,
        stacklevel=3,
    )


def _replay_elements_bracketed(
    emitter: Any,
    recs: "list[Any]",
    *,
    ndm: int,
    envelope_ndf: int,
) -> None:
    """Re-emit rehydrated element records, wrapping builder-ndf-gated
    element runs in a ``model basic -ndf K`` bracket (ADR 0074).

    The forward emit brackets each gated element block (``quad`` / ``tri6n``
    / ``LadrunoQuad`` / ``LadrunoCST`` / equal-order ``LadrunoUP``, whose
    upstream parsers hard-gate on the BUILDER ndf) with a ``model`` re-issue
    + envelope restore.  Replay must do the same, or a mixed-ndf archive
    (e.g. envelope ndf=3 with a 3D equal-order LadrunoUP region needing
    builder ndf=4) re-emits element lines the fork parser refuses
    (OPS_LadrunoUP.cpp:162).  Runs of the same needed ndf are coalesced so
    a big deck does not gain two ``model`` lines per element; the envelope
    is restored after the last gated run so downstream fixes / patterns see
    it.
    """
    from .._element_capabilities import element_builder_ndf
    from .tag_resolution import set_current_fem_element_id, set_element_nodes

    cur = int(envelope_ndf)
    for rec in recs:
        need = element_builder_ndf(rec.type_token, ndm)
        target = int(need) if need is not None else int(envelope_ndf)
        if target != cur:
            emitter.model(ndm=int(ndm), ndf=target)
            cur = target
        if rec.connectivity:
            set_element_nodes(emitter, tuple(int(c) for c in rec.connectivity))
        set_current_fem_element_id(emitter, int(rec.fem_eid))
        emitter.element(rec.type_token, int(rec.tag), *rec.args)
    if cur != int(envelope_ndf):
        emitter.model(ndm=int(ndm), ndf=int(envelope_ndf))


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
    regions: "Sequence[Any]" = (),
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
    deck_ordering: bool = True,
    stage_mp_keys: "frozenset[tuple[Any, ...]]" = frozenset(),
    commands: "Sequence[Any]" = (),
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
      4b. the builder-ndf-GATED elements, as one bracketed run (ADR 0099
          INV-1, ``deck_ordering=True`` only) — see the note below
      5. ``emitter.geomTransf``
      6. ``emitter.beamIntegration``
      7. ``emitter.timeSeries`` (+ ``emitter.damping`` for tagged damping
         objects — ADR 0053 D3b — after the series a ``-factor`` references,
         before the elements an element-flag ``-damp`` references)
      8. ``emitter.element``  (the UNGATED elements when the 4b hoist
         ran; all of them otherwise) — with ``set_element_nodes`` /
         ``set_current_fem_element_id`` side channels for the H5 path
      8b. ``emit_reinforce_ties`` — g.reinforce ``LadrunoEmbeddedRebar``
          couplings, re-emitted from the neutral-zone ``fem`` (no dedicated
          deck record; tags allocated past the max element tag). Skipped when
          ``fem`` is ``None`` (the H5 re-emit path; ties persist via the
          neutral zone there). Scoped/best-effort — see the inline note.
      9. ``emitter.fix`` / ``emitter.mass``
      9a. the global ``/opensees/commands`` rows (``rayleigh``, ``eigen``,
          ``modal_damping``) in their emit order: the bridge's slot 7,
          after the masses and before the patterns and the chain
      9b. ``emitter.region`` for the top-level ``/opensees/regions`` rows,
          verbatim and in store order: after the global
          ``rayleigh`` so a region-scoped one still wins per element, and
          before the recorders that reference a fan-out region by ``-R``
      10. ``emitter.pattern_open`` (+ load / sp / eleLoad +
          pattern_close)
      11. ``emitter.recorder`` (wrapped in declaration-begin/end when
          ``decl_context`` is present)
      12. ``emitter.constraints`` / ``numberer`` / ``system`` /
          ``test`` / ``algorithm`` / ``integrator`` / ``analysis``
          + ``emitter.analyze`` if present

    .. note::

        **``deck_ordering`` — ADR 0099 INV-1.**  A builder-ndf bracket
        re-issues ``model BasicBuilder``, which deletes the Tcl model
        builder and purges the process-global ``timeSeries`` /
        ``geomTransf`` / ``beamIntegration`` / ``damping`` registries
        (fork ``TclModelBuilder.cpp:681``).  Steps 5-7b declare exactly
        those kinds, so with the elements all at step 8 the bracket
        destroys them and the deck dies at the first later reference —
        ``pattern Plain`` — or, for ``damping``, runs silently undamped.
        With ``deck_ordering`` the GATED elements hoist to step 4b, above
        the declarations, matching what ``_emit_flat`` does.  Tag identity
        is untouched: the records carry their tags, so only line
        positions move.

        Pass ``deck_ordering=False`` from a caller that produces no deck.
        The H5 re-emit path (``OpenSeesModel._populate_emitter_h5``) does:
        ``H5Emitter.model`` only stores ``ndm``/``ndf``, the bracket is
        never persisted as a line, so the hoist buys that path nothing —
        and skipping it keeps the archive-rewrite fixed point exactly as
        it was, by construction rather than by argument.

        ``build('live')`` deliberately does NOT opt out, even though it
        emits no deck either.  Measured on the fork's in-process module:
        a ``ops.model('basic', ...)`` re-issue does NOT purge the
        registries there (a ``timeSeries`` and a ``geomTransf`` declared
        before it both survive) — openseespy runs no ``TclModelBuilder``
        destructor, so live never had this defect.  The hoist is kept
        anyway because it is measurably harmless in-process and the
        ordering then holds for ANY backend, rather than resting on one
        backend's teardown behaviour staying as it is.

        **Tag identity may diverge from a fresh ``apeSees(fem).run()``**
        (ADR 0019 INV-5).  The bridge's :class:`TagAllocator`
        allocations are lost across H5 round-trip; this helper
        replays exactly the tags stored in the record graph, which
        the bridge picked deterministically — but reordering at the
        bridge level (a future PR) could break that.  Callers who
        need bridge-time tag stability must capture the
        :class:`BuiltModel` from :meth:`apeSees.build` directly and
        not round-trip through H5.

        **Skipped streams warn (D8, #1412).**  When ``fem`` is given (the
        tcl / py / live deck targets), every neutral-zone stream this
        helper has no replay for is named, with its count, in one
        :class:`ReplaySkippedStreamWarning` before anything is emitted.
        ``stage_mp_keys`` is passed by :func:`_replay_staged_into`: the MP
        lines its stage blocks re-emit, which are therefore not skipped.
        The H5 re-emit path passes no ``fem``; its archive keeps those
        streams in the neutral zone that ``_compose_model_h5`` rewrites.

    Parameters mirror :class:`apeGmsh.opensees._internal.typed_records`
    field names; see :mod:`apeGmsh.opensees.opensees_model` for the
    canonical instantiation pattern.
    """
    from .build import node_coords_as_floats

    if fem is not None:
        _warn_skipped_replay_streams(fem, stage_mp_keys=stage_mp_keys)

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
        # ``ndm`` coordinates only — a trailing 0.0 in a 2-D deck
        # swallows the following ``-ndf K`` (node_coords_as_floats).
        # The replay path has to do this itself: it never touches
        # build.py's node-emit helpers.
        cs = node_coords_as_floats(coords)
        if node_ndf is None:
            emitter.node(int(tag), *cs)
        else:
            emitter.node(int(tag), *cs, ndf=int(node_ndf))

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

    from .build import needs_builder_ndf_bracket_for_token

    # 4b. ADR 0099 INV-1 — the builder-ndf-GATED elements, hoisted above
    # the declarations their bracket would destroy.  Everything they need
    # (an NDMaterial) is already emitted at step 3; nothing gated can
    # reference a transform / integration / series / damping (the five
    # non-quad gated classes take ``nodes matTag`` + flag options only,
    # and ``quad``'s ``-damp`` is refused up front — INV-3).  The split is
    # keyed on the ENVELOPE-aware predicate, so a gated element under an
    # envelope that already matches its builder ndf stays in place at
    # step 8 and no deck reorders for nothing.
    # Gatedness is a property of the TOKEN at a fixed (ndm, envelope_ndf),
    # so the predicate is resolved once per DISTINCT token and memoed — a
    # deck carries a handful of them and can carry millions of elements.
    # Per-element evaluation measured +13-15% on the emit of an
    # ungated-only 200k-element deck, i.e. a cost on exactly the models
    # this hoist does nothing for.
    _kept = [rec for rec in elements if int(rec.tag) not in skip_element_tags]
    _gated: "list[Any]" = []
    _ungated = _kept
    # A bracket is only harmful when there is a builder-scoped declaration
    # for it to destroy.  With none present, INV-1 is satisfied vacuously:
    # skip the hoist, and this deck keeps the line order it had — an O(1)
    # test on four already-materialised sequences, before touching the
    # element list at all.
    if deck_ordering and (
        transforms or beam_integrations or time_series or dampings
    ):
        # Gatedness is a property of the TOKEN at a fixed
        # (ndm, envelope_ndf), so resolve it once per DISTINCT token —
        # a deck carries a handful of those and can carry millions of
        # elements.  Per-element evaluation measured +13-15% on the emit
        # of an ungated-only 200k-element deck.
        _gated_tokens = {
            tok for tok in {rec.type_token for rec in _kept}
            if needs_builder_ndf_bracket_for_token(
                tok, ndm=int(ndm), envelope_ndf=int(ndf),
            )
        }
        if _gated_tokens:
            _gated = [r for r in _kept if r.type_token in _gated_tokens]
            _ungated = [
                r for r in _kept if r.type_token not in _gated_tokens
            ]
            # ADR 0099 INV-3 (S4b) — a rehydrated gated element whose arg
            # tail references a builder-scoped declaration cannot be saved
            # by ordering, hoisted or not.  Only a PRE-S1 archive can carry
            # one; refuse rather than emit a deck that aborts at the
            # element line.  Scanned over the ALREADY-PARTITIONED gated
            # list, inside the vacuity branch: with no scoped declaration
            # present a dangling ``-damp`` tag is a corrupt archive that
            # already fails loud at load (measured: OpenSees warns and
            # aborts at the element line), so the guard would buy nothing
            # there while costing a full per-element arg scan — measured
            # +13.8-22.3% on a gated-but-unscoped 200k-element replay.
            from .build import validate_builder_scope_replay

            validate_builder_scope_replay(
                _gated, ndm=int(ndm), envelope_ndf=int(ndf),
                already_gated=True,
            )
    if _gated:
        _replay_elements_bracketed(
            emitter, _gated, ndm=int(ndm), envelope_ndf=int(ndf),
        )

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
    # tag).  Region ``-damp`` attaches replay at 9b with the other
    # top-level regions.
    for rec in dampings:
        emitter.damping(rec.type_token, int(rec.tag), *rec.args)

    # 8. Elements.  The H5 emitter consults two side channels for
    # connectivity (``set_element_nodes``) and FEM element id
    # (``set_current_fem_element_id``); driving them here keeps the
    # H5 round-trip byte-stable AND lets Tcl / Py / Live emitters
    # ignore the calls (their ``set_*`` helpers no-op when the attr
    # is absent).
    _replay_elements_bracketed(
        emitter, _ungated, ndm=int(ndm), envelope_ndf=int(ndf),
    )

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
        from .build import replay_reinforce_ties
        from .tag_allocator import TagAllocator

        # tag-law waiver reinforce-tie-replay (tag_law_ledger.txt): replay
        # mints the tie element tags (the _counters seed below and
        # replay_reinforce_ties) because replay has no archived tie tags.
        _rt_tags = TagAllocator()
        # Seed the element counter past the max replayed element tag so the tie
        # element tags don't collide with the directly-replayed elements.
        max_elem_tag = max((int(e.tag) for e in elements), default=0)
        _rt_tags._counters["element"] = max_elem_tag
        replay_reinforce_ties(
            emitter, fem, _rt_tags,
            name_to_tag=dict(reinforce_name_to_tag or {}),
        )

    # 9. Fix / mass (model-level).
    for rec in fixes:
        emitter.fix(int(rec.tag), *(int(d) for d in rec.dofs))
    for rec in masses:
        emitter.mass(int(rec.tag), *(float(v) for v in rec.values))

    # 9a. Global ``/opensees/commands`` rows (ADR 0114 R3a), in emit
    # order. The bridge emits them at its slot 7 (``_emit_rayleigh`` /
    # ``_emit_modal_damping``, after the masses and before the patterns).
    for cmd in commands:
        if int(cmd.stage) >= 0:
            raise ValueError(
                f"_replay_into: a stage-{cmd.stage} {cmd.method!r} command "
                "row reached the global slot; route it through "
                "_replay_staged_into."
            )
        _replay_command(emitter, cmd, _GLOBAL_COMMAND_METHODS)

    # 9b. Top-level regions: the archived row is the resolved OpenSees
    # call (tag + flag tail), so every one replays verbatim, with
    # or without its K1-6 declaration — region-scoped ``-rayleigh`` and
    # ``-damp`` attaches, recorder fan-out and named regions alike. After
    # the global ``rayleigh`` rows (OpenSees overwrites element Rayleigh
    # per element, so the region must come second to win, as the bridge
    # emits it) and before the recorders that name a fan-out region.
    for rec in regions:
        emitter.region(int(rec.tag), *rec.args)

    # 9c. Global initial stress (ADR 0055 Phase 1).  Emitted BEFORE
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
            replay_initial_stress_global,
        )
        from .tag_allocator import TagAllocator

        # ADR 0055 P2.3: the staged caller threads its SHARED allocator
        # so global + per-stage parameter tags accumulate on one counter
        # (the bridge reuses one ``tags`` across everything).  Flat
        # callers pass None → fresh allocator (unchanged behaviour).
        # tag-law waiver initial-stress-replay (tag_law_ledger.txt): replay
        # mints the parameter tags (replay_initial_stress_global) because
        # the archive stores the declarative record, not the allocated tags.
        _is_tags = initial_stress_tags or TagAllocator()
        # ADR 0065 v2 B3: the emit helpers now take a FemToOpsTagMap.
        fem_eid_to_ops_tag = FemToOpsTagMap.from_pairs(
            (int(e.fem_eid), int(e.tag))
            for e in elements
            if int(e.fem_eid) >= 0
        )
        name_to_param_tags = replay_initial_stress_global(
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
    """Recover the original arg types in a chain ``*_args`` tuple.

    ADR 0055 P2.3: an analysis-chain arg tuple with mixed numeric types
    (``NormDispIncr(tol=1e-4, max_iter=50, p_flag=0, n_type=2)``) is
    stored by ``_set_attr`` as a single ``float64`` array and read back
    all-float (``(1e-4, 50.0, 0.0, 2.0)``).  ``OPS_GetIntInput`` rejects
    a float where it wants an int, and the Tcl deck bytes drift
    (``50.0`` vs ``50``).  Recover any float that is exactly integral to
    ``int`` — a genuine float tol like ``1e-4`` is untouched.

    A tuple that mixes flags with values (``("-matrixType", 2)`` from
    ``system Pardiso``) is stored as string tokens instead, and h5py
    hands those back as ``bytes``.  Decode them and re-read any token
    that parses as a number; a flag like ``-matrixType`` parses as
    neither and survives as a string.  Applies to BOTH the flat and
    staged replay so neither drifts.
    """
    out: list[Any] = []
    for a in args:
        if isinstance(a, bytes):
            a = a.decode("utf-8", errors="replace")
        if isinstance(a, str):
            out.append(_from_token(a))
        elif isinstance(a, float) and a.is_integer():
            out.append(int(a))
        else:
            out.append(a)
    return tuple(out)


def _from_token(token: str) -> "int | float | str":
    """Re-read a stored token as ``int`` / ``float``, else keep the string.

    ``int`` first: ``"2"`` must come back as an ``int`` (the fork parses
    ``-matrixType`` with ``OPS_GetIntInput``), while ``"2.0"`` fails that
    parse and stays a ``float``.
    """
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


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


#: The ``/opensees/commands`` methods each replay slot carries. A row
#: whose method has no slot raises rather than land at a guessed line.
_GLOBAL_COMMAND_METHODS: "frozenset[str]" = frozenset(
    {"rayleigh", "eigen", "modal_damping"})
_STAGE_COMMAND_METHODS: "frozenset[str]" = frozenset({"profiler"})


def _replay_command(emitter: Any, cmd: Any, methods: "frozenset[str]") -> None:
    """Re-emit one ``/opensees/commands`` row (a reader ``CommandRecordRO``)."""
    if cmd.method not in methods:
        raise NotImplementedError(
            f"replay: a {cmd.method!r} command row (stage {cmd.stage}) has "
            f"no replay slot here; this slot carries {sorted(methods)}."
        )
    getattr(emitter, cmd.method)(*cmd.positional, **cmd.keywords)


def _replay_staged_into(
    emitter: Any,
    *,
    stages: "Sequence[Any]",
    commands: "Sequence[Any]" = (),
    program: "Sequence[Any]" = (),
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

    ``commands`` are every ``/opensees/commands`` row: the global ones
    go to the prefix, and a stage's ``profiler`` rows bracket its
    ``analyze``. ``program`` (the ``/opensees/program`` runs) says which
    side of the analyze each row was emitted on.
    """
    from .build import (
        ActivateAbsorbingRecord,
        emit_initial_stress_addtoparameter,
        replay_activate_absorbing,
        replay_initial_stress_global,
    )
    from .tag_allocator import TagAllocator

    # 0. Stage guard — fail clean before any emit (the live emitter's
    # stage_open raises; a deep mid-replay crash would be opaque).
    if not emitter.caps.supports_stages:
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

    # ADR 0099 S7 — a STAGE-OWNED gated element brackets inside its stage
    # block, after the global builder-scoped declarations and (past the
    # first stage) after a completed ``analyze``.  Unlike the global
    # prefix, which the S4a hoist fixes, there is no earlier position to
    # move it to; the fix is a REPLAY of the four declaration kinds at
    # bracket close (per-stage loop below) — Tcl only: measured, the
    # in-process module (live AND the emitted py deck's runtime) purges
    # nothing on a ``model`` re-issue, and re-declaring a still-alive tag
    # hard-errors, so a replay anywhere else collides.  Gatedness is a
    # property of the TOKEN at a fixed (ndm, envelope_ndf); resolve it
    # once per distinct stage-owned token, and only when a builder-scoped
    # declaration exists for a bracket to destroy (otherwise INV-1 holds
    # vacuously and the stage keeps its record order to the byte).
    _scoped_present = bool(
        replay_kwargs.get("transforms")
        or replay_kwargs.get("beam_integrations")
        or replay_kwargs.get("time_series")
        or replay_kwargs.get("dampings")
    )
    _gated_stage_tokens: "set[str]" = set()
    if owned_element_tags and _scoped_present:
        from .build import needs_builder_ndf_bracket_for_token

        # ``ndm`` / ``ndf`` are required keyword args of ``_replay_into``
        # and always present here — indexed, not ``.get``-defaulted,
        # because a wrong envelope would silently mis-evaluate the gate.
        _owned_recs = [
            r for r in elements if int(r.tag) in owned_element_tags
        ]
        _gated_stage_tokens = {
            tok for tok in {r.type_token for r in _owned_recs}
            if needs_builder_ndf_bracket_for_token(
                tok,
                ndm=int(replay_kwargs["ndm"]),
                envelope_ndf=int(replay_kwargs["ndf"]),
            )
        }
        if _gated_stage_tokens:
            # INV-3 still applies and no replay saves it: the purge lands
            # at bracket OPEN, before the element line parses, so a
            # ``quad ... -damp N`` record dies at its own line (measured).
            # The global ``_replay_into`` scan skips stage-owned records,
            # so scan them here.  Only a pre-S1 archive can carry one.
            from .build import validate_builder_scope_replay

            validate_builder_scope_replay(
                [r for r in _owned_recs
                 if r.type_token in _gated_stage_tokens],
                ndm=int(replay_kwargs["ndm"]),
                envelope_ndf=int(replay_kwargs["ndf"]),
                already_gated=True,
            )

    # ONE allocator threaded across the global prefix AND every stage
    # (the bridge reuses a single ``tags``; a per-stage allocator would
    # restart parameter counters at stage boundaries — gate-1 FATAL).
    # tag-law waiver staged-replay-params (tag_law_ledger.txt): replay mints
    # the stage parameter tags (replay_initial_stress_global and
    # replay_activate_absorbing) because the archive stores no parameter tag.
    tags = TagAllocator()

    # 1. Global prefix — _replay_into with stage-owned topology filtered
    # out and the shared allocator threaded for any GLOBAL initial_stress.
    # The stage blocks below re-emit their claimed MP constraints, so the
    # skipped-stream warning must not count those (D8, #1412).
    _replay_into(
        emitter,
        skip_node_tags=owned_node_tags,
        skip_element_tags=owned_element_tags,
        initial_stress_tags=tags,
        stage_mp_keys=_stage_mp_keys(stages),
        commands=tuple(c for c in commands if int(c.stage) < 0),
        **replay_kwargs,
    )
    stage_commands: "dict[int, list[Any]]" = {}
    for cmd in commands:
        if int(cmd.stage) >= 0:
            stage_commands.setdefault(int(cmd.stage), []).append(cmd)
    if any(k >= len(stages) for k in stage_commands):
        raise ValueError(
            f"replay: command rows name stages {sorted(stage_commands)} "
            f"but the archive has {len(stages)} stage(s)."
        )
    from ..emitter.h5_reader import emit_index_of

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
    _ndm = int(replay_kwargs["ndm"])
    _ndf = int(replay_kwargs["ndf"])
    # ADR 0065 v2 B3: the stage emit helpers (initial_stress /
    # activate_absorbing) now take a FemToOpsTagMap.
    from .build import FemToOpsTagMap, node_coords_as_floats

    fem_eid_to_ops_tag = FemToOpsTagMap.from_pairs(
        (int(r.fem_eid), int(r.tag)) for r in elements if int(r.fem_eid) >= 0
    )

    # 2. Per-stage blocks — exact _emit_stages_flat order.
    for k, st in enumerate(stages):
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
            cs = node_coords_as_floats(coords)
            if nndf is None:
                emitter.node(int(nid), *cs)
            else:
                emitter.node(int(nid), *cs, ndf=int(nndf))
        # owned elements (look up the rehydrated record by ops tag).
        owned_recs = [
            rec for etag in st.owned_element_ids
            if (rec := elem_map.get(int(etag))) is not None
        ]
        gated_recs = [
            r for r in owned_recs if r.type_token in _gated_stage_tokens
        ]
        if gated_recs:
            # ADR 0099 S7: hoist first, replay only what's left — the
            # gated records bracket at the top of the stage block, then
            # the purged builder-scoped declarations are re-declared (on
            # a purging runtime only), then the ungated records, whose
            # element lines resolve transforms / integrations / dampings
            # by tag.  Tags are the archive's; only line positions move.
            _replay_elements_bracketed(
                emitter, gated_recs, ndm=_ndm, envelope_ndf=_ndf,
            )
            if emitter.caps.model_reissue_purges:
                # Same kind order as the global prefix (steps 5-7b of
                # ``_replay_into``): transforms, beam integrations, time
                # series, dampings.
                for rec in replay_kwargs.get("transforms", ()):
                    emitter.geomTransf(rec.type_token, int(rec.tag), *rec.vec)
                for rec in replay_kwargs.get("beam_integrations", ()):
                    emitter.beamIntegration(
                        rec.type_token, int(rec.tag), *rec.args)
                for rec in replay_kwargs.get("time_series", ()):
                    emitter.timeSeries(rec.type_token, int(rec.tag), *rec.args)
                for rec in replay_kwargs.get("dampings", ()):
                    emitter.damping(rec.type_token, int(rec.tag), *rec.args)
            _replay_elements_bracketed(
                emitter,
                [r for r in owned_recs
                 if r.type_token not in _gated_stage_tokens],
                ndm=_ndm, envelope_ndf=_ndf,
            )
        else:
            _replay_elements_bracketed(
                emitter, owned_recs, ndm=_ndm, envelope_ndf=_ndf,
            )

        # SSI-2.E removals (before new BCs).
        for n_tag, dof in st.remove_sps:
            emitter.remove_sp(int(n_tag), int(dof))
        for e_tag in st.remove_elements:
            emitter.remove_element(int(e_tag))
        # SSI-2.E SANISAND stage flips (after the removals, after the
        # stage's own element replay above — updateMaterialStage
        # resolves through the Domain's live elements).
        for m_tag, m_stage in st.update_material_stages:
            emitter.update_material_stage(int(m_tag), int(m_stage))

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
            name_to_param_tags = replay_initial_stress_global(
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
            replay_activate_absorbing(
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

        # TIMs A8: the stage's ``profiler`` rows (ADR 0114 R3a) bracket
        # its analyze; the program's emit order says which side each is.
        cmds = stage_commands.get(k, [])
        at = emit_index_of(program, "analyze", 0, stage=k) if cmds else 0
        for cmd in cmds:
            if cmd.emit_index < at:
                _replay_command(emitter, cmd, _STAGE_COMMAND_METHODS)

        # label= mirrors the bridge's _emit_stages_flat call so the
        # fail-loud analyze banner names the stage (deck equality).
        if st.analyze_dt is None:
            emitter.analyze(steps=int(st.analyze_steps), label=st.name)
        else:
            emitter.analyze(
                steps=int(st.analyze_steps), dt=float(st.analyze_dt),
                label=st.name,
            )
        for cmd in cmds:
            if cmd.emit_index > at:
                _replay_command(emitter, cmd, _STAGE_COMMAND_METHODS)
        emitter.stage_close()
