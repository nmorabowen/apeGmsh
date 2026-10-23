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
    ``g.reinforce`` ties, and the global ``/opensees/constraints`` MP rows
    (#2405). It has no deck record for the equation ties, ``g.embed``
    ties and contacts, nor for an MP constraint whose rows the archive
    lacks. A replayed deck that leaves any of them out differs from the
    forward ``apeSees`` deck, so the replay warns and names each one. A
    stream it cannot write at all refuses instead
    (:func:`refuse_unreplayable_streams`).
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
    from .sequence_h5 import write_sequence_into

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
        # ADR 0112 D2b: /sequence, derived by reading the finished
        # /opensees back (after lineage, for its @model_hash; no hash
        # reads it).
        write_sequence_into(f)


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


def _stage_mp_keys(stages: "Sequence[Any]") -> "Counter[tuple[Any, ...]]":
    """The keys of every MP line the stage blocks of ``stages`` re-emit,
    counted: a multiset, so N equal records need N lines.

    A key is ``(bucket, *what it ties)`` over the five ``StageRecordRO``
    MP buckets (the ``equationConstraint`` lines from opensees 2.37.0);
    :func:`_constraint_stage_keys` / :func:`_interpolation_stage_keys`
    build the matching side.
    Keys carry the bucket and the tied nodes, not the declaration name.
    The archived name is no identity: a ``tied_contact`` slave row has
    ``name=None`` (only its parent is named), and a ``rigid_body`` group
    writes its name on its first ``rigidLink`` only.
    """
    keys: "Counter[tuple[Any, ...]]" = Counter()
    for s in stages:
        keys += _mp_row_keys((*s.equal_dofs, *s.rigid_links,
                              *s.rigid_diaphragms, *s.embedded_nodes,
                              *s.equation_constraints))
    return keys


def _mp_row_keys(rows: "Sequence[Any]") -> "Counter[tuple[Any, ...]]":
    """The keys of archived MP rows (``EqualDOFRecord`` /
    ``RigidLinkRecord`` / ``RigidDiaphragmRecord`` /
    ``EmbeddedNodeRecord`` / ``EquationConstraintRecordRO``), counted, shaped as
    :func:`_constraint_stage_keys` / :func:`_interpolation_stage_keys`
    build them for the FEMData side. An unknown row type raises."""
    from ..emitter.h5_reader import EquationConstraintRecordRO
    from .typed_records import (
        EmbeddedNodeRecord, EqualDOFRecord, RigidDiaphragmRecord,
        RigidLinkRecord,
    )

    keys: "Counter[tuple[Any, ...]]" = Counter()
    for r in rows:
        if isinstance(r, EquationConstraintRecordRO):
            keys[_equation_line_key(r.cnode, r.cdof, r.ccoef, r.retained)] += 1
        elif isinstance(r, EqualDOFRecord):
            keys[("equal_dof", int(r.master), int(r.slave),
                  tuple(int(d) for d in r.dofs))] += 1
        elif isinstance(r, RigidLinkRecord):
            keys[("rigid_link", str(r.kind), int(r.master),
                  int(r.slave))] += 1
        elif isinstance(r, RigidDiaphragmRecord):
            keys[("rigid_diaphragm", int(r.master),
                  tuple(sorted(int(n) for n in r.slaves)))] += 1
        elif isinstance(r, EmbeddedNodeRecord):
            keys[("embedded_node", int(r.cnode),
                  tuple(int(n) for n in r.args))] += 1
        else:
            raise TypeError(
                f"replay: {type(r).__name__} is no MP-constraint row.")
    return keys


def _emit_mp_row(emitter: Any, r: Any) -> None:
    """Re-emit one archived MP row, its ``name`` label first, as the
    bridge wrote it. An unknown row type raises."""
    from ..emitter.h5_reader import EquationConstraintRecordRO
    from .typed_records import (
        EmbeddedNodeRecord, EqualDOFRecord, RigidDiaphragmRecord,
        RigidLinkRecord,
    )

    if not isinstance(r, (EqualDOFRecord, RigidLinkRecord,
                          RigidDiaphragmRecord, EmbeddedNodeRecord,
                          EquationConstraintRecordRO)):
        raise TypeError(f"replay: {type(r).__name__} is no MP-constraint row.")
    if r.name:
        emitter.mp_constraint_comment(r.name)
    if isinstance(r, EquationConstraintRecordRO):
        emitter.equationConstraint(
            int(r.cnode), int(r.cdof), float(r.ccoef), list(r.retained))
    elif isinstance(r, EqualDOFRecord):
        emitter.equalDOF(int(r.master), int(r.slave),
                         *(int(d) for d in r.dofs))
    elif isinstance(r, RigidLinkRecord):
        emitter.rigidLink(r.kind, int(r.master), int(r.slave))
    elif isinstance(r, RigidDiaphragmRecord):
        emitter.rigidDiaphragm(int(r.perp_dir), int(r.master),
                               *(int(n) for n in r.slaves))
    else:
        emitter.embeddedNode(
            int(r.ele_tag), int(r.cnode), *(int(a) for a in r.args),
            stiffness=r.stiffness, stiffness_p=r.stiffness_p,
            rotational=r.rotational, pressure=r.pressure,
        )


def _refuse_replayed_partly_fixed_diaphragms(
    *, ndm: int, fixes: "Sequence[Any]", diaphragms: "Sequence[Any]",
    handlers: "Sequence[Any]", staged_fixes: "Sequence[Any]" = (),
    staged_removals: "Sequence[tuple[int, int]]" = (),
    pattern_sps: "Sequence[tuple[int, int]]" = (),
) -> None:
    """#2427 on the replay path (#2435): an archive written before the
    emit-time refusal can hold a ``rigidDiaphragm`` whose master is fixed
    on some in-plane DOFs only, under Transformation. Replaying it would
    re-emit that silently wrong deck, so the same check runs on the
    archived rows (fix masks, stage fixes and removals, pattern SPs, the
    recorded ``constraints`` handlers), as ``fixed_dof_map`` reads them
    at emit, so an archive the emit accepted always replays.
    """
    from types import SimpleNamespace

    from .build.constraints import FixedDofs, _refuse_partly_fixed_diaphragm

    rows = list(diaphragms)
    if not rows or "Transformation" not in {str(h) for h in handlers if h}:
        return
    fixed: "dict[int, set[int]]" = {}
    for rec in fixes:
        fixed.setdefault(int(rec.tag), set()).update(
            i + 1 for i, m in enumerate(rec.dofs) if int(m))
    staged = {(int(r.tag), i + 1) for r in staged_fixes
              for i, m in enumerate(r.dofs) if int(m)}
    staged |= {(int(n), int(d)) for n, d in staged_removals}
    staged |= {(int(n), int(d)) for n, d in pattern_sps}
    fd = FixedDofs(fixed={n: frozenset(d) for n, d in fixed.items()},
                   staged=frozenset(staged), ndf={}, default_ndf=0,
                   owner={}, ndm=int(ndm), transformation=True)
    for r in rows:
        _refuse_partly_fixed_diaphragm(
            None, SimpleNamespace(master_node=int(r.master), name=r.name),
            int(r.perp_dir), fd)


def refuse_archived_partly_fixed_diaphragms(
    *, ndm: int, fixes: "Sequence[Any]", mp_constraints: "Sequence[Any]",
    analysis_attrs: "dict[str, Any] | None", patterns: "Sequence[Any]",
    stages: "Sequence[Any]" = (),
) -> None:
    """:func:`_refuse_replayed_partly_fixed_diaphragms` over one
    rehydrated archive, global rows and every stage's. Called by
    ``OpenSeesModel.build`` before any emitter exists (the live one
    wipes the domain on creation)."""
    from .typed_records import RigidDiaphragmRecord

    def sps(pats: "Sequence[Any]") -> "list[tuple[int, int]]":
        return [(int(sp.target), int(sp.dof)) for pat in pats
                for sp in (getattr(pat, "sps", None) or ())] + [
            (int(n), int(d)) for pat in pats
            for n, d in (getattr(pat, "sp_holds", None) or ())]

    _refuse_replayed_partly_fixed_diaphragms(
        ndm=int(ndm), fixes=fixes,
        diaphragms=[
            *(r for _i, r in mp_constraints
              if isinstance(r, RigidDiaphragmRecord)),
            *(r for st in stages for r in st.rigid_diaphragms)],
        handlers=[(analysis_attrs or {}).get("handler"),
                  *(dict(st.chain_attrs or {}).get("handler")
                    for st in stages)],
        staged_fixes=[r for st in stages for r in st.fixes],
        staged_removals=[(n, d) for st in stages for n, d in st.remove_sps],
        pattern_sps=[*sps(patterns),
                     *(x for st in stages for x in sps(st.patterns))])


def refuse_archived_stage_routes(
    *, stages: "Sequence[Any]", elements: "Sequence[Any]",
    declarations: Any, fem: Any, requires: "Sequence[str]",
) -> None:
    """The stage-close route checks on the replay path (#2465).

    An archive written before #2442, #2443, #2445, #2458 or #2460 landed can
    hold a stage flip that the stage close now refuses: an
    ``updateMaterialStage`` through ``updateParameter``, a ``materialState``
    through a host that drops it, a tag-blind neighbour, a target that
    carries no material, a multi-element plastic flip, or part of a
    class-wide group.  Replaying it would emit that silent wrong answer.
    So the archived stages run the stage close's own checks
    (``stage._builder.check_archived_stage_routes``) on a
    :class:`~apeGmsh.opensees.stage._builder.RouteModel` read from the
    archive's declarations.  The model is the final one, while the stage
    close sees only the elements registered so far, so an archive whose
    elements were registered after a flip may be refused here although the
    bridge accepted it (#2464).  ``requires`` are the archive's replay requirements: a deck
    that needs the fork is known to run it (#2443).
    """
    from ..stage._builder import (
        STAGE_FLAG_PARAMETERS,
        ArchivedStageRoutes,
        check_archived_stage_routes,
    )
    from .build.records import MaterialStageRecord

    if not any(
        st.update_material_stages
        or any(p.name in STAGE_FLAG_PARAMETERS for p in st.update_parameters)
        for st in stages
    ):
        return
    model = _archived_route_model(
        stages=stages, elements=elements, declarations=declarations,
        fem=fem, fork="fork" in requires)
    fem_eid_of = {int(r.tag): int(r.fem_eid) for r in elements}
    check_archived_stage_routes(model, tuple(
        ArchivedStageRoutes(
            name=str(st.name),
            mat_stage_records=tuple(
                MaterialStageRecord(mat_tags=(int(tag),), stage=int(stage))
                for tag, stage in st.update_material_stages),
            parameter_records=tuple(st.update_parameters),
            activated_pgs=tuple(str(pg) for pg in st.activated_pgs),
            removed_eids=frozenset(
                fem_eid_of[int(t)] for t in st.remove_elements
                if fem_eid_of.get(int(t), -1) >= 0),
        )
        for st in stages))


def _archived_route_model(
    *, stages: "Sequence[Any]", elements: "Sequence[Any]",
    declarations: Any, fem: Any, fork: bool,
) -> Any:
    """The :class:`~apeGmsh.opensees.stage._builder.RouteModel` of an
    archive, read from ``/opensees/decls`` and ``/opensees/decl_params``.

    Each element declaration is one spec.  Its class, ``pg`` and the
    declarations it references come from its ``decl_params`` row.  Its
    elements are the archived element rows of its tags, and its owner is
    the stage whose ``owned_element_ids`` hold them.  That is the graph the
    stage close walks through ``Primitive.dependencies()``.  An archive
    without the rows cannot be checked, so it is refused: it may hold a
    flip the checks refuse.
    """
    import functools
    from collections.abc import Mapping

    from ..emitter.h5_reader import DeclRef, DeclStruct
    from ..stage._builder import (
        FIRST_ELEMENT_ONLY_PLASTIC_INIT,
        TAG_BLIND_STAGE_PARAMETER,
        RouteModel,
        RouteSpec,
    )
    from .build.errors import BridgeError
    from .build.pg_expand import expand_pg_to_elements

    def refuse(why: str) -> BridgeError:
        return BridgeError(
            "OpenSeesModel replay: this archive flips a stage flag "
            f"(updateMaterialStage or materialState), but {why}, so the "
            "replay cannot tell which elements carry each flipped "
            "material.  The stage-close checks of issues #2442, #2443, "
            "#2445, #2458 and #2460 cannot run, and an archive written "
            "before them can hold a flip they refuse, a silent wrong "
            "answer (issue #2465).  Rebuild the model with the current "
            "bridge and archive it again."
        )

    if declarations is None or not declarations.params:
        raise refuse(
            "it records no /opensees/decl_params (a file written below "
            "opensees 2.26.0, or one no bridge emit wrote)")
    decls = declarations.decls
    index = {d.key: i for i, d in enumerate(decls)}
    nd_tag: dict[int, int] = {}
    ele_tags: dict[int, list[int]] = {}
    for kind, first, count, decl in declarations.tags:
        if kind == "nDMaterial":
            if int(count) != 1:
                raise refuse(
                    f"nDMaterial {decls[decl].key!r} holds {count} tags")
            nd_tag[int(decl)] = int(first)
        elif kind == "element":
            ele_tags.setdefault(int(decl), []).extend(
                range(int(first), int(first) + int(count)))

    def entry(i: int) -> Any:
        rows = declarations.params.get(i, ())
        if len(rows) != 1:
            if decls[i].family in ("element", "nDMaterial"):
                raise refuse(
                    f"{decls[i].key!r} has {len(rows)} /opensees/decl_params "
                    "rows, not one")
            return None
        return rows[0]

    def type_of(i: int) -> str:
        e = entry(i)
        return f"<unrecorded {decls[i].family}>" if e is None else str(e.type)

    def refs(value: Any, out: "list[str]") -> None:
        if isinstance(value, DeclRef):
            out.append(value.key)
        elif isinstance(value, DeclStruct):
            for v in value.fields.values():
                refs(v, out)
        elif isinstance(value, (tuple, list)):
            for v in value:
                refs(v, out)
        elif isinstance(value, Mapping):
            for v in value.values():
                refs(v, out)

    def children(i: int) -> "list[int]":
        e = entry(i)
        if e is None:
            return []
        out: "list[str]" = []
        for v in e.params.values():
            refs(v, out)
        out.extend((*e.transf_ref, *e.integration_ref, *e.section_ref))
        return [index[k] for k in dict.fromkeys(out)]

    nd_class = {
        nd_tag[i]: type_of(i) for i, d in enumerate(decls)
        if d.family == "nDMaterial" and i in nd_tag
    }
    record_of = {int(r.tag): r for r in elements}
    owner_of = {
        int(t): k for k, st in enumerate(stages) for t in st.owned_element_ids
    }
    specs: "list[Any]" = []
    for i, d in enumerate(decls):
        if d.family != "element":
            continue
        cls = type_of(i)
        tags = ele_tags.get(i, [])
        lost = [t for t in tags if t not in record_of]
        if lost:
            raise refuse(
                f"element {d.key!r} ({cls}) declares tags {lost} that no "
                "archived element row holds")
        nd_tags: set[int] = set()
        dep_classes: set[str] = set()
        blind: list[tuple[str, int | None]] = []
        first_only: dict[int, set[str]] = {}
        walked: set[int] = set()
        todo: "list[tuple[int, tuple[str, ...]]]" = [
            (c, ()) for c in children(i)]
        while todo:
            j, via = todo.pop()
            if j in walked:
                continue
            walked.add(j)
            dep = type_of(j)
            todo.extend((c, (*via, dep)) for c in children(j))
            tag = nd_tag.get(j) if decls[j].family == "nDMaterial" else None
            dep_classes.add(dep)
            if tag is not None:
                nd_tags.add(tag)
            if dep in TAG_BLIND_STAGE_PARAMETER:
                blind.append((dep, tag))
            if dep in FIRST_ELEMENT_ONLY_PLASTIC_INIT and tag is not None:
                first_only.setdefault(tag, set()).update(via)
        pg = entry(i).params.get("pg")
        eids = tuple(int(record_of[t].fem_eid) for t in tags)
        specs.append(RouteSpec(
            cls=cls, pg=None if pg is None else str(pg),
            nd_tags=frozenset(nd_tags), dep_classes=frozenset(dep_classes),
            blind=tuple(blind),
            first_only={t: frozenset(v) for t, v in first_only.items()},
            fanout=functools.partial(tuple, eids),
            owner=min((owner_of[t] for t in tags if t in owner_of),
                      default=-1),
        ))
    return RouteModel(
        specs=tuple(specs), nd_class=nd_class,
        expand_pg=lambda pg: expand_pg_to_elements(fem, pg).eids,
        fork=fork,
    )


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
    """Stage-bucket keys of one non-element interpolation row: its
    ``embeddedNode`` line, or on the equation route its
    ``equationConstraint`` lines (:func:`_equation_tie_keys`).

    The key carries the retained nodes the forward emit writes
    (``_embedded_retained_nodes``): a quad tie goes out on the triangle of
    its face that holds the point, three of the record's four corners."""
    from .build.constraints import _embedded_retained_nodes

    if rec.enforce == "equation":
        return _equation_tie_keys(rec)
    return [("embedded_node", int(rec.slave_node),
             tuple(int(n) for n in _embedded_retained_nodes(rec)))]


def _equation_line_key(
    cnode: Any, cdof: Any, ccoef: Any, retained: Any,
) -> "tuple[Any, ...]":
    """The key of one ``equationConstraint`` line, from either side."""
    return ("equation", int(cnode), int(cdof), float(ccoef),
            tuple((int(n), int(d), float(c)) for n, d, c in retained))


def _equation_tie_keys(rec: Any) -> "list[tuple[Any, ...]]":
    """The keys of the ``equationConstraint`` lines an ``enforce="equation"``
    interpolation writes, in order: the bridge's own expansion
    (``_emit_equation_tie``) run on a recording emitter, so the zero-weight
    filter and the DOF order are the bridge's."""
    from ..emitter.recording import RecordingEmitter
    from .build.constraints import _emit_equation_tie

    rec_em = RecordingEmitter()
    _emit_equation_tie(rec_em, rec)
    return [_equation_line_key(*args) for method, args, _kw in rec_em.calls
            if method == "equationConstraint"]


def _skipped_replay_streams(
    fem: Any, *, stage_mp_keys: "Counter[tuple[Any, ...]] | frozenset[tuple[Any, ...]]",
) -> "dict[str, Counter[str]]":
    """Return ``{stream: Counter(kind -> count)}`` for every non-empty
    neutral-zone stream that :func:`_replay_into` does not re-emit.

    ``stage_mp_keys`` (:func:`_stage_mp_keys`) holds the MP lines that a
    staged archive's stage blocks re-emit.  A record counts as replayed
    there only when every line its fan-out writes is among them, matched
    on bucket and nodes, so a stage that claims the equalDOF named ``x``
    does not also excuse a rigidDiaphragm named ``x``. The keys are a
    multiset: a matched record consumes its lines, so two equal records
    need two lines, and one line never excuses a second record.

    The streams are read without defaults: a ``fem`` that lacks one fails
    here rather than passing for empty.
    """
    out: "dict[str, Counter[str]]" = {}

    available: "Counter[tuple[Any, ...]]" = Counter(stage_mp_keys)

    def _staged(keys: "list[tuple[Any, ...]]") -> bool:
        need = Counter(keys)
        if not need or any(available[k] < n for k, n in need.items()):
            return False
        available.subtract(need)
        return True

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
    # With no row left to match, a record is skipped without expanding its
    # lines (an equation tie's expansion refuses a record with no weights).
    surface_skipped: "Counter[str]" = Counter(
        str(rec.kind) for rec in fem.elements.constraints.interpolations()
        if not _interpolation_replays_as_element(rec, kind_cls)
        and not (+available and _staged(_interpolation_stage_keys(rec)))
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


def refuse_unreplayable_streams(
    fem: Any, *, equation_rows: "Sequence[Any]" = (),
) -> None:
    """Raise :class:`BridgeError` iff ``fem`` carries a stream the deck
    replay cannot write, so it would run untied. Each kind is named, with
    its count and the issue that tracks it:

    - a ``node_to_surface`` / ``node_to_surface_spring`` coupling or a
      mixed-ndf interface, whose lines tie through phantom nodes the
      archive does not store (#2411);
    - an interpolation on the ``enforce="equation"`` route whose
      ``equationConstraint`` lines are not all among ``equation_rows``,
      the archive's equation-tie rows: the global ones (opensees 2.35.0,
      the ``apeSees.equation_constraint`` rows taken off) and the stage
      blocks' (2.37.0). An archive below 2.35.0 stores no line (#2416),
      and one below 2.37.0 no stage-claimed tie's line (#2448). A matched
      record consumes its lines, so N equal ties need N sets of rows.

    :meth:`OpenSeesModel.build` calls it for the ``tcl`` / ``py`` /
    ``live`` targets before any emitter is made, so a refused live build
    leaves the domain as it was.
    """
    from .build.errors import BridgeError

    kind_cls = fem.nodes.constraints.Kind
    phantom_kinds = (kind_cls.NODE_TO_SURFACE, kind_cls.NODE_TO_SURFACE_SPRING)
    phantom: "Counter[str]" = Counter(
        str(rec.kind) for rec in fem.nodes.constraints
        if rec.kind in phantom_kinds)
    phantom.update(
        f"{rec.kind} (mixed-ndf)" for rec in fem.elements.interfaces
        if rec.phantom_node is not None)
    available: "Counter[tuple[Any, ...]]" = _mp_row_keys(equation_rows)

    def _archived(rec: Any) -> bool:
        if not +available:
            return False
        need = Counter(_equation_tie_keys(rec))
        if any(available[k] < n for k, n in need.items()):
            return False
        available.subtract(need)
        return True

    equation: "Counter[str]" = Counter(
        f"{rec.kind} (enforce='equation')"
        for rec in fem.elements.constraints.interpolations()
        if rec.enforce == "equation" and not _archived(rec))
    parts = []
    for found, why in (
        (phantom, "tie through phantom nodes the archive does not store "
                  "(#2411)"),
        (equation, "are equationConstraint lines the archive does not "
                   "store (an archive below opensees 2.35.0 stores none, "
                   "#2416; one below opensees 2.37.0 stores none of a "
                   "stage-claimed tie's, #2448)"),
    ):
        if found:
            detail = ", ".join(f"{k}: {n}" for k, n in sorted(found.items()))
            parts.append(f"{detail}, whose deck lines {why}")
    if parts:
        raise BridgeError(
            "Deck replay refuses this archive: it carries "
            + "; and ".join(parts) + ". The replay would run untied. Load "
            "FEMData.from_h5(path) and emit it through apeSees(fem).")


def _warn_skipped_replay_streams(
    fem: Any, *, stage_mp_keys: "Counter[tuple[Any, ...]] | frozenset[tuple[Any, ...]]",
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


def _archives_param_tags(opensees_version: Any) -> bool:
    """The replay's version check (K1-8, #1465): True iff an archive at
    ``opensees_version`` stores its parameter tags (schema 2.27.0,
    :data:`~apeGmsh.opensees.emitter.h5_reader.PARAM_TAGS_FROM`), so the
    replay reads them and mints none; False for an older archive, which
    the pre-2.27 tag-law waivers re-mint. A caller that replays a
    parameter-bearing record without saying which archive it came from
    refuses, rather than guess either way."""
    from ..emitter.h5_reader import archives_param_tags

    if opensees_version is None:
        raise ValueError(
            "replay: an initial stress, absorbing flip or update_parameter "
            "record needs the archive's opensees schema version "
            "(opensees_version=) — at 2.27.0 or above the replay reads its "
            "stored parameter tags, below it the replay re-mints them."
        )
    return archives_param_tags(opensees_version)


def _replays_tie_fallback(opensees_version: Any, n_ties: int) -> bool:
    """The replay's version check (K2b): True iff an archive at
    ``opensees_version`` predates the stored reinforce ties (schema 2.29.0,
    :data:`~apeGmsh.opensees.emitter.h5_reader.EMBEDDED_REBAR_FROM`), so
    the replay re-derives its ``n_ties`` ties from the neutral zone. A
    caller that replays ties without saying which archive they came from
    refuses, rather than guess either way."""
    from ..emitter.h5_reader import archives_embedded_rebar

    if opensees_version is None:
        raise ValueError(
            f"replay: {n_ties} reinforce tie(s) need the archive's opensees "
            "schema version (opensees_version=) — at 2.29.0 or above the "
            "replay reads the stored ties, below it the replay re-derives "
            "them from the neutral zone."
        )
    return not archives_embedded_rebar(opensees_version)


def _rebar_cell_tags(elements: "Sequence[Any]", fem: Any) -> "set[int]":
    """The element tags of the archived ``g.rebar.place(emit_elements=True)``
    cells (#1828), which the replay writes in the bridge's slot 7b'', after
    the reinforce ties, not with the other elements at step 8.

    A cell is a bridge-minted ``CorotTruss`` row (sentinel fem id; only
    ``emit_rebar_elements`` mints one) whose node pair is a line cell of a
    bar in the neutral zone, matched one for one. Refuses both ways: a
    neutral cell with no row, or a minted row the neutral zone does not name.
    """
    from collections import Counter

    from .build.constraints_plan import rebar_element_records
    from .tag_resolution import MISSING_FEM_ELEMENT_ID

    # One pair per line cell of each bar, as ``rebar_cell_entries`` plans them.
    entries = [
        (int(i), int(j))
        for rec in rebar_element_records(fem)
        for i, j in rec.connectivity
    ]
    want: "Counter[tuple[int, int]]" = Counter(entries)
    tags: "set[int]" = set()
    minted = 0
    for rec in elements:
        if (rec.type_token != "CorotTruss"
                or int(rec.fem_eid) != MISSING_FEM_ELEMENT_ID):
            continue
        minted += 1
        conn = tuple(rec.connectivity)
        if len(conn) != 2:
            continue
        pair = (int(conn[0]), int(conn[1]))
        if want[pair] > 0:
            want[pair] -= 1
            tags.add(int(rec.tag))
    if len(tags) != len(entries) or minted != len(entries):
        raise ValueError(
            f"replay: the neutral zone carries {len(entries)} rebar cell(s) "
            f"but the archive's element store holds {minted} minted "
            f"CorotTruss row(s), {len(tags)} of them matching; the two "
            "zones disagree."
        )
    return tags


def _write_archived_ramps(
    emitter: Any,
    records: "Sequence[Any]",
    param_tags: "Sequence[tuple[int, int, int]]",
    where: str,
) -> "dict[str, tuple[int, int, int]]":
    """Write each initial stress's ramp under its stored tags; mint none.

    ``param_tags`` holds one ``(xx, yy, zz)`` per record (a 2.27 archive's
    ``stress_NNN@param_tags``). Returns ``{name: tags}`` for the
    ``addToParameter`` fan-out, as the minting writer does.
    """
    from .build.stage_emit import PlannedParameter, write_planned_ramps

    if len(param_tags) != len(records):
        raise ValueError(
            f"replay: {where} holds {len(records)} initial stress record(s) "
            f"but {len(param_tags)} stored param_tags; a 2.27.0 archive "
            "stores them on every record."
        )
    return write_planned_ramps(emitter, [
        PlannedParameter(rec, None, "step_hook_ramp", tuple(tags))
        for rec, tags in zip(records, param_tags)
    ])


def _write_archived_flips(
    emitter: Any,
    verb: str,
    records: "Sequence[Any]",
    param_tags: "Sequence[Sequence[tuple[int, int]]]",
    fem: Any,
    fem_eid_to_ops_tag: Any,
    where: str,
) -> None:
    """Write each absorbing flip or ``s.update_parameter`` under its stored
    tag; mint none.

    The replay deck is flat (partition-blind), so each record addresses
    every element it resolves to on the whole mesh
    (:func:`~.build.absorbing_ele_tags` /
    :func:`~.build.update_parameter_ele_tags`, no rank). ``param_tags``
    holds each record's ``(rank, pid)`` rows: one pid, written on every
    rank holding the record's elements.
    """
    from .build.stage_emit import (
        PlannedParameter,
        absorbing_ele_tags,
        update_parameter_ele_tags,
        write_planned_flips,
    )

    if verb not in ("flip_element_stage", "update_parameter"):
        raise ValueError(f"replay: unknown parameter verb {verb!r}.")
    if len(param_tags) != len(records):
        raise ValueError(
            f"replay: {where} holds {len(records)} record(s) but "
            f"{len(param_tags)} stored param_tags entries; a 2.27.0 archive "
            "stores them on every record."
        )
    lines = []
    ele_tags = []
    for rec, rows in zip(records, param_tags):
        pids = {int(pid) for _rank, pid in rows}
        ops_tags = (
            absorbing_ele_tags(rec, fem, fem_eid_to_ops_tag)
            if verb == "flip_element_stage"
            else update_parameter_ele_tags(rec, fem, fem_eid_to_ops_tag))
        if not rows:
            # A record that wrote no flip (a rewritten pre-2.27 record
            # whose target holds no element): it must still hold none.
            if ops_tags:
                raise ValueError(
                    f"replay: {where}: a record stores no parameter tag but "
                    f"resolves {len(ops_tags)} element(s) on this mesh.")
            continue
        if len(pids) != 1:
            raise ValueError(
                f"replay: {where}: a record's param_tags rows {tuple(rows)!r} "
                "do not name one parameter tag.")
        if not ops_tags:
            raise ValueError(
                f"replay: {where}: a record stores parameter tag "
                f"{next(iter(pids))} but resolves no element on this mesh.")
        lines.append(PlannedParameter(rec, None, verb, (pids.pop(),), ops_tags))
        ele_tags.append(ops_tags)
    write_planned_flips(emitter, lines, ele_tags)


#: The builder-scoped declaration kinds a ``model BasicBuilder`` re-issue
#: purges (ADR 0099 INV-1): no emit-order merge may put one of these units
#: above the hoisted gated-element bracket.
_BUILDER_SCOPED_UNITS: "frozenset[str]" = frozenset(
    {"geomTransf", "beamIntegration", "timeSeries", "damping"})

#: The section of ``BuiltModel._emit_flat`` that writes each replay unit
#: kind, in deck order. A partitioned archive's program
#: records the ADR 0027 emit (declarations above the partition blocks
#: that hold the nodes, elements, fixes and patterns), so its replay sorts
#: by ``(section, emit index)`` to write the flat deck. Two kinds have no
#: single section: a ``"chain"`` unit is a user declaration (4a) when the
#: program wrote it above the first partition block, else the auto-emitted
#: handler (7c, :data:`_AUTO_CHAIN_SECTION`); a ``"region"`` unit whose
#: declaration is a recorder (an MPCO filter region) sits with its
#: recorder (:meth:`_ReplayOrder.tagged`).
_FLAT_SECTIONS: "dict[str, int]" = {
    "nodes": 0,  # 1a
    "uniaxialMaterial": 1, "nDMaterial": 1, "section": 1,  # 4a
    "gated_elements": 2,  # 5a
    "timeSeries": 3, "beamIntegration": 3, "damping": 3,  # 5b
    "geomTransf": 4,  # 6
    "elements": 5,  # 6b
    "fixes": 6, "masses": 7,  # 7
    "region": 8, "command": 8,  # 7: regions, rayleigh, attaches, modal
    "mp_constraints": 9, "equation_constraints": 9,  # 7b
    "reinforce_ties": 10,  # 7b'
    "rebar_cells": 11,  # 7b''
    "initial_stress": 13,  # 7d
    "pattern": 14, "recorder": 14,  # 8
    "analyze": 15,
}
#: 4a, where ``_emit_flat`` writes a user's analysis-chain declarations.
_USER_CHAIN_SECTION = 1
#: 7c, where ``_emit_flat`` writes the auto-emitted constraint handler.
_AUTO_CHAIN_SECTION = 12


class _ReplayOrder:
    """The bridge's emit order for one archive, read from
    ``/opensees/program`` (K R1b-S1, #2193).

    The deck replay (:func:`_replay_into`) groups its calls into units:
    one per declaration record (a material, section, transform,
    integration, series, damping, region, pattern or recorder), one per
    ``/opensees/commands`` row, one per analysis-chain component, and one
    per block (the nodes, each element run class, the fixes, the masses,
    the equation constraints, the reinforce ties, the initial stress, the
    ``analyze``). Each unit's key is the 1-based emit index of the call
    that wrote it, so a stable sort by key replays the units in the
    bridge's own order.

    Only an archive whose program records its declaration joins
    (:func:`~apeGmsh.opensees.emitter.h5_reader.records_program_decl`)
    builds one; the caller keeps the fixed category order otherwise.

    A tagged record is joined as ``_materials_in_emit_order`` (#1814)
    joins a material: a ``method`` call names its declaration, and
    ``/opensees/decls/tags`` names that declaration's tags, so the k-th
    fresh call of a declaration wrote its k-th tag. A partition replica
    repeats the ``(stage, row)`` of its first capture and is skipped. A
    record that no call joins (a side pass minted it, ``decl = -1``) and a
    block with no call in the program have no key: :func:`_replay_into`
    keeps such a unit in its category place, right after the unit before
    it. An analysis-chain component always has a call, so a missing one
    raises (:func:`chain`).

    A partitioned archive (a ``partition_open`` call in its program)
    records the ADR 0027 emit, not the flat deck the replay writes, so
    its units sort by ``(flat section, emit index)`` instead
    (:meth:`section`, :data:`_FLAT_SECTIONS`; K R1b-S3b, #2277). Every
    other archive sorts by the emit index alone (section 0).
    """

    __slots__ = (
        "_program", "_decls", "_tag_runs", "_first", "_joined",
        "_partition_at", "_recorder_keys")

    def __init__(self, program: "Sequence[Any]", declarations: Any) -> None:
        from ..emitter.h5_reader import records_program_decl

        if not records_program_decl(declarations):
            raise ValueError(
                "_ReplayOrder: the archive does not record its program's "
                "declaration joins (/opensees/decls@program_decl is not "
                "'complete'); replay it in the fixed category order.")
        self._program = tuple(program)
        self._decls = tuple(declarations.decls)
        # kind -> decl -> [(first tag, count), ...] in table order.
        self._tag_runs: "dict[str, dict[int, list[tuple[int, int]]]]" = {}
        for kind, first, count, decl in declarations.tags:
            self._tag_runs.setdefault(str(kind), {}).setdefault(
                int(decl), []).append((int(first), int(count)))
        self._first: "dict[tuple[str, int], int]" = {}
        for run in self._program:
            self._first.setdefault(
                (str(run.method), int(run.stage)), int(run.first))
        self._joined: "dict[tuple[str, frozenset[str]], dict[int, int]]" = {}
        # The emit index of the first global partition block; ``None`` for
        # an archive with no partition block, which sorts by emit index.
        self._partition_at: "int | None" = None
        self._recorder_keys: "frozenset[int]" = frozenset()
        if any(run.method == "partition_open" for run in self._program):
            self._partition_at = self.first("partition_open")
            if self._partition_at is None:
                raise ValueError(
                    "_ReplayOrder: the program opens partition blocks only "
                    "inside stages; a partitioned emit opens one per rank "
                    "in the global pass, so this is not a bridge emit.")
            self._recorder_keys = frozenset(
                first + i
                for _decl, first, count in self._fresh_calls(
                    frozenset({"recorder"}))
                for i in range(count))

    @property
    def partitioned(self) -> bool:
        """Whether the program records a partitioned (ADR 0027) emit."""
        return self._partition_at is not None

    def section(self, kind: str, key: "int | None") -> int:
        """The ``_emit_flat`` section of a replay unit of ``kind`` keyed
        ``key`` (its emit index, ``None`` when no call wrote it): 0 for
        every unit of an archive that is not partitioned, so those sort
        by emit index alone. Raises :class:`ValueError` on a kind the
        table does not place."""
        if self._partition_at is None:
            return 0
        if kind == "chain":
            if key is None:
                raise ValueError(
                    "replay: an analysis-chain unit has no emit index.")
            return (_USER_CHAIN_SECTION if key < self._partition_at
                    else _AUTO_CHAIN_SECTION)
        if kind == "region" and key in self._recorder_keys:
            return _FLAT_SECTIONS["recorder"]
        section = _FLAT_SECTIONS.get(kind)
        if section is None:
            raise ValueError(
                f"replay: no _emit_flat section places a {kind!r} unit; "
                "a partitioned archive cannot be replayed flat without one.")
        return section

    def first(self, method: str, stage: int = -1) -> "int | None":
        """The emit index of the first ``method`` call in ``stage``;
        ``None`` when the program holds none."""
        return self._first.get((method, stage))

    def chain(self, method: str) -> int:
        """The emit index of the global analysis-chain ``method`` call.
        Raises :class:`LookupError` when the program holds none: the
        archive stores a chain component no call wrote."""
        from ..emitter.h5_reader import emit_index_of

        return emit_index_of(self._program, method, 0, stage=-1)

    def row(self, method: str, row: int) -> int:
        """The emit index of global ``row`` of ``method``'s store
        (:func:`~apeGmsh.opensees.emitter.h5_reader.emit_index_of`)."""
        from ..emitter.h5_reader import emit_index_of

        return emit_index_of(self._program, method, row, stage=-1)

    def _fresh_calls(
        self, methods: "frozenset[str]",
    ) -> "list[tuple[int, int, int]]":
        """``(decl, first emit, count)`` of each global ``methods`` run
        that writes fresh rows (a partition replica re-writes rows a run
        before it captured, and is skipped), in emit order."""
        covered: "dict[str, int]" = {}
        out: "list[tuple[int, int, int]]" = []
        for run in self._program:
            if run.method not in methods or int(run.stage) != -1:
                continue
            start = int(run.row)
            if start < covered.get(run.method, 0):
                continue
            covered[run.method] = start + int(run.count)
            out.append((int(run.decl), int(run.first), int(run.count)))
        return out

    @staticmethod
    def _tag_at(runs: "list[tuple[int, int]]", k: int) -> "int | None":
        """The ``k``-th tag of a declaration's tag runs, or ``None``."""
        for first, count in runs:
            if k < count:
                return first + k
            k -= count
        return None

    def tagged(self, kind: str, methods: "frozenset[str]") -> "dict[int, int]":
        """``{tag: emit index}`` for the tags of allocator ``kind`` that a
        global ``methods`` call declared (the first call wins).

        In a partitioned archive a ``region`` tag whose declaration is a
        recorder (an MPCO filter region, which ADR 0027 writes once per
        rank, above the recorders) takes that recorder's emit index, as
        ``_emit_flat`` writes the region just above its recorder (#2277).
        A recorder that a stage claims keeps its region with the other
        regions. Raises :class:`LookupError` when the program holds no
        call of that recorder at all."""
        cache_key = (kind, methods)
        found = self._joined.get(cache_key)
        if found is not None:
            return found
        tag_runs = self._tag_runs.get(kind, {})
        calls_of_decl: "dict[int, int]" = {}
        out: "dict[int, int]" = {}
        for decl, first, count in self._fresh_calls(methods):
            if decl < 0:
                continue
            runs = tag_runs.get(decl, [])
            k0 = calls_of_decl.get(decl, 0)
            calls_of_decl[decl] = k0 + count
            for i in range(count):
                tag = self._tag_at(runs, k0 + i)
                if tag is None:
                    break
                out.setdefault(tag, first + i)
        if kind == "region" and self.partitioned:
            recorder_at: "dict[int, int]" = {}
            for decl, first, _count in self._fresh_calls(
                    frozenset({"recorder"})):
                recorder_at.setdefault(decl, first)
            staged = {
                int(run.decl) for run in self._program
                if run.method == "recorder" and int(run.stage) >= 0}
            for decl, runs in tag_runs.items():
                if decl < 0 or self._decls[decl].family != "recorder":
                    continue
                if decl not in recorder_at:
                    if decl in staged:
                        # A stage claims the recorder: the global region
                        # replays with the other regions, as the category
                        # order wrote it.
                        continue
                    raise LookupError(
                        f"replay: region declaration "
                        f"{self._decls[decl].key!r} belongs to a recorder "
                        "that no 'recorder' call in /opensees/program "
                        "wrote.")
                for first, count in runs:
                    for tag in range(first, first + count):
                        out[tag] = recorder_at[decl]
        self._joined[cache_key] = out
        return out

    def first_element(self, member: "Any") -> "int | None":
        """The emit index of the first global ``element`` call whose
        element ``member(tag)`` accepts; ``None`` when no joined call
        does. Classifies whole runs by their first tag (one run is one
        declaration, so one element class), never element by element."""
        tag_runs = self._tag_runs.get("element", {})
        calls_of_decl: "dict[int, int]" = {}
        for decl, first, count in self._fresh_calls(frozenset({"element"})):
            if decl < 0:
                continue
            k0 = calls_of_decl.get(decl, 0)
            calls_of_decl[decl] = k0 + count
            tag = self._tag_at(tag_runs.get(decl, []), k0)
            if tag is not None and member(tag):
                return first
        return None


def _run_units(
    units: "list[tuple[int | None, str, Any]]",
    order: "_ReplayOrder | None",
) -> None:
    """Run each ``(key, kind, thunk)`` unit: in list order (the fixed
    category order) when ``order`` is ``None``, else stably sorted by
    emit index, a keyless unit keeping its place after the unit before
    it (K R1b-S1, #2193). A partitioned ``order`` sorts by its
    ``_emit_flat`` section first (:meth:`_ReplayOrder.section`, #2277).

    Fails closed on ADR 0099 INV-1: a builder-scoped declaration sorted
    above the gated-element bracket would be purged by it, so an archive
    whose program says so raises instead of replaying a broken deck.
    """
    if order is None:
        for _key, _kind, thunk in units:
            thunk()
        return
    keyed: "list[tuple[tuple[int, int], int, str, Any]]" = []
    last = 0
    for seq, (key, kind, thunk) in enumerate(units):
        if key is not None:
            last = key
        keyed.append(((order.section(kind, key), last), seq, kind, thunk))
    keyed.sort(key=lambda u: (u[0], u[1]))
    gated_at = next(
        (i for i, u in enumerate(keyed) if u[2] == "gated_elements"), None)
    if gated_at is not None:
        early = sorted({
            u[2] for u in keyed[:gated_at] if u[2] in _BUILDER_SCOPED_UNITS})
        if early:
            raise ValueError(
                f"replay: /opensees/program emits {early} above the "
                "builder-ndf-gated elements, whose 'model BasicBuilder' "
                "bracket would purge them (ADR 0099 INV-1); the archive's "
                "emit order is not one the bridge writes.")
    for _sort, _seq, _kind, thunk in keyed:
        thunk()


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
    initial_stress_param_tags: "Sequence[tuple[int, int, int]]" = (),
    opensees_version: Any = None,
    reinforce_name_to_tag: "dict[str, int] | None" = None,
    embedded_rebars: "Sequence[Any]" = (),
    deck_ordering: bool = True,
    stage_mp_keys: "Counter[tuple[Any, ...]] | frozenset[tuple[Any, ...]]" = frozenset(),
    commands: "Sequence[Any]" = (),
    equation_constraints: "Sequence[Any]" = (),
    mp_constraints: "Sequence[tuple[int | None, Any]]" = (),
    flat: bool = False,
    order: "_ReplayOrder | None" = None,
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
      9. ``emitter.fix`` / ``emitter.mass``
      9a. the global ``/opensees/commands`` rows (``rayleigh``, ``eigen``,
          ``modal_damping``) in their emit order: the bridge's slot 7,
          after the masses and before the patterns and the chain
      9b. ``emitter.region`` for the top-level ``/opensees/regions`` rows,
          verbatim and in store order: after the global
          ``rayleigh`` so a region-scoped one still wins per element, and
          before the recorders that reference a fan-out region by ``-R``
      9b'. the global ``/opensees/constraints`` MP rows
          (``mp_constraints``: ``(emit index | None, row)`` pairs, in the
          bridge's order; #2405): its slot 7b, one unit keyed by the first
          row's index
      9b''. ``emitter.equationConstraint`` for ``equation_constraints``,
          in call order: an archive below opensees 2.35.0, whose user rows
          live only in ``/opensees/decl_params``. From 2.35.0 they are
          ``mp_constraints`` rows (9b')
      9c'. the g.reinforce ``LadrunoEmbeddedRebar`` ties: the stored
          ``embedded_rebars`` rows (opensees 2.29.0), else, for an older
          archive, re-derived from the neutral-zone ``fem``; the bridge's
          slot 7b', after the fixes, masses and equation
          constraints
      9c''. the ``g.rebar.place(emit_elements=True)`` ``CorotTruss``
          cells, taken out of step 8 when ``fem`` is given: the bridge's
          slot 7b'', after the ties (#1828)
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

        **Parameter tags (K1-8, #1465).**  ``opensees_version`` is the
        archive's opensees-zone version, required whenever an initial
        stress is replayed. At or above
        :data:`~apeGmsh.opensees.emitter.h5_reader.PARAM_TAGS_FROM` the
        archive stores every ramp's tags (``initial_stress_param_tags``,
        one per record) and the replay writes them, minting none; below
        it the replay re-mints them under the pre-2.27 tag-law waiver.

        **The bridge's order (K R1b-S1, #2193).**  The numbered steps
        above are the fixed category order. With ``order`` (a
        :class:`_ReplayOrder`, built from an archive whose
        ``/opensees/decls@program_decl`` is ``"complete"``), every unit
        those steps write is merged by its ``/opensees/program`` emit
        index instead, so declarations, the global commands and the
        analysis chain interleave as the bridge emitted them (a section
        between two materials, a ``geomTransf`` after the integrations,
        ``system`` above the declarations, ``rayleigh`` between two
        regions). A unit with no recorded call keeps its category place
        after the unit before it (:func:`_run_units`). A partitioned
        archive's ``order`` sorts by the ``_emit_flat`` section of each
        unit first, then by emit index, since its program records the
        ADR 0027 emit rather than this flat deck (#2277). ``order=None``
        (a legacy archive, a ``to_h5`` rewrite of one, the H5 target)
        keeps the category order, unchanged. ``order`` needs
        ``deck_ordering=True``: the H5 rewrite keeps its store order.

    Parameters mirror :class:`apeGmsh.opensees._internal.typed_records`
    field names; see :mod:`apeGmsh.opensees.opensees_model` for the
    canonical instantiation pattern.
    """
    from .build.ndf import node_coords_as_floats

    if order is not None and not deck_ordering:
        raise ValueError(
            "_replay_into: order= merges a deck by its emit index and needs "
            "deck_ordering=True; the H5 rewrite keeps its store order.")

    if fem is not None:
        # A record whose every line is an archived global row replays.
        _warn_skipped_replay_streams(
            fem, stage_mp_keys=Counter(stage_mp_keys) + _mp_row_keys(
                tuple(row for _i, row in mp_constraints)))

    # Every step below appends ``(emit index | None, kind, thunk)`` units;
    # _run_units runs them in this (category) order, or merged by emit
    # index when ``order`` is given. ``_key`` reads a
    # unit's index only when there is an order to merge by.
    units: "list[tuple[int | None, str, Any]]" = []

    def _key(read: Any) -> "int | None":
        return None if order is None else read(order)

    _ELEMENT = frozenset({"element"})

    # 1. Model directive.
    emitter.model(ndm=int(ndm), ndf=int(ndf))

    # 2. Nodes.  S2 (ADR 0033): the OpenSeesModel build path widens
    # the per-node tuple to ``(tag, coords, ndf|None)`` so per-node
    # ``-ndf K`` declarations survive the H5 round-trip.  Legacy
    # 2-tuples ``(tag, coords)`` are tolerated for callers that
    # haven't rebound to the wider shape.
    def _nodes() -> None:
        for entry in nodes:
            if len(entry) == 3:
                tag, coords, node_ndf = entry
            else:
                tag, coords = entry
                node_ndf = None
            # ADR 0055 P2.3: stage-owned nodes emit INSIDE their stage
            # block (_replay_staged_into), not in the global prefix.
            if int(tag) in skip_node_tags:
                continue
            # ``ndm`` coordinates only — a trailing 0.0 in a 2-D deck
            # swallows the following ``-ndf K`` (node_coords_as_floats).
            # The replay path has to do this itself: it never touches
            # build/'s node-emit helpers.
            cs = node_coords_as_floats(coords)
            if node_ndf is None:
                emitter.node(int(tag), *cs)
            else:
                emitter.node(int(tag), *cs, ndf=int(node_ndf))

    units.append((_key(lambda o: o.first("node")), "nodes", _nodes))

    def _tagged(
        kind: str, methods: "frozenset[str]", recs: "Sequence[Any]",
        emit: Any,
    ) -> None:
        """One unit per tagged declaration record, keyed by its join."""
        joined = {} if order is None else order.tagged(kind, methods)
        for rec in recs:
            units.append((
                joined.get(int(rec.tag)), kind,
                lambda rec=rec: emit(rec),
            ))

    # 3. Materials — uniaxial first, then nD.  ADR 0011 schema mirrors
    # this group nesting; the emitter doesn't enforce order, but the
    # OpenSees domain does (a section_open referencing matTag=1 needs
    # the uniaxialMaterial 1 already present).
    _tagged(
        "uniaxialMaterial", frozenset({"uniaxialMaterial"}),
        uniaxial_materials,
        lambda rec: emitter.uniaxialMaterial(
            rec.type_token, int(rec.tag), *rec.params),
    )
    _tagged(
        "nDMaterial", frozenset({"nDMaterial"}), nd_materials,
        lambda rec: emitter.nDMaterial(
            rec.type_token, int(rec.tag), *rec.params),
    )

    # 4. Sections — simple (Elastic / Aggregator) then complex
    # (Fiber).  Same ordering rule: a section_open referencing a
    # matTag needs the material registered first (already done).
    def _complex_section(rec: Any) -> None:
        emitter.section_open(rec.type_token, int(rec.tag), *rec.params)
        for patch in rec.patches:
            emitter.patch(patch.kind, *patch.args)
        for fiber in rec.fibers:
            emitter.fiber(fiber.y, fiber.z, fiber.area, int(fiber.mat_tag))
        for layer in rec.layers:
            emitter.layer(layer.kind, *layer.args)
        emitter.section_close()

    _SECTION = frozenset({"section", "section_open"})
    _tagged(
        "section", _SECTION, simple_sections,
        lambda rec: emitter.section(rec.type_token, int(rec.tag), *rec.params),
    )
    _tagged("section", _SECTION, complex_sections, _complex_section)

    from .build.builder_scope import needs_builder_ndf_bracket_for_token

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
    # The g.rebar.place(emit_elements=True) cells leave step 8: the bridge
    # writes them after the reinforce ties (slot 7b''), and so does 9c''
    # below. Only a deck target (``fem`` given) moves them; the H5
    # rewrite keeps its element order.
    _cell_tags = (
        _rebar_cell_tags(elements, fem) if fem is not None else set())
    _cells = [
        rec for rec in elements
        if int(rec.tag) in _cell_tags
        and int(rec.tag) not in skip_element_tags
    ]
    _kept = [
        rec for rec in elements
        if int(rec.tag) not in skip_element_tags
        and int(rec.tag) not in _cell_tags
    ]
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
            from .build.builder_scope import validate_builder_scope_replay

            validate_builder_scope_replay(
                _gated, ndm=int(ndm), envelope_ndf=int(ndf),
                already_gated=True,
            )
    if _gated:
        _gated_tags = {int(r.tag) for r in _gated}
        units.append((
            _key(lambda o: o.first_element(_gated_tags.__contains__)),
            "gated_elements",
            lambda: _replay_elements_bracketed(
                emitter, _gated, ndm=int(ndm), envelope_ndf=int(ndf),
            ),
        ))

    # 5. Transforms.  Schema deviation (TransformRecord docstring):
    # one record per ``geomTransf`` call, not per spec.  Replay
    # produces the same call shape: one ``geomTransf`` line per
    # record's stored ``vec``.
    _tagged(
        "geomTransf", frozenset({"geomTransf"}), transforms,
        lambda rec: emitter.geomTransf(rec.type_token, int(rec.tag), *rec.vec),
    )

    # 6. Beam integration.
    _tagged(
        "beamIntegration", frozenset({"beamIntegration"}), beam_integrations,
        lambda rec: emitter.beamIntegration(
            rec.type_token, int(rec.tag), *rec.args),
    )

    # 7. Time series.
    _tagged(
        "timeSeries", frozenset({"timeSeries"}), time_series,
        lambda rec: emitter.timeSeries(rec.type_token, int(rec.tag), *rec.args),
    )

    # 7b. Damping objects (ADR 0053 D3b).  After time_series (a ``-factor``
    # tail may reference a series tag) and before elements (an element's
    # ``-damp $tag`` rides in its own arg tail and resolves the object by
    # tag).  Region ``-damp`` attaches replay at 9b with the other
    # top-level regions.
    _tagged(
        "damping", frozenset({"damping"}), dampings,
        lambda rec: emitter.damping(rec.type_token, int(rec.tag), *rec.args),
    )

    # 8. Elements.  The H5 emitter consults two side channels for
    # connectivity (``set_element_nodes``) and FEM element id
    # (``set_current_fem_element_id``); driving them here keeps the
    # H5 round-trip byte-stable AND lets Tcl / Py / Live emitters
    # ignore the calls (their ``set_*`` helpers no-op when the attr
    # is absent).
    _ungated_out = (
        {int(r.tag) for r in _gated} | _cell_tags | set(skip_element_tags)
        if order is not None else set())
    units.append((
        _key(lambda o: o.first_element(
            lambda t: t not in _ungated_out)),
        "elements",
        lambda: _replay_elements_bracketed(
            emitter, _ungated, ndm=int(ndm), envelope_ndf=int(ndf),
        ),
    ))

    # 9. Fix / mass (model-level).
    def _fixes() -> None:
        for rec in fixes:
            emitter.fix(int(rec.tag), *(int(d) for d in rec.dofs))

    def _masses() -> None:
        for rec in masses:
            emitter.mass(int(rec.tag), *(float(v) for v in rec.values))

    units.append((_key(lambda o: o.first("fix")), "fixes", _fixes))
    units.append((_key(lambda o: o.first("mass")), "masses", _masses))

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
        units.append((
            None if order is None else int(cmd.emit_index), "command",
            lambda cmd=cmd: _replay_command(
                emitter, cmd, _GLOBAL_COMMAND_METHODS),
        ))

    # 9b. Top-level regions: the archived row is the resolved OpenSees
    # call (tag + flag tail), so every one replays verbatim, with
    # or without its K1-6 declaration — region-scoped ``-rayleigh`` and
    # ``-damp`` attaches, recorder fan-out and named regions alike. After
    # the global ``rayleigh`` rows (OpenSees overwrites element Rayleigh
    # per element, so the region must come second to win, as the bridge
    # emits it) and before the recorders that name a fan-out region.
    _tagged(
        "region", frozenset({"region"}), regions,
        lambda rec: emitter.region(int(rec.tag), *rec.args),
    )

    # The ndf each node replays with (``None`` = the envelope), for the
    # equationConstraint node and DOF checks below.
    eq_node_ndf: "dict[int, int]" = {}
    for entry in nodes:
        entry_ndf = entry[2] if len(entry) == 3 else None
        if entry_ndf is not None:
            eq_node_ndf[int(entry[0])] = int(entry_ndf)

    # 9b'. The global MP rows, at the bridge's slot 7b: after the
    # regions, before the equation constraints. One unit, keyed by its
    # first row's emit index; the caller passes the rows in bridge order.
    # From opensees 2.35.0 they include every global equationConstraint
    # line (a tie's and a user row's), each checked here against the
    # replayed nodes by the bridge's own helper before a deck is written.
    if mp_constraints:
        from ..emitter.h5_reader import EquationConstraintRecordRO

        eq_rows = [row for _i, row in mp_constraints
                   if isinstance(row, EquationConstraintRecordRO)]
        # The H5 rewrite passes no fem and writes no deck: it echoes them.
        if eq_rows and fem is not None:
            from ..emitter.recording import RecordingEmitter
            from .build.constraints import emit_equation_constraints
            from .build.records import EquationConstraintRecord

            emit_equation_constraints(
                RecordingEmitter(), fem,
                [EquationConstraintRecord(
                    cnode=r.cnode, cdof=r.cdof, ccoef=r.ccoef,
                    retained=r.retained) for r in eq_rows],
                node_ndf=eq_node_ndf, default_ndf=int(ndf))
        first_mp = min((int(i) for i, _r in mp_constraints if i is not None),
                       default=None)

        def _mp_rows() -> None:
            for _i, row in mp_constraints:
                _emit_mp_row(emitter, row)

        units.append((None if order is None else first_mp,
                      "mp_constraints", _mp_rows))

    # 9b''. User ``equationConstraint`` rows, in call order,
    # where the bridge emits them: after the regions, before the initial
    # stress. The archive keeps each one only as its ``/opensees/decl_params``
    # row (``H5Emitter`` ledgers the verb), which the caller decodes into
    # ``EquationConstraintRecord``s. The bridge's own emit helper re-checks
    # every node and DOF against the replayed nodes' ndf.
    if equation_constraints:
        if fem is None:
            raise ValueError(
                "_replay_into: equation_constraints need the broker fem to "
                "check their nodes; this caller passed none."
            )
        from .build.constraints import emit_equation_constraints

        units.append((
            _key(lambda o: o.first("equationConstraint")),
            "equation_constraints",
            lambda: emit_equation_constraints(
                emitter, fem, equation_constraints,
                node_ndf=eq_node_ndf, default_ndf=int(ndf),
            ),
        ))

    # 9c'. Embedded-reinforcement ties (g.reinforce → LadrunoEmbeddedRebar),
    # after the fixes, masses, global commands, regions and the equation
    # constraints (9b'): the bridge's slot 7b', so a reinforced deck
    # replays in the bridge's line order.
    # From opensees 2.29.0 (K2b) the archive stores every tie the build
    # wrote (``embedded_rebars``, its tag and arguments), and the replay
    # writes those, minting nothing and reading nothing else; the H5
    # rewrite path (no ``fem``) echoes them the same way. An archive below
    # 2.29.0 stores none, so the replay re-derives its ties from the
    # neutral-zone ``fem`` behind the version check below.
    from .build.constraints_plan import reinforce_tie_records

    # The bridge's own reading of the stream (``emit_reinforce_ties``).
    fem_ties = reinforce_tie_records(fem) if fem is not None else []
    if embedded_rebars:
        if fem is not None and len(fem_ties) != len(embedded_rebars):
            raise ValueError(
                f"replay: the archive stores {len(embedded_rebars)} reinforce "
                f"tie(s) but its neutral zone carries {len(fem_ties)}; the "
                "two zones disagree."
            )
        def _stored_ties() -> None:
            for tie in embedded_rebars:
                if tie.name:
                    emitter.mp_constraint_comment(tie.name)
                emitter.embedded_rebar(int(tie.ele_tag), *tie.args)

        units.append((
            _key(lambda o: o.first("embedded_rebar")), "reinforce_ties",
            _stored_ties,
        ))
    elif fem_ties:
        if not _replays_tie_fallback(opensees_version, len(fem_ties)):
            raise ValueError(
                f"replay: the neutral zone carries {len(fem_ties)} reinforce "
                "tie(s) but the archive, at opensees 2.29.0 or above, "
                "stores none; it is not an archive this writer produced."
            )
        from .build.constraints import replay_reinforce_ties
        from .tag_allocator import TagAllocator

        # tag-law waiver pre-2-29-reinforce-tie-replay (tag_law_ledger.txt):
        # an archive below 2.29.0 stores no tie, so the replay mints the
        # tie element tags (the _counters seed below and
        # replay_reinforce_ties), past the highest replayed element tag;
        # pre-2.29 archives only, behind the version check above. The
        # allocator is seeded here and the ties emit in their unit's turn.
        _rt_tags = TagAllocator()
        max_elem_tag = max((int(e.tag) for e in elements), default=0)
        _rt_tags._counters["element"] = max_elem_tag
        units.append((
            _key(lambda o: o.first("embedded_rebar")), "reinforce_ties",
            lambda: replay_reinforce_ties(
                emitter, fem, _rt_tags,
                name_to_tag=dict(reinforce_name_to_tag or {}),
            ),
        ))

    # 9c''. The g.rebar.place(emit_elements=True) CorotTruss cells, in
    # archive order: the bridge's slot 7b'', after the ties. Of what
    # the bridge writes in between, the embed ties and contacts are streams
    # this replay skips, and an interface's zeroLength stays at step 8.
    if _cells:
        units.append((
            _key(lambda o: o.first_element(_cell_tags.__contains__)),
            "rebar_cells",
            lambda: _replay_elements_bracketed(
                emitter, _cells, ndm=int(ndm), envelope_ndf=int(ndf),
            ),
        ))

    # 9c. Global initial stress (ADR 0055 Phase 1).  Emitted BEFORE
    # patterns / the analysis chain so ``step_hook_ramp`` registers and
    # the trailing ``analyze`` re-wraps into the hook-driven loop — without
    # this ordering the ramp procs declare but never fire (the emitter's
    # ``analyze`` emits the bare form when ``_step_hooks_registered`` is
    # False; tcl.py:402).  Mirrors the bridge's 7d-before-8 order.  Re-runs
    # the bridge emit helpers against the rehydrated declarative records,
    # under the parameter tags the archive stores (K1-8, schema 2.27.0);
    # an archive below 2.27.0 stores none and re-mints them, the same tags
    # the bridge minted.  The H5 target never reaches here (its
    # initial-stress persists via the side-channel, not _replay_into).
    if initial_stress:
        from .build.ownership import FemToOpsTagMap
        from .build.stage_emit import emit_initial_stress_addtoparameter

        # ADR 0065 v2 B3: the emit helpers now take a FemToOpsTagMap.
        fem_eid_to_ops_tag = FemToOpsTagMap.from_pairs(
            (int(e.fem_eid), int(e.tag))
            for e in elements
            if int(e.fem_eid) >= 0
        )
        # Each unit writes the ramps (the keyword argument is evaluated
        # first), then one addToParameter per element and component.
        _is_key = _key(lambda o: o.first("step_hook_ramp"))
        if _archives_param_tags(opensees_version):
            units.append((_is_key, "initial_stress", lambda: (
                emit_initial_stress_addtoparameter(
                    initial_stress, emitter, fem,
                    name_to_param_tags=_write_archived_ramps(
                        emitter, initial_stress, initial_stress_param_tags,
                        "/opensees/initial_stress",
                    ),
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                ))))
        else:
            from .build.stage_emit import replay_initial_stress_global
            from .tag_allocator import TagAllocator

            # ADR 0055 P2.3: the staged caller threads its SHARED allocator
            # so global + per-stage parameter tags accumulate on one counter
            # (the bridge reuses one ``tags`` across everything).  Flat
            # callers pass None → fresh allocator (unchanged behaviour).
            # tag-law waiver pre-2-27-initial-stress-replay (ledger): an
            # archive below 2.27.0 stores the declarative record and not
            # its parameter tags, so replay mints them
            # (replay_initial_stress_global); pre-2.27 archives only,
            # behind the version check above.
            _is_tags = initial_stress_tags or TagAllocator()
            units.append((_is_key, "initial_stress", lambda: (
                emit_initial_stress_addtoparameter(
                    initial_stress, emitter, fem,
                    name_to_param_tags=replay_initial_stress_global(
                        initial_stress, emitter, _is_tags,
                    ),
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                ))))

    # 10. Patterns.  ``args`` are the original ``pattern_open`` args
    # (e.g. ``(ts_tag,)`` for Plain).  Inside each pattern: loads,
    # sps, ele_loads in the order the bridge emitted them.
    def _pattern(rec: Any) -> None:
        emitter.pattern_open(rec.type_token, int(rec.tag), *rec.args)
        for load in rec.loads:
            emitter.load(int(load.target), *load.forces)
        for sp in rec.sps:
            emitter.sp(int(sp.target), int(sp.dof), float(sp.value))
        for ele_load in rec.ele_loads:
            emitter.eleLoad(*ele_load.args)
        emitter.pattern_close()

    _tagged("pattern", frozenset({"pattern_open"}), patterns, _pattern)

    # 11. Recorders.  Schema 2.3.0 wraps declared recorders in a
    # begin/end context; the helper replays the wrapping so the
    # downstream emitter sees the same calls the bridge originally
    # produced.  Typed primitives (``decl_context is None``) emit
    # bare.  The reader returns them in store-row order.
    for i, rec in enumerate(recorders):
        units.append((
            _key(lambda o, i=i: o.row("recorder", i)), "recorder",
            lambda rec=rec: _replay_recorder(emitter, rec),
        ))

    # 12. Analysis chain.  ``analysis_attrs`` is the same flat dict
    # the H5 emitter accumulates via its constraints / numberer /
    # system / test / algorithm / integrator / analysis methods.
    if analysis_attrs:
        for method, call in _analysis_chain_calls(
                emitter, analysis_attrs, flat=flat):
            units.append((
                _key(lambda o, m=method: o.chain(m)), "chain", call))
    if analyze_call is not None:
        steps, dt = analyze_call

        def _analyze() -> None:
            if dt is None:
                emitter.analyze(steps=int(steps))
            else:
                emitter.analyze(steps=int(steps), dt=float(dt))

        units.append((_key(lambda o: o.first("analyze")), "analyze", _analyze))

    _run_units(units, order)


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
    emitter: Any, attrs: "dict[str, Any]", *, flat: bool = False,
) -> None:
    """Replay an analysis-chain flat dict onto ``emitter``.

    Mirrors how :class:`H5Emitter` accumulates analysis-chain calls
    into ``self._analysis_attrs`` — each key maps to one Protocol
    method; ``<key>_args`` carries the trailing positional args when
    present.  ``*_args`` tuples are int-recovered (see
    :func:`_int_recover`) so a round-tripped ``NormDispIncr`` etc.
    re-renders byte-identically to the bridge.

    ``flat=True`` (#1832) drops the ``numberer`` and the ``system``
    whose attrs carry a ``*_runtime_fallback``: the ADR 0027 INV-5
    auto-chain only a partitioned emit writes. ``apeSees.tcl(flat=True)``
    forces the single-domain emit, which never auto-emits them, so the
    flat replay of a partitioned archive matches that deck. A numberer
    or system without a fallback attr was declared by the user and is
    replayed either way.
    """
    for _method, call in _analysis_chain_calls(emitter, attrs, flat=flat):
        call()


def _analysis_chain_calls(
    emitter: Any, attrs: "dict[str, Any]", *, flat: bool = False,
) -> "list[tuple[str, Any]]":
    """The calls :func:`_replay_analysis_chain` makes, in its order, each
    as ``(program method, thunk)``: the ``/opensees/program`` method name
    is what an emit-order merge keys the call by (K R1b-S1, #2193)."""
    out: "list[tuple[str, Any]]" = []
    if "handler" in attrs:
        args = _int_recover(attrs.get("handler_args", ()))
        out.append(("constraints", lambda: emitter.constraints(
            attrs["handler"], *args)))
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
            if not flat:
                out.append((
                    "parallel_runtime_fallback_numberer",
                    lambda: emitter.parallel_runtime_fallback_numberer(
                        attrs["numberer"], str(numberer_fb)),
                ))
        else:
            out.append((
                "numberer", lambda: emitter.numberer(attrs["numberer"])))
    if "system" in attrs:
        system_fb = attrs.get("system_runtime_fallback")
        if system_fb is not None:
            # The INV-5 auto-emit is the only fallback producer and
            # never carries system args.
            if not flat:
                out.append((
                    "parallel_runtime_fallback_system",
                    lambda: emitter.parallel_runtime_fallback_system(
                        attrs["system"], str(system_fb)),
                ))
        else:
            system_args = _int_recover(attrs.get("system_args", ()))
            out.append(("system", lambda: emitter.system(
                attrs["system"], *system_args)))
    for method in ("test", "algorithm", "integrator"):
        if method in attrs:
            out.append((method, lambda m=method: getattr(emitter, m)(
                attrs[m], *_int_recover(attrs.get(f"{m}_args", ())))))
    if "analysis" in attrs:
        out.append(("analysis", lambda: emitter.analysis(attrs["analysis"])))
    return out


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
    declarations: Any = None,
    requires: "Sequence[str]" = (),
    commands: "Sequence[Any]" = (),
    program: "Sequence[Any]" = (),
    flat: bool = False,
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

    ``flat`` reaches every analysis chain, the global one and each
    stage's (see :func:`_replay_analysis_chain`).

    ``declarations`` (the archive's ``/opensees/decls`` table) and
    ``requires`` (its replay requirements) feed
    :func:`refuse_archived_stage_routes`, which runs before anything is
    emitted (#2465).  Without declarations a stage-flag flip is refused,
    since it cannot be checked; a program with no such flip needs none.
    """
    from .build.records import ActivateAbsorbingRecord
    from .build.stage_emit import (
        emit_initial_stress_addtoparameter,
        emit_zero_velocities,
        replay_activate_absorbing,
        replay_initial_stress_global,
        zero_velocity_target_nodes,
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
    order: "_ReplayOrder | None" = replay_kwargs.get("order")

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
        from .build.builder_scope import needs_builder_ndf_bracket_for_token

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
            from .build.builder_scope import validate_builder_scope_replay

            validate_builder_scope_replay(
                [r for r in _owned_recs
                 if r.type_token in _gated_stage_tokens],
                ndm=int(replay_kwargs["ndm"]),
                envelope_ndf=int(replay_kwargs["ndf"]),
                already_gated=True,
            )

    # K1-8 (schema 2.27.0): a 2.27 archive stores every parameter tag the
    # build planned, and the stage blocks below write those, minting none.
    # Below 2.27.0 the replay re-mints them, with ONE allocator threaded
    # across the global prefix AND every stage (the bridge reuses a single
    # ``tags``; a per-stage allocator would restart parameter counters at
    # stage boundaries — gate-1 FATAL).  ``opensees_version`` is checked
    # where a record needs a tag, so a caller replaying none passes none.
    opensees_version = replay_kwargs.get("opensees_version")
    tags: "TagAllocator | None" = None
    if opensees_version is not None and not _archives_param_tags(
        opensees_version,
    ):
        # tag-law waiver pre-2-27-staged-replay-params (ledger): an archive
        # below 2.27.0 stores no parameter tag, so replay mints the stage
        # tags (replay_initial_stress_global, replay_activate_absorbing);
        # pre-2.27 archives only, behind the version check above.
        tags = TagAllocator()

    # An archived stage flip that the stage close refuses is refused here
    # too, before any emit.
    refuse_archived_stage_routes(
        stages=stages, elements=elements, declarations=declarations,
        fem=fem, requires=requires)

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
        flat=flat,
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
    from .build.ownership import FemToOpsTagMap
    from .build.ndf import node_coords_as_floats

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
                # Same order as the global prefix: steps 5-7b of
                # ``_replay_into`` (transforms, beam integrations, time
                # series, dampings), merged by emit index when there is
                # an ``order``, as the bridge's
                # ``replay_builder_scoped_declarations`` re-emits them.
                _replay_units: "list[tuple[int | None, str, Any]]" = []
                for scoped, recs, emit in (
                    ("geomTransf", replay_kwargs.get("transforms", ()),
                     lambda r: emitter.geomTransf(
                         r.type_token, int(r.tag), *r.vec)),
                    ("beamIntegration",
                     replay_kwargs.get("beam_integrations", ()),
                     lambda r: emitter.beamIntegration(
                         r.type_token, int(r.tag), *r.args)),
                    ("timeSeries", replay_kwargs.get("time_series", ()),
                     lambda r: emitter.timeSeries(
                         r.type_token, int(r.tag), *r.args)),
                    ("damping", replay_kwargs.get("dampings", ()),
                     lambda r: emitter.damping(
                         r.type_token, int(r.tag), *r.args)),
                ):
                    joined = ({} if order is None else
                              order.tagged(scoped, frozenset({scoped})))
                    for rec in recs:
                        _replay_units.append((
                            joined.get(int(rec.tag)), scoped,
                            lambda rec=rec, emit=emit: emit(rec),
                        ))
                _run_units(_replay_units, order)
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
        # kind order for pre-P2.3 archives that carry no seq. A stage's
        # equationConstraint lines (opensees 2.37.0) are the fifth kind;
        # the reader holds their seq to their rows.
        mp_groups = (
            (st.rigid_links, st.rigid_link_seq),
            (st.equal_dofs, st.equal_dof_seq),
            (st.rigid_diaphragms, st.rigid_diaphragm_seq),
            (st.embedded_nodes, st.embedded_node_seq),
            (st.equation_constraints, st.equation_constraint_seq),
        )
        have_seq = all(
            len(seq) == len(recs) for recs, seq in mp_groups
        ) and any(seq for _, seq in mp_groups)
        if have_seq:
            mp_items: "list[tuple[int, Any]]" = []
            for recs, seq in mp_groups:
                for s_idx, rec in zip(seq, recs):
                    mp_items.append((int(s_idx), rec))
            mp_items.sort(key=lambda it: it[0])
            for _s, rec in mp_items:
                _emit_mp_row(emitter, rec)
        else:
            # Pre-P2.3 fallback: fixed kind order (correct unless a
            # stage mixes rigid_diaphragm + kinematic_coupling).
            for recs, _seq in mp_groups:
                for rec in recs:
                    _emit_mp_row(emitter, rec)

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

        # stage initial_stress — re-run the bridge helpers under the
        # stored ramp tags (2.27+), or under tags re-minted from the SHARED
        # allocator (parameter tags accumulate across stages; pre-2.27).
        where = f"/opensees/stages/stage_{k:03d}"
        if st.initial_stress:
            if _archives_param_tags(opensees_version):
                name_to_param_tags = _write_archived_ramps(
                    emitter, st.initial_stress, st.initial_stress_param_tags,
                    f"{where}/initial_stress",
                )
            else:
                assert tags is not None  # set above for a pre-2.27 archive
                name_to_param_tags = replay_initial_stress_global(
                    st.initial_stress, emitter, tags,
                )
            emit_initial_stress_addtoparameter(
                st.initial_stress, emitter, fem,
                name_to_param_tags=name_to_param_tags,
                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            )

        # activate_absorbing — declarative (pg/elements); the helper
        # resolves the elements and writes each flip under its stored tag
        # (2.27+), or under one re-minted from the shared pool (pre-2.27).
        if st.activate_absorbing:
            ab_records = tuple(
                ActivateAbsorbingRecord(pg=pg, elements=els)
                for pg, els in st.activate_absorbing
            )
            if _archives_param_tags(opensees_version):
                _write_archived_flips(
                    emitter, "flip_element_stage", ab_records,
                    st.absorbing_param_tags, fem, fem_eid_to_ops_tag,
                    f"{where}/activate_absorbing",
                )
            else:
                assert tags is not None  # set above for a pre-2.27 archive
                replay_activate_absorbing(
                    ab_records, emitter, fem,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag, tags=tags,
                )

        # s.update_parameter (schema 2.27.0, K1-8; refused before it, so
        # an archive that holds one stores its tags) — the general form of
        # the absorbing flip, in the bridge's slot 6c.
        if st.update_parameters:
            _write_archived_flips(
                emitter, "update_parameter", st.update_parameters,
                st.update_parameter_param_tags, fem, fem_eid_to_ops_tag,
                f"{where}/update_parameter",
            )

        # stage analysis chain.
        if st.chain_attrs:
            _replay_analysis_chain(emitter, dict(st.chain_attrs), flat=flat)

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

        # s.zero_velocities (schema 2.27.0, K1-8) — the bridge's slot 8b:
        # after ``reset`` (which would restore the velocities) and last
        # before the analyze, over the archived nodes at their effective
        # ndf (``/opensees/nodes_ndf``, else the envelope).
        if st.zero_velocities:
            if fem is None:
                raise ValueError(
                    f"replay: {where} zeroes velocities, which needs the "
                    "neutral zone's node ids (pass fem=).")
            emit_zero_velocities(
                zero_velocity_target_nodes(st.zero_velocities, fem.nodes.ids),
                emitter,
                effective_ndf={
                    nid: int(nndf) for nid, (_c, nndf) in node_map.items()
                    if nndf is not None},
                envelope_ndf=_ndf,
            )

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
