"""Emission helpers — ``ResolvedRecorderSpec`` to OpenSees recorder commands.

Lives under :mod:`apeGmsh.results.spec` because the helpers
translate a resolved recorder spec into OpenSees-native recorder
syntax for the live driver, the Tcl/Py emit path, and the MPCO
binding.  Relocated from ``apeGmsh.solvers._recorder_emit`` in
Phase 8.3b.

The spec's canonical components (``displacement_x``, ``stress_yy``, …)
are translated to OpenSees-native recorder syntax:

- For nodes: components map to ``(ops_type, dof_index)`` pairs.
  Components sharing the same ``ops_type`` collapse into one
  recorder command with a ``-dof`` list.
- For element-level categories (``elements`` / ``gauss`` /
  ``line_stations``): emit a single recorder per record using the
  category's broad response token (``globalForce``, ``stress``, etc.).
  The file ends up with the full response vector; the transcoder
  (Phase 6) extracts canonical-component columns from it using the
  spec's manifest as a guide.

Modal, fibers, and layers are stubbed in this module — they need
emission patterns that don't fit the simple recorder form (modal
needs an ``eigen`` analysis call; fibers need per-fiber coords).
For Phase 5 they emit a pass-through comment with a TODO.

File naming
-----------
One recorder command produces one output file. The convention is::

    <output_dir>/<record.name>_<ops_token>.out      (text format)
    <output_dir>/<record.name>_<ops_token>.xml      (xml format)

For records that produce multiple recorders (e.g. nodes with
displacement + velocity in one record), the ``ops_token`` differs
per emitted recorder (``disp``, ``vel`` etc.).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Iterable, Optional

from ._resolved import ResolvedRecorderRecord


# =====================================================================
# Canonical → OpenSees nodal recorder mapping
# =====================================================================

# Translational axis suffix → DOF index (1-based, OpenSees convention)
_AXIS_TO_TRANS_DOF = {"x": 1, "y": 2, "z": 3}
# Rotational axis suffix → DOF index in 3D (ndf=6)
_AXIS_TO_ROT_DOF = {"x": 4, "y": 5, "z": 6}

# Canonical prefix → (ops_recorder_type, axis_kind)
# axis_kind ∈ {"trans", "rot"} controls which DOF table to use.
_NODAL_PREFIX_TABLE: dict[str, tuple[str, str]] = {
    "displacement": ("disp", "trans"),
    "rotation": ("disp", "rot"),
    "velocity": ("vel", "trans"),
    "angular_velocity": ("vel", "rot"),
    "acceleration": ("accel", "trans"),
    "angular_acceleration": ("accel", "rot"),
    "displacement_increment": ("incrDisp", "trans"),
    "reaction_force": ("reaction", "trans"),
    "reaction_moment": ("reaction", "rot"),
    # OpenSees ``unbalance`` returns the residual nodal force vector.
    "force": ("unbalance", "trans"),
    "moment": ("unbalance", "rot"),
}

# Special-case scalar canonical names (no ``_x``/``_y``/``_z`` suffix).
# Map directly to (ops_type, fixed_dof).
_NODAL_SCALAR_TABLE: dict[str, tuple[str, int]] = {
    # Pore pressure recorder is keyed differently in OpenSees:
    # ``recorder Node ... -dof <pdof> pressure``. The pressure DOF
    # depends on the formulation (typically 4 for u-p in 3D).
    # Pass dof=-1 as a sentinel; emitter will use the
    # session-provided ``pressure_dof`` if known, else default 4.
    "pore_pressure": ("pressure", -1),
}


# =====================================================================
# Element-level recorder responses
# =====================================================================

# Category → (ops_recorder_token, comment).
# These are the canonical OpenSees responses for each topology level.
# For categories that don't have a clean single-token response
# (fibers, layers, modal), we emit a TODO comment instead.
#
# Gauss is special: the actual ops keyword (``stresses`` vs ``strains``)
# depends on the record's components, not on the category alone. The
# fallback ``"stresses"`` here is only used if component derivation
# returns nothing — kept so older callers don't silently break.
_ELEMENT_CATEGORY_RESPONSE: dict[str, str] = {
    "elements": "globalForce",        # per-element-node force vector
    "gauss": "stresses",              # fallback only — real choice is per-record
    "line_stations": "section force", # beam section forces along length
}


def _gauss_record_ops_keyword(rec: ResolvedRecorderRecord) -> Optional[str]:
    """Return the ops keyword (``"stresses"`` / ``"strains"``) for a gauss record.

    Raises ``ValueError`` if the record mixes work-conjugate families
    (e.g. stress + strain, or membrane_force + curvature) — those
    can't be emitted under a single Element recorder. Mixing within
    one keyword is OK (membrane_force + bending_moment both go under
    ``stresses``).

    A ``stress_zz`` record whose ``sigma_zz_capable`` says the elements
    can report the material's σ33 is promoted to the 4-component
    plane-strain response, the same record-level promotion the bridge
    and live-capture routes make — the decision (and the warning when it
    is refused) lives in the one shared helper so the three routes cannot
    disagree.  Like ``element_class_name``, this route takes the answer
    from the caller: it has no bridge back-reference to derive it from,
    so an unset field means "unknown" and keeps today's token.
    """
    from apeGmsh.opensees._recorder_translate import _stress_zz_tokens
    from apeGmsh.opensees._response_catalog import gauss_keyword_for_canonical
    keywords: set[str] = set()
    for comp in rec.components:
        keyword = gauss_keyword_for_canonical(comp)
        if keyword is not None:
            keywords.add(keyword)
    if not keywords:
        return None
    if len(keywords) > 1:
        raise ValueError(
            f"Record {rec.name!r} (category=gauss) mixes work-conjugate "
            f"families ({sorted(keywords)}); split into separate records "
            f"(one per ops keyword)."
        )
    keyword = next(iter(keywords))
    if keyword == "stresses" and "stress_zz" in rec.components:
        keyword = _stress_zz_tokens(
            rec.name, getattr(rec, "sigma_zz_capable", None),
        )[0]
    return keyword


def _per_material_strain_skip_reason(
    rec: ResolvedRecorderRecord,
) -> Optional[str]:
    """Detect gauss-strain records the .out path can't capture.

    Tri31 (and any future class listed in
    :data:`apeGmsh.opensees._response_catalog.PER_MATERIAL_STRAIN_CLASSES`)
    has no element-level ``setResponse('strains')`` branch — an
    ``ops.recorder('Element', ..., 'strains')`` against such elements
    writes only the time column, no strain values. The per-material
    path (``ops.recorder('Element', ..., 'material', '<gp>',
    'strain')``) does work, but emitting it requires a per-Gauss-point
    fan-out the .out path doesn't currently handle.

    Returns a user-facing reason string when the record should be
    skipped at .out emission time, else ``None``. Callers either warn
    + return empty (live emission) or embed the reason as a comment
    (Tcl/Python export).
    """
    if rec.category != "gauss":
        return None
    class_name = rec.element_class_name
    if class_name is None:
        return None
    from apeGmsh.opensees._response_catalog import PER_MATERIAL_STRAIN_CLASSES
    if class_name not in PER_MATERIAL_STRAIN_CLASSES:
        return None
    try:
        keyword = _gauss_record_ops_keyword(rec)
    except ValueError:
        return None
    if keyword != "strains":
        return None
    return (
        f"Record {rec.name!r} (gauss strain) targets element class "
        f"{class_name!r}, whose element-level setResponse('strains') "
        f"is unimplemented in OpenSees — an .out recorder would write "
        f"only the time column, no strain data. Use MPCO instead "
        f"(``recorder mpco ... -E material.strain``, read with "
        f"``Results.from_mpco(...)``), or capture in-process via "
        f"``ops.domain_capture(DomainCaptureSpec(opensees=ops), "
        f"path=...)`` (read with ``Results.from_native(...)``)."
    )


def _nodal_record_ops_keyword(rec: ResolvedRecorderRecord) -> Optional[str]:
    """Return the ops keyword (``"globalForce"`` / ``"localForce"``) for an elements record.

    Closed-form line elements (Phase 11b Step 3) expose per-element-
    node forces under both ``globalForce`` and ``localForce`` recorder
    tokens. The frame is derived from the record's canonical
    components: ``nodal_resisting_force_local_*`` /
    ``nodal_resisting_moment_local_*`` → ``localForce``; everything
    else → ``globalForce``. Returns ``None`` if no components route
    through the nodal-forces topology.

    Raises ``ValueError`` if the record mixes the two frames.
    """
    from apeGmsh.opensees._response_catalog import gauss_keyword_for_canonical
    keywords: set[str] = set()
    for comp in rec.components:
        keyword = gauss_keyword_for_canonical(comp, topology="nodal_forces")
        if keyword is not None:
            keywords.add(keyword)
    if not keywords:
        return None
    if len(keywords) > 1:
        raise ValueError(
            f"Record {rec.name!r} (category=elements) mixes global and "
            f"local frames ({sorted(keywords)}); split into separate "
            f"records (one per ops keyword)."
        )
    return next(iter(keywords))


#: File-name suffix of an elastic-beam line-stations record's ``localForce``
#: file (``<name>_line_stations_local_force.<ext>``). Distinct from the
#: ``section force`` file's ``_line_stations`` so the two never alias.
_LINE_STATIONS_LOCAL_FORCE_SUFFIX = "line_stations_local_force"


def line_stations_from_local_force(rec: ResolvedRecorderRecord) -> bool:
    """True when a line-stations record is read from ``localForce``.

    That is the elastic-beam family (``ElasticBeam{2d,3d}``,
    ``ModElasticBeam2d``, ``ElasticTimoshenkoBeam{2d,3d}``): classes with
    no ``section force`` layout and a ``localForce`` line-stations
    synthesis in the response catalog, the same two-way test
    DomainCapture's line-station capturer makes. The emit and the ``.out``
    transcoder both ask this one predicate, so they cannot disagree on
    which file a record wrote.

    The class comes from ``element_class_name``, which this route takes
    from the caller and never infers. Unset, it is not an elastic beam
    here: the record keeps its ``section force`` pair, and the transcoder
    refuses the record for the missing class.
    """
    if rec.category != "line_stations" or rec.element_class_name is None:
        return False
    from apeGmsh.opensees._response_catalog import (
        is_custom_rule_catalogued,
        is_line_station_synthesis_catalogued,
    )
    cls = rec.element_class_name
    return (
        not is_custom_rule_catalogued(cls, "section_force")
        and is_line_station_synthesis_catalogued(cls)
    )


#: OpenSees ``element`` type tokens that build the elastic-beam classes:
#: ``elasticBeamColumn`` gives ``ElasticBeam2d``/``3d`` by ndm,
#: ``ElasticTimoshenkoBeam`` gives ``ElasticTimoshenkoBeam2d``/``3d``.
ELASTIC_BEAM_TYPE_TOKENS = frozenset({
    "elasticBeamColumn", "ElasticTimoshenkoBeam", "ModElasticBeam2d",
})


def refuse_non_elastic_local_force(
    rec: ResolvedRecorderRecord, model: object, *, fem_snapshot_id: str,
) -> None:
    """Raise unless every element of a ``localForce`` record is an elastic beam.

    ``element_class_name`` names one class for the whole record, so a record
    over a force beam and an elastic girder that names the girder's class
    would give the force beam two end stations instead of its section
    stations. ``model`` (an ``OpenSeesModel`` built from the spec's FEMData)
    records each element's type token; any element whose token does not build
    an elastic beam, or that the model does not hold, is refused by FEM id.
    """
    fem_hash = model.lineage.fem_hash  # type: ignore[attr-defined]
    if fem_hash != fem_snapshot_id:
        raise ValueError(
            f"Line-stations record {rec.name!r}: model= was not built from "
            f"the spec's FEMData (lineage.fem_hash {fem_hash!r}, spec "
            f"fem_snapshot_id {fem_snapshot_id!r}), so its element types "
            f"cannot check the record's elastic-beam class."
        )
    types = {
        int(r.fem_eid): r.type_token
        for r in model.elements()  # type: ignore[attr-defined]
        if int(r.fem_eid) >= 0
    }
    assert rec.element_ids is not None
    bad = [int(e) for e in rec.element_ids
           if types.get(int(e)) not in ELASTIC_BEAM_TYPE_TOKENS]
    if bad:
        found = sorted({str(types.get(e)) for e in bad})
        raise ValueError(
            f"Line-stations record {rec.name!r} names the elastic-beam class "
            f"{rec.element_class_name!r}, so it records localForce end "
            f"forces, but FEM element(s) {bad[:10]}"
            f"{' ...' if len(bad) > 10 else ''} are {found} in model=, not "
            f"elastic beams ({sorted(ELASTIC_BEAM_TYPE_TOKENS)}). Split the "
            f"record into one per element class."
        )


# =====================================================================
# Logical recorder spec — backend-agnostic intermediate
# =====================================================================

@dataclass(frozen=True)
class LogicalRecorder:
    """One emitted OpenSees recorder, in backend-agnostic form.

    The Tcl and Python formatters take this and write the right
    command syntax.

    ``precision`` is the significant digits OpenSees writes per value
    (``-precision N``). OpenSees defaults to 6, which caps a ``.out``
    value oracle near rtol 1e-5; the default 17 is the digit count that
    round-trips an IEEE double, so the file holds the value the domain
    held. Every formatter emits it (#1974).
    """

    kind: str                          # "Node" | "Element"
    file_path: str
    file_format: str                   # "out" | "xml"
    target_kind: str                   # "node" | "ele"
    target_ids: tuple[int, ...]
    response_tokens: tuple[str, ...]   # e.g. ("disp",) or ("section", "force")
    dofs: Optional[tuple[int, ...]] = None    # only for nodes
    dt: Optional[float] = None
    record_name: str = ""              # for traceability / comments
    comment: str = ""                  # appended as a trailing comment
    precision: int = 17                # significant digits (-precision)

    def __post_init__(self) -> None:
        p = self.precision
        if isinstance(p, bool) or not isinstance(p, int) or p <= 0:
            raise ValueError(
                f"LogicalRecorder {self.record_name!r}: precision must be "
                f"a positive int (significant digits per written value), "
                f"got {p!r}."
            )


# =====================================================================
# Public emission entry points
# =====================================================================

def emit_logical(
    record: ResolvedRecorderRecord,
    *,
    output_dir: str = "",
    file_format: str = "out",
    stage_id: Optional[str] = None,
) -> list[LogicalRecorder]:
    """Translate one resolved record into 0+ logical recorder specs.

    Some records produce multiple logical recorders (e.g. a node
    record with both displacement and velocity components). Some
    produce zero (modal, fibers, layers — stubbed for Phase 5).

    ``stage_id``: when set, output filenames are prefixed with
    ``<stage_id>__`` so multi-stage live emission can keep separate
    output files per stage. ``None`` (default) keeps the legacy
    flat-naming used by Tcl/Py exports.
    """
    if record.category == "nodes":
        return list(_emit_nodes(record, output_dir, file_format, stage_id))
    if record.category in ("elements", "gauss", "line_stations"):
        return list(_emit_element_simple(
            record, output_dir, file_format, stage_id,
        ))
    if record.category in ("fibers", "layers", "modal"):
        # Deferred — Phase 5 emits a TODO comment via the formatter
        # by returning an empty list. The formatter inserts a comment
        # when it sees an unsupported category.
        return []
    return []


# =====================================================================
# FEM id → OpenSees tag translation at emit
# =====================================================================
#
# A resolved record names elements by FEM id (the contract DomainCapture
# honours too), but an OpenSees ``-ele`` list takes OpenSees tags, and
# under ``element_tags="sequential"`` (the bridge default) the two differ.
# :func:`emit_logical` stays in FEM-id space: the transcoder and the cache
# read it back to label rows and to find files. Only the command that
# reaches OpenSees is translated, through :func:`to_ops_tags`. Node
# records need no map: the bridge emits every node under its FEM id
# (``opensees/_internal/compose.py``, node emit).


def element_tag_map(model: object, *, fem_snapshot_id: str) -> dict[int, int]:
    """Return the ``{fem_eid: ops_tag}`` pairing a built model records.

    ``model`` is an :class:`~apeGmsh.opensees.OpenSeesModel`, and it must
    be built from the FEMData the spec was resolved against: its
    ``lineage.fem_hash`` (equal to ``FEMData.snapshot_id``) must equal the
    spec's ``fem_snapshot_id``. A model from another or a re-meshed
    FEMData can share FEM ids with this one and would translate them to
    the wrong tags, so a mismatch raises. The pairing
    is its element records' ``(fem_eid, tag)``: the same source
    ``ElementTagTranslator.from_model`` reads for ``from_mpco`` and
    domain capture. Rows with the sentinel ``fem_eid < 0`` are skipped.
    Read strictly here: that translator passes an unknown id through,
    which an emit must refuse instead (:func:`to_ops_tags`).
    """
    fem_hash = model.lineage.fem_hash  # type: ignore[attr-defined]
    if fem_hash != fem_snapshot_id:
        raise ValueError(
            f"model= was not built from this spec's FEMData: its "
            f"lineage.fem_hash is {fem_hash!r}, the spec's fem_snapshot_id "
            f"is {fem_snapshot_id!r}. Translating the spec's element ids "
            f"through another model's pairing would record the wrong "
            f"elements. Pass the OpenSeesModel built from the FEMData the "
            f"spec was resolved against."
        )
    out: dict[int, int] = {}
    for rec in model.elements():  # type: ignore[attr-defined]
        fem = int(rec.fem_eid)
        if fem >= 0:
            out[fem] = int(rec.tag)
    return out


def to_ops_tags(
    rec: LogicalRecorder,
    tag_map: "dict[int, int] | None",
    *,
    where: str,
) -> LogicalRecorder:
    """Return ``rec`` with its ``-ele`` ids translated FEM id → ops tag.

    Node recorders pass through. An element recorder with no map raises:
    the FEM ids may not be the OpenSees tags, and identity is never
    assumed. An id the map does not hold raises too (never a partial
    or pass-through translation).
    """
    if rec.target_kind != "ele":
        return rec
    if tag_map is None:
        raise ValueError(
            f"{where}: record {rec.record_name!r} names elements by FEM id, "
            f"and OpenSees records by element tag; under "
            f"element_tags=\"sequential\" (the bridge default) the two "
            f"differ, so emitting the FEM ids would record other elements. "
            f"Pass model= (the built OpenSeesModel, e.g. "
            f"OpenSeesModel.from_h5(<ops.h5(...) path>)) so the ids are "
            f"translated to the tags the model was built with."
        )
    missing = [int(e) for e in rec.target_ids if int(e) not in tag_map]
    if missing:
        raise ValueError(
            f"{where}: record {rec.record_name!r} names FEM element ids "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''} that the "
            f"model= passed has no OpenSees element for. Pass the model "
            f"built from the same FEMData."
        )
    return replace(
        rec, target_ids=tuple(tag_map[int(e)] for e in rec.target_ids),
    )


def format_tcl(rec: LogicalRecorder) -> str:
    """Format a LogicalRecorder as one ``recorder ...`` Tcl command line."""
    parts: list[str] = ["recorder", rec.kind]
    parts.extend(_file_flag_tcl(rec.file_path, rec.file_format))
    parts.append("-time")
    if rec.dt is not None:
        parts.append(f"-dT {rec.dt:.10g}")
    parts.append(f"-{rec.target_kind}")
    parts.extend(str(i) for i in rec.target_ids)
    if rec.dofs is not None:
        parts.append("-dof")
        parts.extend(str(d) for d in rec.dofs)
    # After the target list, before the response: OpenSees reads options
    # in any order and the first unknown token starts the response.
    parts.append(f"-precision {rec.precision}")
    parts.extend(rec.response_tokens)
    line = " ".join(parts)
    if rec.comment:
        line += f"  ;# {rec.comment}"
    return line


def format_python(rec: LogicalRecorder) -> str:
    """Format a LogicalRecorder as one ``ops.recorder(...)`` call."""
    args: list[str] = [repr(rec.kind)]
    args.extend(_file_flag_py(rec.file_path, rec.file_format))
    args.append("'-time'")
    if rec.dt is not None:
        args.extend(["'-dT'", f"{rec.dt:.10g}"])
    args.append(f"'-{rec.target_kind}'")
    args.extend(str(i) for i in rec.target_ids)
    if rec.dofs is not None:
        args.append("'-dof'")
        args.extend(str(d) for d in rec.dofs)
    args.extend(["'-precision'", str(rec.precision)])
    args.extend(repr(t) for t in rec.response_tokens)
    line = f"ops.recorder({', '.join(args)})"
    if rec.comment:
        line += f"  # {rec.comment}"
    return line


def to_ops_args(rec: LogicalRecorder) -> tuple:
    """Argument tuple for a live ``ops.recorder(*args)`` call.

    Mirrors :func:`format_python` exactly, except it returns Python
    objects (strings, ints, floats) instead of the equivalent source
    code. Used by the live recorder strategy
    (``ResolvedRecorderSpec.emit_recorders``) to push the recorder
    into the running openseespy domain without going through a script
    file.

    The returned tuple is ready to splat:

    >>> import openseespy.opensees as ops
    >>> ops.recorder(*to_ops_args(logical))   # doctest: +SKIP
    """
    args: list = [rec.kind]
    if rec.file_format == "xml":
        args.extend(["-xml", rec.file_path])
    else:
        args.extend(["-file", rec.file_path])
    args.append("-time")
    if rec.dt is not None:
        args.extend(["-dT", float(rec.dt)])
    args.append(f"-{rec.target_kind}")
    args.extend(int(i) for i in rec.target_ids)
    if rec.dofs is not None:
        args.append("-dof")
        args.extend(int(d) for d in rec.dofs)
    args.extend(["-precision", int(rec.precision)])
    args.extend(rec.response_tokens)
    return tuple(args)


# =====================================================================
# Per-category emission
# =====================================================================

def _warn_n_steps_unsupported(rec: ResolvedRecorderRecord) -> None:
    """Warn once per record that ``n_steps=`` without ``dt=`` cannot be honoured.

    A classic ``.out`` / ``.xml`` recorder has no step-count option, so
    the record is emitted as an every-step recorder. Both per-category
    emitters call this, so every category warns the same way (#1947).
    ``stacklevel=5`` points past this helper, the emitter generator and
    :func:`emit_logical`, as the nodes warning did before.
    """
    if rec.n_steps is not None and rec.dt is None:
        warnings.warn(
            f"Record {rec.name!r}: ``n_steps={rec.n_steps}`` has no "
            f"equivalent on a classic recorder, which has no step-count "
            f"option. Emitting an every-step recorder. Use ``dt=`` for a "
            f"time cadence on ``.out``, or the MPCO or fork recorder's "
            f"``-T nsteps`` for a step cadence.",
            UserWarning,
            stacklevel=5,
        )


def _emit_nodes(
    rec: ResolvedRecorderRecord,
    output_dir: str,
    file_format: str,
    stage_id: Optional[str] = None,
) -> Iterable[LogicalRecorder]:
    """One logical recorder per ``(ops_type, dof_set)``."""
    if rec.node_ids is None or rec.node_ids.size == 0:
        return

    _warn_n_steps_unsupported(rec)

    by_type: dict[str, list[int]] = {}    # ops_type → sorted DOF list
    for canonical in rec.components:
        type_dof = _canonical_to_ops(canonical)
        if type_dof is None:
            continue
        ops_type, dof = type_dof
        by_type.setdefault(ops_type, []).append(dof)

    target_ids = tuple(int(n) for n in rec.node_ids)
    for ops_type, dofs in by_type.items():
        unique_dofs = tuple(sorted(set(dofs)))
        # Pressure recorder uses a single fixed DOF, no -dof flag in
        # OpenSees Tcl when ``-dof`` is omitted (-pressure has its own
        # syntax). Keep the explicit -dof for now.
        file_path = _build_file_path(
            output_dir, rec.name, ops_type, file_format, stage_id=stage_id,
        )
        yield LogicalRecorder(
            kind="Node",
            file_path=file_path,
            file_format=file_format,
            target_kind="node",
            target_ids=target_ids,
            response_tokens=(ops_type,),
            dofs=unique_dofs,
            dt=rec.dt,
            record_name=rec.name,
            comment=f"{rec.name} {ops_type}",
        )


def _emit_element_simple(
    rec: ResolvedRecorderRecord,
    output_dir: str,
    file_format: str,
    stage_id: Optional[str] = None,
) -> Iterable[LogicalRecorder]:
    """One (or two) logical recorders per element-level record.

    Emits the broad response token for the category. The file
    contains the full response vector per element; the transcoder
    extracts canonical components from it post-hoc.

    Line-stations records emit a *second* paired recorder for
    ``integrationPoints`` to a sibling ``_gpx.<ext>`` file. The
    pair is required because per-element IP locations are not
    derivable from the section-force ``.out`` alone (no META, no
    GP_X) — Phase 11b Step 2c's transcoder reads both to reconstruct
    the same ``LineStationSlab`` shape that MPCO read and
    DomainCapture produce. See
    ``internal_docs/plan_phase_11b_line_stations.md`` and
    ``apeGmsh.results.transcoders._recorder`` for the consumer.

    An elastic-beam line-stations record
    (:func:`line_stations_from_local_force`) emits one ``localForce``
    recorder instead of that pair.
    """
    if rec.element_ids is None or rec.element_ids.size == 0:
        return

    skip_reason = _per_material_strain_skip_reason(rec)
    if skip_reason:
        warnings.warn(skip_reason, RuntimeWarning, stacklevel=3)
        return

    if rec.category == "gauss":
        # Token depends on components (stresses vs strains), not on
        # category alone. None means no recognised gauss components.
        keyword = _gauss_record_ops_keyword(rec)
        if keyword is None:
            return
        response_str = keyword
    elif rec.category == "elements":
        # Phase 11b Step 3c: keyword depends on components
        # (globalForce vs localForce) — local-frame components emit
        # a localForce recorder; everything else falls through to
        # the category default ("globalForce").
        keyword = _nodal_record_ops_keyword(rec)
        if keyword is not None:
            response_str = keyword
        else:
            response_str = _ELEMENT_CATEGORY_RESPONSE.get(rec.category)
            if response_str is None:
                return
    elif line_stations_from_local_force(rec):
        # An elastic beam has no integration points: it answers neither
        # ``section force`` nor ``integrationPoints`` (a recorder writes
        # only the time column). Its stations come from the ``localForce``
        # end vector instead, which the transcoder turns into a
        # two-station slab at xi = -1, +1, as DomainCapture and
        # the MPCO local-force reader do.
        _warn_n_steps_unsupported(rec)
        yield LogicalRecorder(
            kind="Element",
            file_path=_build_file_path(
                output_dir, rec.name, _LINE_STATIONS_LOCAL_FORCE_SUFFIX,
                file_format, stage_id=stage_id,
            ),
            file_format=file_format,
            target_kind="ele",
            target_ids=tuple(int(e) for e in rec.element_ids),
            response_tokens=("localForce",),
            dofs=None,
            dt=rec.dt,
            record_name=rec.name,
            comment=f"{rec.name} line_stations localForce",
        )
        return
    else:
        response_str = _ELEMENT_CATEGORY_RESPONSE.get(rec.category)
        if response_str is None:
            return

    _warn_n_steps_unsupported(rec)
    response_tokens = tuple(response_str.split())
    target_ids = tuple(int(e) for e in rec.element_ids)
    file_path = _build_file_path(
        output_dir, rec.name, rec.category, file_format, stage_id=stage_id,
    )
    yield LogicalRecorder(
        kind="Element",
        file_path=file_path,
        file_format=file_format,
        target_kind="ele",
        target_ids=target_ids,
        response_tokens=response_tokens,
        dofs=None,
        dt=rec.dt,
        record_name=rec.name,
        comment=f"{rec.name} {rec.category}",
    )

    if rec.category == "line_stations":
        # Paired integrationPoints recorder. OpenSees writes per-element
        # physical IP positions ``xi*L`` (per ForceBeamColumn3d.cpp:3338–
        # 3346 ``locs(i) = pts[i] * L``) — one row per step, but only
        # the first row is needed because IPs are static. The transcoder
        # normalises ``xi*L → [-1, +1]`` using element length from the
        # bound FEMData.
        gpx_path = line_station_gpx_path(file_path)
        yield LogicalRecorder(
            kind="Element",
            file_path=gpx_path,
            file_format=file_format,
            target_kind="ele",
            target_ids=target_ids,
            response_tokens=("integrationPoints",),
            dofs=None,
            dt=rec.dt,
            record_name=rec.name,
            comment=f"{rec.name} line_stations gpx",
        )


# =====================================================================
# Helpers
# =====================================================================

def _canonical_to_ops(canonical: str) -> Optional[tuple[str, int]]:
    """Map a canonical name to ``(ops_type, dof_index)`` for nodal recorders.

    Returns ``None`` if the canonical name has no nodal recorder
    translation (the caller should skip).
    """
    # Scalar special cases
    if canonical in _NODAL_SCALAR_TABLE:
        ops_type, dof = _NODAL_SCALAR_TABLE[canonical]
        # Sentinel: pressure DOF defaults to 4 in 3D u-p formulation
        if dof < 0:
            dof = 4
        return (ops_type, dof)

    # Vector form: "<prefix>_<axis>"
    if "_" not in canonical:
        return None
    prefix, axis = canonical.rsplit("_", 1)
    table_entry = _NODAL_PREFIX_TABLE.get(prefix)
    if table_entry is None:
        return None
    ops_type, axis_kind = table_entry
    dof_table = (
        _AXIS_TO_TRANS_DOF if axis_kind == "trans" else _AXIS_TO_ROT_DOF
    )
    if axis not in dof_table:
        return None
    return (ops_type, dof_table[axis])


def _build_file_path(
    output_dir: str, record_name: str, suffix: str, file_format: str,
    *, stage_id: Optional[str] = None,
) -> str:
    """Build the recorder output file path with the right extension.

    When ``stage_id`` is given, the basename is prefixed with
    ``<stage_id>__`` so multi-stage live emission produces distinct
    files per stage. Double-underscore separates the prefix so a
    single-underscore record name remains unambiguous to parse.
    """
    base = f"{record_name}_{suffix}"
    if stage_id is not None:
        base = f"{stage_id}__{base}"
    ext = "xml" if file_format == "xml" else "out"
    fname = f"{base}.{ext}"
    if not output_dir:
        return fname
    sep = "" if output_dir.endswith(("/", "\\")) else "/"
    return f"{output_dir}{sep}{fname}"


def line_station_gpx_path(line_station_file_path: str) -> str:
    """Return the paired ``integrationPoints`` recorder path for a line-stations file.

    Convention: replace the file extension on
    ``<base>_line_stations.<ext>`` with ``_gpx.<ext>``, yielding
    ``<base>_line_stations_gpx.<ext>``. Both files are emitted by
    :func:`_emit_element_simple` for line_stations records and
    consumed together by the .out transcoder.

    Used by both the emitter (when producing the paired recorder
    line) and the transcoder (when locating it on disk to read).
    """
    if line_station_file_path.endswith(".out"):
        return line_station_file_path[:-4] + "_gpx.out"
    if line_station_file_path.endswith(".xml"):
        return line_station_file_path[:-4] + "_gpx.xml"
    return line_station_file_path + "_gpx"


def _file_flag_tcl(path: str, file_format: str) -> list[str]:
    if file_format == "xml":
        return ["-xml", path]
    return ["-file", path]


def _file_flag_py(path: str, file_format: str) -> list[str]:
    if file_format == "xml":
        return ["'-xml'", repr(path)]
    return ["'-file'", repr(path)]


# =====================================================================
# Spec-level emission (called by ResolvedRecorderSpec methods)
# =====================================================================

_DEFERRED_CATEGORIES = frozenset({"fibers", "layers", "modal"})


def emit_spec_tcl(
    records: Iterable[ResolvedRecorderRecord],
    *,
    output_dir: str = "",
    file_format: str = "out",
    model: object = None,
    fem_snapshot_id: str = "",
) -> list[str]:
    """Emit Tcl recorder commands for an iterable of resolved records.

    ``model`` (the built ``OpenSeesModel``) translates element ids to
    OpenSees tags; an element record without it raises
    (:func:`to_ops_tags`).
    """
    tag_map = (None if model is None else
               element_tag_map(model, fem_snapshot_id=fem_snapshot_id))
    lines: list[str] = []
    for rec in records:
        if rec.category in _DEFERRED_CATEGORIES:
            lines.append(_tcl_unsupported_comment(rec))
            continue
        skip_reason = _per_material_strain_skip_reason(rec)
        if skip_reason:
            lines.append(f";# SKIPPED: {skip_reason}")
            continue
        sub = list(emit_logical(
            rec, output_dir=output_dir, file_format=file_format,
        ))
        # Empty sub means "no targets" (e.g. zero IDs) — silently skip.
        for lr in sub:
            lines.append(format_tcl(to_ops_tags(
                lr, tag_map, where="ResolvedRecorderSpec.to_tcl_commands",
            )))
    return lines


def emit_spec_python(
    records: Iterable[ResolvedRecorderRecord],
    *,
    output_dir: str = "",
    file_format: str = "out",
    model: object = None,
    fem_snapshot_id: str = "",
) -> list[str]:
    """Emit ``ops.recorder(...)`` calls for an iterable of resolved records.

    ``model`` translates element ids as in :func:`emit_spec_tcl`.
    """
    tag_map = (None if model is None else
               element_tag_map(model, fem_snapshot_id=fem_snapshot_id))
    lines: list[str] = []
    for rec in records:
        if rec.category in _DEFERRED_CATEGORIES:
            lines.append(_py_unsupported_comment(rec))
            continue
        skip_reason = _per_material_strain_skip_reason(rec)
        if skip_reason:
            lines.append(f"# SKIPPED: {skip_reason}")
            continue
        sub = list(emit_logical(
            rec, output_dir=output_dir, file_format=file_format,
        ))
        for lr in sub:
            lines.append(format_python(to_ops_tags(
                lr, tag_map, where="ResolvedRecorderSpec.to_python_commands",
            )))
    return lines


def _tcl_unsupported_comment(rec: ResolvedRecorderRecord) -> str:
    if rec.category in ("fibers", "layers"):
        return (
            f";# recorder category {rec.category!r} "
            f"(record {rec.name!r}) is MPCO-only in apeGmsh Phase 11c. "
            f"Add an MPCO recorder (e.g. ``recorder mpco {rec.name}.mpco "
            f"-E section.fiber.stress``) instead of relying on .out "
            f"emission, then read with ``Results.from_mpco(...)``."
        )
    return (
        f";# TODO Phase 5+: recorder category {rec.category!r} "
        f"(record {rec.name!r}) — emission deferred. "
        f"Use domain capture (Phase 7) or MPCO bridge (Phase 8)."
    )


def _py_unsupported_comment(rec: ResolvedRecorderRecord) -> str:
    if rec.category in ("fibers", "layers"):
        return (
            f"# recorder category {rec.category!r} "
            f"(record {rec.name!r}) is MPCO-only in apeGmsh Phase 11c. "
            f"Add an MPCO recorder (e.g. ``ops.recorder('mpco', "
            f"'{rec.name}.mpco', '-E', 'section.fiber.stress')``) "
            f"instead of relying on .out emission, then read with "
            f"``Results.from_mpco(...)``."
        )
    return (
        f"# TODO Phase 5+: recorder category {rec.category!r} "
        f"(record {rec.name!r}) — emission deferred. "
        f"Use domain capture (Phase 7) or MPCO bridge (Phase 8)."
    )


# =====================================================================
# MPCO bridge (Phase 8) — single ``recorder mpco`` line
# =====================================================================
#
# MPCO records the WHOLE result tensor/vector for each token (no
# per-DOF selection). We aggregate unique tokens across all records
# and emit one ``recorder mpco`` command. Selection is implicitly
# "everything" — STKO consumers filter at read time.
#
# Token mapping is by canonical *prefix* (the axis suffix doesn't
# matter for MPCO, which always records all components of a result).

# canonical prefix → MPCO -N token
_MPCO_NODE_TOKENS: dict[str, str] = {
    "displacement": "displacement",
    "rotation": "rotation",
    "velocity": "velocity",
    "angular_velocity": "angularVelocity",
    "acceleration": "acceleration",
    "angular_acceleration": "angularAcceleration",
    "displacement_increment": "incrDisp",
    "reaction_force": "reactionForce",
    "reaction_moment": "reactionMoment",
    "force": "unbalancedForce",
    "moment": "unbalancedMoment",
}

# Scalar canonicals → MPCO token
_MPCO_NODE_SCALAR_TOKENS: dict[str, str] = {
    "pore_pressure": "pressure",
}

# Element-level (gauss / line_stations / nodal_forces) — by prefix.
# Continuum stress/strain tokens are plural (``stresses``/``strains``) —
# see the comment on ``_ELEMENT_CATEGORY_RESPONSE`` above.
_MPCO_ELEMENT_PREFIX_TOKENS: dict[str, str] = {
    "stress": "stresses",
    "strain": "strains",
    # Derived scalars — recorded via the raw tensor token, computed on
    # read (see apeGmsh.results._derived / _vocabulary.DERIVED_*_SCALARS).
    "von_mises_stress": "stresses",
    "tresca_stress": "stresses",
    "j2_stress": "stresses",
    "max_shear_stress": "stresses",
    "mean_stress": "stresses",
    "principal_stress": "stresses",
    "pressure_hydrostatic": "stresses",
    "j3_stress": "stresses",
    "lode_angle": "stresses",
    "stress_triaxiality": "stresses",
    "von_mises_shell": "stresses",   # derived from shell resultants ("stresses")
    "von_mises_strain": "strains",
    "volumetric_strain": "strains",
    "j2_strain": "strains",
    "max_shear_strain": "strains",
    "principal_strain": "strains",
    "equivalent_plastic_strain": "strains",
    "axial_force": "section.force",
    "shear": "section.force",
    "torsion": "section.force",
    "bending_moment": "section.force",
    "fiber_stress": "section.fiber.stress",
    "fiber_strain": "section.fiber.strain",
    "nodal_resisting_force": "globalForce",
    "nodal_resisting_force_local": "localForce",
    "nodal_resisting_moment": "globalForce",
    "nodal_resisting_moment_local": "localForce",
    # ZeroLength springs (Phase 11d). Per-spring forces live under
    # ``basicForce`` (not ``force`` — that token writes the global
    # element resisting force vector); per-spring deformations live
    # under ``deformation``. Read-side parity in
    # :data:`apeGmsh.opensees._response_catalog._SPRING_PREFIX_TO_KEYWORD`.
    "spring_force": "basicForce",
    "spring_deformation": "deformation",
}


def _canonical_to_mpco_node_token(canonical: str) -> str | None:
    """Map a canonical nodal name to an MPCO ``-N`` token."""
    if canonical in _MPCO_NODE_SCALAR_TOKENS:
        return _MPCO_NODE_SCALAR_TOKENS[canonical]
    # Strip axis suffix for vector components.
    if "_" in canonical:
        prefix, axis = canonical.rsplit("_", 1)
        if axis in ("x", "y", "z") and prefix in _MPCO_NODE_TOKENS:
            return _MPCO_NODE_TOKENS[prefix]
    return None


def _canonical_to_mpco_element_token(canonical: str) -> str | None:
    """Map a canonical element-level name to an MPCO ``-E`` token.

    Checks the longest prefix first so ``nodal_resisting_force_local``
    wins over ``nodal_resisting_force`` for ``nodal_resisting_force_local_x``.
    """
    sorted_prefixes = sorted(
        _MPCO_ELEMENT_PREFIX_TOKENS.items(),
        key=lambda kv: -len(kv[0]),
    )
    for prefix, token in sorted_prefixes:
        if canonical == prefix or canonical.startswith(prefix + "_"):
            return token
    return None


#: Read-side keywords a gauss record takes verbatim on the MPCO emit.
#: The MPCO reader finds every gauss-points ``stresses`` / ``strains``
#: canonical under the ``ON_ELEMENTS/<keyword>`` group that
#: :func:`~apeGmsh.opensees._response_catalog.gauss_keyword_for_canonical`
#: names, so the emit asks the same function. The generic prefix table
#: above is topology-blind: it sends a shell ``bending_moment_xx`` to the
#: beam ``section.force`` bucket and has no entry for ``membrane_force_*``
#: or ``transverse_shear_*``.
_MPCO_GAUSS_READ_SIDE_KEYWORDS: frozenset[str] = frozenset({"stresses", "strains"})


def _mpco_material_state_token(parent: str) -> str:
    """The ``material.<group>`` token the MPCO reader walks for ``parent``.

    ``parent`` is a material-state parent canonical (``damage``,
    ``equivalent_plastic_strain``, ``plastic_strain``). The group is the
    read side's keyword for it, and the token must be one of the group
    names :func:`mpco_gauss_group_aliases` lists, or the reader would
    never find what the emit wrote.
    """
    from apeGmsh.opensees._response_catalog import (
        gauss_keyword_for_canonical,
        mpco_gauss_group_aliases,
    )
    group = gauss_keyword_for_canonical(parent)
    token = f"material.{group}"
    if group is None or token not in mpco_gauss_group_aliases(group):
        raise RuntimeError(
            f"MPCO emit/read disagree on material state {parent!r}: the "
            f"reader's gauss group is {group!r} and does not list {token!r}."
        )
    return token


def _canonical_to_mpco_gauss_token(canonical: str, record_name: str) -> str:
    """Map a gauss-points canonical to its MPCO ``-E`` token, or raise.

    In order:

    1. The plastic-strain tensor (``plastic_strain_*``) and the scalars
       derived from it record ``material.plasticStrain``.
    2. Material-state scalars (``damage*``,
       ``equivalent_plastic_strain*``) record ``material.<group>``, the
       group the MPCO material-state reader walks.
    3. Continuum stress/strain and the shell resultants and generalized
       strains take the read side's keyword (``stresses`` / ``strains``).
    4. Derived stress/strain scalars keep the generic prefix table; they
       record the raw tensor, so only ``stresses`` / ``strains`` count.

    Anything else raises ``ValueError``: dropping it, or sending it to a
    bucket the reader never reads, would write an MPCO file whose gauss
    slab reads empty, with no warning. That includes the Truss
    ``axial_force``: MPCO writes its ``axialForce`` bucket with no
    ``GAUSS_IDS``, and the reader refuses it.
    """
    from apeGmsh._vocabulary import DERIVED_PLASTIC_STRAIN_SCALARS
    from apeGmsh.opensees._response_catalog import gauss_keyword_for_canonical
    from apeGmsh.results.readers._mpco_material_io import (
        parent_token_for_canonical,
    )
    if canonical in DERIVED_PLASTIC_STRAIN_SCALARS:
        return _mpco_material_state_token("plastic_strain")
    parent = parent_token_for_canonical(canonical)
    if parent is not None:
        return _mpco_material_state_token(parent)
    keyword = gauss_keyword_for_canonical(canonical)
    if keyword in _MPCO_GAUSS_READ_SIDE_KEYWORDS:
        return keyword
    if keyword == "axialForce":
        raise ValueError(
            f"Record {record_name!r} (category=gauss): component "
            f"{canonical!r} (the Truss axial force) cannot go through MPCO: "
            f"MPCO writes its axialForce bucket with no GAUSS_IDS and the "
            f"reader refuses it. Record it with spec.emit_recorders(...), "
            f"a .ladruno recorder, or ops.domain_capture(...)."
        )
    token = _canonical_to_mpco_element_token(canonical)
    if token not in _MPCO_GAUSS_READ_SIDE_KEYWORDS:
        raise ValueError(
            f"Record {record_name!r} (category=gauss): component "
            f"{canonical!r} has no MPCO -E token, so the MPCO file would "
            f"not record it. Drop it from the record, or capture it with "
            f"spec.emit_recorders(...) or ops.domain_capture(...)."
        )
    return token


def collect_mpco_tokens(records) -> tuple[list[str], list[str]]:
    """Return ``(node_tokens, element_tokens)`` deduplicated, in stable order.

    Modal records contribute ``modesOfVibration`` /
    ``modesOfVibrationRotational`` entries on the nodal side. A gauss
    component with no MPCO token raises ``ValueError``
    (:func:`_canonical_to_mpco_gauss_token`).
    """
    node_tokens: list[str] = []
    elem_tokens: list[str] = []
    seen_n: set[str] = set()
    seen_e: set[str] = set()

    def _add(target_list, target_seen, token):
        if token and token not in target_seen:
            target_seen.add(token)
            target_list.append(token)

    for rec in records:
        if rec.category == "modal":
            _add(node_tokens, seen_n, "modesOfVibration")
            # MPCO has a separate token for rotational modes; emit
            # both so users in 3D ndf=6 get them.
            _add(node_tokens, seen_n, "modesOfVibrationRotational")
            continue
        if rec.category == "nodes":
            for c in rec.components:
                _add(node_tokens, seen_n, _canonical_to_mpco_node_token(c))
            continue
        if rec.category == "gauss":
            for c in rec.components:
                _add(elem_tokens, seen_e,
                     _canonical_to_mpco_gauss_token(c, rec.name))
            continue
        # other element-level categories
        for c in rec.components:
            _add(elem_tokens, seen_e, _canonical_to_mpco_element_token(c))

    return node_tokens, elem_tokens


def _aggregate_cadence(records) -> tuple[str, float | int] | None:
    """Pick a single ``-T`` argument for the MPCO recorder.

    Strategy: if any record has ``dt=``, use the smallest dt (covers
    coarser cadences too). Else if any record has ``n_steps=``, use
    the smallest. Else None (every analysis step).
    """
    dts = [r.dt for r in records if getattr(r, "dt", None) is not None]
    if dts:
        return ("dt", min(dts))
    nss = [r.n_steps for r in records if getattr(r, "n_steps", None) is not None]
    if nss:
        return ("nsteps", min(nss))
    return None


def emit_mpco_tcl(
    records,
    *,
    output_dir: str = "",
    filename: str = "run.mpco",
) -> str:
    """Emit a single ``recorder mpco ...`` Tcl command line."""
    file_path = _build_mpco_path(output_dir, filename)
    node_toks, elem_toks = collect_mpco_tokens(records)
    parts = ["recorder", "mpco", file_path]
    if node_toks:
        parts.append("-N")
        parts.extend(node_toks)
    if elem_toks:
        parts.append("-E")
        parts.extend(elem_toks)
    cad = _aggregate_cadence(records)
    if cad is not None:
        kind, val = cad
        if kind == "dt":
            parts.extend(["-T", "dt", f"{val:.10g}"])
        else:
            parts.extend(["-T", "nsteps", str(int(val))])
    return " ".join(parts)


def emit_mpco_python(
    records,
    *,
    output_dir: str = "",
    filename: str = "run.mpco",
) -> str:
    """Emit a single ``ops.recorder('mpco', ...)`` Python call."""
    file_path = _build_mpco_path(output_dir, filename)
    node_toks, elem_toks = collect_mpco_tokens(records)
    args: list[str] = ["'mpco'", repr(file_path)]
    if node_toks:
        args.append("'-N'")
        args.extend(repr(t) for t in node_toks)
    if elem_toks:
        args.append("'-E'")
        args.extend(repr(t) for t in elem_toks)
    cad = _aggregate_cadence(records)
    if cad is not None:
        kind, val = cad
        if kind == "dt":
            args.extend(["'-T'", "'dt'", f"{val:.10g}"])
        else:
            args.extend(["'-T'", "'nsteps'", str(int(val))])
    return f"ops.recorder({', '.join(args)})"


def mpco_ops_args(
    records,
    *,
    output_dir: str = "",
    filename: str = "run.mpco",
) -> tuple:
    """Argument tuple for a live ``ops.recorder('mpco', *args)`` call.

    Mirrors :func:`emit_mpco_python` exactly, except it returns Python
    objects (strings, ints, floats) instead of the equivalent source
    code. The leading ``'mpco'`` argument is included so the caller
    can splat directly:

    >>> import openseespy.opensees as ops
    >>> ops.recorder(*mpco_ops_args(spec.records))   # doctest: +SKIP
    """
    file_path = _build_mpco_path(output_dir, filename)
    node_toks, elem_toks = collect_mpco_tokens(records)
    args: list = ["mpco", file_path]
    if node_toks:
        args.append("-N")
        args.extend(node_toks)
    if elem_toks:
        args.append("-E")
        args.extend(elem_toks)
    cad = _aggregate_cadence(records)
    if cad is not None:
        kind, val = cad
        if kind == "dt":
            args.extend(["-T", "dt", float(val)])
        else:
            args.extend(["-T", "nsteps", int(val)])
    return tuple(args)


def _build_mpco_path(output_dir: str, filename: str) -> str:
    if not output_dir:
        return filename
    sep = "" if output_dir.endswith(("/", "\\")) else "/"
    return f"{output_dir}{sep}{filename}"
