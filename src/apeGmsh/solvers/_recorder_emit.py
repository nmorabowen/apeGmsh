"""Emission helpers — ``ResolvedRecorderSpec`` to OpenSees recorder commands.

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
from dataclasses import dataclass
from typing import Iterable, Optional

from ._recorder_specs import ResolvedRecorderRecord


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
    """
    from ._element_response import gauss_keyword_for_canonical
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
    return next(iter(keywords))


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
    from ._element_response import gauss_keyword_for_canonical
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


# =====================================================================
# Logical recorder spec — backend-agnostic intermediate
# =====================================================================

@dataclass(frozen=True)
class LogicalRecorder:
    """One emitted OpenSees recorder, in backend-agnostic form.

    The Tcl and Python formatters take this and write the right
    command syntax.
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


# =====================================================================
# Public emission entry points
# =====================================================================

def emit_logical(
    record: ResolvedRecorderRecord,
    *,
    output_dir: str = "",
    file_format: str = "out",
) -> list[LogicalRecorder]:
    """Translate one resolved record into 0+ logical recorder specs.

    Some records produce multiple logical recorders (e.g. a node
    record with both displacement and velocity components). Some
    produce zero (modal, fibers, layers — stubbed for Phase 5).
    """
    if record.category == "nodes":
        return list(_emit_nodes(record, output_dir, file_format))
    if record.category in ("elements", "gauss", "line_stations"):
        return list(_emit_element_simple(record, output_dir, file_format))
    if record.category in ("fibers", "layers", "modal"):
        # Deferred — Phase 5 emits a TODO comment via the formatter
        # by returning an empty list. The formatter inserts a comment
        # when it sees an unsupported category.
        return []
    return []


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
    args.extend(repr(t) for t in rec.response_tokens)
    line = f"ops.recorder({', '.join(args)})"
    if rec.comment:
        line += f"  # {rec.comment}"
    return line


# =====================================================================
# Per-category emission
# =====================================================================

def _emit_nodes(
    rec: ResolvedRecorderRecord,
    output_dir: str,
    file_format: str,
) -> Iterable[LogicalRecorder]:
    """One logical recorder per ``(ops_type, dof_set)``."""
    if rec.node_ids is None or rec.node_ids.size == 0:
        return

    if rec.n_steps is not None and rec.dt is None:
        warnings.warn(
            f"Record {rec.name!r}: ``n_steps={rec.n_steps}`` is not "
            f"directly supported by the OpenSees Tcl/Py recorder. "
            f"Emitting as every-step. Use ``dt=`` for time-based "
            f"cadence, or use domain capture (Phase 7) for "
            f"step-based cadence.",
            stacklevel=4,
        )

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
            output_dir, rec.name, ops_type, file_format,
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
    """
    if rec.element_ids is None or rec.element_ids.size == 0:
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
    else:
        response_str = _ELEMENT_CATEGORY_RESPONSE.get(rec.category)
        if response_str is None:
            return

    response_tokens = tuple(response_str.split())
    target_ids = tuple(int(e) for e in rec.element_ids)
    file_path = _build_file_path(
        output_dir, rec.name, rec.category, file_format,
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
) -> str:
    """Build the recorder output file path with the right extension."""
    base = f"{record_name}_{suffix}"
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
) -> list[str]:
    """Emit Tcl recorder commands for an iterable of resolved records."""
    lines: list[str] = []
    for rec in records:
        if rec.category in _DEFERRED_CATEGORIES:
            lines.append(_tcl_unsupported_comment(rec))
            continue
        sub = list(emit_logical(
            rec, output_dir=output_dir, file_format=file_format,
        ))
        # Empty sub means "no targets" (e.g. zero IDs) — silently skip.
        for lr in sub:
            lines.append(format_tcl(lr))
    return lines


def emit_spec_python(
    records: Iterable[ResolvedRecorderRecord],
    *,
    output_dir: str = "",
    file_format: str = "out",
) -> list[str]:
    """Emit ``ops.recorder(...)`` calls for an iterable of resolved records."""
    lines: list[str] = []
    for rec in records:
        if rec.category in _DEFERRED_CATEGORIES:
            lines.append(_py_unsupported_comment(rec))
            continue
        sub = list(emit_logical(
            rec, output_dir=output_dir, file_format=file_format,
        ))
        for lr in sub:
            lines.append(format_python(lr))
    return lines


def _tcl_unsupported_comment(rec: ResolvedRecorderRecord) -> str:
    return (
        f";# TODO Phase 5+: recorder category {rec.category!r} "
        f"(record {rec.name!r}) — emission deferred. "
        f"Use domain capture (Phase 7) or MPCO bridge (Phase 8)."
    )


def _py_unsupported_comment(rec: ResolvedRecorderRecord) -> str:
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
    "von_mises_stress": "stresses",       # derived; recorded via "stresses"
    "principal_stress": "stresses",
    "pressure_hydrostatic": "stresses",
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


def collect_mpco_tokens(records) -> tuple[list[str], list[str]]:
    """Return ``(node_tokens, element_tokens)`` deduplicated, in stable order.

    Modal records contribute ``modesOfVibration`` /
    ``modesOfVibrationRotational`` entries on the nodal side.
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
        # element-level categories
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


def _build_mpco_path(output_dir: str, filename: str) -> str:
    if not output_dir:
        return filename
    sep = "" if output_dir.endswith(("/", "\\")) else "/"
    return f"{output_dir}{sep}{filename}"
