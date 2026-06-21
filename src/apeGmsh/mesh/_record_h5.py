"""Symmetric compound-dtype helpers for the broker's record-set H5 writers.

Master plan
([architecture/phase-8-untangle.md §3](../opensees/architecture/phase-8-untangle.md))
specifies that every record-set group in ``model.h5`` shares the same
outer 4-field compound:

* ``target_kind``  — vlen utf-8, one of ``"node"`` / ``"element"`` / ``"pg"``
* ``target``       — vlen utf-8, tag (str) or PG name
* ``payload_kind`` — vlen utf-8, record subtype (e.g. ``"rigid_beam"``)
* ``payload``      — compound, per-kind nested dtype

:func:`make_record_dtype` returns the outer compound for any payload
dtype.  Per-record-type payload dtype factories live alongside:
:func:`node_pair_payload_dtype`, :func:`node_group_payload_dtype`,
:func:`interpolation_payload_dtype`, :func:`surface_coupling_payload_dtype`,
:func:`node_to_surface_payload_dtype`, :func:`nodal_load_payload_dtype`,
:func:`element_load_payload_dtype`, :func:`sp_payload_dtype`,
:func:`mass_payload_dtype`.

The actual broker-side writer that builds rows of these dtypes lives in
:mod:`apeGmsh.mesh._femdata_h5_io` (Phase 8.5).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


__all__ = [
    "element_load_payload_dtype",
    "interpolation_payload_dtype",
    "make_record_dtype",
    "mass_payload_dtype",
    "node_group_payload_dtype",
    "node_pair_payload_dtype",
    "node_to_surface_payload_dtype",
    "nodal_load_payload_dtype",
    "sp_payload_dtype",
    "surface_coupling_payload_dtype",
]


def _utf8() -> Any:
    """Return the h5py variable-length UTF-8 string dtype."""
    import h5py
    return h5py.string_dtype(encoding="utf-8")


def _vlen(scalar_dtype: Any) -> Any:
    """Return an h5py variable-length array dtype for the given scalar."""
    import h5py
    return h5py.vlen_dtype(np.dtype(scalar_dtype))


def make_record_dtype(payload_dtype: np.dtype) -> np.dtype:
    """Return the 4-field outer compound for a record-set dataset.

    Every record-set dataset on disk has rows of this dtype.  The
    ``payload`` field's nested compound type is what differs from
    kind to kind; the outer three string fields are universal so a
    viewer can dispatch on ``payload_kind`` with one reader.
    """
    return np.dtype([
        ("target_kind", _utf8()),
        ("target", _utf8()),
        ("payload_kind", _utf8()),
        ("payload", payload_dtype),
    ])


# ---------------------------------------------------------------------------
# Constraint payload dtypes
# ---------------------------------------------------------------------------


def node_pair_payload_dtype() -> np.dtype:
    """Payload dtype for :class:`NodePairRecord`.

    Used by kinds ``equal_dof``, ``rigid_beam``, ``rigid_beam_stiff``,
    ``rigid_rod``, ``penalty``.  ``offset`` is filled with NaN when
    the record carries no offset (e.g. ``equal_dof``);
    ``penalty_stiffness`` is NaN when not a ``penalty`` record.

    The ``name`` field (added in neutral schema 2.5.0) carries the
    pre-mesh declaration name (e.g. ``"wall_pin"``).  Empty string
    means no name was set.  Old 2.4.0 files lack this field and the
    reader probes it presence-detected.
    """
    return np.dtype([
        ("master_node", np.int64),
        ("slave_node", np.int64),
        ("dofs", _vlen(np.int64)),
        ("offset", np.float64, (3,)),
        ("penalty_stiffness", np.float64),
        ("name", _utf8()),
    ])


def node_group_payload_dtype() -> np.dtype:
    """Payload dtype for :class:`NodeGroupRecord`.

    Used by kinds ``rigid_diaphragm``, ``rigid_body``,
    ``kinematic_coupling``.  ``offsets`` is stored as a flat
    ``vlen-float64`` of length ``3 * n_slaves`` (reshape on read);
    ``plane_normal`` is NaN-filled when absent.

    The ``name`` field (added in neutral schema 2.5.0) carries the
    pre-mesh declaration name.  Old 2.4.0 files lack this field.
    """
    return np.dtype([
        ("master_node", np.int64),
        ("slave_nodes", _vlen(np.int64)),
        ("dofs", _vlen(np.int64)),
        ("offsets", _vlen(np.float64)),
        ("plane_normal", np.float64, (3,)),
        ("name", _utf8()),
        # Fork coupling knobs (schema 2.12.0; kinematic_coupling only).
        *_coupling_control_fields(),
    ])


def interpolation_payload_dtype() -> np.dtype:
    """Payload dtype for :class:`InterpolationRecord`.

    Used by kinds ``tie``, ``distributing``, ``embedded``.
    ``projected_point`` and ``parametric_coords`` are NaN-filled when
    absent.

    The ``name`` field (added in neutral schema 2.5.0) carries the
    pre-mesh declaration name.  Old 2.4.0 files lack this field.

    Schema 2.8.0 adds five typed columns so the
    ASDEmbeddedNodeElement options carried on the record round-trip
    through H5: ``stiffness`` (``-K``, float64), ``stiffness_p``
    (``-KP``, float64, NaN when ``None``) + ``has_stiffness_p``
    (uint8 presence flag — NaN isn't a clean sentinel because 0 is
    valid penalty data), ``rotational`` (``-rot``, uint8 0/1),
    ``pressure`` (``-p``, uint8 0/1), and ``excess`` (float64, NaN
    when ``None`` — populated by ``resolve_embedded`` as the
    barycentric excess of the slave w.r.t. its host).  Pre-2.8.0
    files lack these columns; the reader probes presence via
    ``p.dtype.names`` and falls back to dataclass defaults
    (``stiffness=1e18``, the rest ``None``/``False``).
    """
    return np.dtype([
        ("slave_node", np.int64),
        ("master_nodes", _vlen(np.int64)),
        ("weights", _vlen(np.float64)),
        ("dofs", _vlen(np.int64)),
        ("projected_point", np.float64, (3,)),
        ("parametric_coords", np.float64, (2,)),
        ("name", _utf8()),
        # ASDEmbeddedNodeElement options (schema 2.8.0)
        ("stiffness", np.float64),
        ("stiffness_p", np.float64),
        ("has_stiffness_p", np.uint8),
        ("rotational", np.uint8),
        ("pressure", np.uint8),
        ("excess", np.float64),
        # Enforcement route (ADR 0068): "penalty" | "penalty_al" |
        # "equation". Pre-2.14.0 files lack this column; the reader probes
        # ``p.dtype.names`` and falls back to "penalty".
        ("enforce", _utf8()),
        # Fork coupling knobs (schema 2.12.0; distributing only).
        *_coupling_control_fields(),
    ])


def _coupling_control_fields() -> list[tuple]:
    """Structured-dtype columns for :class:`CouplingControl` (schema 2.12.0;
    host auto-scalers added in 2.13.0).

    Carries the explicit fork-coupling knobs so they round-trip:
    ``cpl_has`` (uint8 presence flag — distinguishes "no control / use
    fork defaults" from "control with all-default values", and lets a
    NaN-free reader reconstruct ``None`` vs a real object), ``cpl_k`` /
    ``cpl_kr`` / ``cpl_dtcr`` (float64, NaN when the knob is unset),
    ``cpl_enforce`` (uint8: 0=penalty, 1=al), ``cpl_absolute`` (uint8
    0/1). Pre-2.12.0 files lack these columns; the reader probes
    ``p.dtype.names`` and falls back to ``control=None``.

    Schema 2.13.0 adds the host auto-scalers: ``cpl_k_auto`` (uint8 0/1 —
    ``k="auto"``; ``cpl_k`` is NaN then), ``cpl_k_alpha`` (float64, NaN
    when unset), ``cpl_host`` (int64 **FEM element id** — stable across
    emits, deliberately NOT the emit-time ops tag; ``-1`` = none) and
    ``cpl_wcap`` (float64, NaN when unset). 2.12.0 files lack these four;
    the reader probes ``cpl_k_auto`` and falls back to the v1 knobs only.
    """
    return [
        ("cpl_has", np.uint8),
        ("cpl_k", np.float64),
        ("cpl_kr", np.float64),
        ("cpl_enforce", np.uint8),
        ("cpl_dtcr", np.float64),
        ("cpl_absolute", np.uint8),
        # Host auto-scalers (schema 2.13.0).
        ("cpl_k_auto", np.uint8),
        ("cpl_k_alpha", np.float64),
        ("cpl_host", np.int64),
        ("cpl_wcap", np.float64),
    ]


def surface_coupling_payload_dtype() -> np.dtype:
    """Payload dtype for :class:`SurfaceCouplingRecord`.

    Used by kinds ``tied_contact``, ``mortar``.  The dense
    ``mortar_operator`` matrix is stored as a flat
    ``vlen-float64`` whose row-major shape is recorded in
    ``mortar_operator_shape`` (``(rows, cols)``); ``(0, 0)`` and an
    empty flat array signal "no mortar operator".

    The per-slave-node ``slave_records`` list (one
    :class:`InterpolationRecord` per slave for ``tied_contact``;
    also carried by ``mortar``) **is** persisted, CSR-flattened
    across the ``sr_*`` fields, so a ``tied_contact`` (which has no
    ``mortar_operator``) round-trips losslessly instead of decoding
    to a hollow coupling.  Older files lack the ``sr_*`` fields; the
    reader detects their absence structurally (by payload-field
    signature, not schema version) and yields ``slave_records=[]``
    (the legacy behaviour) for them.

    Schema 2.8.0 mirrors the ``interpolation_payload_dtype``
    extension into the sr_* lane so the ASDEmbeddedNodeElement
    options on each slave record round-trip too — without them, a
    tied_contact / mortar whose slave records carry a custom ``-K``
    silently reverts to the dataclass defaults on read, then the
    bridge emits ``-K 1e18`` instead.  Pre-2.8.0 files lack
    ``sr_stiffness`` / ``sr_stiffness_p`` / ``sr_has_stiffness_p`` /
    ``sr_rotational`` / ``sr_pressure`` / ``sr_excess``; presence is
    probed structurally (same pattern as the original ``sr_*``
    detection).

    Schema 2.12.0 likewise mirrors the ``cpl_*`` CouplingControl
    columns (see :func:`_coupling_control_fields`) into the sr_* lane
    as per-slave vlen arrays, so a slave record carrying explicit
    fork-coupling knobs round-trips.  Pre-2.12.0 files lack the
    ``sr_cpl_*`` fields; presence is probed structurally.  Schema
    2.13.0 extends the mirror with the host auto-scalers
    (``sr_cpl_k_auto`` / ``sr_cpl_k_alpha`` / ``sr_cpl_host`` /
    ``sr_cpl_wcap``), probed independently via ``sr_cpl_k_auto``.
    """
    return np.dtype([
        ("master_nodes", _vlen(np.int64)),
        ("slave_nodes", _vlen(np.int64)),
        ("dofs", _vlen(np.int64)),
        ("mortar_operator_shape", np.int64, (2,)),
        ("mortar_operator", _vlen(np.float64)),
        # slave_records (per-slave InterpolationRecord) — CSR-flattened
        # so the ragged list round-trips without a nested dtype.
        ("sr_slave_nodes", _vlen(np.int64)),     # (n_sr,)
        ("sr_master_counts", _vlen(np.int64)),   # (n_sr,) split sizes
        ("sr_master_nodes", _vlen(np.int64)),    # concat, split by counts
        ("sr_weights", _vlen(np.float64)),       # concat, same split
        ("sr_dof_counts", _vlen(np.int64)),      # (n_sr,) split sizes
        ("sr_dofs", _vlen(np.int64)),            # concat, split by dof_counts
        ("sr_projected", _vlen(np.float64)),     # 3*n_sr (NaN per missing)
        ("sr_parametric", _vlen(np.float64)),    # 2*n_sr (NaN per missing)
        # name (neutral schema 2.5.0): pre-mesh declaration name.
        # Old 2.4.0 files lack this field; reader probes presence.
        ("name", _utf8()),
        # ASDEmbeddedNodeElement options per slave record (schema 2.8.0)
        ("sr_stiffness", _vlen(np.float64)),       # (n_sr,)
        ("sr_stiffness_p", _vlen(np.float64)),     # (n_sr,) NaN when None
        ("sr_has_stiffness_p", _vlen(np.uint8)),   # (n_sr,) 0/1 presence
        ("sr_rotational", _vlen(np.uint8)),        # (n_sr,) 0/1
        ("sr_pressure", _vlen(np.uint8)),          # (n_sr,) 0/1
        ("sr_excess", _vlen(np.float64)),          # (n_sr,) NaN when None
        # Enforcement route per slave record (ADR 0068, schema 2.14.0):
        # uint8 code 0=penalty 1=penalty_al 2=equation. Pre-2.14.0 files
        # lack this; reader probes presence and falls back to penalty.
        ("sr_enforce", _vlen(np.uint8)),           # (n_sr,)
        # CouplingControl per slave record (schema 2.12.0 mirror of
        # the cpl_* columns; see _coupling_control_fields).
        ("sr_cpl_has", _vlen(np.uint8)),           # (n_sr,) 0/1 presence
        ("sr_cpl_k", _vlen(np.float64)),           # (n_sr,) NaN when unset
        ("sr_cpl_kr", _vlen(np.float64)),          # (n_sr,) NaN when unset
        ("sr_cpl_enforce", _vlen(np.uint8)),       # (n_sr,) 0=penalty 1=al
        ("sr_cpl_dtcr", _vlen(np.float64)),        # (n_sr,) NaN when unset
        ("sr_cpl_absolute", _vlen(np.uint8)),      # (n_sr,) 0/1
        # Host auto-scalers per slave record (schema 2.13.0 mirror).
        ("sr_cpl_k_auto", _vlen(np.uint8)),        # (n_sr,) 0/1
        ("sr_cpl_k_alpha", _vlen(np.float64)),     # (n_sr,) NaN when unset
        ("sr_cpl_host", _vlen(np.int64)),          # (n_sr,) FEM eid, -1=none
        ("sr_cpl_wcap", _vlen(np.float64)),        # (n_sr,) NaN when unset
    ])


def node_to_surface_payload_dtype() -> np.dtype:
    """Payload dtype for :class:`NodeToSurfaceRecord`.

    Used by kinds ``node_to_surface`` and ``node_to_surface_spring``.

    Stores the high-level fields directly (``master_node``,
    ``slave_nodes``, ``phantom_nodes``, ``phantom_coords``, ``dofs``).
    The nested ``rigid_link_records`` and ``equal_dof_records`` lists
    of :class:`NodePairRecord` are NOT stored — they are derivable
    from the high-level fields by the same expansion the resolver
    runs (one rigid_beam link per phantom + one equal_dof from each
    phantom to its slave for translational DOFs).
    """
    return np.dtype([
        ("master_node", np.int64),
        ("slave_nodes", _vlen(np.int64)),
        ("phantom_nodes", _vlen(np.int64)),
        ("phantom_coords", _vlen(np.float64)),
        ("dofs", _vlen(np.int64)),
        # name (neutral schema 2.5.0): pre-mesh declaration name.
        # Old 2.4.0 files lack this field; reader probes presence.
        ("name", _utf8()),
    ])


def reinforce_tie_payload_dtype() -> np.dtype:
    """Payload dtype for :class:`ReinforceTieRecord` (neutral schema 2.15.0).

    One resolved ``LadrunoEmbeddedRebar`` tie (g.reinforce / ADR 20 R2b):
    a 1-to-N coupling of one rebar node to the host element's nodes — so
    this is simpler than ``surface_coupling_payload_dtype`` (no CSR-of-CSR;
    ``host_nodes`` + the parallel ``weights`` are single vlen arrays).
    Optional scalars use the NaN sentinel (floats) / ``""`` (strings); a
    ``has_*`` flag disambiguates "absent" from "empty" for the vlen /
    fixed-shape geometric fields.  Stored in a dedicated ``/reinforce_ties``
    group (NOT under ``/constraints/``, whose subset-match reader dispatch
    would mis-route it).
    """
    return np.dtype([
        ("rebar_node", np.int64),
        ("host_nodes", _vlen(np.int64)),     # the -shape host node list
        ("weights", _vlen(np.float64)),      # Nᵢ(ξ), parallel to host_nodes
        ("has_weights", np.uint8),           # 0 ⇒ weights is None
        ("direction", np.float64, (3,)),     # unit bar axis d̂ (NaN ⇒ None)
        ("has_direction", np.uint8),
        ("bond_scale", np.float64),          # π·d_b·L_trib  (NaN ⇒ None)
        ("bond", _utf8()),                   # LadrunoBondSlip name ("" ⇒ None)
        ("perfect", np.float64),             # perfect-bond kAxial (NaN ⇒ None)
        ("kt", np.float64),                  # transverse penalty (NaN ⇒ None)
        ("kt_alpha", np.float64),            # (NaN ⇒ None)
        ("enforce", _utf8()),                # "penalty" | ...
        ("bipenalty", np.uint8),             # 0/1
        ("dtcr", np.float64),                # (NaN ⇒ None)
        ("excess", np.float64),              # inverse-map diag (NaN ⇒ None)
        ("in_bounds", np.uint8),             # 0/1
        ("name", _utf8()),                   # pre-mesh declaration ("" ⇒ None)
    ])


# ---------------------------------------------------------------------------
# Load payload dtypes
# ---------------------------------------------------------------------------


def nodal_load_payload_dtype() -> np.dtype:
    """Payload dtype for :class:`NodalLoadRecord`.

    ``force_xyz`` / ``moment_xyz`` are stored as ``(3,)`` float64
    arrays; absent components are NaN-filled.  The pattern name lives
    in the outer ``target_kind`` / ``target`` layer — every record
    set is one pattern per dataset
    (``/loads/nodal/{pattern}/``).

    The ``name`` field (added in neutral schema 2.5.0) carries the
    pre-mesh declaration name.  Old 2.4.0 files lack this field.
    """
    return np.dtype([
        ("node_id", np.int64),
        ("force_xyz", np.float64, (3,)),
        ("moment_xyz", np.float64, (3,)),
        ("name", _utf8()),
    ])


def element_load_payload_dtype() -> np.dtype:
    """Payload dtype for :class:`ElementLoadRecord`.

    The ``params: dict`` field is arbitrary (different keys per
    ``load_type`` — ``beamUniform`` carries wx/wy/wz,
    ``surfacePressure`` carries pressure, …).  No fixed typed
    compound captures the union, so we serialise it to a JSON string
    in ``params_json``.  Documented compromise; consumers parse the
    JSON or interpret per ``load_type``.

    The ``name`` field (added in neutral schema 2.5.0) carries the
    pre-mesh declaration name.  Old 2.4.0 files lack this field.
    """
    return np.dtype([
        ("element_id", np.int64),
        ("load_type", _utf8()),
        ("params_json", _utf8()),
        ("name", _utf8()),
    ])


def sp_payload_dtype() -> np.dtype:
    """Payload dtype for :class:`SPRecord` (single-point constraint).

    ``is_homogeneous`` is stored as int64 (0 / 1) since HDF5
    compound types don't have a native boolean.

    The ``name`` field (added in neutral schema 2.5.0) carries the
    pre-mesh declaration name.  Old 2.4.0 files lack this field.
    """
    return np.dtype([
        ("node_id", np.int64),
        ("dof", np.int64),
        ("value", np.float64),
        ("is_homogeneous", np.int64),
        ("name", _utf8()),
    ])


# ---------------------------------------------------------------------------
# Mass payload dtype
# ---------------------------------------------------------------------------


def mass_payload_dtype() -> np.dtype:
    """Payload dtype for :class:`MassRecord`.

    Always six components: ``(mx, my, mz, Ixx, Iyy, Izz)``.  The
    OpenSees bridge slices to ``ndf`` when emitting ``mass``
    commands; the broker zone stores the full 6-vector so a
    consumer can pick whichever components it needs.

    The ``name`` field (added in neutral schema 2.5.0) carries the
    pre-mesh declaration name.  Old 2.4.0 files lack this field.
    """
    return np.dtype([
        ("node_id", np.int64),
        ("mass", np.float64, (6,)),
        ("name", _utf8()),
    ])
