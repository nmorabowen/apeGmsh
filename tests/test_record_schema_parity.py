"""Reflection-driven parity between record dataclasses and their H5
payload dtypes.

This is the regression test class that would have caught the gap
fixed by PR #337 (ADR 0035 follow-up): :class:`InterpolationRecord`
gained four ASDEmbeddedNodeElement option fields
(``stiffness`` / ``stiffness_p`` / ``rotational`` / ``pressure``)
plus ``excess``, but the kernel H5 schema
:func:`interpolation_payload_dtype` silently dropped them.  Round-
trip through ``g.save()`` → ``FEMData.from_h5(...)`` reverted those
fields to dataclass defaults; the bridge then emitted ``-K 1e18``
unconditionally regardless of what the user set.

The fix landed in #337 by extending the dtype.  This file adds the
*structural* check that locks the invariant for every record type
going forward: every dataclass field appears as a payload dtype
column, modulo a whitelist of fields whose absence from the
payload dtype is documented and intentional (e.g. CSR-flattened
sub-records, group-hierarchy keys).

The pattern is symmetric: when a new field is added to a record
dataclass, this test fails with a message that names the missing
column.  The author then either adds the column (the usual case)
or adds the field to the whitelist with a one-line reason (the
rare intentional case).  Whitelist additions are reviewed.
"""
from __future__ import annotations

import dataclasses

import pytest

from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
    NodeToSurfaceRecord,
    SurfaceCouplingRecord,
)
from apeGmsh._kernel.records._loads import (
    ElementLoadRecord,
    NodalLoadRecord,
    SPRecord,
)
from apeGmsh._kernel.records._masses import MassRecord
from apeGmsh.mesh._record_h5 import (
    element_load_payload_dtype,
    interpolation_payload_dtype,
    mass_payload_dtype,
    nodal_load_payload_dtype,
    node_group_payload_dtype,
    node_pair_payload_dtype,
    node_to_surface_payload_dtype,
    sp_payload_dtype,
    surface_coupling_payload_dtype,
)


# ---------------------------------------------------------------------------
# Record dataclass ↔ payload dtype factory map.
# ---------------------------------------------------------------------------

# One row per (record_cls, dtype_factory) pair.  Every concrete
# record dataclass exported from :mod:`apeGmsh._kernel.records`
# MUST appear here; if you add a new record subclass, add its
# sibling dtype factory too — and if the dataclass introduces
# fields that have no payload column, document them in
# :data:`PAYLOAD_WHITELIST` below.
RECORD_TO_DTYPE: dict[type, callable] = {
    NodePairRecord:       node_pair_payload_dtype,
    NodeGroupRecord:      node_group_payload_dtype,
    InterpolationRecord:  interpolation_payload_dtype,
    SurfaceCouplingRecord: surface_coupling_payload_dtype,
    NodeToSurfaceRecord:  node_to_surface_payload_dtype,
    NodalLoadRecord:      nodal_load_payload_dtype,
    ElementLoadRecord:    element_load_payload_dtype,
    SPRecord:             sp_payload_dtype,
    MassRecord:           mass_payload_dtype,
}


# ---------------------------------------------------------------------------
# Per-record whitelist of dataclass fields intentionally absent from
# the payload dtype.  Every entry MUST carry a non-empty reason.
# ---------------------------------------------------------------------------

# Fields carried by the OUTER compound dtype that wraps every
# payload (``make_record_dtype`` adds ``target_kind`` / ``target`` /
# ``payload_kind`` / ``payload``); record subclasses thread ``kind``
# into ``payload_kind`` and stamp ``name`` separately.  Both are
# exempt from the per-payload check for every record class.
GLOBAL_EXEMPT: frozenset[str] = frozenset({"kind", "name"})

# Per-record exempt fields.  Adding to this dict is a deliberate
# review action: every entry's reason is the contract the writer is
# relying on (and the contract a reviewer can hold the author to).
PAYLOAD_WHITELIST: dict[type, dict[str, str]] = {
    NodeGroupRecord: {
        # The CouplingControl object is persisted decomposed into the
        # cpl_* columns (cpl_has / cpl_k / cpl_kr / cpl_enforce /
        # cpl_dtcr / cpl_absolute, schema 2.12.0; + cpl_k_auto /
        # cpl_k_alpha / cpl_host / cpl_wcap host auto-scalers, schema
        # 2.13.0); _encode_control / _decode_control in
        # apeGmsh.mesh._femdata_h5_io reconstruct the object (or None)
        # on round-trip.
        "control":
            "persisted decomposed into the cpl_* columns "
            "(see _encode_control / _decode_control)",
    },
    InterpolationRecord: {
        "control":
            "persisted decomposed into the cpl_* columns "
            "(see _encode_control / _decode_control)",
    },
    SurfaceCouplingRecord: {
        # The list of per-slave-node InterpolationRecord objects is
        # persisted via the CSR-flattened ``sr_*`` lane in
        # surface_coupling_payload_dtype (sr_slave_nodes,
        # sr_master_counts, sr_master_nodes, sr_weights,
        # sr_dof_counts, sr_dofs, sr_projected, sr_parametric, and
        # the schema-2.8.0 sr_stiffness / sr_stiffness_p /
        # sr_has_stiffness_p / sr_rotational / sr_pressure /
        # sr_excess).  _decode_surface_coupling reconstructs the
        # list on read; no nested compound column is needed.  The
        # sr_* lane is itself parity-checked against
        # interpolation_payload_dtype by
        # test_surface_coupling_sr_lane_mirrors_interpolation_dtype.
        "slave_records":
            "persisted via the sr_* CSR-flattened lane "
            "(see test_surface_coupling_sr_lane_mirrors_interpolation_dtype)",
    },
    NodeToSurfaceRecord: {
        # Both sub-record lists are deterministically re-derived on
        # decode from the persisted high-level fields (master_node
        # + slave_nodes + phantom_nodes + phantom_coords + dofs)
        # via the same expansion the resolver runs — one
        # rigid_beam per master→phantom plus one equalDOF per
        # phantom→slave (translational DOFs only).  Persisting
        # them would duplicate the resolver's logic and let the
        # two diverge.  See _decode_node_to_surface in
        # apeGmsh.mesh._femdata_h5_io.
        "rigid_link_records":
            "re-derived on decode from high-level fields",
        "equal_dof_records":
            "re-derived on decode from high-level fields",
    },
    NodalLoadRecord: {
        # The pattern name is encoded as the H5 group key under
        # /loads/nodal/{pattern}/ — one dataset per pattern.  The
        # writer iterates over patterns and the reader reads the
        # group name back into the LoadRecord.pattern field, so
        # pattern survives round-trip without a payload column.
        "pattern": "stored in H5 group hierarchy (/loads/nodal/{pattern}/)",
    },
    ElementLoadRecord: {
        "pattern": "stored in H5 group hierarchy (/loads/element/{pattern}/)",
        # ``params`` is an arbitrary dict whose schema varies per
        # ``load_type`` (beamUniform / surfacePressure / bodyForce
        # each have different keys).  No fixed compound dtype
        # captures the union, so the encoder JSON-serialises it
        # into the ``params_json`` payload column; the decoder
        # parses the JSON back into a dict.  Documented compromise:
        # see _encode_element_load / _decode_element_load.
        "params":
            "JSON-serialised into the params_json payload column",
    },
    SPRecord: {
        "pattern": "stored in H5 group hierarchy (/loads/sp/{pattern}/)",
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record_cls,dtype_factory",
    list(RECORD_TO_DTYPE.items()),
    ids=lambda v: v.__name__,
)
def test_record_fields_appear_in_payload_dtype(
    record_cls: type, dtype_factory,
) -> None:
    """Every dataclass field on ``record_cls`` is either a column of
    ``dtype_factory()`` or appears in :data:`PAYLOAD_WHITELIST`
    with a documented reason.

    This is the parity check that would have failed at PR #334
    merge time and prevented the silent data loss fixed by #337:
    if InterpolationRecord gains a ``stiffness`` field without a
    matching dtype column, this test surfaces the gap by name.
    """
    dtype_names = set(dtype_factory().names or ())
    record_field_names = {f.name for f in dataclasses.fields(record_cls)}

    whitelist = PAYLOAD_WHITELIST.get(record_cls, {})
    expected = record_field_names - GLOBAL_EXEMPT - set(whitelist.keys())
    missing = expected - dtype_names

    assert not missing, (
        f"{record_cls.__name__} fields {sorted(missing)} are not "
        f"persisted by {dtype_factory.__name__}() and are not in the "
        f"PAYLOAD_WHITELIST.\n\n"
        f"If the absence is intentional, add a one-line reason to "
        f"PAYLOAD_WHITELIST[{record_cls.__name__}] — the reviewer "
        f"will hold you to the contract you write there.\n\n"
        f"If it is unintentional, this is the silent-data-loss bug "
        f"fixed by PR #337 (ADR 0035 follow-up): on H5 round-trip "
        f"the field will snap back to its dataclass default, "
        f"breaking downstream code that depends on the user's value."
    )


# ---------------------------------------------------------------------------
# sr_* lane ↔ interpolation_payload_dtype mapping for tied_contact /
# mortar slave_records.  CSR flattening renames a few columns (e.g.
# ``slave_node`` becomes ``sr_slave_nodes`` because each row carries
# the FULL list of slave nodes for that coupling), so the mapping
# isn't a strict prefix-strip.  Two columns are pure CSR bookkeeping
# (the split sizes that let the decoder un-flatten the ragged
# per-slave arrays) with no top-level counterpart.
# ---------------------------------------------------------------------------

# Map each sr_* column in surface_coupling_payload_dtype to the
# interpolation_payload_dtype column it carries.  Keep both sides
# in lockstep when new fields are added.
SR_TO_INTERP_COLUMN: dict[str, str] = {
    "sr_slave_nodes":     "slave_node",
    "sr_master_nodes":    "master_nodes",
    "sr_weights":         "weights",
    "sr_dofs":            "dofs",
    "sr_projected":       "projected_point",
    "sr_parametric":      "parametric_coords",
    "sr_stiffness":       "stiffness",
    "sr_stiffness_p":     "stiffness_p",
    "sr_has_stiffness_p": "has_stiffness_p",
    "sr_rotational":      "rotational",
    "sr_pressure":        "pressure",
    "sr_excess":          "excess",
    "sr_enforce":         "enforce",   # ADR 0068 (schema 2.14.0)
    # CouplingControl knobs (schema 2.12.0; host auto-scalers 2.13.0)
    # — per-slave vlen mirror of the scalar cpl_* columns.
    "sr_cpl_has":         "cpl_has",
    "sr_cpl_k":           "cpl_k",
    "sr_cpl_kr":          "cpl_kr",
    "sr_cpl_enforce":     "cpl_enforce",
    "sr_cpl_dtcr":        "cpl_dtcr",
    "sr_cpl_absolute":    "cpl_absolute",
    "sr_cpl_k_auto":      "cpl_k_auto",
    "sr_cpl_k_alpha":     "cpl_k_alpha",
    "sr_cpl_host":        "cpl_host",
    "sr_cpl_wcap":        "cpl_wcap",
}

# sr_* columns that exist purely to let the decoder un-flatten the
# ragged per-slave arrays; no counterpart needed in the top-level
# interpolation_payload_dtype.
SR_CSR_BOOKKEEPING: frozenset[str] = frozenset({
    "sr_master_counts",
    "sr_dof_counts",
})

def test_surface_coupling_sr_lane_mirrors_interpolation_dtype() -> None:
    """Every InterpolationRecord field column in
    ``interpolation_payload_dtype`` has a matching ``sr_*`` column
    in ``surface_coupling_payload_dtype`` (modulo CSR bookkeeping
    and the per-record ``name``).

    Why this matters: a tied_contact's ``slave_records`` flow
    through the bridge as ``emitter.embeddedNode(stiffness=...)``
    at ``src/apeGmsh/opensees/_internal/build.py:1831`` — the same
    callsite as a standalone ``embedded`` record.  If a field is
    added to ``InterpolationRecord`` and to the top-level
    ``interpolation_payload_dtype`` but NOT mirrored into the
    ``sr_*`` lane, ``tied_contact``'s embedded options are silently
    lost on round-trip while ``/constraints/embedded`` works
    correctly.  The PR #337 fix had to extend both lanes; this
    test locks the invariant going forward.
    """
    interp_names = set(interpolation_payload_dtype().names or ())
    sc_names = set(surface_coupling_payload_dtype().names or ())
    sc_sr_columns = {n for n in sc_names if n.startswith("sr_")}

    # Every sr_* column is either in the rename map or in the
    # bookkeeping set.  An unrecognised sr_* column is a sign the
    # rename map drifted from the dtype.
    unaccounted = sc_sr_columns - set(SR_TO_INTERP_COLUMN) - SR_CSR_BOOKKEEPING
    assert not unaccounted, (
        f"surface_coupling_payload_dtype has sr_* columns "
        f"{sorted(unaccounted)} that this test does not recognise.\n\n"
        f"Add them to SR_TO_INTERP_COLUMN (with the matching "
        f"interpolation_payload_dtype column) or to "
        f"SR_CSR_BOOKKEEPING (if they're pure CSR split-size or "
        f"presence-flag bookkeeping)."
    )

    covered_interp = {
        SR_TO_INTERP_COLUMN[sr] for sr in sc_sr_columns
        if sr in SR_TO_INTERP_COLUMN
    }

    # ``name`` is intentionally not carried per-slave: the parent
    # SurfaceCouplingRecord stamps its own name, and the resolver
    # never sets a per-slave name on the tied_contact slave records
    # (they're synthetic, not user-declared).
    expected = interp_names - {"name"}

    missing_from_sr = expected - covered_interp
    assert not missing_from_sr, (
        f"surface_coupling_payload_dtype's sr_* lane is missing "
        f"counterparts for these interpolation_payload_dtype "
        f"columns: {sorted(missing_from_sr)}.\n\n"
        f"Both lanes must cover the same InterpolationRecord field "
        f"set so tied_contact slave_records (which flow through "
        f"the same bridge callsite as /constraints/embedded) "
        f"round-trip every field that the standalone embedded "
        f"path covers.\n\n"
        f"Add an ``sr_<...>`` column to "
        f"surface_coupling_payload_dtype, wire its encode/decode in "
        f"_femdata_h5_io._encode_surface_coupling / "
        f"_decode_surface_coupling, and add the rename entry to "
        f"SR_TO_INTERP_COLUMN above."
    )


def test_payload_whitelist_entries_carry_a_reason() -> None:
    """Every PAYLOAD_WHITELIST entry has a non-empty reason string.

    The whole point of the whitelist is to force the author of any
    "field intentionally not persisted" claim to write down WHY,
    so reviewers can verify the claim and future readers can find
    the contract.  An empty-string entry would silently re-admit
    the very bug pattern this file exists to catch.
    """
    for record_cls, entries in PAYLOAD_WHITELIST.items():
        for field_name, reason in entries.items():
            assert reason and reason.strip(), (
                f"PAYLOAD_WHITELIST[{record_cls.__name__}]"
                f"[{field_name!r}] has an empty reason. "
                f"Fill in why this field is intentionally absent "
                f"from the payload dtype."
            )


def test_record_to_dtype_covers_every_concrete_record() -> None:
    """Every concrete record dataclass exported by
    :mod:`apeGmsh._kernel.records` appears in :data:`RECORD_TO_DTYPE`.

    Without this guard, a new record subclass could land without
    a parity check — silently re-admitting the ADR 0035 bug
    pattern for a different record type.  The :data:`RECORD_TO_DTYPE`
    mapping is the explicit catalogue of "every record that is
    persisted to H5".
    """
    from apeGmsh._kernel import records as records_pkg

    # Walk the package's public surface for dataclass subclasses
    # of ConstraintRecord / LoadRecord / MassRecord that have
    # concrete (non-base) semantics.  Base classes
    # (ConstraintRecord, LoadRecord) are excluded — they're
    # not persisted on their own.
    from apeGmsh._kernel.records._constraints import ConstraintRecord
    from apeGmsh._kernel.records._loads import LoadRecord

    base_classes = {ConstraintRecord, LoadRecord}
    concrete_records: set[type] = set()
    for name in records_pkg.__all__:
        obj = getattr(records_pkg, name, None)
        if not isinstance(obj, type):
            continue
        if not dataclasses.is_dataclass(obj):
            continue
        if obj in base_classes:
            continue
        # Only the kernel record types are payload-persisted via
        # _record_h5.  Filter out the Def hierarchy (pre-mesh
        # intent) and PartitionRecord (lives in its own
        # /partitions/ group, not under /constraints|loads|masses).
        if name.endswith("Def"):
            continue
        if name == "PartitionRecord":
            continue
        # ComposeRecord lives in its own /composed_from/{label}/ group
        # as attrs (ADR 0038 Phase 3A.1), not under
        # /constraints|loads|masses, so it has no _record_h5
        # payload-dtype factory by design — same exemption as
        # PartitionRecord.
        if name == "ComposeRecord":
            continue
        concrete_records.add(obj)

    catalogued = set(RECORD_TO_DTYPE.keys())
    missing = concrete_records - catalogued
    assert not missing, (
        f"These record dataclasses are exported from "
        f"apeGmsh._kernel.records but have no entry in "
        f"RECORD_TO_DTYPE: {sorted(c.__name__ for c in missing)}.\n\n"
        f"Add an entry mapping each to its sibling *_payload_dtype "
        f"factory so the parity check runs against them.  If a "
        f"record is intentionally NOT persisted to H5, add it to "
        f"the explicit-skip list inside this test with a comment."
    )
