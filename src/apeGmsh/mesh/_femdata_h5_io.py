"""Broker-side neutral-zone writer for ``model.h5``.

Phase 8.5 makes the :class:`apeGmsh.mesh.FEMData` broker write the
neutral-zone groups that the master plan
([architecture/phase-8-untangle.md §3](../opensees/architecture/phase-8-untangle.md))
places at the root of ``model.h5``:

* ``/meta``                — file-level metadata (schema_version, ndm,
                             snapshot_id, …).
* ``/nodes``               — ids, coords.
* ``/elements/{type}``     — per-type ids + connectivity.
* ``/physical_groups``     — top-level index for viewer discovery.
* ``/labels``              — apeGmsh-internal labels.
* ``/mesh_selections``     — post-mesh selection sets (Phase 8.7).
* ``/constraints/{kind}``  — MP-style records, symmetric compound shape.
* ``/loads/{kind}/{pattern}``  — per-pattern load records.
* ``/masses``              — per-node mass vectors.

Per ADR 0020 (Phase 4 cleanup, May 2026) the same rich layout is also
embedded under a ``/model/`` sub-group inside composed ``results.h5``
files via :func:`write_neutral_zone_into_group`.  This module is now
the single neutral-zone writer; the only difference between root and
sub-group embeds is the parent passed in.

Public entry points:

* :func:`write_fem_h5` — open a fresh file at ``path``, write meta +
  neutral zone, close.  This is what ``FEMData.to_h5(path)`` delegates
  to.
* :func:`write_neutral_zone` — write the seven neutral-zone groups
  into an already-open :class:`h5py.File`.  Used by the bridge in
  Phase 8.5 commit 4 to compose neutral + ``/opensees/`` in one file.
* :func:`write_meta` — write ``/meta`` attrs.  Caller-owned so the
  bridge can stamp its own ``schema_version`` / ``ndf`` while the
  broker fills in the geometry-derived attrs.

Reader-side helpers live in
:mod:`apeGmsh.opensees.emitter.h5_reader` (Phase 8.5 commit 3 extends
it with typed accessors for the new groups).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from ._record_h5 import (
    element_load_payload_dtype,
    interpolation_payload_dtype,
    make_record_dtype,
    mass_payload_dtype,
    node_group_payload_dtype,
    contact_payload_dtype,
    contact_plane_payload_dtype,
    embed_tie_payload_dtype,
    node_pair_payload_dtype,
    node_to_surface_payload_dtype,
    nodal_load_payload_dtype,
    rebar_element_payload_dtype,
    reinforce_tie_payload_dtype,
    sp_payload_dtype,
    surface_coupling_payload_dtype,
)

if TYPE_CHECKING:
    from apeGmsh._kernel.record_sets import MassSet

    from .FEMData import FEMData


__all__ = [
    "COMPOSED_FROM_SCHEMA_VERSION",
    "NEUTRAL_SCHEMA_VERSION",
    "read_fem_h5",
    "read_neutral_zone_from_group",
    "write_fem_h5",
    "write_meta",
    "write_neutral_zone",
    "write_neutral_zone_into_group",
]


#: Schema version stamped by :func:`write_fem_h5` and the standalone
#: ``FEMData.to_h5(path)`` flow.  Phase 8.5 added the neutral zone
#: (`2.0.0 → 2.1.0`); Phase 8.6 added the ``fem_eids`` dataset under
#: each ``/opensees/element_meta/{type_token}/`` group
#: (`2.1.0 → 2.2.0`).  Phase 8.7 commit 2 added the
#: ``/mesh_selections/`` neutral-zone group, mirroring
#: ``/physical_groups`` for post-mesh selection sets
#: (`2.3.0 → 2.4.0`).
#:
#: v2.5.0 (May 2026, Phase 2 of the major architectural refactor):
#: additive — adds the ``name`` field to every record payload dtype
#: (constraints, loads, masses), the new ``/partitions/`` and
#: ``/parts/`` neutral-zone groups, and reader-side verification of
#: ``/meta/snapshot_id`` against the recomputed hash.  Per ADR 0023
#: (per-zone-schema-versioning), readers in the two-version window
#: accept 2.4.x and 2.5.x; 2.4.0 files silently lack the new
#: ``name`` fields and the new groups (reader probes presence per
#: payload field and per group).
#:
#: v2.6.0 (May 2026, Phase 6 of the major architectural refactor):
#: additive — adds the ``/meta/lineage/`` sub-group (per ADR 0021
#: §"Surface — warn, not raise") carrying the git-style hash chain
#: ``fem_hash → model_hash → results_hash``.  Per ADR 0023's
#: two-version reader window, readers tolerate 2.5.x and 2.6.x;
#: 2.5.x files silently lack the lineage sub-group (legacy-file
#: warning surfaces at the :class:`Lineage` layer, never raises).
#:
#: v2.7.0 (May 2026, shell-to-solid coupling broker foundation, S1b):
#: additive — adds the optional ``/nodes/ndf`` (int8) dataset
#: carrying per-node DOF count metadata declared via
#: ``g.node_ndf.set(...)`` / ``g.node_ndf.set_default(...)``.  Per
#: ADR 0023's two-version reader window, readers tolerate 2.6.x and
#: 2.7.x; 2.6.x files silently lack the dataset and round-trip with
#: ``fem.nodes._ndf is None`` (every ``fem.nodes.ndf_for(...)`` call
#: raises ``LookupError`` until the user re-declares).
#:
#: v2.8.0 (May 2026, ASDEmbeddedNodeElement option round-trip):
#: additive — extends ``interpolation_payload_dtype`` and the sr_*
#: lane of ``surface_coupling_payload_dtype`` with six typed
#: columns each so the per-record ``stiffness`` (``-K``),
#: ``stiffness_p`` (``-KP``, with a separate ``has_stiffness_p``
#: presence flag because 0 is valid penalty data), ``rotational``
#: (``-rot``), ``pressure`` (``-p``), and ``excess`` (barycentric
#: excess from ``resolve_embedded``) survive H5 round-trip.  Pre-
#: 2.8.0 files lacked these and silently snapped back to the
#: :class:`InterpolationRecord` defaults (``stiffness=1e18``, the
#: rest ``None`` / ``False``) on read, which the bridge then
#: emitted unconditionally as ``-K 1e18`` after ADR 0035 made
#: ``-K`` mandatory.  Per ADR 0023's two-version reader window,
#: readers tolerate 2.7.x and 2.8.x; 2.7.x files probe
#: ``p.dtype.names`` (the same structural-detection pattern that
#: gates ``sr_*`` and ``name``) and decode with dataclass defaults.
#:
#: v2.9.0 (May 2026, Compose v1 Phase 3A.1 / ADR 0038):
#: additive — adds the optional ``/composed_from/`` provenance
#: sub-group (one ``{label}`` per composed source module, with
#: ``source_fem_hash`` / ``source_neutral_schema_version`` /
#: ``source_path`` / ``translate`` / optional ``rotate`` / optional
#: ``partition_rank`` / ``composed_at`` attrs plus an optional
#: ``properties`` sub-attribute group), the ``meta/@tag_span_max``
#: (int64) attribute used by Phase 3B's ``_compute_source_span`` to
#: size per-module reservations, and the optional ``module_label``
#: (variable-length string) parallel dataset on ``/nodes/`` and each
#: ``/elements/{type}/`` group identifying the source module each row
#: came from (empty string for host-owned rows; populated by Phase
#: 3B's merge engine).  Per ADR 0023's two-version reader window,
#: readers tolerate 2.8.x and 2.9.x; 2.8.x files lack all three
#: additions and decode cleanly with ``composed_from=()`` and no
#: ``module_label`` plumbed.
#:
#: v2.13.0 (June 2026, fork-coupling host auto-scalers — handoff item A):
#: additive — extends the ``cpl_*`` coupling-control lane on
#: ``node_group_payload_dtype`` / ``interpolation_payload_dtype`` with
#: four columns: ``cpl_k_auto`` (uint8, ``k="auto"``), ``cpl_k_alpha``
#: (float64, ``-kAlpha``), ``cpl_host`` (int64 **FEM element id**,
#: ``-1`` = none — deliberately NOT the emit-time ops tag) and
#: ``cpl_wcap`` (float64, ``-bipenalty -wcap``), with matching
#: per-slave vlen ``sr_cpl_*`` mirrors on
#: ``surface_coupling_payload_dtype`` (lane-parity contract from PR
#: #337 / #632).  Per ADR 0023's two-version reader window, readers
#: tolerate 2.12.x and 2.13.x; 2.12.x files probe ``cpl_k_auto`` /
#: ``sr_cpl_k_auto`` via ``p.dtype.names`` and decode with the v1
#: knobs only.
#:
#: v2.14.0 (June 2026, ADR 0068 — equation-constraint tied interface):
#: additive — adds the ``enforce`` route to the interpolation lane
#: (``interpolation_payload_dtype``, a utf8 column "penalty"|"penalty_al"|
#: "equation") and its per-slave mirror ``sr_enforce`` (uint8 code 0/1/2)
#: on ``surface_coupling_payload_dtype``. Per ADR 0023's two-version reader
#: window, readers tolerate 2.13.x and 2.14.x; 2.13.x files lack ``enforce``
#: / ``sr_enforce`` (probed via ``p.dtype.names``) and decode with the
#: dataclass default ``enforce="penalty"``.
#:
#: v2.15.0 (June 2026, ADR 0067 P5.1 — embedded-reinforcement ties):
#: additive — persists ``fem.elements.reinforce_ties`` (a list of
#: :class:`ReinforceTieRecord`, the g.reinforce ``LadrunoEmbeddedRebar``
#: couplings) into a NEW dedicated ``/reinforce_ties`` group via
#: ``reinforce_tie_payload_dtype``.  Previously these were dropped on
#: ``to_h5`` (a deferral warning), so a reinforced model lost its
#: reinforcement on round-trip; now it survives, unblocking composed-Part
#: cage libraries.  The group is omitted entirely when there are no ties,
#: so a tie-free model stays byte-identical and its ``snapshot_id`` is
#: unchanged (the hash does not cover ties — consistent with constraints).
#: Per ADR 0023's two-version reader window, readers tolerate 2.14.x and
#: 2.15.x; a 2.14.x file simply has no ``/reinforce_ties`` group (absence
#: ⇒ no ties).
#:
#: v2.16.0 (June 2026, ADR 0067 P5.2 / B1a.2 — auto-emitted rebar elements):
#: additive — persists ``fem.elements.rebar_elements`` (a list of
#: :class:`RebarElementRecord`, the cage's ``place(emit_elements=True)``
#: structural elements) into a NEW dedicated ``/rebar_elements`` group via
#: ``rebar_element_payload_dtype``.  Previously these were dropped on
#: ``to_h5`` (a deferral warning), so an auto-emitted cage lost its rebar
#: elements on round-trip; now they survive.  The group is omitted when
#: there are none, so a cage that didn't opt in stays byte-identical (and
#: ``snapshot_id`` is unchanged — the hash does not cover rebar elements,
#: consistent with constraints / ties).  Per ADR 0023's two-version reader
#: window, readers tolerate 2.15.x and 2.16.x; a 2.15.x file simply has no
#: ``/rebar_elements`` group.
#:
#: v2.17.0 (June 2026, ADR 0069 — equalDOF_Mixed): additive — adds the
#: ``master_dofs`` vlen-int64 column to ``node_pair_payload_dtype`` so the
#: retained-node DOFs of an ``equal_dof_mixed`` :class:`NodePairRecord`
#: round-trip (every other kind stores a zero-length array ⇒ decoded back
#: to ``None``).  Per ADR 0023's two-version reader window, readers tolerate
#: 2.16.x and 2.17.x; a 2.16.x file lacks ``master_dofs`` (probed via
#: ``p.dtype.names``) and decodes it as ``None``.  The field-signature
#: dispatch (``NODE_PAIR_FIELDS``) is a subset match, so the added column
#: does not perturb decoder routing.
#:
#: v2.18.0 (June 2026, ADR 0069 follow-up — EmbeddedNodeControl pressure
#: tie): additive — adds ``cpl_pressure`` (uint8) + ``cpl_kp`` (float64) to
#: ``_coupling_control_fields`` and their ``sr_cpl_*`` per-slave mirrors, so
#: a ``tie``/``tied_contact``/``embedded`` carrying an
#: :class:`EmbeddedNodeControl` (the ``-pressure``/``-kp`` LadrunoEmbeddedNode
#: knobs) round-trips — the decoded control is an ``EmbeddedNodeControl``
#: when ``cpl_pressure`` is set, else a base ``CouplingControl``.  Per ADR
#: 0023's two-version reader window, readers tolerate 2.17.x and 2.18.x;
#: a 2.17.x file lacks the columns (probed via ``p.dtype.names``) and
#: decodes the base control.
#:
#: v2.19.0 (June 2026, ADR 0071 — LadrunoRigidBody): additive — adds
#: ``rb_as_element`` (uint8) + ``rb_mass`` (float64) to
#: ``node_group_payload_dtype`` so a ``rigid_body`` declared with
#: ``as_element=True`` (emit the fork ``element LadrunoRigidBody`` instead
#: of the rigidLink chain) and its optional ``-mass`` round-trip.  Per ADR
#: 0023's two-version reader window, readers tolerate 2.18.x and 2.19.x; a
#: 2.18.x file lacks the columns (probed via ``p.dtype.names``) and decodes
#: ``as_element=False`` / ``mass=None`` (the rigidLink-chain form).
#:
#: v2.20.0 (June 2026, ADR 0071 follow-up — LadrunoRigidBody -omega): additive
#: — adds an ``omega`` (3,)-float64 column to ``node_group_payload_dtype`` so
#: a rigid_body's initial body-frame angular velocity (the ``-omega``
#: explicit-dynamics IC) round-trips (NaN-filled ⇒ ``None``). Per ADR 0023's
#: two-version reader window, readers tolerate 2.19.x and 2.20.x; a 2.19.x
#: file lacks ``omega`` (probed via ``p.dtype.names``) and decodes ``None``.
#:
#: v2.21.0 (June 2026, ADR 0073 follow-up — fork contact persistence): additive
#: — persists ``fem.elements.contacts`` (a list of :class:`ContactRecord`, the
#: g.constraints.contact / g.constraints.mortar fork NTS/mortar interactions)
#: into a NEW dedicated ``/contacts`` group via ``contact_payload_dtype``.
#: Previously these were dropped on the OpenSees deck zone with a one-time
#: ``H5FeatureDeferredWarning`` and had NO neutral persistence, so a contact
#: model lost its contact on round-trip; now it survives in the neutral zone
#: (the deck-zone no-op is consequently silent, mirroring reinforce ties). The
#: group is omitted when there are no contacts, so a contact-free model stays
#: byte-identical (and ``snapshot_id`` is unchanged — the hash does not cover
#: contacts, consistent with constraints / ties). Per ADR 0023's two-version
#: reader window, readers tolerate 2.20.x and 2.21.x; a 2.20.x file simply has
#: no ``/contacts`` group (absence ⇒ no contacts).
#:
#: v2.22.0 (June 2026, ADR 0073 follow-up — g.embed persistence): additive —
#: persists ``fem.elements.embed_ties`` (a list of :class:`EmbedTieRecord`, the
#: g.embed ``LadrunoEmbeddedNode`` node-to-host couplings) into a NEW dedicated
#: ``/embed_ties`` group via ``embed_tie_payload_dtype`` (the isotropic sibling
#: of ``/reinforce_ties``). Previously these were dropped on the OpenSees deck
#: zone with a one-time ``H5FeatureDeferredWarning`` and had NO neutral
#: persistence, so an embedded model lost its embedment on round-trip; now it
#: survives in the neutral zone (the deck-zone no-op is consequently silent,
#: mirroring reinforce / contact). The group is omitted when there are none, so
#: a model with no embedment stays byte-identical (``snapshot_id`` unchanged —
#: the hash does not cover embed ties). Per ADR 0023's two-version reader
#: window, readers tolerate 2.21.x and 2.22.x; a 2.21.x file simply has no
#: ``/embed_ties`` group (absence ⇒ no ties).
#:
#: v2.23.0 (June 2026, ADR 0073 follow-up — contact broad-phase knob): additive
#: — adds the optional ``cell`` field (the ``-cell`` broad-phase cell-size scale)
#: to ``contact_payload_dtype``. Presence-probed on read (``"cell" in
#: p.dtype.names``); a 2.22.x file simply lacks the column and decodes
#: ``cell=None``. Omitted-knob round-trip stays byte-identical. Per ADR 0023's
#: two-version reader window, readers tolerate 2.22.x and 2.23.x.
#:
#: v2.24.0 (June 2026, ADR 0073 follow-up — rigid-plane contact): additive —
#: persists ``fem.elements.contact_planes`` (a list of
#: :class:`ContactPlaneRecord`, the g.constraints.contact_plane ``contactPlane``
#: rigid-plane contacts) into a NEW dedicated ``/contact_planes`` group via
#: ``contact_plane_payload_dtype`` (the analytical-plane sibling of
#: ``/contacts``). The group is omitted when there are none, so a plane-free
#: model stays byte-identical (``snapshot_id`` unchanged — the hash does not
#: cover contact planes). Per ADR 0023's two-version reader window, readers
#: tolerate 2.23.x and 2.24.x; a 2.23.x file simply has no ``/contact_planes``
#: group (absence ⇒ no planes).
#:
#: v2.25.0 (June 2026, ADR 0073 follow-up — edge-edge contact fallback): additive
#: — adds the edge-edge fallback columns (the ``-edgeedge`` + ``-edge*`` knobs,
#: fork ADR-57 E2–E7) to ``contact_payload_dtype``: ``edge_edge`` (0/1),
#: ``edge_kn`` + ``edge_kn_mode`` (auto/None/numeric), ``edge_band``,
#: ``edge_mu``/``edge_kt``/``edge_cohesion``/``edge_tau_max``,
#: ``edge_consistent_tan`` (0/1), ``edge_soft`` + ``edge_soft_mode``
#: (None/bare/numeric), ``edge_alm`` (0/1), ``edge_aug_tol``. Presence-probed on
#: read (``"edge_edge" in p.dtype.names``); a 2.24.x file simply lacks the
#: columns and decodes the edge-edge fallback off (``edge_edge=False``,
#: everything else None). Omitted-knob round-trip stays byte-identical. Per ADR
#: 0023's two-version reader window, readers tolerate 2.24.x and 2.25.x.
#:
#: v2.26.0 (June 2026, ADR 20 R3c — g.reinforce co-rotated bar axis): additive
#: — adds the ``corot`` (0/1) + ``shape_b`` (vlen) + ``has_shape_b`` columns
#: (the ``-corot -shapeB`` point-B weights, ADR 20 §10.5) to
#: ``reinforce_tie_payload_dtype``. Presence-probed on read (``"corot" in
#: p.dtype.names``); a 2.25.x file lacks the columns and decodes ``corot=False``,
#: ``shape_b=None`` (the frozen ``-dir`` path). Omitted-knob round-trip stays
#: byte-identical. Per ADR 0023's two-version reader window, readers tolerate
#: 2.25.x and 2.26.x.
#:
#: Broker-only files (no `/opensees/...`) still stamp the current
#: minor — the field is additive and old readers tolerate its
#: absence.
NEUTRAL_SCHEMA_VERSION: str = "2.26.0"

#: Inner schema-version stamp written on the ``/composed_from/`` group
#: when ``fem.composed_from`` is non-empty.  Independent of the
#: neutral-zone schema; ADR 0038 §"Implementation pointer" locks the
#: initial value at ``"1.0.0"`` so future provenance-shape additions
#: can bump this independently.
COMPOSED_FROM_SCHEMA_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def write_fem_h5(
    fem: "FEMData",
    path: str,
    *,
    schema_version: str = NEUTRAL_SCHEMA_VERSION,
    model_name: str = "",
    apegmsh_version: str = "",
    ndf: int = 0,
) -> None:
    """Write a fresh ``model.h5`` with the neutral zone.

    No ``/opensees/`` content is emitted — absent enrichment is the
    right "no solver" signal.

    Phase 6 (ADR 0021) — stamps ``/meta/lineage/fem_hash`` after the
    neutral zone is written.  Lazy import keeps the apeGmsh.mesh
    import-time graph free of apeGmsh.opensees.
    """
    import h5py

    from ..opensees._internal.lineage import write_lineage_attrs

    with h5py.File(path, "w") as f:
        write_meta(
            fem, f,
            schema_version=schema_version,
            model_name=model_name,
            apegmsh_version=apegmsh_version,
            ndf=ndf,
        )
        write_neutral_zone(fem, f)
        # ADR 0021 lineage — broker-only files carry just ``fem_hash``
        # (no ``/opensees/`` ⇒ no ``model_hash``).  The fem snapshot
        # is authoritative; recompute happens at read time per the
        # warn-not-raise contract.
        if "meta" in f:
            try:
                fem_hash = str(fem.snapshot_id)
            except Exception:
                fem_hash = ""
            if fem_hash:
                write_lineage_attrs(f["meta"], fem_hash=fem_hash)


def write_neutral_zone_into_group(
    fem: "FEMData",
    parent: Any,
    *,
    schema_version: str = NEUTRAL_SCHEMA_VERSION,
    model_name: str = "",
    apegmsh_version: str = "",
    ndf: int = 0,
) -> None:
    """Write ``meta`` + neutral zone as children of ``parent``.

    The composed-results pattern (ADR 0020) embeds a FEMData snapshot
    under ``/model/`` of a ``results.h5``.  This helper accepts an
    open ``h5py.Group`` (the ``/model/`` sub-group) and writes the
    same rich layout :func:`write_fem_h5` writes to the file root —
    only the prefix differs.  ``write_fem_h5(path)`` itself uses
    ``parent = h5py.File(...)`` and so produces byte-identical output
    when this helper is given the same fem and ``parent = file``.
    """
    write_meta(
        fem, parent,
        schema_version=schema_version,
        model_name=model_name,
        apegmsh_version=apegmsh_version,
        ndf=ndf,
    )
    write_neutral_zone(fem, parent)


def write_meta(
    fem: "FEMData",
    f: Any,
    *,
    schema_version: str,
    model_name: str = "",
    apegmsh_version: str = "",
    ndf: int = 0,
) -> None:
    """Create ``/meta`` and stamp the file-level attrs.

    Caller-owned so the bridge can supply its own ``ndf`` /
    ``schema_version``.  Broker-only writes pass ``ndf=0``.

    Per ADR 0023 (per-zone schema versioning, Phase 7a) this also
    stamps ``/meta/neutral_schema_version`` as the neutral-zone-specific
    marker, **and** stamps ``/meta/opensees_schema_version`` with the
    current bridge writer version even on broker-only files (so they
    open cleanly through :func:`h5_reader.open` whose validation runs
    against the opensees per-zone window — an absent per-zone key
    would fall back to the envelope and fail validation whenever the
    neutral and opensees windows have diverged).  The envelope
    ``/meta/schema_version`` is preserved for one-key legacy readers;
    the bridge composer may later overwrite the envelope with its own
    ``SCHEMA_VERSION`` so envelope readers see "whichever wrote last"
    (single-stamp era semantics).
    """
    # Lazy import: keep apeGmsh.mesh's import-time graph free of
    # apeGmsh.opensees.  Only the version constant is needed.
    from apeGmsh.opensees.emitter.h5 import SCHEMA_VERSION as OPENSEES_VERSION

    meta = f.create_group("meta")
    meta.attrs["schema_version"] = schema_version
    # ADR 0023 §"Three per-zone version stamps" — per-zone neutral key,
    # independent of the envelope. Broker-only files now also stamp
    # the opensees per-zone key (see docstring above); composed files
    # overwrite this stamp in ``_compose_model_h5`` after writing the
    # /opensees/ content.
    meta.attrs["neutral_schema_version"] = schema_version
    meta.attrs["opensees_schema_version"] = OPENSEES_VERSION
    meta.attrs["apeGmsh_version"] = apegmsh_version
    meta.attrs["created_iso"] = datetime.now(tz=timezone.utc).isoformat()
    meta.attrs["ndm"] = int(_derive_ndm(fem))
    meta.attrs["ndf"] = int(ndf)
    meta.attrs["snapshot_id"] = str(fem.snapshot_id)
    meta.attrs["model_name"] = str(model_name)
    # ADR 0038 §"Schema" — tag-span-max (max(max_node, max_elem) -
    # min(min_node, min_elem) + 1) used by Phase 3B's
    # ``_compute_source_span`` to size per-module reservations.  Stored
    # on /meta rather than the file root because the codebase's
    # convention places every file-level attr under /meta (file-root
    # attrs are atypical in h5py).  3B reads from
    # ``meta.attrs["tag_span_max"]``; pre-2.9.0 files lack the attr
    # and trigger 3B's fallback dataset scan.
    meta.attrs["tag_span_max"] = int(_compute_tag_span_max(fem))


def write_neutral_zone(fem: "FEMData", f: Any) -> None:
    """Write the seven neutral-zone groups into an open HDF5 file.

    Does NOT write ``/meta`` — the caller owns that, so the bridge
    can stamp its own ``schema_version`` / ``ndf`` while the broker
    just contributes geometry.
    """
    _write_nodes(fem, f)
    _write_elements(fem, f)
    _write_physical_groups(fem, f)
    _write_labels(fem, f)
    _write_mesh_selections(fem, f)
    _write_partitions(fem, f)
    _write_parts(fem, f)
    _write_constraints(fem, f)
    _write_reinforce_ties(fem, f)
    _write_embed_ties(fem, f)
    _write_rebar_elements(fem, f)
    _write_contacts(fem, f)
    _write_contact_planes(fem, f)
    _write_loads(fem, f)
    _write_masses(fem, f)
    # ADR 0038 §"Schema" — optional /composed_from/ provenance group.
    # Omitted entirely when ``fem.composed_from`` is empty (the
    # uncomposed case); absence is the right "uncomposed" signal on
    # disk and keeps pre-2.9.0 round-trip diffs minimal.
    _write_composed_from(fem, f)


# ---------------------------------------------------------------------------
# Per-group writers
# ---------------------------------------------------------------------------


def _derive_ndm(fem: "FEMData") -> int:
    """Best-effort spatial dimension from the broker's element types."""
    try:
        dims = [int(t.dim) for t in fem.info.types]
        if dims:
            return max(dims)
    except (AttributeError, ValueError):
        pass
    return 3


def _vlen_utf8() -> Any:
    """Return the h5py variable-length UTF-8 string dtype.

    Local helper to keep the writer's lazy-import discipline — h5py is
    pulled in only at the write callsites that need it.
    """
    import h5py
    return h5py.string_dtype(encoding="utf-8")


def _compute_tag_span_max(fem: "FEMData") -> int:
    """Compute ``max_tag - min_tag + 1`` across nodes + elements.

    ADR 0038 §"Schema" — written as ``/meta/@tag_span_max`` and read
    by Phase 3B's ``_compute_source_span`` to size the per-module tag
    reservation without a full dataset scan.  Combines nodes AND
    elements into a single span because the per-module reservation
    must cover both classes uniformly.  Empty mesh → 0.
    """
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    eid_parts: list[np.ndarray] = []
    for group in fem.elements:
        ids = np.asarray(group.ids, dtype=np.int64)
        if ids.size > 0:
            eid_parts.append(ids)
    if node_ids.size == 0 and not eid_parts:
        return 0
    mins: list[int] = []
    maxs: list[int] = []
    if node_ids.size > 0:
        mins.append(int(node_ids.min()))
        maxs.append(int(node_ids.max()))
    for arr in eid_parts:
        mins.append(int(arr.min()))
        maxs.append(int(arr.max()))
    return int(max(maxs) - min(mins) + 1)


def _write_nodes(fem: "FEMData", f: Any) -> None:
    """Write ``/nodes/{ids, coords[, ndf]}`` from ``fem.nodes``.

    Neutral schema 2.7.0 added the per-node ``ndf`` dataset
    (shell-to-solid coupling, S1b).  Written whenever the broker has
    a populated ``_ndf`` array (``from_gmsh`` always populates;
    ``from_msh`` and direct test fixtures leave ``_ndf=None`` per
    the S2 locked design).  Readers tolerate absence by leaving
    ``_ndf=None`` on the rehydrated broker, or — for pre-2.7.0
    files — synthesising the all-sentinel array so the recomputed
    hash equals what was stored.  Both shapes (None and all-zero)
    hash identically thanks to the empty-channel gate in
    ``_femdata_hash._hash_nodes``.
    """
    nodes_grp = f.create_group("nodes")
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    node_coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    nodes_grp.create_dataset("ids", data=node_ids)
    nodes_grp.create_dataset("coords", data=node_coords)

    # Per-node ndf (additive in 2.7.0; absent when the composite
    # carries no populated array).  int8 since values are always
    # in [1, 6] with 0 as the "undeclared" sentinel.
    ndf = getattr(fem.nodes, "_ndf", None)
    if ndf is not None:
        nodes_grp.create_dataset(
            "ndf", data=np.asarray(ndf, dtype=np.int8),
        )

    # Per-node provenance (decoupled nodes — ADR 0049; neutral schema
    # 2.11.0).  int8 (0=mesh, 1=decoupled).  Written only when the
    # broker carries decoupled nodes (``_provenance is not None``); the
    # common no-decoupled case omits the dataset so the file stays
    # byte-identical to a pre-2.11.0 write.  2.10.x readers ignore it.
    prov = getattr(fem.nodes, "_provenance", None)
    if prov is not None:
        nodes_grp.create_dataset(
            "provenance", data=np.asarray(prov, dtype=np.int8),
        )

    # ADR 0038 — per-node ``module_label`` parallel dataset (neutral
    # schema 2.9.0).  Always written so the compose-aware reader has a
    # stable shape contract.  Populated with the broker's stored
    # ``_module_label`` array when present (Phase 3B.2c onwards); falls
    # back to empty strings for legacy fixtures + the uncomposed case.
    # 2.8.x readers ignore the extra dataset.
    mlbl = getattr(fem.nodes, "_module_label", None)
    if mlbl is not None and len(mlbl) == node_ids.size:
        ml_data = np.array(
            [str(x) if x is not None else "" for x in mlbl],
            dtype=object,
        )
    else:
        ml_data = np.array([""] * node_ids.size, dtype=object)
    nodes_grp.create_dataset(
        "module_label",
        data=ml_data,
        dtype=_vlen_utf8(),
    )


def _write_elements(fem: "FEMData", f: Any) -> None:
    """Write ``/elements/{type}/{ids, connectivity}`` per element type.

    ``{type}`` is the broker's GMSH-style alias (``tet4``, ``hex8``,
    ``triangle3``, …).  These deliberately do NOT match the bridge's
    OpenSees type tokens (``forceBeamColumn``, ``FourNodeTetrahedron``);
    the two namespaces serve different consumers and live in
    different zones (root vs ``/opensees/element_meta``).
    """
    elements_grp = f.create_group("elements")
    for elem_group in fem.elements:
        if elem_group.ids.size == 0:
            continue
        type_name = elem_group.type_name.replace("/", "_")
        sub = elements_grp.create_group(type_name)
        et = elem_group.element_type
        sub.attrs["code"] = int(et.code)
        sub.attrs["gmsh_name"] = str(et.gmsh_name)
        sub.attrs["npe"] = int(et.npe)
        sub.attrs["dim"] = int(et.dim)
        sub.attrs["order"] = int(et.order)
        eids = np.asarray(elem_group.ids, dtype=np.int64)
        sub.create_dataset("ids", data=eids)
        sub.create_dataset(
            "connectivity",
            data=np.asarray(elem_group.connectivity, dtype=np.int64),
        )
        # ADR 0038 — per-element ``module_label`` parallel dataset
        # (neutral schema 2.9.0).  Same shape contract as
        # ``/nodes/module_label`` — Phase 3B.2c onwards stamps real
        # values for compose-merged rows; falls back to empty strings
        # for the uncomposed case + legacy fixtures.
        ml_dict = getattr(fem.elements, "_module_label", None)
        if (
            ml_dict is not None
            and elem_group.type_code in ml_dict
            and len(ml_dict[elem_group.type_code]) == eids.size
        ):
            elem_ml_data = np.array(
                [
                    str(x) if x is not None else ""
                    for x in ml_dict[elem_group.type_code]
                ],
                dtype=object,
            )
        else:
            elem_ml_data = np.array([""] * eids.size, dtype=object)
        sub.create_dataset(
            "module_label",
            data=elem_ml_data,
            dtype=_vlen_utf8(),
        )


def _write_physical_groups(fem: "FEMData", f: Any) -> None:
    """Write ``/physical_groups/{node_side,element_side}/{safe_name}/...``.

    Neutral schema 2.10.0 split — node-side and element-side PG
    taxonomies live in separate sub-trees rather than the pre-2.10
    flat union.  Eliminates the read-time heuristic that misclassified
    element-side-only PGs as also-node-side (root cause of the
    snapshot_id drift bug).  Omitted entirely if neither side
    declared any PGs.
    """
    _write_named_index_at_root(
        fem, f, group_name="physical_groups",
        node_side=getattr(fem.nodes, "physical", None),
        element_side=getattr(fem.elements, "physical", None),
    )


def _write_labels(fem: "FEMData", f: Any) -> None:
    """Write ``/labels/{node_side,element_side}/{safe_name}/...``.

    Same shape as ``/physical_groups`` (neutral schema 2.10.0 split);
    the only difference is the source-side label set.
    """
    _write_named_index_at_root(
        fem, f, group_name="labels",
        node_side=getattr(fem.nodes, "labels", None),
        element_side=getattr(fem.elements, "labels", None),
    )


def _write_mesh_selections(fem: "FEMData", f: Any) -> None:
    """Write ``/mesh_selections/{name}/{node_ids, node_coords,
    element_ids, connectivity}``.

    Mirrors ``/physical_groups`` and ``/labels`` shape so the same
    ``H5Reader._read_named_index`` helper handles all three — that
    helper reads only the node/element id datasets, so the extra
    ``connectivity`` dataset is transparent to it and is consumed only
    by :meth:`MeshSelectionStore.get_elements` on reload.  Sourced
    from :attr:`FEMData.mesh_selection` (a
    :class:`apeGmsh.mesh.MeshSelectionSet.MeshSelectionStore` captured
    at ``get_fem_data()`` time).  Omitted entirely when the snapshot
    has no selection store or the store is empty — absence is the
    right "no selections" signal.

    Added in Phase 8.7 commit 2 (schema 2.3.0 → 2.4.0, additive) so
    the viewer's ``selection=`` selector round-trips through
    ``model.h5``.  ``connectivity`` was added afterwards (still
    additive, presence-detected on read → no schema bump; old files
    just lack the dataset) so a reloaded element-bearing set no
    longer raises from ``get_elements`` for want of connectivity.
    """
    store = getattr(fem, "mesh_selection", None)
    if store is None:
        return
    try:
        keys = store.get_all()
    except (AttributeError, TypeError):
        return
    if not keys:
        return

    parent = f.create_group("mesh_selections")
    seen_safe: set[str] = set()
    for dim, tag in keys:
        d, t = int(dim), int(tag)
        try:
            name = store.get_name(d, t)
        except (KeyError, ValueError, AttributeError):
            name = ""
        if not name:
            name = f"_unnamed_{d}_{t}"
        safe = name.replace("/", "_")
        if safe in seen_safe:
            safe = f"{safe}__{d}_{t}"
        seen_safe.add(safe)

        sub = parent.create_group(safe)
        sub.attrs["dim"] = d
        sub.attrs["tag"] = t
        sub.attrs["name"] = name

        try:
            node_data = store.get_nodes(d, t)
            nids = np.asarray(node_data["tags"], dtype=np.int64)
            ncoords = np.asarray(node_data["coords"], dtype=np.float64)
        except (KeyError, ValueError, AttributeError):
            nids = np.array([], dtype=np.int64)
            ncoords = np.zeros((0, 3), dtype=np.float64)
        sub.create_dataset("node_ids", data=nids)
        sub.create_dataset("node_coords", data=ncoords)

        if d >= 1:
            try:
                elem_data = store.get_elements(d, t)
                eids = np.asarray(elem_data["element_ids"], dtype=np.int64)
                conn = np.asarray(elem_data["connectivity"], dtype=np.int64)
            except (KeyError, ValueError, AttributeError):
                eids = np.array([], dtype=np.int64)
                conn = np.empty((0, 0), dtype=np.int64)
            if eids.size > 0:
                sub.create_dataset("element_ids", data=eids)
                # Persist connectivity alongside element_ids: without
                # it a reloaded element set keeps element_ids but
                # MeshSelectionStore.get_elements() raises "no element
                # data".  Rows align 1:1 with element_ids.
                sub.create_dataset("connectivity", data=conn)


def _write_partitions(fem: "FEMData", f: Any) -> None:
    """Write ``/partitions/{id}/{node_ids, element_ids}`` for each partition.

    Sourced from :attr:`NodeComposite._partitions` — populated by
    ``_fem_factory`` when the mesh was extracted from a partitioned
    Gmsh session.  Omitted when the snapshot has no partitions
    (absence is the right "not partitioned" signal on disk; matches
    the omit-empty-groups convention).

    Added in neutral schema 2.5.0 (Phase 2 of the major refactor) so
    ``fem.partitions`` / ``select(partition=k)`` survive the H5
    round-trip.  Per ADR 0023's two-version window, readers in 2.4.x
    silently lack this group.
    """
    parts = getattr(fem.nodes, "_partitions", None) or {}
    if not parts:
        return

    parent = f.create_group("partitions")
    for pid in sorted(parts.keys()):
        pdata = parts[pid]
        sub = parent.create_group(str(int(pid)))
        sub.attrs["id"] = int(pid)
        node_ids = np.asarray(
            sorted(int(x) for x in pdata.get("node_ids", [])),
            dtype=np.int64,
        )
        elem_ids = np.asarray(
            sorted(int(x) for x in pdata.get("element_ids", [])),
            dtype=np.int64,
        )
        sub.create_dataset("node_ids", data=node_ids)
        sub.create_dataset("element_ids", data=elem_ids)


def _write_composed_from(fem: "FEMData", f: Any) -> None:
    """Write ``/composed_from/{label}/`` for each composed source module.

    ADR 0038 §"Schema" — neutral schema 2.9.0.  One sub-group per
    :class:`ComposeRecord` on ``fem.composed_from``, attrs carrying
    the provenance fields (``source_path`` / ``source_fem_hash`` /
    ``source_neutral_schema_version`` / ``translate`` / optional
    ``rotate`` / optional ``partition_rank`` / ``composed_at``).
    Optional ``properties`` mapping is stored under a child
    ``properties/`` sub-group (one attr per key).

    Omitted entirely when ``fem.composed_from`` is empty — absence is
    the canonical "uncomposed" signal on disk.  The group carries a
    ``composed_from_schema_version`` attr (initial value
    :data:`COMPOSED_FROM_SCHEMA_VERSION`) so future provenance-shape
    additions can bump independently of the neutral-zone schema.
    """
    composed = getattr(fem, "composed_from", None)
    if not composed:
        return

    parent = f.create_group("composed_from")
    parent.attrs["composed_from_schema_version"] = COMPOSED_FROM_SCHEMA_VERSION

    seen_safe: set[str] = set()
    for rec in composed:
        label = rec.label
        # Sanitize joined labels (e.g. "bayP/frameA" from nested
        # compose, PR #369) into HDF5-legal group names by replacing
        # "/" with "_". KNOWN COSMETIC: two distinct labels "a/b" and
        # "a_b" both sanitize to "a_b" — the dedup suffix below makes
        # the file legal but the suffix is order-dependent
        # (uses len(seen_safe) as a counter), so re-saving the same
        # FEMData in a different iteration order can produce different
        # __N suffixes on the affected groups. The verbatim ``label``
        # attribute (line below) round-trips correctly regardless, so
        # this is an INTERNAL LAYOUT detail, not a public-surface
        # hazard — readers pull from ``attrs["label"]``, never from
        # the group name. ADR 0038 amendment 2026-05-27 documents
        # this as cosmetic-only; see PR #369 implementation audit.
        safe = label.replace("/", "_") or "_unnamed"
        if safe in seen_safe:
            safe = f"{safe}__{len(seen_safe)}"
        seen_safe.add(safe)

        sub = parent.create_group(safe)
        sub.attrs["label"] = str(label)
        sub.attrs["source_path"] = str(rec.source_path)
        sub.attrs["source_fem_hash"] = str(rec.source_fem_hash)
        sub.attrs["source_neutral_schema_version"] = str(
            rec.source_neutral_schema_version
        )
        sub.attrs["translate"] = np.asarray(rec.translate, dtype=np.float64)
        sub.attrs["composed_at"] = str(rec.composed_at)
        if rec.rotate is not None:
            sub.attrs["rotate"] = np.asarray(rec.rotate, dtype=np.float64)
        if rec.partition_rank is not None:
            sub.attrs["partition_rank"] = int(rec.partition_rank)

        if rec.properties:
            props = sub.create_group("properties")
            for k, v in rec.properties.items():
                # Coerce numeric / string values onto h5py-friendly
                # attr types.  Phase 3A.1 only persists scalar
                # str/int/float — anything else surfaces an h5py error
                # at write time (intentionally; richer property
                # shapes need a follow-up schema bump).
                props.attrs[str(k)] = v


def _write_parts(fem: "FEMData", f: Any) -> None:
    """Write ``/parts/{label}/{node_ids, element_ids}`` for each Part label.

    Sourced from :attr:`NodeComposite._part_node_map` (and
    :attr:`ElementComposite._part_elem_map`) — populated by
    ``_fem_factory`` from the apeGmsh ``parts`` registry.  The two
    maps are written together as the union of label keys; a label
    may have only nodes (no elements) and vice-versa.  Omitted when
    both maps are empty.

    Added in neutral schema 2.5.0 (Phase 2 of the major refactor) so
    ``fem.nodes.select(target=part_label)`` / ``fem.elements.select
    (target=part_label)`` survive the H5 round-trip.  Per ADR 0023's
    two-version window, readers in 2.4.x silently lack this group.
    """
    node_map = getattr(fem.nodes, "_part_node_map", None) or {}
    elem_map = getattr(fem.elements, "_part_elem_map", None) or {}
    if not node_map and not elem_map:
        return

    parent = f.create_group("parts")
    labels = sorted(set(node_map.keys()) | set(elem_map.keys()))
    seen_safe: set[str] = set()
    for label in labels:
        safe = label.replace("/", "_")
        if safe in seen_safe:
            safe = f"{safe}__{labels.index(label)}"
        seen_safe.add(safe)

        sub = parent.create_group(safe)
        sub.attrs["label"] = label
        nids = np.asarray(
            sorted(int(x) for x in node_map.get(label, set())),
            dtype=np.int64,
        )
        eids = np.asarray(
            sorted(int(x) for x in elem_map.get(label, set())),
            dtype=np.int64,
        )
        sub.create_dataset("node_ids", data=nids)
        sub.create_dataset("element_ids", data=eids)


def _write_named_index_at_root(
    fem: "FEMData",
    f: Any,
    *,
    group_name: str,
    node_side: Any,
    element_side: Any,
) -> None:
    """Write ``/{group_name}/{node_side,element_side}/{safe_name}/...``.

    Neutral schema 2.10.0 sub-tree split (B2).  Each side's PG /
    label set lives in its own sub-tree so the reader never has to
    infer which side an entry came from.  An entry that exists on
    both sides (same (dim, tag) declared on both
    :attr:`NodeComposite.physical` and
    :attr:`ElementComposite.physical`) is written into both sub-trees
    — the round-trip preserves the original taxonomies exactly.

    The element-side sub-group writes its OWN ``node_ids`` /
    ``node_coords`` pulled from the element-side group_set (fixes a
    latent pre-2.10.0 writer bug where element-side PGs lost their
    node membership on disk).
    """
    node_keys = _safe_get_all(node_side)
    elem_keys = _safe_get_all(element_side)
    if not node_keys and not elem_keys:
        return

    parent = f.create_group(group_name)

    if node_keys:
        node_parent = parent.create_group("node_side")
        _write_named_index_side(
            node_parent, node_side, node_keys,
            include_element_ids=False,
        )
    if elem_keys:
        elem_parent = parent.create_group("element_side")
        _write_named_index_side(
            elem_parent, element_side, elem_keys,
            include_element_ids=True,
        )


def _write_named_index_side(
    parent: Any,
    group_set: Any,
    keys: list[tuple[int, int]],
    *,
    include_element_ids: bool,
) -> None:
    """Write one side (node or element) of the named index split.

    Each (dim, tag) becomes a sub-group named by the sanitized
    ``name`` attr.  Always writes ``node_ids`` + ``node_coords`` from
    ``group_set``; the element side additionally writes
    ``element_ids`` when present and non-empty.
    """
    seen_safe: set[str] = set()
    for dim, tag in keys:
        name = _safe_get_name(group_set, dim, tag) or f"_unnamed_{dim}_{tag}"
        safe = name.replace("/", "_")
        if safe in seen_safe:
            safe = f"{safe}__{dim}_{tag}"
        seen_safe.add(safe)

        sub = parent.create_group(safe)
        sub.attrs["dim"] = int(dim)
        sub.attrs["tag"] = int(tag)
        sub.attrs["name"] = name

        nids, ncoords = _named_node_arrays(group_set, dim, tag)
        sub.create_dataset("node_ids", data=nids)
        sub.create_dataset("node_coords", data=ncoords)

        if include_element_ids and dim >= 1:
            eids = _named_element_ids(group_set, dim, tag)
            if eids.size > 0:
                sub.create_dataset("element_ids", data=eids)


def _safe_get_all(group_set: Any) -> list[tuple[int, int]]:
    if group_set is None:
        return []
    try:
        keys = group_set.get_all()
    except (AttributeError, TypeError):
        return []
    return [(int(d), int(t)) for d, t in keys]


def _safe_get_name(group_set: Any, dim: int, tag: int) -> str | None:
    if group_set is None:
        return None
    try:
        name = group_set.get_name(dim, tag)
    except (AttributeError, KeyError, ValueError):
        return None
    return None if name is None else str(name)


def _named_node_arrays(
    group_set: Any, dim: int, tag: int,
) -> tuple[np.ndarray, np.ndarray]:
    if group_set is None:
        return (
            np.array([], dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
        )
    try:
        nids = np.asarray(group_set.node_ids((dim, tag)), dtype=np.int64)
        ncoords = np.asarray(
            group_set.node_coords((dim, tag)), dtype=np.float64,
        )
    except (KeyError, ValueError, AttributeError):
        return (
            np.array([], dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
        )
    return nids, ncoords


def _named_element_ids(group_set: Any, dim: int, tag: int) -> np.ndarray:
    if group_set is None:
        return np.array([], dtype=np.int64)
    try:
        eids = np.asarray(
            group_set.element_ids((dim, tag)), dtype=np.int64,
        )
    except (KeyError, ValueError, AttributeError):
        return np.array([], dtype=np.int64)
    return eids


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def _write_constraints(fem: "FEMData", f: Any) -> None:
    """Write ``/constraints/{kind}`` datasets using the symmetric compound.

    Iterates the broker's node-side and element-side constraint
    composites separately, binning each record by ``kind``.  Per-kind
    datasets use a per-record-type payload dtype from
    :mod:`apeGmsh.mesh._record_h5`.
    """
    from apeGmsh._kernel.records._constraints import (
        InterpolationRecord,
        NodeGroupRecord,
        NodePairRecord,
        NodeToSurfaceRecord,
        SurfaceCouplingRecord,
    )

    by_kind: dict[str, list[Any]] = {}

    def _bucket(rec: Any) -> None:
        kind = getattr(rec, "kind", None)
        if kind is None:
            return
        by_kind.setdefault(str(kind), []).append(rec)

    node_constraints = getattr(fem.nodes, "constraints", None)
    if node_constraints is not None:
        for rec in node_constraints:
            _bucket(rec)
    elem_constraints = getattr(fem.elements, "constraints", None)
    if elem_constraints is not None:
        for rec in elem_constraints:
            _bucket(rec)

    if not by_kind:
        return

    parent = f.create_group("constraints")
    for kind, records in by_kind.items():
        safe_kind = kind.replace("/", "_")
        first = records[0]
        if isinstance(first, NodePairRecord):
            _write_kind_dataset(
                parent, safe_kind, kind, records,
                node_pair_payload_dtype(), _encode_node_pair,
                target_kind="node",
            )
        elif isinstance(first, NodeGroupRecord):
            _write_kind_dataset(
                parent, safe_kind, kind, records,
                node_group_payload_dtype(), _encode_node_group,
                target_kind="node",
            )
        elif isinstance(first, InterpolationRecord):
            _write_kind_dataset(
                parent, safe_kind, kind, records,
                interpolation_payload_dtype(), _encode_interpolation,
                target_kind="node",
            )
        elif isinstance(first, SurfaceCouplingRecord):
            _write_kind_dataset(
                parent, safe_kind, kind, records,
                surface_coupling_payload_dtype(),
                _encode_surface_coupling,
                target_kind="element",
            )
        elif isinstance(first, NodeToSurfaceRecord):
            _write_kind_dataset(
                parent, safe_kind, kind, records,
                node_to_surface_payload_dtype(),
                _encode_node_to_surface,
                target_kind="node",
            )
        else:
            # Unknown record type — preserve the kind name but log via
            # an attribute so consumers can detect we lost detail.
            sub = parent.create_group(safe_kind)
            sub.attrs["__deviation__"] = (
                f"unrecognised record type {type(first).__name__}; "
                f"{len(records)} records skipped"
            )


def _write_kind_dataset(
    parent: Any,
    safe_kind: str,
    kind_label: str,
    records: list[Any],
    payload_dtype: np.dtype,
    encode_payload: Any,
    *,
    target_kind: str,
) -> None:
    """Build the symmetric-compound rows for one kind and emit the dataset."""
    outer = make_record_dtype(payload_dtype)
    rows = np.empty(len(records), dtype=outer)
    for i, rec in enumerate(records):
        rows[i] = (
            target_kind,
            _target_for(rec, target_kind),
            kind_label,
            encode_payload(rec),
        )
    parent.create_dataset(safe_kind, data=rows)


def _target_for(rec: Any, target_kind: str) -> str:
    """Best-effort string identifier for ``target`` (per symmetric contract)."""
    if target_kind == "node":
        for attr in ("slave_node", "master_node", "rebar_node"):
            v = getattr(rec, attr, None)
            if v is not None:
                return str(int(v))
    elif target_kind == "pg":
        # Rebar structural elements: the bar's physical-group label.
        return str(getattr(rec, "pg", "") or "")
    elif target_kind == "element":
        # Surface coupling: pick the first slave node as a stand-in
        # identifier (no single "element id" applies — the constraint
        # spans many).  Consumers walk the payload for full info.
        slaves = getattr(rec, "slave_nodes", None)
        if slaves:
            return str(int(slaves[0]))
    elif target_kind == "contact":
        # Contact interaction: no single node/element id applies (it spans two
        # surfaces). Use the declaration name as a best-effort identifier;
        # consumers walk the payload for the faces / formulation.
        return str(getattr(rec, "name", "") or "")
    return ""


def _encode_node_pair(rec: Any) -> tuple[Any, ...]:
    nan = float("nan")
    offset = rec.offset
    offset_arr: tuple[float, ...]
    if offset is None:
        offset_arr = (nan, nan, nan)
    else:
        offset_arr = tuple(float(x) for x in np.asarray(offset).reshape(-1)[:3])
    penalty = float(rec.penalty_stiffness) if rec.penalty_stiffness is not None else nan
    # equal_dof_mixed carries retained DOFs; every other kind leaves this
    # empty (encoded as a zero-length vlen array → decoded back to None).
    master_dofs = getattr(rec, "master_dofs", None)
    master_dofs_arr = np.asarray(
        master_dofs if master_dofs is not None else [], dtype=np.int64,
    )
    return (
        int(rec.master_node),
        int(rec.slave_node),
        np.asarray(rec.dofs, dtype=np.int64),
        offset_arr,
        penalty,
        rec.name or "",
        master_dofs_arr,
    )


def _encode_control(ctrl: Any) -> tuple[Any, ...]:
    """Encode a :class:`CouplingControl` (or ``None``) into the ``cpl_*``
    columns (schema 2.12.0; host auto-scalers 2.13.0).  ``cpl_has`` is the
    presence flag; unset numeric knobs encode as NaN; ``enforce`` encodes
    0=penalty/1=al.  ``k="auto"`` encodes as ``cpl_k_auto=1`` + ``cpl_k``
    NaN; ``cpl_host`` carries the **FEM element id** (``-1`` = none) — the
    eid is stable across emits, ops tags are not.
    """
    nan = float("nan")
    if ctrl is None:
        return (
            np.uint8(0), nan, nan, np.uint8(0), nan, np.uint8(0),
            np.uint8(0), nan, np.int64(-1), nan,
            # EmbeddedNodeControl pressure tie (schema 2.18.0).
            np.uint8(0), nan,
        )
    k_auto = ctrl.k == "auto"
    # pressure / kp live on EmbeddedNodeControl only; a base CouplingControl
    # lacks the attrs (getattr defaults keep the encode uniform).
    pressure = bool(getattr(ctrl, "pressure", False))
    kp = getattr(ctrl, "kp", None)
    return (
        np.uint8(1),
        float(ctrl.k) if ctrl.k is not None and not k_auto else nan,
        float(ctrl.kr) if ctrl.kr is not None else nan,
        np.uint8(1 if ctrl.enforce == "al" else 0),
        float(ctrl.bipenalty_dtcr) if ctrl.bipenalty_dtcr is not None else nan,
        np.uint8(1 if ctrl.absolute else 0),
        np.uint8(1 if k_auto else 0),
        float(ctrl.k_alpha) if ctrl.k_alpha is not None else nan,
        np.int64(ctrl.host) if ctrl.host is not None else np.int64(-1),
        float(ctrl.bipenalty_wcap) if ctrl.bipenalty_wcap is not None else nan,
        np.uint8(1 if pressure else 0),
        float(kp) if kp is not None else nan,
    )


def _decode_control(p: Any) -> Any:
    """Reconstruct a :class:`CouplingControl` from the ``cpl_*`` columns,
    or ``None`` (pre-2.12.0 files lack the columns → probe ``dtype.names``;
    ``cpl_has == 0`` means the record carried no control).  The 2.13.0
    host auto-scaler columns are presence-probed independently
    (``cpl_k_auto``) so 2.12.0 files decode with the v1 knobs only."""
    names = set(p.dtype.names or ())
    if "cpl_has" not in names:
        return None
    # schema 2.18.0 pressure-tie columns — presence-probed independently.
    pk: dict[str, Any] = {}
    if "cpl_pressure" in names:
        pk = dict(pressure=p["cpl_pressure"], kp=p["cpl_kp"])
    if "cpl_k_auto" in names:
        return _control_from_values(
            p["cpl_has"], p["cpl_k"], p["cpl_kr"],
            p["cpl_enforce"], p["cpl_dtcr"], p["cpl_absolute"],
            k_auto=p["cpl_k_auto"], k_alpha=p["cpl_k_alpha"],
            host=p["cpl_host"], wcap=p["cpl_wcap"], **pk,
        )
    return _control_from_values(
        p["cpl_has"], p["cpl_k"], p["cpl_kr"],
        p["cpl_enforce"], p["cpl_dtcr"], p["cpl_absolute"], **pk,
    )


def _control_from_values(
    has: Any, k: Any, kr: Any, enforce: Any, dtcr: Any, absolute: Any,
    *, k_auto: Any = None, k_alpha: Any = None,
    host: Any = None, wcap: Any = None,
    pressure: Any = None, kp: Any = None,
) -> Any:
    """Values-level core of :func:`_decode_control` — also used by the
    ``sr_cpl_*`` lane decode in :func:`_decode_surface_coupling`, where
    the columns arrive as per-slave vlen array elements instead of
    scalar payload columns.  The host auto-scaler values (schema
    2.13.0) are keyword-only and stay ``None`` for 2.12.0 files that
    lack the columns — the decode then yields the v1 knobs only.  The
    EmbeddedNodeControl pressure-tie values (schema 2.18.0) are likewise
    keyword-only; when ``pressure`` is present and set, the reconstructed
    object is an :class:`EmbeddedNodeControl` (else a base
    :class:`CouplingControl`)."""
    if not int(has):
        return None
    from apeGmsh._kernel._coupling_control import CouplingControl
    k_val: Any = _opt_scalar(k)
    extras: dict[str, Any] = {}
    if k_auto is not None:
        if int(k_auto):
            k_val = "auto"
        host_eid = int(host)
        extras = dict(
            k_alpha=_opt_scalar(k_alpha),
            host=host_eid if host_eid >= 0 else None,
            bipenalty_wcap=_opt_scalar(wcap),
        )
    common = dict(
        k=k_val,
        kr=_opt_scalar(kr),
        enforce=("al" if int(enforce) else "penalty"),
        bipenalty_dtcr=_opt_scalar(dtcr),
        absolute=bool(int(absolute)),
        **extras,
    )
    # schema 2.18.0 — EmbeddedNodeControl iff the pressure tie is set.
    if pressure is not None and int(pressure):
        from apeGmsh._kernel._coupling_control import EmbeddedNodeControl
        return EmbeddedNodeControl(
            pressure=True, kp=_opt_scalar(kp), **common,
        )
    return CouplingControl(**common)


def _encode_node_group(rec: Any) -> tuple[Any, ...]:
    nan = float("nan")
    offsets = rec.offsets
    if offsets is None:
        offsets_flat = np.array([], dtype=np.float64)
    else:
        offsets_flat = np.asarray(offsets, dtype=np.float64).reshape(-1)
    plane = rec.plane_normal
    plane_arr: tuple[float, ...]
    if plane is None:
        plane_arr = (nan, nan, nan)
    else:
        plane_arr = tuple(float(x) for x in np.asarray(plane).reshape(-1)[:3])
    # LadrunoRigidBody emission (schema 2.19.0; rigid_body only).
    as_element = bool(getattr(rec, "as_element", False))
    mass = getattr(rec, "mass", None)
    omega = getattr(rec, "omega", None)
    omega_arr: tuple[float, ...] = (
        (nan, nan, nan) if omega is None
        else tuple(float(w) for w in omega)
    )
    return (
        int(rec.master_node),
        np.asarray(rec.slave_nodes, dtype=np.int64),
        np.asarray(rec.dofs, dtype=np.int64),
        offsets_flat,
        plane_arr,
        rec.name or "",
        *_encode_control(getattr(rec, "control", None)),
        np.uint8(1 if as_element else 0),
        float(mass) if mass is not None else nan,
        omega_arr,
    )


def _encode_interpolation(rec: Any) -> tuple[Any, ...]:
    nan = float("nan")
    weights = rec.weights
    if weights is None:
        weights_arr = np.array([], dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    pp = rec.projected_point
    pp_arr = (
        tuple(float(x) for x in np.asarray(pp).reshape(-1)[:3])
        if pp is not None else (nan, nan, nan)
    )
    pc = rec.parametric_coords
    pc_arr = (
        tuple(float(x) for x in np.asarray(pc).reshape(-1)[:2])
        if pc is not None else (nan, nan)
    )
    # ASDEmbeddedNodeElement options (schema 2.8.0).  ``has_stiffness_p``
    # is the presence flag because 0 is valid penalty data; ``excess``
    # NaN-encodes None.
    has_kp = rec.stiffness_p is not None
    return (
        int(rec.slave_node),
        np.asarray(rec.master_nodes, dtype=np.int64),
        weights_arr,
        np.asarray(rec.dofs, dtype=np.int64),
        pp_arr,
        pc_arr,
        rec.name or "",
        float(rec.stiffness),
        float(rec.stiffness_p) if has_kp else nan,
        np.uint8(1 if has_kp else 0),
        np.uint8(1 if rec.rotational else 0),
        np.uint8(1 if rec.pressure else 0),
        float(rec.excess) if rec.excess is not None else nan,
        getattr(rec, "enforce", "penalty") or "penalty",   # ADR 0068
        *_encode_control(getattr(rec, "control", None)),
    )


#: enforce route ↔ uint8 code for the sr_enforce surface-coupling column
#: (ADR 0068). The single-tie InterpolationRecord lane stores enforce as a
#: string; the per-slave-record sr_ arrays use a compact code to match the
#: other sr_ uint8 columns.
_SR_ENFORCE_CODE = {"penalty": 0, "penalty_al": 1, "equation": 2}
_SR_ENFORCE_NAME = {0: "penalty", 1: "penalty_al", 2: "equation"}


def _encode_surface_coupling(rec: Any) -> tuple[Any, ...]:
    op = rec.mortar_operator
    op_shape: tuple[int, ...]
    if op is None:
        op_arr = np.array([], dtype=np.float64)
        op_shape = (0, 0)
    else:
        op_np = np.asarray(op, dtype=np.float64)
        op_shape = tuple(int(s) for s in op_np.shape[:2])
        if len(op_shape) < 2:
            op_shape = (op_shape[0] if op_shape else 0, 0)
        op_arr = op_np.reshape(-1)
    # slave_records (ragged list[InterpolationRecord]) — CSR-flatten so
    # tied_contact (no mortar_operator) round-trips losslessly.
    srs = list(getattr(rec, "slave_records", []) or [])
    sr_slave_nodes: list[int] = []
    sr_master_counts: list[int] = []
    sr_master_nodes: list[int] = []
    sr_weights: list[float] = []
    sr_dof_counts: list[int] = []
    sr_dofs: list[int] = []
    sr_projected: list[float] = []
    sr_parametric: list[float] = []
    # ASDEmbeddedNodeElement options per slave record (schema 2.8.0)
    sr_stiffness: list[float] = []
    sr_stiffness_p: list[float] = []
    sr_has_stiffness_p: list[int] = []
    sr_rotational: list[int] = []
    sr_pressure: list[int] = []
    sr_excess: list[float] = []
    sr_enforce: list[int] = []   # ADR 0068 (uint8 code per slave record)
    # CouplingControl per slave record (schema 2.12.0 sr_cpl_* mirror;
    # host auto-scalers 2.13.0)
    sr_cpl_has: list[Any] = []
    sr_cpl_k: list[float] = []
    sr_cpl_kr: list[float] = []
    sr_cpl_enforce: list[Any] = []
    sr_cpl_dtcr: list[float] = []
    sr_cpl_absolute: list[Any] = []
    sr_cpl_k_auto: list[Any] = []
    sr_cpl_k_alpha: list[float] = []
    sr_cpl_host: list[Any] = []
    sr_cpl_wcap: list[float] = []
    sr_cpl_pressure: list[Any] = []
    sr_cpl_kp: list[float] = []
    nan = float("nan")
    for ir in srs:
        m = [int(x) for x in np.asarray(ir.master_nodes).reshape(-1)]
        if ir.weights is None:
            w = [nan] * len(m)
        else:
            w = [float(x) for x in np.asarray(ir.weights).reshape(-1)]
        d = [int(x) for x in np.asarray(ir.dofs).reshape(-1)]
        sr_slave_nodes.append(int(ir.slave_node))
        sr_master_counts.append(len(m))
        sr_master_nodes.extend(m)
        sr_weights.extend(w)
        sr_dof_counts.append(len(d))
        sr_dofs.extend(d)
        pp = ir.projected_point
        sr_projected.extend(
            tuple(float(x) for x in np.asarray(pp).reshape(-1)[:3])
            if pp is not None else (nan, nan, nan))
        pc = ir.parametric_coords
        sr_parametric.extend(
            tuple(float(x) for x in np.asarray(pc).reshape(-1)[:2])
            if pc is not None else (nan, nan))
        has_kp = ir.stiffness_p is not None
        sr_stiffness.append(float(ir.stiffness))
        sr_stiffness_p.append(float(ir.stiffness_p) if has_kp else nan)
        sr_has_stiffness_p.append(1 if has_kp else 0)
        sr_rotational.append(1 if ir.rotational else 0)
        sr_pressure.append(1 if ir.pressure else 0)
        sr_excess.append(float(ir.excess) if ir.excess is not None else nan)
        sr_enforce.append(
            _SR_ENFORCE_CODE.get(getattr(ir, "enforce", "penalty"), 0))
        (c_has, c_k, c_kr, c_enf, c_dtcr, c_abs,
         c_auto, c_alpha, c_host, c_wcap,
         c_pressure, c_kp) = _encode_control(ir.control)
        sr_cpl_has.append(c_has)
        sr_cpl_k.append(c_k)
        sr_cpl_kr.append(c_kr)
        sr_cpl_enforce.append(c_enf)
        sr_cpl_dtcr.append(c_dtcr)
        sr_cpl_absolute.append(c_abs)
        sr_cpl_k_auto.append(c_auto)
        sr_cpl_k_alpha.append(c_alpha)
        sr_cpl_host.append(c_host)
        sr_cpl_wcap.append(c_wcap)
        sr_cpl_pressure.append(c_pressure)
        sr_cpl_kp.append(c_kp)
    return (
        np.asarray(rec.master_nodes, dtype=np.int64),
        np.asarray(rec.slave_nodes, dtype=np.int64),
        np.asarray(rec.dofs, dtype=np.int64),
        op_shape,
        op_arr,
        np.asarray(sr_slave_nodes, dtype=np.int64),
        np.asarray(sr_master_counts, dtype=np.int64),
        np.asarray(sr_master_nodes, dtype=np.int64),
        np.asarray(sr_weights, dtype=np.float64),
        np.asarray(sr_dof_counts, dtype=np.int64),
        np.asarray(sr_dofs, dtype=np.int64),
        np.asarray(sr_projected, dtype=np.float64),
        np.asarray(sr_parametric, dtype=np.float64),
        rec.name or "",
        np.asarray(sr_stiffness, dtype=np.float64),
        np.asarray(sr_stiffness_p, dtype=np.float64),
        np.asarray(sr_has_stiffness_p, dtype=np.uint8),
        np.asarray(sr_rotational, dtype=np.uint8),
        np.asarray(sr_pressure, dtype=np.uint8),
        np.asarray(sr_excess, dtype=np.float64),
        np.asarray(sr_enforce, dtype=np.uint8),
        np.asarray(sr_cpl_has, dtype=np.uint8),
        np.asarray(sr_cpl_k, dtype=np.float64),
        np.asarray(sr_cpl_kr, dtype=np.float64),
        np.asarray(sr_cpl_enforce, dtype=np.uint8),
        np.asarray(sr_cpl_dtcr, dtype=np.float64),
        np.asarray(sr_cpl_absolute, dtype=np.uint8),
        np.asarray(sr_cpl_k_auto, dtype=np.uint8),
        np.asarray(sr_cpl_k_alpha, dtype=np.float64),
        np.asarray(sr_cpl_host, dtype=np.int64),
        np.asarray(sr_cpl_wcap, dtype=np.float64),
        np.asarray(sr_cpl_pressure, dtype=np.uint8),
        np.asarray(sr_cpl_kp, dtype=np.float64),
    )


def _encode_node_to_surface(rec: Any) -> tuple[Any, ...]:
    coords = rec.phantom_coords
    if coords is None:
        coords_flat = np.array([], dtype=np.float64)
    else:
        coords_flat = np.asarray(coords, dtype=np.float64).reshape(-1)
    return (
        int(rec.master_node),
        np.asarray(rec.slave_nodes, dtype=np.int64),
        np.asarray(rec.phantom_nodes, dtype=np.int64),
        coords_flat,
        np.asarray(rec.dofs, dtype=np.int64),
        rec.name or "",
    )


# ---------------------------------------------------------------------------
# Loads
# ---------------------------------------------------------------------------


def _write_reinforce_ties(fem: "FEMData", f: Any) -> None:
    """Write ``/reinforce_ties/ties`` from ``fem.elements.reinforce_ties``.

    A dedicated group (not under ``/constraints/``, whose reader dispatches
    by payload-field subset match and would mis-route these) holding one
    symmetric-compound dataset of :class:`ReinforceTieRecord` rows. Omitted
    entirely when there are no ties, so a tie-free model stays byte-stable.
    """
    ties = getattr(fem.elements, "reinforce_ties", None)
    if not ties:
        return
    parent = f.create_group("reinforce_ties")
    _write_kind_dataset(
        parent, "ties", "reinforce_tie", list(ties),
        reinforce_tie_payload_dtype(), _encode_reinforce_tie,
        target_kind="node",
    )


def _encode_reinforce_tie(rec: Any) -> tuple[Any, ...]:
    """Encode a :class:`ReinforceTieRecord` into the reinforce-tie payload.

    Optional floats → NaN sentinel; optional strings → ``""``; the
    geometric vlen/fixed fields carry a ``has_*`` flag so None survives
    distinct from empty/zero.
    """
    nan = float("nan")
    host = np.asarray(rec.host_nodes, dtype=np.int64).reshape(-1)
    if rec.weights is None:
        weights = np.empty(0, dtype=np.float64)
        has_w = np.uint8(0)
    else:
        weights = np.asarray(rec.weights, dtype=np.float64).reshape(-1)
        has_w = np.uint8(1)
    # Fail loud on a malformed tie at the serialization boundary, rather
    # than writing a record that decodes to garbage or emits an invalid
    # LadrunoEmbeddedRebar (adversarial-review findings C0/C1/C2):
    #   * a tie must couple the rebar to >= 1 host node, and
    #   * weights (when present) must be parallel to host_nodes.
    # An empty weights array round-trips as the has_weights=0 "None" case,
    # so reject it explicitly to keep None vs [] unambiguous.
    if host.size == 0:
        raise ValueError(
            f"reinforce tie at rebar node {rec.rebar_node}: host_nodes is "
            f"empty — a tie must couple the rebar to at least one host node.")
    if has_w and weights.size == 0:
        raise ValueError(
            f"reinforce tie at rebar node {rec.rebar_node}: weights is an "
            f"empty array — pass weights=None or the shape-function weights.")
    if has_w and weights.size != host.size:
        raise ValueError(
            f"reinforce tie at rebar node {rec.rebar_node}: weights length "
            f"{weights.size} != host_nodes length {host.size} (weights must "
            f"be parallel to host_nodes).")
    if rec.direction is None:
        direction = (nan, nan, nan)
        has_d = np.uint8(0)
    else:
        direction = tuple(
            float(x) for x in np.asarray(rec.direction, dtype=np.float64).reshape(-1)[:3])
        has_d = np.uint8(1)

    # Co-rotated bar-axis point-B weights (ADR 20 §10.5). Like `weights`, when
    # present they must be parallel to host_nodes (the fork reads them as
    # NshapeB over the same host node list). A corot tie must carry them.
    if rec.shape_b is None:
        shape_b = np.empty(0, dtype=np.float64)
        has_sb = np.uint8(0)
    else:
        shape_b = np.asarray(rec.shape_b, dtype=np.float64).reshape(-1)
        has_sb = np.uint8(1)
    if has_sb and shape_b.size != host.size:
        raise ValueError(
            f"reinforce tie at rebar node {rec.rebar_node}: shape_b length "
            f"{shape_b.size} != host_nodes length {host.size} (the corot "
            f"point-B weights must be parallel to host_nodes).")
    if rec.corot and not has_sb:
        raise ValueError(
            f"reinforce tie at rebar node {rec.rebar_node}: corot=True needs "
            f"shape_b (the -shapeB point-B weights).")

    def _f(v: Any) -> float:
        return float(v) if v is not None else nan

    return (
        int(rec.rebar_node),
        host,
        weights,
        has_w,
        direction,
        has_d,
        np.uint8(1 if rec.corot else 0),
        shape_b,
        has_sb,
        _f(rec.bond_scale),
        rec.bond or "",
        _f(rec.perfect),
        _f(rec.kt),
        _f(rec.kt_alpha),
        rec.enforce or "penalty",
        np.uint8(1 if rec.bipenalty else 0),
        _f(rec.dtcr),
        _f(rec.excess),
        np.uint8(1 if rec.in_bounds else 0),
        rec.name or "",
    )


def _write_embed_ties(fem: "FEMData", f: Any) -> None:
    """Write ``/embed_ties/ties`` from ``fem.elements.embed_ties``.

    A dedicated group (its own group, like ``/reinforce_ties`` — not under
    ``/constraints/``) holding one symmetric-compound dataset of
    :class:`EmbedTieRecord` rows (the g.embed ``LadrunoEmbeddedNode``
    node-to-host couplings). Omitted entirely when there are no ties, so an
    embedment-free model stays byte-stable.
    """
    ties = getattr(fem.elements, "embed_ties", None)
    if not ties:
        return
    parent = f.create_group("embed_ties")
    _write_kind_dataset(
        parent, "ties", "embed_tie", list(ties),
        embed_tie_payload_dtype(), _encode_embed_tie,
        target_kind="node",
    )


def _encode_embed_tie(rec: Any) -> tuple[Any, ...]:
    """Encode an :class:`EmbedTieRecord` into the embed-tie payload.

    Fails loud on a malformed tie (empty host_nodes, or weights not parallel
    to host_nodes) so a corrupt record can't decode to garbage or emit an
    invalid LadrunoEmbeddedNode (mirror of :func:`_encode_reinforce_tie`).
    """
    nan = float("nan")
    host = np.asarray(rec.host_nodes, dtype=np.int64).reshape(-1)
    if rec.weights is None:
        weights = np.empty(0, dtype=np.float64)
        has_w = np.uint8(0)
    else:
        weights = np.asarray(rec.weights, dtype=np.float64).reshape(-1)
        has_w = np.uint8(1)
    if host.size == 0:
        raise ValueError(
            f"embed tie at node {rec.node}: host_nodes is empty — a tie must "
            f"couple the node to at least one host node.")
    if has_w and weights.size == 0:
        raise ValueError(
            f"embed tie at node {rec.node}: weights is an empty array — pass "
            f"weights=None or the shape-function weights.")
    if has_w and weights.size != host.size:
        raise ValueError(
            f"embed tie at node {rec.node}: weights length {weights.size} != "
            f"host_nodes length {host.size} (weights must be parallel to "
            f"host_nodes).")

    def _f(v: Any) -> float:
        return float(v) if v is not None else nan

    return (
        int(rec.node),
        host,
        weights,
        has_w,
        _f(rec.k),
        _f(rec.k_alpha),
        rec.enforce or "penalty",
        np.uint8(1 if rec.bipenalty else 0),
        _f(rec.dtcr),
        np.uint8(1 if rec.staged else 0),
        _f(rec.excess),
        np.uint8(1 if rec.in_bounds else 0),
        rec.name or "",
    )


def _write_rebar_elements(fem: "FEMData", f: Any) -> None:
    """Write ``/rebar_elements/elements`` from ``fem.elements.rebar_elements``.

    A dedicated group (its own group, like ``/reinforce_ties`` — not under
    ``/constraints/``) holding one symmetric-compound dataset of
    :class:`RebarElementRecord` rows (the cage's auto-emitted structural
    elements, ADR 0067 P5.2 / B1a.2). Omitted when there are none, so a
    cage that didn't opt into ``emit_elements`` stays byte-stable.
    """
    recs = getattr(fem.elements, "rebar_elements", None)
    if not recs:
        return
    parent = f.create_group("rebar_elements")
    _write_kind_dataset(
        parent, "elements", "rebar_element", list(recs),
        rebar_element_payload_dtype(), _encode_rebar_element,
        target_kind="pg",
    )


def _encode_rebar_element(rec: Any) -> tuple[Any, ...]:
    """Encode a :class:`RebarElementRecord` into the rebar-element payload.

    Fails loud on a malformed record (empty connectivity, or a flat array
    whose length isn't even) so a corrupt record can't decode to garbage or
    emit a dangling element.
    """
    flat = np.asarray(
        [int(n) for pair in rec.connectivity for n in pair], dtype=np.int64)
    if flat.size == 0:
        raise ValueError(
            f"rebar element on PG {rec.pg!r}: connectivity is empty — a bar "
            f"must have at least one 2-node line cell.")
    if flat.size % 2 != 0:
        raise ValueError(
            f"rebar element on PG {rec.pg!r}: connectivity flat length "
            f"{flat.size} is odd — cells must be (i, j) node pairs.")
    return (
        rec.pg or "",
        rec.element or "",
        rec.material or "",
        float(rec.area),
        rec.role or "",
        flat,
        int(flat.size // 2),
    )


def _write_contacts(fem: "FEMData", f: Any) -> None:
    """Write ``/contacts/contacts`` from ``fem.elements.contacts``.

    A dedicated group (its own group, like ``/reinforce_ties`` — not under
    ``/constraints/``, whose subset-match reader would mis-route it; contacts
    are a serial-only fork subsystem on ``fem.elements.contacts``) holding one
    symmetric-compound dataset of :class:`ContactRecord` rows (ADR 0073).
    Omitted entirely when there are no contacts, so a contact-free model stays
    byte-stable (and ``snapshot_id`` is unchanged — the hash does not cover
    contacts).
    """
    contacts = getattr(fem.elements, "contacts", None)
    if not contacts:
        return
    parent = f.create_group("contacts")
    _write_kind_dataset(
        parent, "contacts", "contact", list(contacts),
        contact_payload_dtype(), _encode_contact,
        target_kind="contact",
    )


def _auto_or_pos_mode(v: Any) -> tuple[float, int]:
    """Encode a ``float | "auto" | None`` penalty as (value, mode): mode 0 ⇒
    None, 1 ⇒ "auto", 2 ⇒ numeric (value carried; NaN otherwise)."""
    nan = float("nan")
    if v is None:
        return nan, 0
    if isinstance(v, str):
        if v != "auto":
            raise ValueError(f"contact penalty: expected a number or 'auto', "
                             f"got {v!r}")
        return nan, 1
    return float(v), 2


def _encode_contact(rec: Any) -> tuple[Any, ...]:
    """Encode a :class:`ContactRecord` into the contact payload (inverse of
    :func:`_decode_contact`).

    Fails loud on a malformed record (empty master faces, a flat length not a
    multiple of the stride, or a slave side inconsistent with the formulation)
    so a corrupt record can't decode to garbage or emit an invalid contact.
    """
    nan = float("nan")

    def _f(v: Any) -> float:
        return float(v) if v is not None else nan

    if rec.master_faces is None:
        raise ValueError("contact: master_faces is None — a contact needs a "
                         "faceted master surface.")
    master = np.asarray(rec.master_faces, dtype=np.int64).reshape(-1)
    m_nps = int(rec.master_nps)
    if master.size == 0:
        raise ValueError("contact: master_faces is empty.")
    if m_nps not in (3, 4) or master.size % m_nps != 0:
        raise ValueError(
            f"contact: master flat length {master.size} is not a multiple of "
            f"master_nps={m_nps} (expected 3 or 4).")

    # Slave side: exactly one of node-set (NTS) / faceted (mortar).
    if rec.formulation == "nts":
        if not rec.slave_nodes:
            raise ValueError("contact (nts): slave_nodes is empty.")
        slave_nodes = np.asarray(
            [int(n) for n in rec.slave_nodes], dtype=np.int64)
        has_sn = np.uint8(1)
        slave_faces = np.empty(0, dtype=np.int64)
        s_nps = 0
        has_sf = np.uint8(0)
    else:  # mortar — faceted slave
        if rec.slave_faces is None:
            raise ValueError("contact (mortar): slave_faces is None.")
        slave_faces = np.asarray(rec.slave_faces, dtype=np.int64).reshape(-1)
        s_nps = int(rec.slave_nps)
        if slave_faces.size == 0 or s_nps not in (3, 4) \
                or slave_faces.size % s_nps != 0:
            raise ValueError(
                f"contact (mortar): slave flat length {slave_faces.size} is "
                f"not a multiple of slave_nps={s_nps} (expected 3 or 4).")
        has_sf = np.uint8(1)
        slave_nodes = np.empty(0, dtype=np.int64)
        has_sn = np.uint8(0)

    if rec.outward is None:
        outward = (nan, nan, nan)
        has_o = np.uint8(0)
    else:
        outward = tuple(
            float(x) for x in np.asarray(rec.outward, dtype=np.float64).reshape(-1)[:3])
        has_o = np.uint8(1)

    kn_v, kn_mode = _auto_or_pos_mode(rec.kn)
    eps_n_v, eps_n_mode = _auto_or_pos_mode(rec.eps_n)
    eps_t_v, eps_t_mode = _auto_or_pos_mode(rec.eps_t)

    # soft: None/False ⇒ 0; True ⇒ 1 (bare); numeric ⇒ 2.
    if rec.soft is None or rec.soft is False:
        soft_v, soft_mode = nan, 0
    elif rec.soft is True:
        soft_v, soft_mode = nan, 1
    else:
        soft_v, soft_mode = float(rec.soft), 2

    # Edge-edge fallback (ADR-57 E2–E7, neutral 2.25.0). edge_kn is the
    # auto/None/numeric tri-state; edge_soft mirrors soft (None/bare/numeric).
    edge_kn_v, edge_kn_mode = _auto_or_pos_mode(rec.edge_kn)
    if rec.edge_soft is None or rec.edge_soft is False:
        edge_soft_v, edge_soft_mode = nan, 0
    elif rec.edge_soft is True:
        edge_soft_v, edge_soft_mode = nan, 1
    else:
        edge_soft_v, edge_soft_mode = float(rec.edge_soft), 2

    return (
        rec.formulation,
        master,
        m_nps,
        slave_nodes,
        has_sn,
        slave_faces,
        s_nps,
        has_sf,
        outward,
        has_o,
        kn_v, np.uint8(kn_mode),
        _f(rec.kt),
        _f(rec.mu),
        eps_n_v, np.uint8(eps_n_mode),
        eps_t_v, np.uint8(eps_t_mode),
        _f(rec.cohesion),
        _f(rec.tau_max),
        _f(rec.aug_tol),
        _f(rec.max_aug),
        _f(rec.ngp),
        np.uint8(1 if rec.tie else 0),
        soft_v, np.uint8(soft_mode),
        _f(rec.visc),
        np.uint8(1 if rec.consistent_tan else 0),
        np.uint8(1 if rec.geom_tan else 0),
        _f(rec.cell),
        np.uint8(1 if rec.edge_edge else 0),
        edge_kn_v, np.uint8(edge_kn_mode),
        _f(rec.edge_band),
        _f(rec.edge_mu),
        _f(rec.edge_kt),
        _f(rec.edge_cohesion),
        _f(rec.edge_tau_max),
        np.uint8(1 if rec.edge_consistent_tan else 0),
        edge_soft_v, np.uint8(edge_soft_mode),
        np.uint8(1 if rec.edge_alm else 0),
        _f(rec.edge_aug_tol),
        rec.name or "",
    )


def _write_contact_planes(fem: "FEMData", f: Any) -> None:
    """Write ``/contact_planes/contact_planes`` from
    ``fem.elements.contact_planes``. Its own group (like ``/contacts``);
    omitted entirely when there are none, so a plane-free model stays
    byte-stable (``snapshot_id`` is unchanged — the hash does not cover
    contact planes)."""
    planes = getattr(fem.elements, "contact_planes", None)
    if not planes:
        return
    parent = f.create_group("contact_planes")
    _write_kind_dataset(
        parent, "contact_planes", "contact_plane", list(planes),
        contact_plane_payload_dtype(), _encode_contact_plane,
        target_kind="contact_plane",
    )


def _encode_contact_plane(rec: Any) -> tuple[Any, ...]:
    """Encode a :class:`ContactPlaneRecord` into the contact-plane payload
    (inverse of :func:`_decode_contact_plane`). Fails loud on a malformed
    record (empty slave set, missing normal/point/kn)."""
    nan = float("nan")
    if not rec.slave_nodes:
        raise ValueError("contact_plane: slave_nodes is empty.")
    slave_nodes = np.asarray([int(n) for n in rec.slave_nodes], dtype=np.int64)
    if rec.normal is None or rec.point is None:
        raise ValueError("contact_plane: normal and point are required.")
    normal = tuple(
        float(x) for x in np.asarray(rec.normal, dtype=np.float64).reshape(-1)[:3])
    point = tuple(
        float(x) for x in np.asarray(rec.point, dtype=np.float64).reshape(-1)[:3])
    if len(normal) != 3 or len(point) != 3:
        raise ValueError("contact_plane: normal/point must be 3-vectors.")
    if rec.kn is None:
        raise ValueError("contact_plane: kn (normal penalty) is required.")

    # soft: None/False ⇒ 0; True ⇒ 1 (bare); numeric ⇒ 2.
    if rec.soft is None or rec.soft is False:
        soft_v, soft_mode = nan, 0
    elif rec.soft is True:
        soft_v, soft_mode = nan, 1
    else:
        soft_v, soft_mode = float(rec.soft), 2

    return (
        slave_nodes,
        normal,
        point,
        float(rec.kn),
        float(rec.visc) if rec.visc is not None else nan,
        soft_v, np.uint8(soft_mode),
        rec.name or "",
    )


def _write_loads(fem: "FEMData", f: Any) -> None:
    """Write ``/loads/{kind}/{pattern}`` per pattern + kind.

    Nodal loads land under ``/loads/nodal/{pattern}/``; element loads
    under ``/loads/element/{pattern}/``.  SP (single-point) records
    land under ``/loads/sp/{pattern_or_default}`` for symmetry with
    the other load kinds.
    """
    nodal_loads = getattr(fem.nodes, "loads", None)
    elem_loads = getattr(fem.elements, "loads", None)
    sp_loads = getattr(fem.nodes, "sp", None)

    has_nodal = bool(nodal_loads) if nodal_loads is not None else False
    has_elem = bool(elem_loads) if elem_loads is not None else False
    has_sp = bool(sp_loads) if sp_loads is not None else False

    if not (has_nodal or has_elem or has_sp):
        return

    parent = f.create_group("loads")

    if has_nodal:
        _write_nodal_loads(parent.create_group("nodal"), nodal_loads)
    if has_elem:
        _write_element_loads(parent.create_group("element"), elem_loads)
    if has_sp:
        _write_sp_loads(parent.create_group("sp"), sp_loads)


def _write_nodal_loads(parent: Any, load_set: Any) -> None:
    nan = float("nan")
    outer = make_record_dtype(nodal_load_payload_dtype())
    for pattern in load_set.patterns():
        records = load_set.by_pattern(pattern)
        if not records:
            continue
        rows = np.empty(len(records), dtype=outer)
        for i, rec in enumerate(records):
            force = rec.force_xyz or (nan, nan, nan)
            moment = rec.moment_xyz or (nan, nan, nan)
            rows[i] = (
                "node", str(int(rec.node_id)), "nodal",
                (int(rec.node_id), tuple(float(x) for x in force),
                 tuple(float(x) for x in moment), rec.name or ""),
            )
        safe = str(pattern).replace("/", "_") or "default"
        parent.create_dataset(safe, data=rows)


def _write_element_loads(parent: Any, load_set: Any) -> None:
    outer = make_record_dtype(element_load_payload_dtype())
    for pattern in load_set.patterns():
        records = load_set.by_pattern(pattern)
        if not records:
            continue
        rows = np.empty(len(records), dtype=outer)
        for i, rec in enumerate(records):
            params_json = json.dumps(rec.params, default=_json_default)
            rows[i] = (
                "element", str(int(rec.element_id)), "element",
                (int(rec.element_id), str(rec.load_type), params_json,
                 rec.name or ""),
            )
        safe = str(pattern).replace("/", "_") or "default"
        parent.create_dataset(safe, data=rows)


def _write_sp_loads(parent: Any, sp_set: Any) -> None:
    outer = make_record_dtype(sp_payload_dtype())
    # SPSet has no pattern attr per record beyond the LoadRecord base;
    # group all records under a single ``default`` dataset.
    rows = np.empty(len(sp_set), dtype=outer)
    for i, rec in enumerate(sp_set):
        rows[i] = (
            "node", str(int(rec.node_id)), "sp",
            (
                int(rec.node_id), int(rec.dof),
                float(rec.value), int(bool(rec.is_homogeneous)),
                rec.name or "",
            ),
        )
    parent.create_dataset("default", data=rows)


def _json_default(obj: Any) -> Any:
    """Fallback for ``json.dumps`` on non-JSON types (numpy scalars)."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Masses
# ---------------------------------------------------------------------------


def _write_masses(fem: "FEMData", f: Any) -> None:
    """Write ``/masses`` — one symmetric-compound row per :class:`MassRecord`.

    ADR 0065 v2 / plan_emit_memory_columnar.md C1–C3: fills the compound
    payload directly from the columnar :class:`MassSet` columns
    (``node_ids()`` / ``mass_array()``) so a multi-million-node save never
    boxes one ``MassRecord`` per node just to write. Byte-identical to the
    per-record fill: the stored ``node_id`` / ``float64[6]`` payload and
    the ``str(node_id)`` target are the same values.
    """
    mass_set = getattr(fem.nodes, "masses", None)
    if not mass_set:
        return

    outer = make_record_dtype(mass_payload_dtype())
    n = len(mass_set)
    rows = np.empty(n, dtype=outer)

    # Prefer the columnar fast path; fall back to record iteration for any
    # non-columnar set (e.g. a stub in tests).
    node_ids_fn = getattr(mass_set, "node_ids", None)
    mass_arr_fn = getattr(mass_set, "mass_array", None)
    if callable(node_ids_fn) and callable(mass_arr_fn):
        node_ids = np.asarray(node_ids_fn(), dtype=np.int64)
        mass = np.asarray(mass_arr_fn(), dtype=np.float64)
        names = getattr(mass_set, "_names", {}) or {}
        rows["target_kind"] = "node"
        rows["payload_kind"] = "mass"
        rows["target"] = [str(int(t)) for t in node_ids]
        payload = rows["payload"]
        payload["node_id"] = node_ids
        payload["mass"] = mass
        payload["name"] = [names.get(i, "") for i in range(n)]
    else:  # pragma: no cover - defensive record-path fallback
        for i, rec in enumerate(mass_set):
            mass_tuple = tuple(float(x) for x in tuple(rec.mass)[:6])
            if len(mass_tuple) < 6:
                mass_tuple = mass_tuple + (0.0,) * (6 - len(mass_tuple))
            rows[i] = (
                "node", str(int(rec.node_id)), "mass",
                (int(rec.node_id), mass_tuple, rec.name or ""),
            )
    f.create_dataset("masses", data=rows)


# ===========================================================================
# Reader (Phase session-save 2 — root-layout inverse of write_fem_h5)
# ===========================================================================
#
# Mirrors the writer's seven neutral-zone groups plus ``/meta``.  Schema
# major version is checked; minor differences are tolerated (additive).
# Companion to :func:`write_fem_h5` — together they form the contract
# that ``apeGmsh(save_to=...)`` round-trips through.


def read_fem_h5(path: str, *, root: str = "/") -> "FEMData":
    """Reconstruct a :class:`FEMData` from a root-layout ``model.h5``.

    Inverse of :func:`write_fem_h5`.  Reads the seven neutral-zone
    groups (``/nodes``, ``/elements``, ``/physical_groups``,
    ``/labels``, ``/mesh_selections``, ``/constraints``, ``/loads``,
    ``/masses``) plus ``/meta``, and rebuilds a fully populated
    ``FEMData`` snapshot.  Empty groups are skipped — absence is the
    right "no records" signal on disk.

    Parameters
    ----------
    path : str
        Path to a model.h5 written by ``g.save()``,
        ``apeGmsh(save_to=...)``, or :func:`write_fem_h5`.
    root : str, default ``"/"``
        Sub-group root inside ``path`` to read from.  Default reads a
        standalone ``model.h5`` (rich layout at the file root).  Per
        ADR 0020, composed ``results.h5`` files carry the same rich
        layout under ``/model/``; pass ``root="/model"`` to rehydrate
        from there.  Backcompat: the default produces byte-identical
        behaviour to the pre-refactor implementation.

    Raises
    ------
    apeGmsh.opensees.emitter.h5_reader.SchemaVersionError
        If ``/meta/schema_version`` major != 2.
    apeGmsh.opensees.emitter.h5_reader.MalformedH5Error
        If ``/meta`` is missing.
    """
    import h5py

    with h5py.File(path, "r") as f:
        # Resolve the sub-group root.  ``/`` keeps the existing
        # file-root behaviour byte-identical.
        if root in ("", "/"):
            parent = f
            label = path
        else:
            key = root.lstrip("/")
            if key not in f:
                from apeGmsh.opensees.emitter.h5_reader import (
                    MalformedH5Error,
                )
                raise MalformedH5Error(
                    f"{path}: missing sub-group {root!r}; not a "
                    "composed apeGmsh results.h5"
                )
            parent = f[key]
            label = f"{path}{root}"
        return read_neutral_zone_from_group(parent, label=label)


def read_neutral_zone_from_group(
    parent: Any,
    *,
    label: str = "<h5 group>",
) -> "FEMData":
    """Rebuild a :class:`FEMData` from an open HDF5 group.

    Mirrors the inverse of :func:`write_neutral_zone_into_group`:
    expects ``meta`` + neutral-zone children directly under
    ``parent``.  ``label`` is a display string used only in error
    messages (typically the file path plus sub-group root).
    """
    from apeGmsh.opensees.emitter.h5_reader import (
        MalformedH5Error,
        SchemaVersionError,
    )
    from apeGmsh.opensees._internal.schema_version import (
        NEUTRAL,
        read_zone_version,
        reader_version,
        validate_zone_version,
    )

    from ._element_types import ElementGroup, ElementTypeInfo, make_type_info
    from ._group_set import LabelSet, PhysicalGroupSet
    from .FEMData import (
        ElementComposite, FEMData, MeshInfo, NodeComposite,
        _compute_bandwidth,
    )

    # -- meta + schema check (ADR 0023 two-version window) --
    if "meta" not in parent:
        raise MalformedH5Error(
            f"{label}: missing /meta group; not an apeGmsh model.h5"
        )
    meta_attrs = parent["meta"].attrs
    if "schema_version" not in meta_attrs or not str(
        meta_attrs["schema_version"]
    ):
        raise MalformedH5Error(
            f"{label}: /meta/schema_version attribute is empty"
        )
    try:
        file_version = read_zone_version(meta_attrs, NEUTRAL)
    except ValueError as exc:
        raise MalformedH5Error(
            f"{label}: /meta schema-version attr is not "
            f"semver-shaped: {exc}"
        ) from exc
    if file_version is None:
        raise MalformedH5Error(
            f"{label}: /meta carries no neutral zone version"
        )
    try:
        validate_zone_version(
            file_version, reader_version(NEUTRAL), zone=NEUTRAL,
        )
    except SchemaVersionError as exc:
        raise SchemaVersionError(f"{label}: {exc}") from None

    # -- nodes --
    nodes_grp = parent["nodes"]
    node_ids = np.asarray(nodes_grp["ids"][...], dtype=np.int64)
    node_coords = np.asarray(nodes_grp["coords"][...], dtype=np.float64)
    # Per-node ndf (additive in 2.7.0; absent in 2.6.x files — readers
    # must tolerate the omission per the two-version window).  Probe
    # with ``in`` not ``Group.get`` per the h5py optional-child .get()
    # hazard (project_h5py_optional_child_get_hazard).
    #
    # If the dataset is present, use it.  If absent:
    #   * Pre-2.7.0 file: the writer didn't know about ndf at all.
    #     Synthesise the all-sentinel array so the loaded FEM hashes
    #     identically to what ``from_gmsh``-with-no-declarations
    #     would produce — both are empty-channel cases that the hash
    #     fold gate skips.
    #   * 2.7.0+ file with no ``/nodes/ndf``: the writer intentionally
    #     omitted it because ``_ndf=None`` at write time (hand-built
    #     test fixture or from_msh).  Stored snapshot_id was computed
    #     without the ndf fold, so leave ``_ndf=None`` to keep the
    #     recomputed hash symmetric.  The S2 hash gate skips both
    #     None and all-zero, so the two empty-channel shapes hash
    #     identically.
    #
    # The explicit-only API contract is unaffected: ``ndf_for`` still
    # raises the helpful LookupError for sentinel-0 nodes (or for the
    # ``_ndf=None`` case).
    if "ndf" in nodes_grp:
        node_ndf = np.asarray(nodes_grp["ndf"][...], dtype=np.int8)
        if node_ndf.shape != node_ids.shape:
            raise MalformedH5Error(
                f"{label}: /nodes/ndf shape {node_ndf.shape} does not "
                f"match /nodes/ids shape {node_ids.shape}."
            )
    else:
        # Tuple-compare the dataclass fields (SchemaVersion isn't
        # ordering-enabled, but its fields are comparable).
        _fv = (file_version.major, file_version.minor, file_version.patch)
        if _fv < (2, 7, 0):
            node_ndf = np.zeros(node_ids.shape, dtype=np.int8)
        else:
            node_ndf = None

    # Per-node provenance (decoupled nodes — ADR 0049; neutral schema
    # 2.11.0).  Absent in pre-2.11.0 files and in any 2.11.0+ file with
    # no decoupled nodes — both decode to ``_provenance=None`` (the
    # all-mesh case), which the hash gate treats identically.  Probe
    # with ``in`` not ``Group.get`` per the h5py optional-child hazard.
    node_provenance: "np.ndarray | None" = None
    if "provenance" in nodes_grp:
        node_provenance = np.asarray(
            nodes_grp["provenance"][...], dtype=np.int8)
        if node_provenance.shape != node_ids.shape:
            raise MalformedH5Error(
                f"{label}: /nodes/provenance shape "
                f"{node_provenance.shape} does not match /nodes/ids "
                f"shape {node_ids.shape}."
            )

    # ADR 0038 — per-node module_label parallel dataset (2.9.0).  Only
    # carry it when at least one entry is non-empty (uncomposed case
    # keeps ``_module_label=None`` so the snapshot_id hash + repr
    # stay byte-identical with pre-2.9.0 fixtures).
    node_module_label: "np.ndarray | None" = None
    if "module_label" in nodes_grp:
        raw_ml = nodes_grp["module_label"][...]
        decoded_ml = np.array(
            [
                x.decode("utf-8", errors="replace")
                if isinstance(x, (bytes, bytearray))
                else str(x)
                for x in raw_ml
            ],
            dtype=object,
        )
        if any(s != "" for s in decoded_ml):
            node_module_label = decoded_ml

    # -- elements (per-type subgroups) --
    element_groups: dict[int, ElementGroup] = {}
    types_meta: list[ElementTypeInfo] = []
    # Per-type module_label arrays (neutral schema 2.9.0).  Populated
    # only when the dataset is non-empty (i.e. at least one row carries
    # a non-empty label) — saves the uncomposed common case from
    # round-tripping with an all-empty-string array.
    elem_module_labels: dict[int, np.ndarray] = {}
    elem_grp = parent["elements"] if "elements" in parent else None
    if elem_grp is not None:
        for type_name in sorted(elem_grp.keys()):
            sub = elem_grp[type_name]
            if not hasattr(sub, "keys"):
                continue
            ids = np.asarray(sub["ids"][...], dtype=np.int64)
            conn = np.asarray(sub["connectivity"][...], dtype=np.int64)
            attrs = sub.attrs
            npe = int(
                attrs.get("npe", conn.shape[1] if conn.ndim == 2 else 0)
            )
            info = make_type_info(
                code=int(attrs.get("code", 0)),
                gmsh_name=str(attrs.get("gmsh_name", type_name)),
                dim=int(attrs.get("dim", 0)),
                order=int(attrs.get("order", 1)),
                npe=npe,
                count=ids.shape[0],
            )
            types_meta.append(info)
            element_groups[info.code] = ElementGroup(
                element_type=info, ids=ids, connectivity=conn,
            )
            # ADR 0038 — module_label parallel dataset; presence-
            # detected via ``in`` per the h5py optional-child .get()
            # hazard.  Only carry it when at least one entry is
            # non-empty (the compose-empty fast path).
            if "module_label" in sub:
                raw = sub["module_label"][...]
                decoded = np.array(
                    [
                        x.decode("utf-8", errors="replace")
                        if isinstance(x, (bytes, bytearray))
                        else str(x)
                        for x in raw
                    ],
                    dtype=object,
                )
                if any(s != "" for s in decoded):
                    elem_module_labels[info.code] = decoded

    # -- physical_groups + labels (root-level union of node + elem sides) --
    node_pgs, elem_pgs = _read_named_index_at_root(
        parent["physical_groups"] if "physical_groups" in parent else None
    )
    node_labels, elem_labels = _read_named_index_at_root(
        parent["labels"] if "labels" in parent else None
    )

    # -- mesh_selections --
    mesh_selection = _read_mesh_selections(
        parent["mesh_selections"] if "mesh_selections" in parent else None
    )

    # -- partitions (neutral schema 2.5.0; absent in 2.4.0 files) --
    partitions = _read_partitions(
        parent["partitions"] if "partitions" in parent else None
    )

    # -- parts (neutral schema 2.5.0; absent in 2.4.0 files) --
    part_node_map, part_elem_map = _read_parts(
        parent["parts"] if "parts" in parent else None
    )

    # -- composed_from (neutral schema 2.9.0; absent in 2.8.x files) --
    # ADR 0038 / h5py optional-child .get() hazard: probe with ``in``,
    # NEVER Group.get on the child name — the latter returns phantom
    # truthy proxies on some optional-child layouts (the bug fixed in
    # PR #261).  See ``project_h5py_optional_child_get_hazard``.
    composed_from = _read_composed_from(
        parent["composed_from"] if "composed_from" in parent else None
    )

    # -- constraints (split node-side vs element-side by record type) --
    # node_xyz lets _decode_node_to_surface re-derive the exact
    # rigid-beam offsets (phantom_coord − master_coord).
    node_xyz = {
        int(t): node_coords[i]
        for i, t in enumerate(node_ids.tolist())
    }
    node_constraints, elem_constraints = _read_constraints(
        parent["constraints"] if "constraints" in parent else None,
        node_xyz,
    )

    # -- embedded-reinforcement ties (neutral schema 2.15.0) --
    reinforce_ties = _read_reinforce_ties(
        parent["reinforce_ties"] if "reinforce_ties" in parent else None
    )

    # -- node-to-host embedment ties (neutral schema 2.22.0) --
    embed_ties = _read_embed_ties(
        parent["embed_ties"] if "embed_ties" in parent else None
    )

    # -- auto-emitted structural rebar elements (neutral schema 2.16.0) --
    rebar_elements = _read_rebar_elements(
        parent["rebar_elements"] if "rebar_elements" in parent else None
    )

    # -- fork contact interactions (neutral schema 2.21.0) --
    contacts = _read_contacts(
        parent["contacts"] if "contacts" in parent else None
    )

    # -- fork rigid-plane contacts (neutral schema 2.24.0) --
    contact_planes = _read_contact_planes(
        parent["contact_planes"] if "contact_planes" in parent else None
    )

    # -- loads --
    nodal_loads, element_loads, sp_records = _read_loads(
        parent["loads"] if "loads" in parent else None
    )

    # -- masses --
    mass_records = _read_masses(
        parent["masses"] if "masses" in parent else None
    )

    # -- assemble composites --
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=node_coords,
        physical=PhysicalGroupSet(node_pgs),
        labels=LabelSet(node_labels),
        constraints=node_constraints,
        loads=nodal_loads,
        sp=sp_records,
        masses=mass_records,
        partitions=partitions or None,
        part_node_map=part_node_map or None,
        ndf=node_ndf,
        module_label=node_module_label,
        provenance=node_provenance,
    )
    elements = ElementComposite(
        groups=element_groups,
        physical=PhysicalGroupSet(elem_pgs),
        labels=LabelSet(elem_labels),
        constraints=elem_constraints,
        loads=element_loads,
        partitions=partitions or None,
        part_elem_map=part_elem_map or None,
        module_label=elem_module_labels or None,
        reinforce_ties=reinforce_ties or None,
        embed_ties=embed_ties or None,
        rebar_elements=rebar_elements or None,
        contacts=contacts or None,
        contact_planes=contact_planes or None,
    )
    info = MeshInfo(
        n_nodes=len(node_ids),
        n_elems=sum(len(g) for g in element_groups.values()),
        # Recomputed from connectivity (B3): the writer never
        # stored bandwidth, so we derive it deterministically from
        # the reloaded per-type groups — matches what from_gmsh
        # would compute for the same connectivity.
        bandwidth=_compute_bandwidth(element_groups),
        types=types_meta,
    )
    rebuilt = FEMData(
        nodes=nodes, elements=elements, info=info,
        mesh_selection=mesh_selection,
        composed_from=composed_from,
    )

    # B4 — verify /meta/snapshot_id matches the recomputed hash
    # of the rebuilt FEM.  Per ADR 0021, FEM round-trip integrity
    # is a hard guarantee: /meta/snapshot_id must equal the
    # recomputed hash of the rebuilt FEM.  (Lineage CHAIN mismatch
    # is warn-not-raise, a separate surface.)
    # Probe with ``in`` rather than ``Group.get`` per the h5py
    # optional-child .get() hazard noted in
    # ``project_h5py_optional_child_get_hazard``.
    if "snapshot_id" in parent["meta"].attrs:
        stored = str(parent["meta"].attrs["snapshot_id"])
        if stored:
            recomputed = rebuilt.snapshot_id
            if recomputed != stored:
                raise MalformedH5Error(
                    f"{label}: /meta/snapshot_id mismatch — "
                    f"stored={stored!r}, recomputed={recomputed!r}. "
                    "The neutral zone has been corrupted or "
                    "tampered with since the file was written."
                )

    return rebuilt


def _read_named_index_at_root(
    parent: Any,
) -> tuple[dict[tuple[int, int], dict], dict[tuple[int, int], dict]]:
    """Read a root ``/physical_groups`` or ``/labels`` index.

    Neutral schema 2.10.0 sub-tree split (B2): walks
    ``parent/node_side/`` and ``parent/element_side/`` independently
    and returns ``(node_dict, elem_dict)`` deterministically — no
    heuristic, no inference, no shared dataset reuse across sides.
    Either sub-tree may be absent (writer omits it when its side has
    no entries); both being absent is the canonical "no PGs / no
    labels" signal.
    """
    node_dict: dict[tuple[int, int], dict] = {}
    elem_dict: dict[tuple[int, int], dict] = {}
    if parent is None:
        return node_dict, elem_dict

    if "node_side" in parent:
        _read_named_index_side(parent["node_side"], node_dict)
    if "element_side" in parent:
        _read_named_index_side(parent["element_side"], elem_dict)

    return node_dict, elem_dict


def _read_named_index_side(
    parent: Any,
    out: dict[tuple[int, int], dict],
) -> None:
    """Populate ``out`` from one side of the named-index split."""
    for safe_name in parent.keys():
        sub = parent[safe_name]
        if not hasattr(sub, "keys"):
            continue
        attrs = sub.attrs
        dim = int(attrs.get("dim", 0))
        tag = int(attrs.get("tag", 0))
        name = str(attrs.get("name", safe_name))

        nids = (
            np.asarray(sub["node_ids"][...], dtype=np.int64)
            if "node_ids" in sub
            else np.array([], dtype=np.int64)
        )
        ncoords = (
            np.asarray(sub["node_coords"][...], dtype=np.float64)
            if "node_coords" in sub
            else np.zeros((0, 3), dtype=np.float64)
        )

        entry: dict = {
            "name": name,
            "node_ids": nids,
            "node_coords": ncoords,
        }
        if "element_ids" in sub:
            entry["element_ids"] = np.asarray(
                sub["element_ids"][...], dtype=np.int64,
            )
        out[(dim, tag)] = entry


def _read_partitions(
    parent: Any,
) -> dict[int, dict]:
    """Reconstruct the per-partition node/element membership dict.

    Inverse of :func:`_write_partitions`.  Returns an empty dict when
    ``/partitions`` is absent (legacy 2.4.0 files) — the FEM
    composites then resolve as un-partitioned, matching pre-2.5.0
    behaviour.
    """
    if parent is None:
        return {}
    result: dict[int, dict] = {}
    for key in parent.keys():
        sub = parent[key]
        if not hasattr(sub, "keys"):
            continue
        # Prefer the integer id attr; fall back to the group name
        # (which the writer sets to ``str(int(pid))``).
        attrs = sub.attrs
        if "id" in attrs:
            pid = int(attrs["id"])
        else:
            pid = int(key)
        nids = (
            np.asarray(sub["node_ids"][...], dtype=np.int64)
            if "node_ids" in sub
            else np.array([], dtype=np.int64)
        )
        eids = (
            np.asarray(sub["element_ids"][...], dtype=np.int64)
            if "element_ids" in sub
            else np.array([], dtype=np.int64)
        )
        result[pid] = {"node_ids": nids, "element_ids": eids}
    return result


def _read_parts(
    parent: Any,
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Reconstruct the Part-label maps written by :func:`_write_parts`.

    Returns ``(part_node_map, part_elem_map)``.  Either side may be
    empty for a given label when that label only had nodes (or only
    elements) at FEM-build time.  Returns ``({}, {})`` when
    ``/parts`` is absent (legacy 2.4.0 files).
    """
    node_map: dict[str, set[int]] = {}
    elem_map: dict[str, set[int]] = {}
    if parent is None:
        return node_map, elem_map
    for key in parent.keys():
        sub = parent[key]
        if not hasattr(sub, "keys"):
            continue
        label = str(sub.attrs.get("label", key))
        if "node_ids" in sub:
            nids = np.asarray(sub["node_ids"][...], dtype=np.int64)
            if nids.size > 0:
                node_map[label] = {int(x) for x in nids}
        if "element_ids" in sub:
            eids = np.asarray(sub["element_ids"][...], dtype=np.int64)
            if eids.size > 0:
                elem_map[label] = {int(x) for x in eids}
    return node_map, elem_map


def _read_composed_from(parent: Any) -> tuple:
    """Reconstruct ``ComposeRecord`` tuple from ``/composed_from/``.

    Inverse of :func:`_write_composed_from`.  Returns an empty tuple
    when ``parent`` is ``None`` (the 2.8.x compatibility path) — the
    caller turns this into an empty :class:`ComposeSet`.

    Probes optional children with ``in`` rather than ``Group.get`` per
    the h5py optional-child .get() hazard
    (``project_h5py_optional_child_get_hazard``).
    """
    from apeGmsh._kernel.records._compose import ComposeRecord

    if parent is None:
        return ()

    records: list[ComposeRecord] = []
    for key in sorted(parent.keys()):
        sub = parent[key]
        if not hasattr(sub, "keys"):
            continue
        attrs = sub.attrs
        label = str(attrs.get("label", key))
        source_path = str(attrs.get("source_path", ""))
        source_fem_hash = str(attrs.get("source_fem_hash", ""))
        source_neutral_schema_version = str(
            attrs.get("source_neutral_schema_version", "")
        )
        translate_raw = attrs.get("translate", np.zeros(3, dtype=np.float64))
        translate = tuple(
            float(x) for x in np.asarray(translate_raw, dtype=np.float64)
            .reshape(-1)[:3]
        )
        rotate: tuple[float, float, float, float] | None = None
        if "rotate" in attrs:
            rotate_raw = np.asarray(attrs["rotate"], dtype=np.float64).reshape(-1)
            rotate = tuple(float(x) for x in rotate_raw[:4])  # type: ignore[assignment]
        partition_rank: int | None = None
        if "partition_rank" in attrs:
            partition_rank = int(attrs["partition_rank"])
        composed_at = str(attrs.get("composed_at", ""))

        properties: dict = {}
        # Optional sub-group probed with ``in`` (NOT ``.get``).
        if "properties" in sub:
            prop_attrs = sub["properties"].attrs
            for pk in prop_attrs:
                raw = prop_attrs[pk]
                # numpy scalars / 0-d arrays decode to Python scalars
                if isinstance(raw, np.ndarray) and raw.shape == ():
                    raw = raw.item()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                properties[str(pk)] = raw

        records.append(ComposeRecord(
            label=label,
            source_path=source_path,
            source_fem_hash=source_fem_hash,
            source_neutral_schema_version=source_neutral_schema_version,
            translate=translate,  # type: ignore[arg-type]
            rotate=rotate,
            partition_rank=partition_rank,
            composed_at=composed_at,
            properties=properties,
        ))
    return tuple(records)


def _read_mesh_selections(parent: Any):
    """Reconstruct a ``MeshSelectionStore`` from ``/mesh_selections``."""
    if parent is None:
        return None
    from .MeshSelectionSet import MeshSelectionStore

    sets: dict[tuple[int, int], dict] = {}
    for safe_name in parent.keys():
        sub = parent[safe_name]
        if not hasattr(sub, "keys"):
            continue
        attrs = sub.attrs
        dim = int(attrs.get("dim", 0))
        tag = int(attrs.get("tag", 0))
        info: dict = {"name": str(attrs.get("name", safe_name))}

        if "node_ids" in sub:
            info["node_ids"] = np.asarray(
                sub["node_ids"][...], dtype=np.int64,
            )
            info["node_coords"] = np.asarray(
                sub["node_coords"][...], dtype=np.float64,
            )
        if "element_ids" in sub:
            info["element_ids"] = np.asarray(
                sub["element_ids"][...], dtype=np.int64,
            )
        # Presence-detected: pre-connectivity files simply lack this
        # dataset and round-trip exactly as before.
        if "connectivity" in sub:
            info["connectivity"] = np.asarray(
                sub["connectivity"][...], dtype=np.int64,
            )

        sets[(dim, tag)] = info

    if not sets:
        return None
    return MeshSelectionStore(sets)


# ---------------------------------------------------------------------------
# Constraint decoders
# ---------------------------------------------------------------------------


def _read_constraints(
    parent: Any,
    node_xyz: dict[int, Any] | None = None,
) -> tuple[list[Any], list[Any]]:
    """Decode ``/constraints/{kind}`` datasets into node + element record lists.

    Routing matches the writer:

    * ``NodePair*``, ``NodeGroup*``, ``NodeToSurface*`` → node-side
    * ``Interpolation*``, ``SurfaceCoupling*``         → element-side
    """
    from apeGmsh._kernel.records._constraints import (
        InterpolationRecord,
        NodeGroupRecord,
        NodePairRecord,
        NodeToSurfaceRecord,
        SurfaceCouplingRecord,
    )

    node_records: list[Any] = []
    elem_records: list[Any] = []
    if parent is None:
        return node_records, elem_records

    # Maps payload-field signature → (decoder, target_list)
    NODE_PAIR_FIELDS = {
        "master_node", "slave_node", "dofs", "offset", "penalty_stiffness",
    }
    NODE_GROUP_FIELDS = {
        "master_node", "slave_nodes", "dofs", "offsets", "plane_normal",
    }
    INTERPOLATION_FIELDS = {
        "slave_node", "master_nodes", "weights", "dofs",
        "projected_point", "parametric_coords",
    }
    SURFACE_COUPLING_FIELDS = {
        "master_nodes", "slave_nodes", "dofs",
        "mortar_operator_shape", "mortar_operator",
    }
    NODE_TO_SURFACE_FIELDS = {
        "master_node", "slave_nodes", "phantom_nodes",
        "phantom_coords", "dofs",
    }

    for kind_name in parent.keys():
        ds = parent[kind_name]
        if hasattr(ds, "keys"):
            # Skipped/unknown record type was written as a group with
            # __deviation__ attr — nothing to reconstruct.
            continue

        rows = ds[...]
        if rows.shape == ():       # scalar → make 1-D
            rows = np.array([rows])
        if rows.size == 0:
            continue

        payload_fields = set(rows.dtype["payload"].names or ())

        # Dispatch via subset match: every EXPECTED field-set is the
        # CORE minimum (pre-2.5.0 fields).  Newer files (2.5.0+) add
        # the optional ``name`` field; SurfaceCoupling also adds the
        # ``sr_*`` slave-records fields (presence-detected by its
        # decoder).  Subset matching keeps both old and new files
        # dispatching to the right decoder.  The unique core field on
        # each kind (e.g. ``slave_node`` vs ``slave_nodes``,
        # ``phantom_nodes`` vs ``offsets``) keeps the subset match
        # unambiguous.
        if NODE_PAIR_FIELDS <= payload_fields:
            for row in rows:
                node_records.append(_decode_node_pair(row, NodePairRecord))
        elif NODE_GROUP_FIELDS <= payload_fields:
            for row in rows:
                node_records.append(_decode_node_group(row, NodeGroupRecord))
        elif INTERPOLATION_FIELDS <= payload_fields:
            for row in rows:
                elem_records.append(
                    _decode_interpolation(row, InterpolationRecord)
                )
        elif SURFACE_COUPLING_FIELDS <= payload_fields:
            # Subset already; mortar_operator_shape is unique to this
            # record so the subset match stays unambiguous and
            # back-compatible across the 2.4.x → 2.5.x bump.
            for row in rows:
                elem_records.append(
                    _decode_surface_coupling(row, SurfaceCouplingRecord)
                )
        elif NODE_TO_SURFACE_FIELDS <= payload_fields:
            for row in rows:
                node_records.append(
                    _decode_node_to_surface(
                        row, NodeToSurfaceRecord, node_xyz)
                )
        # else: unknown payload schema — skip silently (forward-compat)

    return node_records, elem_records


def _kind(row: Any) -> str:
    return _str(row["payload_kind"])


def _str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _opt_vec3(arr: np.ndarray) -> np.ndarray | None:
    """Return arr if any component is finite, else None (NaN-sentinel decode)."""
    a = np.asarray(arr, dtype=np.float64).reshape(-1)[:3]
    if not np.any(np.isfinite(a)):
        return None
    return a


def _opt_vec2(arr: np.ndarray) -> np.ndarray | None:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)[:2]
    if not np.any(np.isfinite(a)):
        return None
    return a


def _opt_scalar(value: float) -> float | None:
    v = float(value)
    if np.isnan(v):
        return None
    return v


def _opt_name(payload: Any) -> str | None:
    """Decode the optional payload ``name`` field.

    Neutral schema 2.5.0 adds ``name`` to every record dtype.  Old
    2.4.0 files lack the field — probe ``dtype.names`` (not h5py
    ``Group.get``; this is a payload sub-dtype) and fall back to
    ``None``.  Empty stored string round-trips back to ``None`` to
    preserve the ``name: str | None`` semantics on the record
    dataclasses.
    """
    names = payload.dtype.names or ()
    if "name" not in names:
        return None
    raw = _str(payload["name"])
    return raw or None


def _decode_node_pair(row: Any, cls: type) -> Any:
    p = row["payload"]
    # master_dofs (equal_dof_mixed, schema 2.17.0) — probe presence for
    # pre-2.17.0 files; an empty stored array round-trips back to None.
    master_dofs: list[int] | None = None
    if "master_dofs" in (p.dtype.names or ()):
        md = np.asarray(p["master_dofs"], dtype=np.int64).reshape(-1)
        if md.size:
            master_dofs = [int(x) for x in md]
    return cls(
        kind=_kind(row),
        name=_opt_name(p),
        master_node=int(p["master_node"]),
        slave_node=int(p["slave_node"]),
        dofs=[int(x) for x in np.asarray(p["dofs"]).reshape(-1)],
        offset=_opt_vec3(p["offset"]),
        penalty_stiffness=_opt_scalar(p["penalty_stiffness"]),
        master_dofs=master_dofs,
    )


def _decode_node_group(row: Any, cls: type) -> Any:
    p = row["payload"]
    slaves = np.asarray(p["slave_nodes"], dtype=np.int64).reshape(-1)
    offsets_flat = np.asarray(p["offsets"], dtype=np.float64).reshape(-1)
    offsets = (
        offsets_flat.reshape(-1, 3)
        if offsets_flat.size and offsets_flat.size == 3 * len(slaves)
        else None
    )
    # LadrunoRigidBody emission (schema 2.19.0) — probe presence for
    # pre-2.19.0 files (decode as_element=False / mass=None).
    rb_extras: dict[str, Any] = {}
    if "as_element" in (p.dtype.names or ()):
        rb_extras = dict(
            as_element=bool(int(p["as_element"])),
            mass=_opt_scalar(p["mass"]),
        )
    # omega (schema 2.20.0) probed independently — all-NaN ⇒ None.
    if "omega" in (p.dtype.names or ()):
        om = _opt_vec3(p["omega"])
        rb_extras["omega"] = None if om is None else tuple(
            float(w) for w in om
        )
    return cls(
        kind=_kind(row),
        name=_opt_name(p),
        master_node=int(p["master_node"]),
        slave_nodes=[int(x) for x in slaves],
        dofs=[int(x) for x in np.asarray(p["dofs"]).reshape(-1)],
        offsets=offsets,
        plane_normal=_opt_vec3(p["plane_normal"]),
        control=_decode_control(p),
        **rb_extras,
    )


def _decode_interpolation(row: Any, cls: type) -> Any:
    p = row["payload"]
    weights_flat = np.asarray(p["weights"], dtype=np.float64).reshape(-1)
    weights = weights_flat if weights_flat.size > 0 else None
    # ASDEmbeddedNodeElement options (schema 2.8.0).  Pre-2.8.0 files
    # lack these columns; probe ``p.dtype.names`` (same pattern as
    # ``_opt_name`` / the sr_* surface-coupling fallback) and fall
    # back to InterpolationRecord dataclass defaults so old files
    # decode as before.
    names = set(p.dtype.names or ())
    if "stiffness" in names:
        kp_present = bool(int(p["has_stiffness_p"]))
        extras: dict[str, Any] = dict(
            stiffness=float(p["stiffness"]),
            stiffness_p=float(p["stiffness_p"]) if kp_present else None,
            rotational=bool(int(p["rotational"])),
            pressure=bool(int(p["pressure"])),
            excess=_opt_scalar(p["excess"]),
        )
    else:
        extras = {}
    # enforce route (ADR 0068, schema 2.14.0) — probed independently of the
    # 2.8.0 stiffness block; pre-2.14.0 files fall back to "penalty".
    if "enforce" in names:
        extras["enforce"] = _str(p["enforce"]) or "penalty"
    return cls(
        kind=_kind(row),
        name=_opt_name(p),
        slave_node=int(p["slave_node"]),
        master_nodes=[
            int(x) for x in np.asarray(p["master_nodes"]).reshape(-1)
        ],
        weights=weights,
        dofs=[int(x) for x in np.asarray(p["dofs"]).reshape(-1)],
        projected_point=_opt_vec3(p["projected_point"]),
        parametric_coords=_opt_vec2(p["parametric_coords"]),
        control=_decode_control(p),
        **extras,
    )


def _decode_surface_coupling(row: Any, cls: type) -> Any:
    from apeGmsh._kernel.records._constraints import InterpolationRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind

    p = row["payload"]
    shape = tuple(int(x) for x in np.asarray(p["mortar_operator_shape"]))
    flat = np.asarray(p["mortar_operator"], dtype=np.float64).reshape(-1)
    if shape == (0, 0) or flat.size == 0:
        operator = None
    else:
        operator = flat.reshape(shape)

    # slave_records: present only when the payload carries the sr_*
    # fields.  Older files lack them → slave_records=[] (the historical
    # behaviour, now explicit and detected structurally).
    slave_records: list[Any] = []
    names = set(p.dtype.names or ())
    if "sr_slave_nodes" in names:
        sn = np.asarray(p["sr_slave_nodes"], dtype=np.int64).reshape(-1)
        mcount = np.asarray(p["sr_master_counts"], dtype=np.int64).reshape(-1)
        mnodes = np.asarray(p["sr_master_nodes"], dtype=np.int64).reshape(-1)
        wts = np.asarray(p["sr_weights"], dtype=np.float64).reshape(-1)
        dcount = np.asarray(p["sr_dof_counts"], dtype=np.int64).reshape(-1)
        dofs_flat = np.asarray(p["sr_dofs"], dtype=np.int64).reshape(-1)
        proj = np.asarray(p["sr_projected"], dtype=np.float64).reshape(-1)
        para = np.asarray(p["sr_parametric"], dtype=np.float64).reshape(-1)
        # ASDEmbeddedNodeElement options per slave (schema 2.8.0).
        # Pre-2.8.0 files lack these — fall back to dataclass defaults
        # the same way ``_decode_interpolation`` does at the top level.
        has_sr_opts = "sr_stiffness" in names
        if has_sr_opts:
            sr_K = np.asarray(p["sr_stiffness"], dtype=np.float64).reshape(-1)
            sr_KP = np.asarray(p["sr_stiffness_p"], dtype=np.float64).reshape(-1)
            sr_hKP = np.asarray(p["sr_has_stiffness_p"], dtype=np.uint8).reshape(-1)
            sr_rot = np.asarray(p["sr_rotational"], dtype=np.uint8).reshape(-1)
            sr_p = np.asarray(p["sr_pressure"], dtype=np.uint8).reshape(-1)
            sr_ex = np.asarray(p["sr_excess"], dtype=np.float64).reshape(-1)
        # CouplingControl per slave (schema 2.12.0 sr_cpl_* mirror;
        # host auto-scalers 2.13.0, presence-probed independently).
        # Pre-2.12.0 files lack these — fall back to control=None.
        has_sr_cpl = "sr_cpl_has" in names
        has_sr_cpl_host = "sr_cpl_k_auto" in names
        if has_sr_cpl:
            sr_c_has = np.asarray(p["sr_cpl_has"], dtype=np.uint8).reshape(-1)
            sr_c_k = np.asarray(p["sr_cpl_k"], dtype=np.float64).reshape(-1)
            sr_c_kr = np.asarray(p["sr_cpl_kr"], dtype=np.float64).reshape(-1)
            sr_c_enf = np.asarray(
                p["sr_cpl_enforce"], dtype=np.uint8).reshape(-1)
            sr_c_dtcr = np.asarray(
                p["sr_cpl_dtcr"], dtype=np.float64).reshape(-1)
            sr_c_abs = np.asarray(
                p["sr_cpl_absolute"], dtype=np.uint8).reshape(-1)
        if has_sr_cpl_host:
            sr_c_auto = np.asarray(
                p["sr_cpl_k_auto"], dtype=np.uint8).reshape(-1)
            sr_c_alpha = np.asarray(
                p["sr_cpl_k_alpha"], dtype=np.float64).reshape(-1)
            sr_c_host = np.asarray(
                p["sr_cpl_host"], dtype=np.int64).reshape(-1)
            sr_c_wcap = np.asarray(
                p["sr_cpl_wcap"], dtype=np.float64).reshape(-1)
        # EmbeddedNodeControl pressure tie per slave (schema 2.18.0;
        # probed independently). Pre-2.18.0 files fall back to base control.
        has_sr_cpl_pressure = "sr_cpl_pressure" in names
        if has_sr_cpl_pressure:
            sr_c_pressure = np.asarray(
                p["sr_cpl_pressure"], dtype=np.uint8).reshape(-1)
            sr_c_kp = np.asarray(
                p["sr_cpl_kp"], dtype=np.float64).reshape(-1)
        # enforce route per slave (ADR 0068, schema 2.14.0; probed
        # independently). Pre-2.14.0 files fall back to "penalty".
        has_sr_enforce = "sr_enforce" in names
        if has_sr_enforce:
            sr_enf = np.asarray(p["sr_enforce"], dtype=np.uint8).reshape(-1)
        m_off = 0
        d_off = 0
        for i in range(sn.size):
            mc = int(mcount[i])
            dc = int(dcount[i])
            m = [int(x) for x in mnodes[m_off:m_off + mc]]
            w_slice = wts[m_off:m_off + mc]
            w = None if (w_slice.size and np.all(np.isnan(w_slice))) \
                else [float(x) for x in w_slice]
            d = [int(x) for x in dofs_flat[d_off:d_off + dc]]
            m_off += mc
            d_off += dc
            pp = proj[3 * i:3 * i + 3]
            pc = para[2 * i:2 * i + 2]
            if has_sr_opts:
                opt_extras: dict[str, Any] = dict(
                    stiffness=float(sr_K[i]),
                    stiffness_p=(float(sr_KP[i]) if int(sr_hKP[i]) else None),
                    rotational=bool(int(sr_rot[i])),
                    pressure=bool(int(sr_p[i])),
                    excess=_opt_scalar(sr_ex[i]),
                )
            else:
                opt_extras = {}
            if has_sr_cpl:
                control = _control_from_values(
                    sr_c_has[i], sr_c_k[i], sr_c_kr[i],
                    sr_c_enf[i], sr_c_dtcr[i], sr_c_abs[i],
                    **(dict(
                        k_auto=sr_c_auto[i], k_alpha=sr_c_alpha[i],
                        host=sr_c_host[i], wcap=sr_c_wcap[i],
                    ) if has_sr_cpl_host else {}),
                    **(dict(
                        pressure=sr_c_pressure[i], kp=sr_c_kp[i],
                    ) if has_sr_cpl_pressure else {}),
                )
            else:
                control = None
            slave_records.append(InterpolationRecord(
                kind=ConstraintKind.TIE,
                slave_node=int(sn[i]),
                master_nodes=m,
                weights=w,
                dofs=d,
                projected_point=(None if np.all(np.isnan(pp))
                                 else pp.astype(np.float64)),
                parametric_coords=(None if np.all(np.isnan(pc))
                                   else pc.astype(np.float64)),
                control=control,
                **opt_extras,
                **({"enforce": _SR_ENFORCE_NAME.get(int(sr_enf[i]),
                                                    "penalty")}
                   if has_sr_enforce else {}),
            ))

    return cls(
        kind=_kind(row),
        name=_opt_name(p),
        master_nodes=[
            int(x) for x in np.asarray(p["master_nodes"]).reshape(-1)
        ],
        slave_nodes=[
            int(x) for x in np.asarray(p["slave_nodes"]).reshape(-1)
        ],
        dofs=[int(x) for x in np.asarray(p["dofs"]).reshape(-1)],
        mortar_operator=operator,
        slave_records=slave_records,
    )


def _decode_node_to_surface(
    row: Any, cls: type,
    node_xyz: dict[int, Any] | None = None,
) -> Any:
    from apeGmsh._kernel.records._constraints import NodePairRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind

    p = row["payload"]
    slaves = [int(x) for x in
              np.asarray(p["slave_nodes"], dtype=np.int64).reshape(-1)]
    phantoms = [int(x) for x in
                np.asarray(p["phantom_nodes"], dtype=np.int64).reshape(-1)]
    dofs = [int(x) for x in np.asarray(p["dofs"], dtype=np.int64).reshape(-1)]
    master = int(p["master_node"])
    coords_flat = np.asarray(p["phantom_coords"], dtype=np.float64).reshape(-1)
    if coords_flat.size and coords_flat.size == 3 * len(slaves):
        phantom_coords = coords_flat.reshape(-1, 3)
    else:
        phantom_coords = None

    # Re-derive the sub-records exactly as resolve_node_to_surface does
    # (rigid_beam master->phantom, equalDOF phantom->slave).  These are
    # NOT persisted but are fully determined by the high-level fields;
    # without this, every iterator (rigid_link_groups / equal_dofs /
    # pairs / stiff_beam_groups) returns empty after from_h5 and the
    # reloaded model is silently disconnected.
    m_xyz = None
    if node_xyz is not None and master in node_xyz:
        m_xyz = np.asarray(node_xyz[master], dtype=np.float64)
    rigid_records: list[Any] = []
    edof_records: list[Any] = []
    for i, (ph, sl) in enumerate(zip(phantoms, slaves)):
        offset = None
        if m_xyz is not None and phantom_coords is not None:
            offset = phantom_coords[i] - m_xyz
        rigid_records.append(NodePairRecord(
            kind=ConstraintKind.RIGID_BEAM,
            master_node=master, slave_node=ph, offset=offset,
        ))
        edof_records.append(NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=ph, slave_node=sl, dofs=list(dofs),
        ))

    return cls(
        kind=_kind(row),
        name=_opt_name(p),
        master_node=master,
        slave_nodes=slaves,
        phantom_nodes=phantoms,
        phantom_coords=phantom_coords,
        rigid_link_records=rigid_records,
        equal_dof_records=edof_records,
        dofs=dofs,
    )


# ---------------------------------------------------------------------------
# Reinforce-tie decoder + reader
# ---------------------------------------------------------------------------


def _read_reinforce_ties(parent: Any) -> list[Any]:
    """Decode the ``/reinforce_ties/ties`` dataset into a list of
    :class:`ReinforceTieRecord` (empty when the group/dataset is absent)."""
    if parent is None or "ties" not in parent:
        return []
    from apeGmsh._kernel.records._constraints import ReinforceTieRecord
    rows = parent["ties"][...]
    if rows.shape == ():
        rows = np.array([rows])
    return [_decode_reinforce_tie(row, ReinforceTieRecord) for row in rows]


def _decode_reinforce_tie(row: Any, cls: type) -> Any:
    """Reconstruct a :class:`ReinforceTieRecord` from a payload row
    (inverse of :func:`_encode_reinforce_tie`)."""
    p = row["payload"]

    def _f(name: str) -> float | None:
        v = float(p[name])
        return v if np.isfinite(v) else None

    has_w = int(p["has_weights"]) == 1
    weights = (np.asarray(p["weights"], dtype=np.float64).reshape(-1)
               if has_w else None)
    has_d = int(p["has_direction"]) == 1
    direction = (np.asarray(p["direction"], dtype=np.float64).reshape(-1)[:3]
                 if has_d else None)
    bond = _str(p["bond"]) or None
    name = _str(p["name"]) or None
    host_nodes = [int(x) for x in
                  np.asarray(p["host_nodes"], dtype=np.int64).reshape(-1)]
    # Defensive: a corrupt file with weights desynced from host_nodes
    # would silently emit a wrong LadrunoEmbeddedRebar; refuse it loudly
    # (mirror of the encode-side invariant, adversarial-review C0).
    if weights is not None and len(weights) != len(host_nodes):
        raise ValueError(
            f"corrupted reinforce tie (rebar node {int(p['rebar_node'])}): "
            f"weights length {len(weights)} != host_nodes length "
            f"{len(host_nodes)}.")
    # corot / shape_b added in neutral 2.26.0 — presence-probe so an in-window
    # 2.25.x file (no columns) decodes corot=False, shape_b=None.
    if "corot" in p.dtype.names:
        corot = int(p["corot"]) == 1
        has_sb = int(p["has_shape_b"]) == 1
        shape_b = (np.asarray(p["shape_b"], dtype=np.float64).reshape(-1)
                   if has_sb else None)
        if shape_b is not None and len(shape_b) != len(host_nodes):
            raise ValueError(
                f"corrupted reinforce tie (rebar node {int(p['rebar_node'])}): "
                f"shape_b length {len(shape_b)} != host_nodes length "
                f"{len(host_nodes)}.")
        corot_kw = {"corot": corot, "shape_b": shape_b}
    else:
        corot_kw = {}
    return cls(
        kind="reinforce",
        name=name,
        rebar_node=int(p["rebar_node"]),
        host_nodes=host_nodes,
        weights=weights,
        direction=direction,
        **corot_kw,
        bond_scale=_f("bond_scale"),
        bond=bond,
        perfect=_f("perfect"),
        kt=_f("kt"),
        kt_alpha=_f("kt_alpha"),
        enforce=_str(p["enforce"]) or "penalty",
        bipenalty=int(p["bipenalty"]) == 1,
        dtcr=_f("dtcr"),
        excess=_f("excess"),
        in_bounds=int(p["in_bounds"]) == 1,
    )


def _read_embed_ties(parent: Any) -> list[Any]:
    """Decode the ``/embed_ties/ties`` dataset into a list of
    :class:`EmbedTieRecord` (empty when the group/dataset is absent)."""
    if parent is None or "ties" not in parent:
        return []
    from apeGmsh._kernel.records._constraints import EmbedTieRecord
    rows = parent["ties"][...]
    if rows.shape == ():
        rows = np.array([rows])
    return [_decode_embed_tie(row, EmbedTieRecord) for row in rows]


def _decode_embed_tie(row: Any, cls: type) -> Any:
    """Reconstruct an :class:`EmbedTieRecord` from a payload row
    (inverse of :func:`_encode_embed_tie`)."""
    p = row["payload"]

    def _f(name: str) -> float | None:
        v = float(p[name])
        return v if np.isfinite(v) else None

    has_w = int(p["has_weights"]) == 1
    weights = (np.asarray(p["weights"], dtype=np.float64).reshape(-1)
               if has_w else None)
    host_nodes = [int(x) for x in
                  np.asarray(p["host_nodes"], dtype=np.int64).reshape(-1)]
    if weights is not None and len(weights) != len(host_nodes):
        raise ValueError(
            f"corrupted embed tie (node {int(p['node'])}): weights length "
            f"{len(weights)} != host_nodes length {len(host_nodes)}.")
    return cls(
        kind="embed",
        name=_str(p["name"]) or None,
        node=int(p["node"]),
        host_nodes=host_nodes,
        weights=weights,
        k=_f("k"),
        k_alpha=_f("k_alpha"),
        enforce=_str(p["enforce"]) or "penalty",
        bipenalty=int(p["bipenalty"]) == 1,
        dtcr=_f("dtcr"),
        staged=int(p["staged"]) == 1,
        excess=_f("excess"),
        in_bounds=int(p["in_bounds"]) == 1,
    )


def _read_rebar_elements(parent: Any) -> list[Any]:
    """Decode the ``/rebar_elements/elements`` dataset into a list of
    :class:`RebarElementRecord` (empty when the group/dataset is absent)."""
    if parent is None or "elements" not in parent:
        return []
    from apeGmsh._kernel.records._rebar import RebarElementRecord
    rows = parent["elements"][...]
    if rows.shape == ():
        rows = np.array([rows])
    return [_decode_rebar_element(row, RebarElementRecord) for row in rows]


def _decode_rebar_element(row: Any, cls: type) -> Any:
    """Reconstruct a :class:`RebarElementRecord` from a payload row
    (inverse of :func:`_encode_rebar_element`)."""
    p = row["payload"]
    flat = np.asarray(p["connectivity"], dtype=np.int64).reshape(-1)
    if flat.size % 2 != 0:
        raise ValueError(
            f"corrupted rebar element (PG {_str(p['pg'])!r}): connectivity "
            f"flat length {flat.size} is odd — cells must be (i, j) pairs.")
    conn = tuple(
        (int(flat[k]), int(flat[k + 1])) for k in range(0, flat.size, 2))
    return cls(
        pg=_str(p["pg"]),
        element=_str(p["element"]),
        material=_str(p["material"]),
        area=float(p["area"]),
        role=_str(p["role"]),
        connectivity=conn,
    )


def _read_contacts(parent: Any) -> list[Any]:
    """Decode the ``/contacts/contacts`` dataset into a list of
    :class:`ContactRecord` (empty when the group/dataset is absent)."""
    if parent is None or "contacts" not in parent:
        return []
    from apeGmsh._kernel.records._constraints import ContactRecord
    rows = parent["contacts"][...]
    if rows.shape == ():
        rows = np.array([rows])
    return [_decode_contact(row, ContactRecord) for row in rows]


def _decode_contact(row: Any, cls: type) -> Any:
    """Reconstruct a :class:`ContactRecord` from a payload row
    (inverse of :func:`_encode_contact`)."""
    p = row["payload"]

    def _f(name: str) -> float | None:
        v = float(p[name])
        return v if np.isfinite(v) else None

    def _i(name: str) -> int | None:
        v = float(p[name])
        return int(round(v)) if np.isfinite(v) else None

    def _auto_or_pos(val_name: str, mode_name: str) -> float | str | None:
        mode = int(p[mode_name])
        if mode == 0:
            return None
        if mode == 1:
            return "auto"
        return float(p[val_name])

    formulation = _str(p["formulation"])
    master_nps = int(p["master_nps"])
    master = np.asarray(p["master_faces"], dtype=np.int64).reshape(-1)
    if master_nps not in (3, 4) or master.size % master_nps != 0:
        raise ValueError(
            f"corrupted contact: master flat length {master.size} is not a "
            f"multiple of master_nps={master_nps}.")
    master_faces = master.reshape(-1, master_nps)

    has_sn = int(p["has_slave_nodes"]) == 1
    slave_nodes = ([int(x) for x in
                    np.asarray(p["slave_nodes"], dtype=np.int64).reshape(-1)]
                   if has_sn else None)
    has_sf = int(p["has_slave_faces"]) == 1
    slave_nps = int(p["slave_nps"])
    if has_sf:
        sf = np.asarray(p["slave_faces"], dtype=np.int64).reshape(-1)
        if slave_nps not in (3, 4) or sf.size % slave_nps != 0:
            raise ValueError(
                f"corrupted contact: mortar slave flat length {sf.size} is not "
                f"a multiple of slave_nps={slave_nps}.")
        slave_faces = sf.reshape(-1, slave_nps)
    else:
        slave_faces = None
        slave_nps = 0

    has_o = int(p["has_outward"]) == 1
    outward = (tuple(float(x) for x in
                     np.asarray(p["outward"], dtype=np.float64).reshape(-1)[:3])
               if has_o else None)

    soft_mode = int(p["soft_mode"])
    if soft_mode == 0:
        soft: float | bool | None = None
    elif soft_mode == 1:
        soft = True
    else:
        soft = float(p["soft"])

    # Edge-edge fallback columns added in neutral 2.25.0 — presence-probe so an
    # in-window 2.24.x file (no columns) decodes to the off defaults.
    has_edge = "edge_edge" in p.dtype.names
    if has_edge:
        edge_edge = int(p["edge_edge"]) == 1
        edge_kn = _auto_or_pos("edge_kn", "edge_kn_mode")
        edge_soft_mode = int(p["edge_soft_mode"])
        if edge_soft_mode == 0:
            edge_soft: float | bool | None = None
        elif edge_soft_mode == 1:
            edge_soft = True
        else:
            edge_soft = float(p["edge_soft"])
        edge_kw = dict(
            edge_edge=edge_edge,
            edge_kn=edge_kn,
            edge_band=_f("edge_band"),
            edge_mu=_f("edge_mu"),
            edge_kt=_f("edge_kt"),
            edge_cohesion=_f("edge_cohesion"),
            edge_tau_max=_f("edge_tau_max"),
            edge_consistent_tan=int(p["edge_consistent_tan"]) == 1,
            edge_soft=edge_soft,
            edge_alm=int(p["edge_alm"]) == 1,
            edge_aug_tol=_f("edge_aug_tol"),
        )
    else:
        edge_kw = {}

    return cls(
        kind="contact",
        name=_str(p["name"]) or None,
        formulation=formulation,
        master_faces=master_faces,
        master_nps=master_nps,
        slave_nodes=slave_nodes,
        slave_faces=slave_faces,
        slave_nps=slave_nps,
        outward=outward,
        kn=_auto_or_pos("kn", "kn_mode"),
        kt=_f("kt"),
        mu=_f("mu"),
        eps_n=_auto_or_pos("eps_n", "eps_n_mode"),
        eps_t=_auto_or_pos("eps_t", "eps_t_mode"),
        cohesion=_f("cohesion"),
        tau_max=_f("tau_max"),
        aug_tol=_f("aug_tol"),
        max_aug=_i("max_aug"),
        ngp=_i("ngp"),
        tie=int(p["tie"]) == 1,
        soft=soft,
        visc=_f("visc"),
        consistent_tan=int(p["consistent_tan"]) == 1,
        geom_tan=int(p["geom_tan"]) == 1,
        # cell added in neutral 2.23.0 — presence-probe so an in-window 2.22.x
        # file (no column) decodes cell=None.
        cell=(_f("cell") if "cell" in p.dtype.names else None),
        **edge_kw,
    )


def _read_contact_planes(parent: Any) -> list[Any]:
    """Decode the ``/contact_planes/contact_planes`` dataset into a list of
    :class:`ContactPlaneRecord` (empty when the group/dataset is absent)."""
    if parent is None or "contact_planes" not in parent:
        return []
    from apeGmsh._kernel.records._constraints import ContactPlaneRecord
    rows = parent["contact_planes"][...]
    if rows.shape == ():
        rows = np.array([rows])
    return [_decode_contact_plane(row, ContactPlaneRecord) for row in rows]


def _decode_contact_plane(row: Any, cls: type) -> Any:
    """Reconstruct a :class:`ContactPlaneRecord` from a payload row
    (inverse of :func:`_encode_contact_plane`)."""
    p = row["payload"]
    slave_nodes = [int(x) for x in
                   np.asarray(p["slave_nodes"], dtype=np.int64).reshape(-1)]
    normal = tuple(float(x) for x in
                   np.asarray(p["normal"], dtype=np.float64).reshape(-1)[:3])
    point = tuple(float(x) for x in
                  np.asarray(p["point"], dtype=np.float64).reshape(-1)[:3])
    visc_v = float(p["visc"])
    soft_mode = int(p["soft_mode"])
    if soft_mode == 0:
        soft: float | bool | None = None
    elif soft_mode == 1:
        soft = True
    else:
        soft = float(p["soft"])
    return cls(
        kind="contact_plane",
        name=_str(p["name"]) or None,
        slave_nodes=slave_nodes,
        normal=normal,
        point=point,
        kn=float(p["kn"]),
        visc=visc_v if np.isfinite(visc_v) else None,
        soft=soft,
    )


# ---------------------------------------------------------------------------
# Load decoders
# ---------------------------------------------------------------------------


def _read_loads(
    parent: Any,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Decode ``/loads/{nodal|element|sp}/{pattern}`` datasets."""
    from apeGmsh._kernel.records._loads import ElementLoadRecord, NodalLoadRecord, SPRecord

    nodal: list[Any] = []
    element: list[Any] = []
    sp: list[Any] = []
    if parent is None:
        return nodal, element, sp

    if "nodal" in parent:
        nodal_grp = parent["nodal"]
        for pattern_safe in nodal_grp.keys():
            ds = nodal_grp[pattern_safe]
            rows = np.atleast_1d(ds[...])
            for row in rows:
                p = row["payload"]
                force = tuple(
                    float(x) for x in np.asarray(p["force_xyz"]).reshape(-1)[:3]
                )
                moment = tuple(
                    float(x) for x in np.asarray(p["moment_xyz"]).reshape(-1)[:3]
                )
                nodal.append(NodalLoadRecord(
                    pattern=_str(pattern_safe),
                    name=_opt_name(p),
                    node_id=int(p["node_id"]),
                    force_xyz=force if any(np.isfinite(force)) else None,
                    moment_xyz=moment if any(np.isfinite(moment)) else None,
                ))

    if "element" in parent:
        elem_grp = parent["element"]
        for pattern_safe in elem_grp.keys():
            ds = elem_grp[pattern_safe]
            rows = np.atleast_1d(ds[...])
            for row in rows:
                p = row["payload"]
                params_str = _str(p["params_json"])
                params = json.loads(params_str) if params_str else {}
                element.append(ElementLoadRecord(
                    pattern=_str(pattern_safe),
                    name=_opt_name(p),
                    element_id=int(p["element_id"]),
                    load_type=_str(p["load_type"]),
                    params=params,
                ))

    if "sp" in parent:
        sp_grp = parent["sp"]
        for pattern_safe in sp_grp.keys():
            ds = sp_grp[pattern_safe]
            rows = np.atleast_1d(ds[...])
            for row in rows:
                p = row["payload"]
                sp.append(SPRecord(
                    pattern=_str(pattern_safe) if pattern_safe != "default" else "default",
                    name=_opt_name(p),
                    node_id=int(p["node_id"]),
                    dof=int(p["dof"]),
                    value=float(p["value"]),
                    is_homogeneous=bool(int(p["is_homogeneous"])),
                ))

    return nodal, element, sp


# ---------------------------------------------------------------------------
# Mass decoder
# ---------------------------------------------------------------------------


def _read_masses(ds: Any) -> "MassSet":
    """Decode the ``/masses`` dataset into a columnar :class:`MassSet`.

    ADR 0065 v2 / plan_emit_memory_columnar.md C1–C3: the ``/masses``
    dataset is *already* columnar on disk (one compound row per node).
    Adopt its ``node_id`` / ``mass`` columns straight into ``MassSet``'s
    ``int64[N]`` / ``float64[N, 6]`` arrays with a single copy instead of
    boxing 7M ``MassRecord`` dataclasses at ``from_h5`` (that rehydration
    was ~2 GB at LOH.1 scale). Float values are copied verbatim from the
    stored ``float64`` payload, so deck ``repr`` stays bit-identical.
    """
    from apeGmsh._kernel.record_sets import MassSet

    if ds is None:
        return MassSet()
    rows = np.atleast_1d(ds[...])
    n = len(rows)
    if n == 0:
        return MassSet()

    payload = rows["payload"]
    # Vectorized column extraction (single copy each). ``node_id`` is a
    # scalar field, ``mass`` a (6,) sub-array field on the compound dtype.
    node_ids = np.ascontiguousarray(payload["node_id"], dtype=np.int64)
    mass = np.ascontiguousarray(payload["mass"], dtype=np.float64)
    if mass.ndim == 1:  # degenerate single-row squeeze guard
        mass = mass.reshape(n, -1)
    if mass.shape[1] < 6:  # tolerate a narrow legacy payload
        pad = np.zeros((n, 6 - mass.shape[1]), dtype=np.float64)
        mass = np.concatenate([mass, pad], axis=1)
    elif mass.shape[1] > 6:
        mass = mass[:, :6]

    # Names are sparse — only rows carrying a non-empty label get an entry.
    # (Pre-2.5.0 files lack the ``name`` field; probe presence like
    # ``_opt_name``, then decode per non-empty row.)
    names: dict[int, str] = {}
    if "name" in (payload.dtype.names or ()):
        raw_names = payload["name"]
        for i in range(n):
            nm = _str(raw_names[i])
            if nm:
                names[i] = nm

    return MassSet(node_ids=node_ids, mass=np.ascontiguousarray(mass),
                   names=names)
