"""Single source of truth for schema versions used in test fixtures.

Tests stamping ``/meta/schema_version`` or ``/meta/opensees_schema_version``
in synthetic h5 fixtures must import from here so the next minor bump
is a one-file edit.  Per ADR 0023's two-version reader window,
``*_PRIOR_MINOR`` is the oldest version the current reader accepts.
"""
OPENSEES_CURRENT     = "2.18.0"  # ADR 0055 Phase 2 (/opensees/stages staged archival)
OPENSEES_PRIOR_MINOR = "2.17.0"  # ADR 0049 (node-pair zeroLength: inline_connectivity)
NEUTRAL_CURRENT      = "2.12.0"  # fork coupling knobs (CouplingControl: cpl_* columns on node_group + interpolation payloads; sr_cpl_* mirror on surface_coupling)
NEUTRAL_PRIOR_MINOR  = "2.11.0"  # ADR 0049 PR-4: /nodes/provenance (decoupled nodes); snapshot_id hash folds provenance
