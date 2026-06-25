"""Single source of truth for schema versions used in test fixtures.

Tests stamping ``/meta/schema_version`` or ``/meta/opensees_schema_version``
in synthetic h5 fixtures must import from here so the next minor bump
is a one-file edit.  Per ADR 0023's two-version reader window,
``*_PRIOR_MINOR`` is the oldest version the current reader accepts.
"""
OPENSEES_CURRENT     = "2.19.0"  # ADR 0055 Phase 5 P5.1 (partitioned staged archival; no layout change)
OPENSEES_PRIOR_MINOR = "2.18.0"  # ADR 0055 Phase 2 (/opensees/stages staged archival)
NEUTRAL_CURRENT      = "2.18.0"  # ADR 0069 follow-up EmbeddedNodeControl pressure tie (cpl_pressure/cpl_kp)
NEUTRAL_PRIOR_MINOR  = "2.17.0"  # ADR 0069 equalDOF_Mixed (NodePairRecord.master_dofs column)
