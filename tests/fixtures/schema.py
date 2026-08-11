"""Single source of truth for schema versions used in test fixtures.

Tests stamping ``/meta/schema_version`` or ``/meta/opensees_schema_version``
in synthetic h5 fixtures must import from here so the next minor bump
is a one-file edit.  Per ADR 0023's two-version reader window,
``*_PRIOR_MINOR`` is the oldest version the current reader accepts.
"""
OPENSEES_CURRENT     = "2.20.0"  # ADR 0078 A1 (/opensees/computed_sections provenance sidecar)
OPENSEES_PRIOR_MINOR = "2.19.0"  # ADR 0055 Phase 5 P5.1 (partitioned staged archival; no layout change)
NEUTRAL_CURRENT      = "2.28.0"  # ADR 0091 load basis (NodalLoadRecord.basis column)
NEUTRAL_PRIOR_MINOR  = "2.27.0"  # tie stiffness="auto" (InterpolationRecord.stiffness_auto + sr_ mirror columns)
