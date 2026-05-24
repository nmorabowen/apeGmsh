"""Single source of truth for schema versions used in test fixtures.

Tests stamping ``/meta/schema_version`` or ``/meta/opensees_schema_version``
in synthetic h5 fixtures must import from here so the next minor bump
is a one-file edit.  Per ADR 0023's two-version reader window,
``*_PRIOR_MINOR`` is the oldest version the current reader accepts.
"""
OPENSEES_CURRENT     = "2.11.0"  # fix: 0-based runtime ranks (was Gmsh 1-based)
OPENSEES_PRIOR_MINOR = "2.10.0"  # ADR 0027 (/opensees/partitions/ zone)
NEUTRAL_CURRENT      = "2.6.0"   # Phase 6 (lineage chain)
NEUTRAL_PRIOR_MINOR  = "2.5.0"
