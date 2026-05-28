"""Single source of truth for schema versions used in test fixtures.

Tests stamping ``/meta/schema_version`` or ``/meta/opensees_schema_version``
in synthetic h5 fixtures must import from here so the next minor bump
is a one-file edit.  Per ADR 0023's two-version reader window,
``*_PRIOR_MINOR`` is the oldest version the current reader accepts.
"""
OPENSEES_CURRENT     = "2.12.0"  # ADR 0035 (ASDEmbeddedNodeElement option exposure)
OPENSEES_PRIOR_MINOR = "2.11.0"  # fix: 0-based runtime ranks (was Gmsh 1-based)
NEUTRAL_CURRENT      = "2.10.0"  # B2: physical_groups/labels split into node_side/element_side sub-trees; snapshot_id hash widened
NEUTRAL_PRIOR_MINOR  = "2.9.0"   # ADR 0038 / Phase 3A.1: composed_from + tag_span_max + module_label
