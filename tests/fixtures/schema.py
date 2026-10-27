"""Single source of truth for schema versions used in test fixtures.

Tests stamping ``/meta/schema_version`` or ``/meta/opensees_schema_version``
in synthetic h5 fixtures must import from here so the next minor bump
is a one-file edit.  Each zone's reader accepts every minor from its
``*_FLOOR`` up to ``*_CURRENT`` on the same major (ADR 0113 (#1303)); the
floors move only with a major bump.  ``*_PRIOR_MINOR`` is the previous
minor, kept for the tests of reader shims keyed on the newest minor.
Comparing a schema version against a literal anywhere else in ``tests/``
fails ``scripts/check_quirks.py`` (rule ``schema-literal``): it went stale
and turned main red at 2.12.0, 2.13.0 and 2.16.0.
"""
OPENSEES_CURRENT     = "2.37.0"  # #2470 joint offsets (a -jntOffset geomTransf line gains a jnt_offset dataset)
OPENSEES_PRIOR_MINOR = "2.36.0"  # #2375 (a pinned OpenSeesTarget.mode is /opensees@target_mode)
OPENSEES_SOLVE_STAMP_FROM = "2.24.0"  # the minor that added the /opensees solve stamp (K1-5 #1462); fixed history
OPENSEES_PARAM_TAGS_FROM = "2.27.0"  # the minor whose archives store every parameter tag (K1-8 #1465); fixed history
OPENSEES_EMBEDDED_REBAR_FROM = "2.29.0"  # the minor whose archives store every reinforce tie (K2b #1795); fixed history
OPENSEES_PROGRAM_DECL_FROM = "2.31.0"  # the minor whose /opensees/program.decl records declarations (A7 #1839); fixed history
OPENSEES_VALUE_REQUIRES_FROM = "2.33.0"  # the minor whose @requires names "mpco" / "mpi" (K #2372); fixed history
OPENSEES_LARGE_ARRAY_DATASETS_FROM = "2.34.0"  # the minor whose oversized params / ids are datasets (K #2418); fixed history
OPENSEES_EQUATION_CONSTRAINT_FROM = "2.35.0"  # the minor whose archives store every global equationConstraint line (K #2447); fixed history
OPENSEES_BEFORE_EQUATION_CONSTRAINT = "2.34.0"  # the minor just below it, which stores none (K #2447); fixed history
OPENSEES_TARGET_MODE_FROM = "2.36.0"  # the minor whose /opensees@target_mode keeps a pinned target (K #2375); fixed history
OPENSEES_BEFORE_TARGET_MODE = "2.35.0"  # the minor just below it, which has no @target_mode (K #2375); fixed history
OPENSEES_FLOOR       = "2.12.0"  # ADR 0113 D3 evidence gate (#1329): no 2.11-era file opens (neutral 2.7 stamps)
NEUTRAL_CURRENT      = "2.36.0"  # #2551 S1: optional /elements/{alias}/entity (per-element source entity)
NEUTRAL_PRIOR_MINOR  = "2.35.0"  # #1338: additive `source` column on /loads/nodal (the definition kind)
NEUTRAL_FLOOR        = "2.10.0"  # B2 layout split; every later minor is additive or shimmed
NEUTRAL_ENTITY_FROM  = "2.36.0"  # the minor that added /elements/{alias}/entity (#2551 S1); fixed history
NEUTRAL_BEFORE_ENTITY = "2.35.0"  # the minor just below it, whose files carry no entity stream (#2551 S1); fixed history
RESULTS_FLOOR        = "1.0.0"   # the results zone's first version
RESULTS_CURRENT      = "1.2.0"   # #1746: optional root element_index_space stamp
RESULTS_PRIOR_MINOR  = "1.1.0"   # Phase 6 (ADR 0021): /meta/lineage
# ADR 0112 D2/D3 root zones (#1304). Both start at 1.0.0; a zone's floor
# is its first version. Geometry has no prior minor yet; add
# GEOMETRY_PRIOR_MINOR at its first minor bump.
GEOMETRY_CURRENT       = "1.0.0"  # V2a: /geometry zone registered (sibling <stem>.geometry.h5)
PROVENANCE_CURRENT     = "1.1.0"  # V2d #1378: additive /provenance/records/origin column
PROVENANCE_PRIOR_MINOR = "1.0.0"  # V2a: /provenance zone registered
GEOMETRY_FLOOR         = "1.0.0"
PROVENANCE_FLOOR       = "1.0.0"
