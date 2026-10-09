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
OPENSEES_CURRENT     = "2.23.0"  # ADR 0114 R2/R3a (/opensees/program + /opensees/commands, #1461)
OPENSEES_PRIOR_MINOR = "2.22.0"  # ADR 0112 am. 5 (/opensees/bcs@mass_from_model marker, #1304)
OPENSEES_FLOOR       = "2.12.0"  # ADR 0113 D3 evidence gate (#1329): no 2.11-era file opens (neutral 2.7 stamps)
NEUTRAL_CURRENT      = "2.35.0"  # #1338: additive `source` column on /loads/nodal (the definition kind)
NEUTRAL_PRIOR_MINOR  = "2.34.0"  # #1291: /meta/ndm is the ops.model spatial dimension (0 = undeclared)
NEUTRAL_FLOOR        = "2.10.0"  # B2 layout split; every later minor is additive or shimmed
RESULTS_FLOOR        = "1.0.0"   # the results zone's first version
# ADR 0112 D2/D3 root zones (#1304). Both start at 1.0.0; a zone's floor
# is its first version. Geometry has no prior minor yet; add
# GEOMETRY_PRIOR_MINOR at its first minor bump.
GEOMETRY_CURRENT       = "1.0.0"  # V2a: /geometry zone registered (sibling <stem>.geometry.h5)
PROVENANCE_CURRENT     = "1.1.0"  # V2d #1378: additive /provenance/records/origin column
PROVENANCE_PRIOR_MINOR = "1.0.0"  # V2a: /provenance zone registered
GEOMETRY_FLOOR         = "1.0.0"
PROVENANCE_FLOOR       = "1.0.0"
