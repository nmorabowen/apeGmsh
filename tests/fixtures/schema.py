"""Single source of truth for schema versions used in test fixtures.

Tests stamping ``/meta/schema_version`` or ``/meta/opensees_schema_version``
in synthetic h5 fixtures must import from here so the next minor bump
is a one-file edit.  Per ADR 0023's two-version reader window,
``*_PRIOR_MINOR`` is the oldest version the current reader accepts.
Comparing a schema version against a literal anywhere else in ``tests/``
fails ``scripts/check_quirks.py`` (rule ``schema-literal``): it went stale
and turned main red at 2.12.0, 2.13.0 and 2.16.0.
"""
OPENSEES_CURRENT     = "2.22.0"  # ADR 0112 am. 5 (/opensees/bcs@mass_from_model marker, #1304)
OPENSEES_PRIOR_MINOR = "2.21.0"  # SSI-2.E (/opensees/stages/*/update_material_stage)
NEUTRAL_CURRENT      = "2.34.0"  # #1291: /meta/ndm is the ops.model spatial dimension (0 = undeclared)
NEUTRAL_PRIOR_MINOR  = "2.33.0"  # fork #839: additive `cpl_al_update` column on the coupling-control lane
# ADR 0112 D2/D3 root zones (#1304). Both start at 1.0.0, so neither has a
# prior minor yet; add *_PRIOR_MINOR at their first minor bump.
GEOMETRY_CURRENT     = "1.0.0"  # V2a: /geometry zone registered (sibling <stem>.geometry.h5)
PROVENANCE_CURRENT   = "1.0.0"  # V2a: /provenance zone registered
