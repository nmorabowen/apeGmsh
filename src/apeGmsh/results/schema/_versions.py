"""Version constants for the results module on-disk formats.

Bump ``SCHEMA_VERSION`` minor (``"1.x"``) for additive changes,
major (``"2.0"``) for breaking changes. Bump ``PARSER_VERSION``
when recorder transcoder logic changes in a way that invalidates
cached transcoded HDF5 files.

Schema history
--------------

* ``"1.0"`` — initial results schema (Phase 1).
* ``"1.1"`` — Phase 4 additive: the ``/opensees/`` bridge zone may
  be embedded at the root of a native results file (Composed-file
  pattern per ADR 0020), paired with a rich ``/model/`` neutral
  zone.  Old result files (no zones) keep working — readers treat
  the absence of ``/opensees/`` as ``Results.model is None``.  The
  pre-cleanup ``/opensees_archive/`` mirror introduced earlier in
  Phase 4 has been removed; readers no longer materialise a temp
  file to rehydrate :class:`OpenSeesModel` from the composed file.

Per-zone stamps (ADR 0023)
--------------------------
The envelope :data:`SCHEMA_VERSION` is the partition-shape stamp
(bumps when a new zone is added or the partition layout changes).
Per ADR 0023 §"Three per-zone version stamps + one envelope", the
results file ALSO writes the per-zone marker:

* ``/meta/results_schema_version`` — :data:`RESULTS_SCHEMA_VERSION`,
  introduced by Phase 4 alongside the embedded ``/opensees/`` zone.
  Bumps only when the *results-zone* content shape changes (stage
  layout, slab dtypes, ...); independent of the envelope.

Phase 7a (ADR 0023 §"Per-zone read validation") will land the
two-version reader window for results consumers; Phase 4 only writes
the new attribute key so old files reach Phase 7a with the marker
already in place.

Phase 6 (ADR 0021) — ``RESULTS_SCHEMA_VERSION`` bumps 1.0.0 → 1.1.0
for the additive ``/meta/lineage/`` sub-group stamped by
:meth:`NativeWriter.close`.  Old result files lacking the sub-group
keep working (``Results.lineage`` surfaces a "lineage absent" warning
without raising).
"""
from __future__ import annotations

SCHEMA_VERSION = "1.1.0"
RESULTS_SCHEMA_VERSION = "1.1.0"
PARSER_VERSION = "1.0"
