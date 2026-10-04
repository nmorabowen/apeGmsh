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

Phase 7a (ADR 0023 §"Per-zone read validation") landed the reader
gate; ADR 0113 (#1303) replaced its two-version window with a floor,
:data:`RESULTS_SCHEMA_FLOOR`, so every version from the floor to the
current one opens.

Phase 6 (ADR 0021) — ``RESULTS_SCHEMA_VERSION`` bumps 1.0.0 → 1.1.0
for the additive ``/meta/lineage/`` sub-group stamped by
:meth:`NativeWriter.close`.  Old result files lacking the sub-group
keep working (``Results.lineage`` surfaces a "lineage absent" warning
without raising).
"""
from __future__ import annotations

SCHEMA_VERSION = "1.1.0"
RESULTS_SCHEMA_VERSION = "1.1.0"
#: Oldest results-zone version the reader opens (ADR 0113 (#1303)). It moves
#: only with a major bump; ``schema_version.reader_floor`` reads it here.
RESULTS_SCHEMA_FLOOR = "1.0.0"
PARSER_VERSION = "1.0"

# Ladruno recorder on-disk format — the integer ``INFO/FORMAT_VERSION``
# the ``.ladruno`` HDF5 writer stamps (currently 1). The reader is pinned
# to a supported set: additive changes *within* a version are tolerated
# (the schema is actively evolving), but a FORMAT_VERSION bump signals a
# breaking layout change and must be added here deliberately. The reader
# rejects any version outside this set with a clear error rather than
# silently misreading (ADR 0023 spirit, integer-keyed).
#
# Policy: maintain a rolling **two-version window** — the current writer
# version N and the immediately prior N-1 — so a reader can straddle a
# writer bump (read both the old and new fixtures during a transition);
# drop versions older than N-1. Only v1 exists today, so the set is just
# ``(1,)``; when the writer ships v2, widen to ``(1, 2)`` (NOT ``(2,)``).
LADRUNO_FORMAT_VERSION = 1
LADRUNO_SUPPORTED_FORMAT_VERSIONS: tuple[int, ...] = (1,)
