"""FEM element id → OpenSees tag bridge, read from a Phase 8.6+ ``model.h5``.

Built on top of :class:`apeGmsh.opensees.emitter.h5_reader.H5Model`. The
section-cut pipeline needs to filter MPCO results by OpenSees element
tag, but the user thinks (and apeGmsh's FEMData composites speak) in
FEM element ids. Phase 8.6 wired the side-channel
``fem_eids`` parallel array under each
``/opensees/element_meta/{type_token}/`` group, exactly so consumers
can rebuild this map.

Sentinel handling
-----------------
``-1`` in ``fem_eids`` (a.k.a.
``apeGmsh.opensees._internal.tag_resolution.MISSING_FEM_ELEMENT_ID``)
marks records emitted outside a bridge fan-out — standalone H5Emitter
test calls, mostly. These entries are silently dropped when building
the map; they're not "errors", they just don't have a FEM eid to map
to. A lookup with a real-but-unknown fem_eid still raises.

Collision handling
------------------
A FEM element id should appear under at most one OpenSees type token
(one FEM element fans out to exactly one OpenSees element). If a
collision is detected during the build, :meth:`from_h5` raises
``ValueError`` — collisions point to a genuine bridge bug, not an
expected case.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


# Mirrors apeGmsh.opensees._internal.tag_resolution.MISSING_FEM_ELEMENT_ID.
# Keep this module dependency-free of `opensees/_internal/`; the value is
# stable and documented in the h5 schema (Phase 8.6).
_MISSING_FEM_EID = -1


class FemToOpsTagMap:
    """Read-only mapping from FEM element ids to OpenSees element tags.

    Built from a ``model.h5`` written by the apeGmsh bridge (schema
    major 2, with the Phase 8.6 ``fem_eids`` side-channel).

    Lookup is exact — :meth:`ops_tags_for_fem_eids` raises on any
    unknown FEM id (rather than silently dropping). This matches the
    section-cut use case: if a physical group resolves to a FEM eid
    that has no OpenSees tag, that's a setup bug and the user wants to
    know loudly.
    """

    __slots__ = ("_by_fem_eid", "_type_of_fem_eid", "_n_sentinel", "_by_ops_tag")

    def __init__(
        self,
        *,
        by_fem_eid: Mapping[int, int],
        type_of_fem_eid: Mapping[int, str],
        n_sentinel: int = 0,
    ) -> None:
        self._by_fem_eid: dict[int, int] = dict(by_fem_eid)
        self._type_of_fem_eid: dict[int, str] = dict(type_of_fem_eid)
        self._n_sentinel: int = int(n_sentinel)
        # Inverse lookup, built lazily on first inverse query. The
        # forward map is bijective by construction (collisions raise in
        # :meth:`from_h5`), so the inverse is single-valued.
        self._by_ops_tag: dict[int, int] | None = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_reader(cls, reader: Any) -> "FemToOpsTagMap":
        """Build the map from any open H5ModelReader-conforming object.

        ADR 0026 PR7-b — the canonical factory.  Walks every
        ``/opensees/element_meta/{type_token}/`` group on the reader,
        pairs each row of ``ids`` (OpenSees tag) with ``fem_eids`` (FEM
        eid), drops sentinel ``-1`` rows, and detects collisions.

        Does **not** close the reader.  Lifecycle is the caller's
        responsibility — typically a ``with h5_reader.open(...) as r``
        block in :meth:`from_h5` or a long-lived
        :class:`ResultsDirector`-owned reader.

        Parameters
        ----------
        reader
            Any object exposing ``element_meta()`` and
            ``element_meta_arrays(type_token)`` per the H5ModelReader
            Protocol — duck-typed as ``Any`` so the cuts subpackage
            does not import the Protocol class (preserving its
            independence from ``apeGmsh.viewers.data``).

        Raises
        ------
        ValueError
            If a FEM eid appears in more than one type group, or if
            a type group is missing the ``fem_eids`` dataset.
        """
        by_fem: dict[int, int] = {}
        type_of: dict[int, str] = {}
        n_sentinel = 0

        for type_token in reader.element_meta():
            arrays = reader.element_meta_arrays(type_token)
            if "fem_eids" not in arrays:
                # Pre-Phase-8.6 file — bail with a clear message.
                raise ValueError(
                    f"/opensees/element_meta/{type_token} is missing the "
                    "'fem_eids' dataset. Re-emit model.h5 with "
                    "apeGmsh ≥ Phase 8.6 (schema 2.2.0)."
                )
            ids = np.asarray(arrays["ids"], dtype=np.int64)
            fem_eids = np.asarray(arrays["fem_eids"], dtype=np.int64)
            for ops_tag, fem_eid in zip(ids, fem_eids):
                fid = int(fem_eid)
                if fid == _MISSING_FEM_EID:
                    n_sentinel += 1
                    continue
                if fid in by_fem:
                    prior_tag = by_fem[fid]
                    prior_type = type_of[fid]
                    raise ValueError(
                        f"FEM eid {fid} is mapped to OpenSees tag "
                        f"{prior_tag} ({prior_type}) and also to "
                        f"{int(ops_tag)} ({type_token}). One FEM "
                        "element must fan out to exactly one OpenSees "
                        "element — this is a bridge bug."
                    )
                by_fem[fid] = int(ops_tag)
                type_of[fid] = str(type_token)

        return cls(
            by_fem_eid=by_fem,
            type_of_fem_eid=type_of,
            n_sentinel=n_sentinel,
        )

    @classmethod
    def from_h5(cls, path: str | Path) -> "FemToOpsTagMap":
        """Build the map by reading ``path`` through the reference reader.

        Convenience shim over :meth:`from_reader`: opens the file via
        :func:`apeGmsh.opensees.emitter.h5_reader.open`, delegates the
        walk, then closes.  ADR 0026 keeps this path-based entry point
        for callers that own only a path string (e.g. legacy session-
        restore) — the canonical reader-based factory is
        :meth:`from_reader`.

        Raises
        ------
        ValueError
            If a FEM eid appears in more than one type group.
        FileNotFoundError, SchemaVersionError, MalformedH5Error
            Propagated from :func:`apeGmsh.opensees.emitter.h5_reader.open`.
        """
        # Local import to keep the cuts subpackage importable without
        # eagerly loading h5py-bearing modules at apeGmsh import time.
        from apeGmsh.opensees.emitter import h5_reader

        with h5_reader.open(str(path)) as reader:
            return cls.from_reader(reader)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def ops_tag(self, fem_eid: int) -> int:
        """Look up one FEM eid → OpenSees tag. Raises ``KeyError`` if absent."""
        try:
            return self._by_fem_eid[int(fem_eid)]
        except KeyError:
            raise KeyError(
                f"FEM eid {int(fem_eid)} not found in tag map. "
                f"Map contains {len(self._by_fem_eid)} entries; check that "
                "the FEM eid actually appears in the bridge's element fan-out."
            ) from None

    def ops_tags_for_fem_eids(
        self,
        fem_eids: Iterable[int] | np.ndarray,
    ) -> tuple[int, ...]:
        """Look up many FEM eids in one call.

        Returns a tuple in the same order as the input. Raises
        ``KeyError`` listing every missing FEM id — so the user sees
        the full set of failures, not just the first.
        """
        arr = (
            fem_eids if isinstance(fem_eids, np.ndarray)
            else np.fromiter((int(x) for x in fem_eids), dtype=np.int64)
        )
        out: list[int] = []
        missing: list[int] = []
        for fid in arr:
            fid_int = int(fid)
            tag = self._by_fem_eid.get(fid_int)
            if tag is None:
                missing.append(fid_int)
            else:
                out.append(tag)
        if missing:
            raise KeyError(
                f"{len(missing)} FEM eid(s) not in tag map: {missing[:10]}"
                + ("…" if len(missing) > 10 else "")
            )
        return tuple(out)

    def type_token_for(self, fem_eid: int) -> str:
        """Return the OpenSees element type token for one FEM eid."""
        return self._type_of_fem_eid[int(fem_eid)]

    def fem_eids_for_ops_tags(
        self,
        ops_tags: Iterable[int] | np.ndarray,
    ) -> tuple[int, ...]:
        """Inverse lookup — OpenSees tags → FEM element ids.

        Mirror of :meth:`ops_tags_for_fem_eids`. Returns a tuple in the
        same order as the input. Raises ``KeyError`` listing every
        missing OpenSees tag so the user sees the full failure set.

        The viewer uses this to walk a :class:`apeGmsh.cuts.SectionCutDef`
        (which carries OpenSees tags) back into FEM eids so it can pick
        out the corresponding rows of the FEM connectivity / coords.
        """
        if self._by_ops_tag is None:
            self._by_ops_tag = {
                int(ops): int(fem)
                for fem, ops in self._by_fem_eid.items()
            }
        arr = (
            ops_tags if isinstance(ops_tags, np.ndarray)
            else np.fromiter((int(x) for x in ops_tags), dtype=np.int64)
        )
        out: list[int] = []
        missing: list[int] = []
        for tag in arr:
            tag_int = int(tag)
            fid = self._by_ops_tag.get(tag_int)
            if fid is None:
                missing.append(tag_int)
            else:
                out.append(fid)
        if missing:
            raise KeyError(
                f"{len(missing)} OpenSees tag(s) not in tag map: {missing[:10]}"
                + ("…" if len(missing) > 10 else "")
            )
        return tuple(out)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def n_sentinel(self) -> int:
        """Count of ``fem_eids == -1`` rows dropped during build.

        Non-zero is fine — it means the h5 contains records emitted
        outside a bridge fan-out (e.g. standalone H5Emitter tests).
        """
        return self._n_sentinel

    @property
    def type_tokens(self) -> tuple[str, ...]:
        """Distinct OpenSees type tokens present in the map."""
        return tuple(sorted(set(self._type_of_fem_eid.values())))

    def __len__(self) -> int:
        return len(self._by_fem_eid)

    def __contains__(self, fem_eid: object) -> bool:
        try:
            return int(fem_eid) in self._by_fem_eid  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return False

    def __repr__(self) -> str:
        return (
            f"FemToOpsTagMap(n={len(self)}, "
            f"types={self.type_tokens}, sentinel_dropped={self._n_sentinel})"
        )
