"""Bind validation — links a results file to a FEMData by snapshot_id.

The bind contract: a FEMData and a results file are compatible iff
their ``snapshot_id`` hashes match. Re-meshing produces a new hash;
old artifacts refuse to bind to the new fem.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from .readers._protocol import ResultsReader


class BindError(ValueError):
    """Raised when a candidate FEMData does not match the embedded snapshot."""


def resolve_bound_fem(
    reader: "ResultsReader",
    candidate: "Optional[FEMData]",
) -> "Optional[FEMData]":
    """Pick the right FEMData for binding, validating consistency.

    Resolution rules:

    1. If ``candidate`` is None: return the reader's embedded fem
       (may itself be None — bare construction is allowed).
    2. If ``candidate`` is provided and the reader has an embedded fem:
       both must have matching ``snapshot_id``. Returns ``candidate``
       (preferred — carries apeGmsh-specific labels and provenance
       that may be richer than the embedded snapshot).
    3. If ``candidate`` is provided and the reader has no embedded fem
       (e.g. a results file written without the ``fem=`` argument):
       returns ``candidate`` without further validation. The user has
       opted in by providing it.
    """
    embedded = reader.fem()

    if candidate is None:
        return embedded

    if embedded is None:
        return candidate

    if candidate.snapshot_id != embedded.snapshot_id:
        raise BindError(
            f"FEMData snapshot_id mismatch: results file has "
            f"{embedded.snapshot_id!r}, candidate has "
            f"{candidate.snapshot_id!r}. The candidate was built from "
            f"a different mesh — re-extract get_fem_data() from the "
            f"session that produced these results, or re-run the analysis."
        )
    return candidate
