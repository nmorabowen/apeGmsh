"""Bind resolution — picks the FEMData to use when opening a results file.

Resolution prefers an explicit candidate when supplied (it typically
carries richer apeGmsh-specific labels and provenance than the
embedded snapshot), and falls back to the embedded FEM otherwise.

The historic ``snapshot_id``-equality check has been removed: it is
on the user to pair a candidate FEMData with a results file from the
same run. The hash is still computed and stored for caching and
metadata, but bind no longer rejects on mismatch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from .readers._protocol import ResultsReader


class BindError(ValueError):
    """Retained for back-compat; no longer raised by :func:`resolve_bound_fem`."""


def resolve_bound_fem(
    reader: "ResultsReader",
    candidate: "Optional[FEMData]",
) -> "Optional[FEMData]":
    """Pick the right FEMData for binding.

    Resolution rules:

    1. If ``candidate`` is None: return the reader's embedded fem
       (may itself be None — bare construction is allowed).
    2. If ``candidate`` is provided: return it (preferred — carries
       apeGmsh-specific labels and provenance that may be richer than
       the embedded snapshot). No hash validation is performed; it is
       the user's responsibility to provide a FEMData consistent with
       the results file.
    """
    embedded = reader.fem()
    if candidate is None:
        return embedded
    return candidate
