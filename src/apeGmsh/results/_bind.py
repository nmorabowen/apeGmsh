"""Bind resolution — picks the FEMData / OpenSeesModel to use when opening a results file.

Resolution prefers an explicit candidate when supplied (it typically
carries richer apeGmsh-specific labels and provenance than the
embedded snapshot), and falls back to the embedded FEM / model
otherwise.

The historic ``snapshot_id``-equality check has been removed: it is
on the user to pair a candidate FEMData with a results file from the
same run. The hash is still computed and stored for caching and
metadata, but bind no longer rejects on mismatch.

Phase 8 (ADR 0021) — :class:`BindError` deleted (was inert through
Phase 6; this is the prune). The remaining helpers are
internal-by-convention; :func:`resolve_bound_model` is the only one
still on the public-ish surface (it brokers the
:class:`OpenSeesModel` chain forward).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from ..opensees.opensees_model import OpenSeesModel
    from .readers._protocol import ResultsReader


def _resolve_fem(
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


def resolve_bound_model(
    reader: "ResultsReader",
    candidate: "Optional[OpenSeesModel]",
) -> "Optional[OpenSeesModel]":
    """Pick the right :class:`OpenSeesModel` for binding (ADR 0020).

    Resolution rules:

    1. If ``candidate`` is provided: return it (user-supplied wins).
       Passing a model explicitly takes precedence over any
       auto-resolve the reader would have done.
    2. If ``candidate`` is None: ask the reader. Native readers
       auto-resolve from the file's ``/opensees/`` zone when present
       (silent, no warning per ADR 0020). MPCO readers always return
       ``None`` (MPCO has no ``/opensees/`` zone, per the
       ``project_mpco_no_vecxz`` memory).
    3. If neither has one: return ``None``. The caller is responsible
       for surfacing the missing-model condition as a ``TypeError``
       per the Phase 8 contract.
    """
    if candidate is not None:
        return candidate
    # Protocol method (Phase 4 extension) — readers added in lockstep
    # with this helper. ``getattr`` cushions the rollout against any
    # third-party reader that hasn't picked up the protocol extension
    # yet; missing the method == "no model available", which matches
    # the contract above.
    fetch = getattr(reader, "opensees_model", None)
    if fetch is None:
        return None
    return fetch()
