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

Third-party captures (``.mpco``) carry no physical groups and no stage
names; the sibling ``model.h5`` that :meth:`Results.from_mpco` requires
carries both.  :func:`_resolve_fem_via_model` and
:func:`_bind_stage_names` bring them across (#1325, #1324).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from ..opensees.opensees_model import OpenSeesModel
    from .readers._protocol import ResultsReader


class ModelFemMismatchWarning(UserWarning):
    """``model_h5=`` holds a FEMData that does not cover the capture's nodes.

    The results were bound to the capture's own synthesized FEMData
    instead (no physical groups, no labels).  Either the ``model_h5``
    belongs to another model, or the run added nodes the archive does
    not know; pass ``fem=`` to bind a FEMData regardless.
    """


class StageCountMismatchWarning(UserWarning):
    """``model_h5=`` declares a different number of stages than the capture.

    The program's stage names cannot be paired positionally, so the
    capture keeps its own ``MODEL_STAGE[<k>]`` names.
    """


def _resolve_fem_via_model(
    reader: "ResultsReader",
    candidate: "Optional[FEMData]",
    model: "OpenSeesModel",
    *,
    model_path: "str | Path",
) -> "Optional[FEMData]":
    """Pick the FEMData for a third-party capture opened beside ``model_h5``.

    Resolution rules (#1325):

    1. ``candidate`` provided: return it, unchecked — same contract as
       :func:`_resolve_fem`.
    2. Otherwise the neutral FEMData archived in ``model_h5`` — the one
       the bridge emitted the run from, with its physical groups and
       labels — provided its node ids cover every node id the capture
       reports.  Node tags are fem node ids on the bridge's emit path,
       so a capture node the archive does not know means the pairing is
       wrong (or the run grew the domain); that case warns
       :class:`ModelFemMismatchWarning` and falls through.
    3. The reader's own synthesized FEMData (may be ``None``).
    """
    if candidate is not None:
        return candidate
    embedded = reader.fem()
    model_fem = model.fem
    if embedded is None:
        return model_fem
    capture_ids = np.asarray(embedded.nodes.ids, dtype=np.int64)
    archive_ids = np.asarray(model_fem.nodes.ids, dtype=np.int64)
    missing = np.setdiff1d(capture_ids, archive_ids)
    if missing.size == 0:
        return model_fem
    shown = ", ".join(str(int(n)) for n in missing[:5])
    more = f", ... ({missing.size} in all)" if missing.size > 5 else ""
    warnings.warn(
        ModelFemMismatchWarning(
            f"model_h5={str(model_path)!r} does not cover the capture's "
            f"nodes: {missing.size} of {capture_ids.size} node ids are "
            f"absent from its FEMData (e.g. {shown}{more}). Binding the "
            f"capture's own MODEL group instead, which has no physical "
            f"groups; pass fem= to bind a FEMData explicitly."
        ),
        stacklevel=3,
    )
    return embedded


def _bind_stage_names(
    reader: "Any",
    model: "OpenSeesModel",
    *,
    model_path: "str | Path",
) -> None:
    """Rename the capture's stages after the archive's program (#1324).

    ``model.stages()`` lists the ``ops.stage(name=...)`` blocks in
    registration order; the bridge emits one ``domainChange`` per
    stage, so the recorder opens one ``MODEL_STAGE[<k>]`` per program
    stage and the pairing is positional.  When the counts differ
    (extra hand-written stages, a stage with no recorded step, a
    foreign archive) nothing is renamed and
    :class:`StageCountMismatchWarning` says so.  A vanilla archive (no
    stages) is silent: there is nothing to map.
    """
    program = [str(s.name) for s in model.stages()]
    if not program:
        return
    capture = reader.stages()
    if len(program) != len(capture):
        warnings.warn(
            StageCountMismatchWarning(
                f"model_h5={str(model_path)!r} declares {len(program)} "
                f"stages ({program}) but the capture holds "
                f"{len(capture)} MODEL_STAGE groups "
                f"({[s.name for s in capture]}); keeping the capture's "
                f"names. Select stages by MODEL_STAGE name or by index."
            ),
            stacklevel=3,
        )
        return
    reader.attach_stage_names(program)


def _resolve_fem(
    reader: "ResultsReader",
    candidate: "Optional[FEMData]",
) -> "Optional[FEMData]":
    """Pick the right FEMData for binding.

    Resolution rules:

    1. If ``candidate`` is provided: return it without touching the
       reader (preferred — carries apeGmsh-specific labels and
       provenance that may be richer than the embedded snapshot). No
       hash validation is performed; it is the user's responsibility to
       provide a FEMData consistent with the results file. The embedded
       zone is never read on this path, so a results file whose
       ``/model`` is below its floor (ADR 0113 D9) binds a supplied fem.
    2. If ``candidate`` is None: return the reader's embedded fem
       (may itself be None — bare construction is allowed). A
       ``NativeReader`` whose embedded ``/model`` is below its floor
       raises here; ``Results.from_native`` checks
       ``unavailable_zones`` first and defers that refusal to
       ``Results.fem``.
    """
    if candidate is not None:
        return candidate
    return reader.fem()


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
