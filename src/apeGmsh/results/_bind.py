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
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from ..opensees.opensees_model import OpenSeesModel
    from .readers._protocol import ResultsReader


class ModelFemMismatchWarning(UserWarning):
    """``model_h5=`` holds a FEMData that does not match the capture's nodes.

    Either a capture node id is absent from the archive, or a shared id
    sits at different coordinates (a different mesh of the same part:
    #1393).  The results were bound to the capture's own synthesized
    FEMData instead (no physical groups, no labels).  Either the
    ``model_h5`` belongs to another model, or the run added nodes the
    archive does not know; pass ``fem=`` to bind a FEMData regardless.
    """


class StageCountMismatchWarning(UserWarning):
    """``model_h5=`` declares a different number of stages than the capture.

    Fewer capture stages than program stages is a partial run: the
    program names are paired onto the prefix and the rest have no
    capture.  More capture stages than program stages cannot be paired
    positionally, so the capture keeps its own ``MODEL_STAGE[<k>]``
    names.
    """


class DuplicateStageNameWarning(UserWarning):
    """``model_h5=`` declares the same ``ops.stage(name=...)`` twice.

    The names are still attached, but ``results.stage(<name>)`` resolves
    to the first stage so named; select the others by id (``stage_<k>``)
    or by their ``MODEL_STAGE[<k>]`` alias.
    """


class ShadowedStageNameWarning(UserWarning):
    """A program stage is named like ANOTHER capture stage's id.

    ``results.stage(x)`` resolves the exact id ``stage_<k>`` before any
    name, so a stage named ``stage_1`` that sits at id ``stage_0`` is
    unreachable by its name: ``stage("stage_1")`` is the second stage.
    The name is still attached; select the shadowed stage by its id or
    its ``MODEL_STAGE[<k>]`` alias, or rename it in the program.
    """


# Coordinates agree when every component differs by less than this
# fraction of the model's bounding-box diagonal.  The deck round-trips
# float64 through ``repr`` (exact) and STKO stores float64, so a real
# pairing agrees to the last bit; a different mesh of the same part
# differs at element-size scale, orders of magnitude above this.
_COORD_RTOL = 1e-6


def _coords_mismatch(
    capture_ids: np.ndarray, capture_xyz: np.ndarray,
    archive_ids: np.ndarray, archive_xyz: np.ndarray,
    *, ndm: int,
) -> "tuple[float, float]":
    """``(max |dx|, tolerance)`` over the capture's ids, both in the archive.

    The caller has already proved every capture id is in the archive.
    Coordinates are ``(N, 3)`` on both sides (``FEMData`` pads 2-D to
    three columns), but only the first ``ndm`` columns are the model's:
    a 2-D deck emits ``x, y`` and the recorder stores two columns that
    the MPCO synthesis pads with ``z = 0``, while the archive keeps
    gmsh's real ``z``.  An ``ndm=2`` model drawn on an offset plane
    (legal since #1346) agrees in ``x, y`` and differs in ``z`` by the
    offset, so the dropped axis is not compared.  The tolerance is
    relative to the archive's bounding box over the same columns.
    """
    order = np.argsort(archive_ids, kind="stable")
    pos = order[np.searchsorted(archive_ids[order], capture_ids)]
    ncol = max(1, min(int(ndm), 3))
    archive_at = np.asarray(archive_xyz, dtype=np.float64)[pos][:, :ncol]
    capture_at = np.asarray(capture_xyz, dtype=np.float64)[:, :ncol]
    diff = np.abs(archive_at - capture_at)
    max_diff = float(diff.max()) if diff.size else 0.0
    extent = archive_at.max(axis=0) - archive_at.min(axis=0)
    diag = float(np.sqrt(np.sum(extent * extent)))
    return max_diff, _COORD_RTOL * max(diag, 1.0)


def _elements_reason(
    embedded: "FEMData", model_fem: "FEMData", model: "OpenSeesModel",
) -> "Optional[str]":
    """Why the archive's elements cannot drive the capture's element reads.

    ``None`` when they can.  An MPCO keys element results by ops tag.
    A bridge archive (``ops.h5``) carries ``element_meta``, the ops
    tag <-> fem element id pairing, and the reader relabels through it
    (ADR 0043).  A broker-only archive (``fem.to_h5``) carries none, so
    ops tags are read as fem element ids: that is right only when the
    capture's elements ARE the archive's, same id and same nodes.  The
    bridge renumbers elements densely (gmsh numbers the lower-dimension
    elements first), so a broker archive beside a bridge-run capture
    usually disagrees: ``gauss.get(pg="Plate")`` would then answer with
    the wrong elements and no error (#1393).  A capture that holds a
    consistent subset of the archive's elements passes.
    """
    from .readers._tag_translation import ElementTagTranslator

    if not ElementTagTranslator.from_model(model).is_empty:
        return None
    archive: dict[int, tuple[int, ...]] = {}
    for group in model_fem.elements:
        ids = np.asarray(group.ids, dtype=np.int64)
        conn = np.asarray(group.connectivity, dtype=np.int64)
        for eid, row in zip(ids, conn):
            archive[int(eid)] = tuple(sorted(int(n) for n in row))
    n_capture = 0
    missing: list[int] = []
    differing: list[int] = []
    for group in embedded.elements:
        ids = np.asarray(group.ids, dtype=np.int64)
        conn = np.asarray(group.connectivity, dtype=np.int64)
        for eid, row in zip(ids, conn):
            n_capture += 1
            known = archive.get(int(eid))
            if known is None:
                missing.append(int(eid))
            elif known != tuple(sorted(int(n) for n in row)):
                differing.append(int(eid))
    if not missing and not differing:
        return None

    def _some(ids: list[int]) -> str:
        shown = ", ".join(str(i) for i in ids[:5])
        more = f", ... ({len(ids)} in all)" if len(ids) > 5 else ""
        return f"{shown}{more}"

    parts = []
    if missing:
        parts.append(
            f"{len(missing)} absent from its elements (e.g. {_some(missing)})",
        )
    if differing:
        parts.append(
            f"{len(differing)} on other nodes (e.g. {_some(differing)})",
        )
    return (
        "declares no element tag map (/opensees/element_meta: written by "
        "fem.to_h5, not ops.h5), so the capture's element ids must be its "
        f"own, and of the capture's {n_capture} elements {'; '.join(parts)}"
        ". Write the archive with ops.h5(...) so ops tags pair with fem "
        "element ids"
    )


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
       reports **and** the shared ids sit at the same coordinates.
       Node tags are fem node ids on the bridge's emit path, so a
       capture node the archive does not know means the pairing is
       wrong (or the run grew the domain).  Ids alone do not prove it:
       a finer mesh of the same part numbers its nodes ``1..N`` too, so
       an 86-node run beside a 272-node archive passes the id check and
       binds the wrong geometry and physical groups (#1393); the
       coordinates at the capture's ids are compared against the MPCO
       ``MODEL/`` ones, relative to the bounding box.  Either failure
       warns :class:`ModelFemMismatchWarning` and falls through.
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
    if missing.size:
        shown = ", ".join(str(int(n)) for n in missing[:5])
        more = f", ... ({missing.size} in all)" if missing.size > 5 else ""
        reason = (
            f"does not cover the capture's nodes: {missing.size} of "
            f"{capture_ids.size} node ids are absent from its FEMData "
            f"(e.g. {shown}{more})"
        )
    else:
        # ``/meta/ndm`` is 0 on a broker-only archive (``fem.to_h5``,
        # no ``ops.model`` call): the capture's own column count is the
        # model's ndm then (a 2-D recorder stores two columns; the
        # synthesis pads the third with 0, which is not the mesh's z).
        ndm = int(model.ndm) if int(model.ndm) >= 1 else int(reader.spatial_dim())
        max_diff, tol = _coords_mismatch(
            capture_ids, embedded.nodes.coords,
            archive_ids, model_fem.nodes.coords, ndm=ndm,
        )
        if max_diff > tol:
            reason = (
                f"holds a different mesh: its {archive_ids.size} nodes "
                f"cover the capture's {capture_ids.size} ids, but the "
                f"coordinates at those ids differ by up to {max_diff:.6g} "
                f"(tolerance {tol:.3g}, from the bounding box)"
            )
        else:
            element_reason = _elements_reason(embedded, model_fem, model)
            if element_reason is None:
                return model_fem
            reason = element_reason
    warnings.warn(
        ModelFemMismatchWarning(
            f"model_h5={str(model_path)!r} {reason}. Binding the "
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
    stage (``apesees.py``, flat and partitioned emit), so the recorder
    opens one ``MODEL_STAGE[<k>]`` per program stage that ran and the
    pairing is positional.  Fewer capture stages than program stages
    is a partial run (the analysis stopped, or the deck was cut
    short): the program names go onto the prefix and
    :class:`StageCountMismatchWarning` names the stages that have no
    capture.  More capture stages than program stages (hand-written
    stages, a foreign archive) cannot be paired; nothing is renamed
    and the same warning says so.  A vanilla archive (no stages) is
    silent: there is nothing to map.  A program that names two stages
    alike warns :class:`DuplicateStageNameWarning`: ``stage(<name>)``
    then resolves to the first, and the ids stay unique.
    """
    program = [str(s.name) for s in model.stages()]
    if not program:
        return
    capture = reader.stages()
    if len(capture) > len(program):
        warnings.warn(
            StageCountMismatchWarning(
                f"model_h5={str(model_path)!r} declares {len(program)} "
                f"stages ({program}) but the capture holds "
                f"{len(capture)} MODEL_STAGE groups "
                f"({[s.name for s in capture]}); keeping the capture's "
                f"names. Select stages by MODEL_STAGE name or by id "
                f"(results.stage('stage_<k>'))."
            ),
            stacklevel=3,
        )
        return
    if len(capture) < len(program):
        unrun = program[len(capture):]
        warnings.warn(
            StageCountMismatchWarning(
                f"model_h5={str(model_path)!r} declares {len(program)} "
                f"stages ({program}) but the capture holds only "
                f"{len(capture)} MODEL_STAGE groups: a partial run. The "
                f"first {len(capture)} program names are paired onto "
                f"the capture in order; {unrun} have no capture."
            ),
            stacklevel=3,
        )
    paired = program[:len(capture)]
    repeated = sorted(n for n, k in Counter(paired).items() if k > 1)
    if repeated:
        warnings.warn(
            DuplicateStageNameWarning(
                f"model_h5={str(model_path)!r} names more than one stage "
                f"{repeated}; results.stage(<name>) resolves to the first "
                f"of each. Select the others by id "
                f"(results.stage('stage_<k>')) or by their MODEL_STAGE "
                f"alias."
            ),
            stacklevel=3,
        )
    ids = [s.id for s in capture]
    raw = [s.name for s in capture]          # the MODEL_STAGE[<k>] aliases
    shadowed = []
    for i, name in enumerate(paired):
        if name in ids and ids.index(name) != i:
            shadowed.append(f"{name!r} (id {ids[i]!r}) by the id of stage {ids.index(name)}")
        elif name in raw and raw.index(name) != i:
            shadowed.append(f"{name!r} (id {ids[i]!r}) by the alias of stage {raw.index(name)}")
    if shadowed:
        warnings.warn(
            ShadowedStageNameWarning(
                f"model_h5={str(model_path)!r} names a stage like another "
                f"stage's id or MODEL_STAGE alias: {', '.join(shadowed)}. "
                f"results.stage(x) resolves ids first and names before "
                f"aliases, so select each stage by its own id, or rename "
                f"it in the program."
            ),
            stacklevel=3,
        )
    reader.attach_stage_names(paired)


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
