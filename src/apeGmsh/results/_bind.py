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
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from ..opensees.opensees_model import OpenSeesModel
    from .readers._ladruno import StageStamp
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


class UnpairedStageWarning(UserWarning):
    """``model_h5=`` declares stages the ``.ladruno`` cannot be paired with.

    The file carries no ``INFO/PROVENANCE@stage_names`` and holds fewer
    stages than the program, and their recorded ``KIND`` s do not prove
    which program stages they are (two program stages of one kind, or a
    stage with no ``KIND`` evidence).  A recorder bound to a later stage
    writes such a file; pairing it with the program's first stages would
    name it wrongly (#2824).  The message names the stages, which keep
    their ``MODEL_STAGE[<k>]`` names; stamp the file with the stages it
    recorded to name them.
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

#: A program stage's ``analysis`` record → its :attr:`StageInfo.kind`.
#: The keys are the tokens ``apeGmsh.opensees.analysis`` emits;
#: a stage's ``analyze_dt`` does not decide it (a ``Static`` stage may
#: carry ``dt=dlam``, a ``Transient`` one may omit ``dt``).
_ANALYSIS_KIND = {
    "Static": "static",
    "Transient": "transient",
    "VariableTransient": "transient",
}


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
    spatial_dim: "Optional[int]" = None,
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

    An archive that records no ``ndm`` (``fem.to_h5``) is compared over
    the capture's own dimension: ``spatial_dim`` when the caller gives
    it (a ``.ladruno`` reader has no ``spatial_dim()``), else
    ``reader.spatial_dim()`` (the MPCO readers).
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
        if int(model.ndm) >= 1:
            ndm = int(model.ndm)
        elif spatial_dim is not None:
            ndm = int(spatial_dim)
        else:
            ndm = int(reader.spatial_dim())
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
    stamp: "Optional[StageStamp]" = None,
) -> None:
    """Name and kind the capture's stages after the archive's program.

    ``stamp`` is a ``.ladruno``'s :func:`read_stage_stamp`; with it the
    binding is :func:`_bind_ladruno_stage_names`. An ``.mpco`` has none.

    #1324 names, #1948 kinds: a stage's kind comes from its archived
    ``analysis`` record (:func:`_stage_kind`), not from ``analyze_dt``.

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

    This is the ``.mpco`` path: an ``.mpco`` records no stage kind and
    no stamp, so its stages pair by position.  A ``.ladruno`` binds by
    its stamped names or a proven kind match instead
    (:func:`_bind_ladruno_stage_names`, #2823).

    The reader has the last word on the pairing (#2101).  A ``.ladruno``
    stage records the ``KIND`` it ran, and where that disagrees with the
    program kind the reader renames nothing and warns
    ``StageKindMismatchWarning`` (#2079 Q7).  The partial-run,
    duplicate and shadow warnings above describe a rename, so they are
    emitted only once the reader has made it: a declined pairing gives
    the reader's one warning and nothing else.  Whether a rename
    happened is read back from ``reader.stages()``, so the single-file
    and partitioned readers of both formats take the same path.
    """
    if stamp is not None:
        # Raised again from here, so each warning names the caller's line.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _bind_ladruno_stage_names(reader, model, model_path=model_path,
                                      stamp=stamp)
        for w in caught:
            warnings.warn(w.message, w.category, stacklevel=3)
        return
    program = [str(s.name) for s in model.stages()]
    if not program:
        return
    # Mode stages are read out of a MODEL_STAGE's modal groups;
    # they are not program stages and never take a program name.
    capture = [s for s in reader.stages() if s.kind != "mode"]
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
    paired = program[:len(capture)]
    kinds = [
        _stage_kind(s, model_path=model_path)
        for s in model.stages()[:len(capture)]
    ]
    before = _stage_naming(reader)
    reader.attach_stage_names(paired, kinds)
    if paired and _stage_naming(reader) == before:
        # The reader declined the pairing and has said why; the
        # warnings below would describe a rename that never happened.
        # An empty pairing renames nothing by construction, and the
        # count warning below is then the only account of the run.
        return
    if len(capture) < len(program):
        _warn_partial_run(program, len(capture), model_path=model_path)
    _warn_named_alike(capture, paired, model_path=model_path)


def _warn_partial_run(
    program: "list[str]", n_capture: int, *, model_path: "str | Path",
    stacklevel: int = 4,
) -> None:
    unrun = program[n_capture:]
    warnings.warn(
        StageCountMismatchWarning(
            f"model_h5={str(model_path)!r} declares {len(program)} "
            f"stages ({program}) but the capture holds only "
            f"{n_capture} MODEL_STAGE groups: a partial run. The "
            f"first {n_capture} program names are paired onto "
            f"the capture in order; {unrun} have no capture."
        ),
        stacklevel=stacklevel,
    )


def _warn_named_alike(
    capture: "Sequence[Any]", paired: "Sequence[Optional[str]]", *,
    model_path: "str | Path",
) -> None:
    """The duplicate and shadow warnings over the stages that took a name.

    ``paired[i]`` is the program name ``capture[i]`` took, ``None`` where
    it kept its own.
    """
    taken = [n for n in paired if n is not None]
    repeated = sorted(n for n, k in Counter(taken).items() if k > 1)
    if repeated:
        warnings.warn(
            DuplicateStageNameWarning(
                f"model_h5={str(model_path)!r} names more than one stage "
                f"{repeated}; results.stage(<name>) resolves to the first "
                f"of each. Select the others by id "
                f"(results.stage('stage_<k>')) or by their MODEL_STAGE "
                f"alias."
            ),
            stacklevel=4,
        )
    ids = [s.id for s in capture]
    raw = [s.name for s in capture]          # the MODEL_STAGE[<k>] aliases
    shadowed = []
    for i, name in enumerate(paired):
        if name is None:
            continue
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
            stacklevel=4,
        )


#: The raw ``KIND`` s that say which kind of program stage ran.
_EVIDENCE = ("static", "transient")
#: The raw ``KIND`` s a stage may record without contradicting a program kind.
_NO_EVIDENCE = (None, "eigen")


def _bind_ladruno_stage_names(
    reader: "Any",
    model: "OpenSeesModel",
    *,
    model_path: "str | Path",
    stamp: "StageStamp",
) -> None:
    """Name and kind a ``.ladruno``'s stages after the archive's program.

    The ``.ladruno`` sibling of :func:`_bind_stage_names` (ADR 0122 D3,
    #2823); ``stamp`` is the file's :func:`read_stage_stamp`. A stage is
    never paired with a program stage by position alone (#2824):

    * **Stamped** (``INFO/PROVENANCE@stage_names``): each stage takes the
      program stage its stamp names, in any order. A stamped name equal to
      the stage's own ``MODEL_STAGE[<k>]`` group (a stage that is no program
      stage: modes) keeps the stage as it is. A name the program does not
      declare, or a program stage of another kind than the ``KIND`` the
      file recorded, raises :class:`StageStampRefused` naming it. No count
      warning: the stamp says which stages ran.
    * **Unstamped**: a modes-only stage (``KIND="eigen"``, no step) is not a
      program stage and keeps its name. The other stages pair with the
      program by kind: as many stages as the program pairs one to one, and
      a conflicting ``KIND`` renames nothing (the reader's
      ``StageKindMismatchWarning``); fewer pair with the program's first
      stages only when no other run of program stages fits their ``KIND`` s
      (a partial run, :class:`StageCountMismatchWarning`); otherwise
      nothing is renamed and :class:`UnpairedStageWarning` names the
      stages. More stages than the program warn the count and rename
      nothing.

    Duplicate and shadowed names warn as in :func:`_bind_stage_names`.
    """
    from .readers._ladruno import StageStampRefused

    program = list(model.stages())
    names = [str(s.name) for s in program]
    if not names:
        return
    hosts = [s for s in reader.stages() if s.kind != "mode"]
    if len(hosts) != len(stamp.stages):
        raise RuntimeError(
            f"{stamp.source}: the reader lists {len(hosts)} MODEL_STAGE "
            f"groups and the file {len(stamp.stages)}.")
    odd = [f"{rec.group} records KIND {rec.kind!r}" for rec in stamp.stages
           if rec.kind not in _EVIDENCE and rec.kind not in _NO_EVIDENCE]
    kinds: "dict[int, str]" = {}

    def kind_of(j: int) -> str:
        if j not in kinds:
            kinds[j] = _stage_kind(program[j], model_path=model_path)
        return kinds[j]

    if stamp.names is not None:
        if odd:
            raise StageStampRefused(
                f"{stamp.source}: {'; '.join(odd)}, which is no stage kind; "
                f"its stamped stage_names {list(stamp.names)} cannot be "
                f"checked against model_h5={str(model_path)!r}.")
        pairing = _pair_by_stamp(stamp, names, kind_of, model_path=model_path)
    else:
        by_kind = _pair_by_kind(stamp, names, kind_of, model_path=model_path)
        if by_kind is None:
            return
        pairing = by_kind
    paired = [None if j is None else names[j] for j in pairing]
    if all(n is None for n in paired):
        return
    before = _stage_naming(reader)
    reader.attach_stage_names(
        [h.name if j is None else names[j] for h, j in zip(hosts, pairing)],
        [h.kind if j is None else kind_of(j) for h, j in zip(hosts, pairing)],
    )
    if _stage_naming(reader) == before:
        return  # the reader declined the pairing and has said why
    if stamp.names is None:
        n_paired = sum(j is not None for j in pairing)
        if n_paired < len(names):
            _warn_partial_run(names, n_paired, model_path=model_path)
    _warn_named_alike(hosts, paired, model_path=model_path)


def _pair_by_stamp(
    stamp: "StageStamp", names: "list[str]", kind_of: "Any", *,
    model_path: "str | Path",
) -> "list[Optional[int]]":
    """The program stage each stamped stage names (``None``: not one)."""
    from .readers._ladruno import StageStampRefused

    assert stamp.names is not None
    out: "list[Optional[int]]" = []
    used: "set[int]" = set()
    for rec, name in zip(stamp.stages, stamp.names):
        candidates = [j for j, n in enumerate(names) if n == name]
        if not candidates:
            if name == rec.group:
                out.append(None)    # the stamp keeps it: not a program stage
                continue
            raise StageStampRefused(
                f"{stamp.source}: {rec.group} is stamped {name!r}, which "
                f"model_h5={str(model_path)!r} does not declare (its stages "
                f"are {names}). The file was stamped against another "
                f"program; nothing is renamed.")
        if rec.kind in _EVIDENCE:
            fits = [j for j in candidates if kind_of(j) == rec.kind]
            if not fits:
                raise StageStampRefused(
                    f"{stamp.source}: {rec.group} is stamped {name!r} and "
                    f"records KIND {rec.kind!r}, but model_h5="
                    f"{str(model_path)!r} declares {name!r} "
                    f"{sorted({kind_of(j) for j in candidates})}. The stamp "
                    f"contradicts the program; nothing is renamed.")
        else:
            fits = candidates
            if len({kind_of(j) for j in fits}) > 1:
                raise StageStampRefused(
                    f"{stamp.source}: {rec.group} is stamped {name!r} and "
                    f"records no stage kind (KIND {rec.kind!r}), and "
                    f"model_h5={str(model_path)!r} declares {name!r} more than "
                    f"once with kinds {sorted({kind_of(j) for j in fits})}; "
                    f"which one it is cannot be told. Nothing is renamed.")
        j = next((j for j in fits if j not in used), fits[0])
        used.add(j)
        out.append(j)
    return out


def _pair_by_kind(
    stamp: "StageStamp", names: "list[str]", kind_of: "Any", *,
    model_path: "str | Path",
) -> "Optional[list[Optional[int]]]":
    """The program stage each unstamped stage pairs with, by kind.

    ``None`` when the caller has nothing to rename and has warned (or
    needs no warning).
    """
    stages = stamp.stages
    steps = [i for i, rec in enumerate(stages) if not rec.modes_only]
    m, n = len(steps), len(names)
    out: "list[Optional[int]]" = [None] * len(stages)
    if not stages:
        _warn_partial_run(names, 0, model_path=model_path, stacklevel=5)
        return None
    if m == 0:
        return None     # modes only: no program stage ran here
    if m > n:
        warnings.warn(
            StageCountMismatchWarning(
                f"model_h5={str(model_path)!r} declares {n} stages "
                f"({names}) but the capture holds {m} MODEL_STAGE groups "
                f"with steps ({[stages[i].group for i in steps]}); keeping "
                f"the capture's names. Select stages by MODEL_STAGE name or "
                f"by id (results.stage('stage_<k>'))."
            ),
            stacklevel=4,
        )
        return None
    evidence = [stages[i].kind if stages[i].kind in _EVIDENCE else None
                for i in steps]
    if m < n:
        program_kinds: "list[Optional[str]]" = []
        for j in range(n):
            try:
                program_kinds.append(kind_of(j))
            except ValueError:
                if j < m:
                    raise
                program_kinds.append(None)   # a wildcard: proves nothing
        prefix_fits = all(e is None or e == program_kinds[j]
                          for j, e in enumerate(evidence))
        if prefix_fits and _embeddings(evidence, program_kinds) > 1:
            groups = [stages[i].group for i in steps]
            warnings.warn(
                UnpairedStageWarning(
                    f"model_h5={str(model_path)!r} declares {n} stages "
                    f"({names}, kinds {program_kinds}) but {stamp.source} "
                    f"holds {m} ({groups}, KIND {[stages[i].kind for i in steps]}) "
                    f"and no INFO/PROVENANCE@stage_names, so which program "
                    f"stages they recorded cannot be told (a recorder bound "
                    f"to a later stage writes such a file). {groups} keep "
                    f"their names; stamp the file with the stages it "
                    f"recorded to name them."
                ),
                stacklevel=4,
            )
            return None
    for j, i in enumerate(steps):
        out[i] = j
    return out


def _embeddings(evidence: "list[Optional[str]]", kinds: "list[Optional[str]]") -> int:
    """How many ways (0, 1 or 2, meaning "more") ``evidence`` runs in order
    over ``kinds``; ``None`` on either side fits anything."""
    m, n = len(evidence), len(kinds)
    # ways[i][j]: embeddings of evidence[i:] into kinds[j:], capped at 2.
    ways = [[0] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        ways[m][j] = 1
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            fit = evidence[i] is None or kinds[j] is None or evidence[i] == kinds[j]
            ways[i][j] = min(2, ways[i][j + 1] + (ways[i + 1][j + 1] if fit else 0))
    return ways[0][0]


def _stage_naming(reader: "Any") -> "list[tuple[str, str, str, tuple[str, ...]]]":
    """Each stage's ``(id, name, kind, aliases)``: what a rename changes."""
    return [(s.id, s.name, s.kind, tuple(s.aliases)) for s in reader.stages()]


def _stage_kind(stage: "Any", *, model_path: "str | Path") -> str:
    """The kind of one program stage, read from its ``analysis`` record.

    Every bridge stage archives its analysis chain
    (``/opensees/stages/stage_NNN/analysis@analysis``); a stage without
    one, or with a token this module does not know, raises rather than
    guess from ``analyze_dt`` or the time vector.
    """
    chain = stage.chain_attrs
    if "analysis" not in chain:
        raise ValueError(
            f"model_h5={str(model_path)!r}: stage {stage.name!r} archives "
            f"no analysis record, so its kind (static or transient) is "
            f"unknown."
        )
    token = str(chain["analysis"])
    if token not in _ANALYSIS_KIND:
        raise ValueError(
            f"model_h5={str(model_path)!r}: stage {stage.name!r} has "
            f"analysis {token!r}, which has no stage kind; known: "
            f"{sorted(_ANALYSIS_KIND)}."
        )
    return _ANALYSIS_KIND[token]


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
