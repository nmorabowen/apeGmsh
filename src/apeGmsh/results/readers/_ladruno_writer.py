"""Write a ``.ladruno`` from Python (ADR 0122 D6; #2079 ruling 1).

The native ``results.h5`` retires, so the sources with no recorder write
the fork's own format instead: :func:`write_modes` for the two Python
mode writers, :func:`write_nodal_history` for the demo (ADR 0131 D10).

Modes
-----

The two Python mode writers (``EigenResult`` and
``ParallelModalResult``) share :func:`write_modes`: one ``eigen``
``MODEL_STAGE`` holding every mode, in the layout the fork recorder
writes for ``-N modesOfVibration modesOfVibrationRotational`` (schema v1
at ``FORK_PIN``, ``Ladruno_NodeResults.cpp``), stamped with
:func:`~apeGmsh.results.readers._ladruno_provenance.stamp_provenance`
against the bridge's model archive.

==================================  ========================================
``.ladruno``                        written from
==================================  ========================================
``INFO``                            ``GENERATOR="Ladruno"``,
                                    ``FORMAT_VERSION=1``,
                                    ``FORMAT_MINOR_VERSION=1`` (one stage,
                                    ``KIND`` from the modal request, as §2a
                                    has it), ``SPATIAL_DIM``,
                                    ``STORED_PRECISION="f64"``, one
                                    partition, ``PROVENANCE``. No
                                    ``SOLVER_NAME`` / ``SOLVER_VERSION``:
                                    the modes may come from a deck run by
                                    another binary, which nothing records
``MODEL_STAGE[1]``                  ``KIND="eigen"``, ``STEP=0``,
                                    ``TIME=0.0``
``MODEL/NODES``                     the bridge's fem: ``ID``, and the first
                                    ``ndm`` coordinates
``MODEL/ELEMENTS``                  an empty group (below)
``MODEL/SECTION_ASSIGNMENTS``       an empty group
``ON_NODES/MODES_OF_VIBRATION(U)``  ``Ux[,Uy[,Uz]]`` (``ndm`` columns)
``ON_NODES/MODES_OF_VIBRATION(R)``  ``Rz`` (``ndm=2``) or ``Rx,Ry,Rz``
                                    (``ndm=3``; ``ndm=1``: zeros), 0.0 at a
                                    node the caller's ``rotations`` mask
                                    clears
``DATA/STEP_0/MODE_<k>``            ``[n x nComp]``, ``k`` 0-based, with
                                    ``MODE``, ``LAMBDA``, ``OMEGA``,
                                    ``FREQUENCY``, ``PERIOD``
==================================  ========================================

**No element buckets.** A ``MODEL/ELEMENTS/<classTag>-<ClassName>[<g>]``
bucket needs the C++ class tag, which apeGmsh does not hold as fact for
every element (the G3 note on #1950), so none is written and the group
stays empty; the fork's ``validate()`` accepts that. ``PROVENANCE@
unsupported`` names ``elements``: read the file with ``model_h5=`` to
bind the element records.

Both modal groups are always written, as the recorder writes them for a
request of both tokens: the rotational group of a model without rotations
holds zeros. ``DIMENSION`` is absent, as the recorder skips an empty
string attribute.

A non-positive eigenvalue warns once for its mode and writes ``OMEGA =
FREQUENCY = PERIOD = 0`` (the native writer's rule, and what the reader
reports for it); the fork writes ``sqrt`` of it, a NaN.

Nodal history
-------------

:func:`write_nodal_history` writes one ``static`` or ``transient``
``MODEL_STAGE[1]`` of nodal results, in the layout the fork recorder
writes for ``-N displacement`` (schema v1 at ``FORK_PIN``): ``INFO`` and
``MODEL`` as above (the stage ``KIND`` the caller's; ``STEP`` / ``TIME``
the first record's), and ``ON_NODES/DISPLACEMENT`` with
``COMPONENTS="Ux[,Uy[,Uz]]"``, ``ID[n x 1]``, ``DATA[T x n x ndm]``,
``STEP[T]`` and ``TIME[T]``. Only the recorder groups in
:data:`NODAL_GROUPS` are written; any other name is refused. It writes no
element buckets either, for the same reason, and stamps ``elements``
unsupported.

:func:`write_nodal_stages` writes several stages, ``MODEL_STAGE[1..k]``,
each holding its own nodes, ``KIND`` and time, from canonical node
components (``displacement_x``, ``reaction_force_z``, ``pore_pressure``):
the ``.out`` recorder transcode writes it. :data:`NODE_COMPONENTS` maps
each canonical name onto the recorder's group and label
(``Ladruno_NodeResults.cpp`` at ``FORK_PIN``). A group's ``COMPONENTS``
lists only the components recorded, in the recorder's order, so a read
of one never recorded finds nothing, as it did in the native file; the
recorder itself writes all ``ndm`` of them. A name the table does not
hold is refused, never dropped.

Both
----

The file is written and stamped in a staging directory under its final
name, then moved into place: either a stamped file is at ``path`` or
nothing new is. ``path`` may not be the model archive, nor any ``.h5``.

Every writer here stamps ``INFO/PROVENANCE@stage_names`` itself (ADR 0122
D3, #2823), because a reader never pairs unnamed stages with the program by
position: :func:`write_modes` names its ``eigen`` stage by its own
``MODEL_STAGE[1]`` (modes are no program stage), and
:func:`write_nodal_stages` takes the program stage each of its stages
recorded (``recorded_stages=``).

``opensees_backend`` defaults to ``"unknown"``: a file Python writes is
never stamped with the engine of the fork's recorder unless its caller
proves that engine solved the values (:func:`solved_backend`).
"""
from __future__ import annotations

import math
import os
import shutil
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..._atomic_io import replace_with_retry
from ._ladruno_provenance import stamp_provenance

if TYPE_CHECKING:
    import h5py

    from ...mesh.FEMData import FEMData
    from ...opensees.opensees_model import OpenSeesModel
    from ..Results import Results
    from ..spec._resolved import ResolvedRecorderSpec

__all__ = [
    "NODE_COMPONENTS", "NodalStage", "check_target", "is_transcode_target",
    "merge_node_records", "nodal_history_misfits", "open_transcode",
    "solved_backend", "stamp_transcode", "transcode_target", "write_modes",
    "write_nodal_history", "write_nodal_stages", "write_transcode",
]

#: ``PROVENANCE@opensees_backend`` when no engine is proven (§7b.1).
UNKNOWN_BACKEND = "unknown"
_FORK_BACKEND = "ladruno"
_U = "MODES_OF_VIBRATION(U)"
_R = "MODES_OF_VIBRATION(R)"
_SCALAR, _VECTORIAL = 0, 1  # the recorder's ResultDataType
_MODAL = 1                  # the recorder's ResultType::Modal
_INT32_MAX = 2**31 - 1
#: What this writer leaves out of the file (``PROVENANCE@unsupported``).
UNSUPPORTED: tuple[str, ...] = ("elements",)
#: ``PROVENANCE@writer``'s source for a recorder transcode.
_TRANSCODE_SOURCE = "recorders"


class MPCOConversionRefused(RuntimeError):
    """:func:`convert_mpco` refused before any target file existed.

    Raised for an existing target (unless ``overwrite=True``), a missing
    or unreadable ``model_h5``, MPCO stages or kinds that do not match the
    program's, a partition set that is incomplete or inconsistent, and a
    source that is unreadable or holds something this converter does not
    know how to carry. The source is never modified.
    """


def write_modes(
    path: "str | os.PathLike[str]",
    *,
    model_h5: "str | os.PathLike[str]",
    node_ids: "np.ndarray | Sequence[int]",
    coords: np.ndarray,
    ndm: int,
    eigenvalues: "np.ndarray | Sequence[float]",
    fields: Callable[[int], np.ndarray],
    rotations: "np.ndarray | Sequence[bool]",
    record_ids: "np.ndarray | Sequence[int] | None" = None,
    source: str,
    opensees_backend: str = UNKNOWN_BACKEND,
    stacklevel: int = 2,
) -> Path:
    """Write the modes as one ``eigen`` stage of a ``.ladruno``; stamp it.

    Parameters
    ----------
    path
        The ``.ladruno`` to write (replaced when it exists).
    model_h5
        The bridge's model archive; ``INFO/PROVENANCE`` pairs with it.
    node_ids, coords
        ``MODEL/NODES``: the fem's node ids and their coordinates
        (``[n x >=ndm]``; the first ``ndm`` columns are written).
    ndm
        The model's spatial dimension.
    eigenvalues
        ``λ_k``, mode 1 first.
    fields
        ``fields(k)`` is mode ``k``'s (1-based) eigenvector field by
        spatial slot, ``[len(record_ids) x 6]`` (``ux, uy, uz, rx, ry,
        rz``), rows in ``record_ids`` order; a slot a node does not carry
        is 0.0. The caller maps each node's DOFs onto the slots.
    rotations
        Per recorded node, whether its rotation slots hold rotations.
        ``False`` writes 0.0 there, so a u-p node's pressure is never
        written as one.
    record_ids
        The nodes the modal groups record; ``None`` records every node.
        Each must be a node of ``node_ids``.
    source
        What the modes came from, for ``PROVENANCE@writer``
        (``"apeGmsh <source>->ladruno <version>"``): ``eigen``, ``modal_deck``.
    opensees_backend
        ``PROVENANCE@opensees_backend``: ``"unknown"`` unless the caller
        proves which engine solved the modes (:func:`solved_backend`).
    stacklevel
        Of every warning this raises, the stamp's
        ``StageNamesOmittedWarning`` included, counted from this function.

    The stage is stamped ``stage_names=["MODEL_STAGE[1]"]``, its own name:
    the modes are no program stage, so a reader with a staged ``model_h5``
    never names the stage after one (#2824: it was labelled with the
    program's first stage). A program with no stages stamps none.

    Raises
    ------
    ValueError
        No modes, an ``ndm`` outside 1-3, a node tag
        outside int32, a recorded node that is not in ``node_ids``, or a
        field of the wrong shape, or a ``path`` refused by
        :func:`check_target`. Nothing is written.
    """
    out = Path(os.fspath(path))
    check_target(out, model_h5)
    lams = [float(v) for v in np.asarray(eigenvalues, dtype=np.float64).ravel()]
    if not lams:
        raise ValueError(
            "write_modes: no eigenvalues, so there is no mode to write; no "
            "file was created.")
    if ndm not in (1, 2, 3):
        raise ValueError(f"write_modes: ndm must be 1, 2 or 3, got {ndm}.")
    nids = _int32_ids(node_ids, "node_ids")
    xyz = np.asarray(coords, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[0] != nids.size or xyz.shape[1] < ndm:
        raise ValueError(
            f"write_modes: coords has shape {xyz.shape}; expected "
            f"[{nids.size} x >={ndm}].")
    rids = nids if record_ids is None else _int32_ids(record_ids, "record_ids")
    stray = np.setdiff1d(rids, nids)
    if stray.size:
        raise ValueError(
            f"write_modes: recorded node(s) {stray[:5].tolist()} are not "
            "nodes of the model archive's fem; the mode shapes and the model "
            "come from different meshes.")
    shapes = []
    for k in range(1, len(lams) + 1):
        f = np.asarray(fields(k), dtype=np.float64)
        if f.shape != (rids.size, 6):
            raise ValueError(
                f"write_modes: mode {k} has a field of shape {f.shape}; "
                f"expected [{rids.size} x 6].")
        shapes.append(f)
    rot = np.asarray(rotations, dtype=bool).ravel()
    if rot.shape != (rids.size,):
        raise ValueError(
            f"write_modes: rotations has {rot.size} entries; expected one "
            f"per recorded node ({rids.size}).")
    # A plain loop, not a comprehension: before Python 3.12 (PEP 709) a
    # comprehension is a frame of its own, which would shift the warning's
    # stacklevel by one and point it into the library.
    scalars: list[tuple[float, float, float]] = []
    for k, lam in enumerate(lams, start=1):
        scalars.append(_mode_scalars(lam, k, stacklevel + 1))

    _place_stamped(
        out, model_h5,
        lambda staged: _write_file(staged, nids, xyz[:, :ndm], rids, ndm,
                                   lams, scalars, shapes, rot),
        source=source, opensees_backend=opensees_backend,
        recorded_stages=[None], stacklevel=stacklevel + 1,
    )
    return out


#: The ``ON_NODES`` groups :func:`write_nodal_history` writes, as the fork
#: recorder writes them (``Ladruno_NodeResults.cpp`` at ``FORK_PIN``):
#: ``(component prefix, DESCRIPTION, DIMENSION, DISPLAY_NAME)``. The
#: components are ``<prefix>x[,<prefix>y[,<prefix>z]]``, one per ``ndm``.
NODAL_GROUPS: dict[str, tuple[str, str, str, str]] = {
    "DISPLACEMENT": ("U", "Nodal displacement field", "L", "Displacement"),
}
_NODAL = 0  # the recorder's ResultType::Nodal

#: Each ``ON_NODES`` group :func:`write_nodal_stages` writes, as the fork
#: recorder writes it (``Ladruno_NodeResults.cpp`` at ``FORK_PIN``):
#: ``(canonical prefix, label prefix, DESCRIPTION, DIMENSION,
#: DISPLAY_NAME)``. A vector group's labels are ``<label prefix><axis>``;
#: ``PRESSURE`` holds the one scalar ``p``.
_NODE_GROUPS: dict[str, tuple[str, str, str, str, str]] = {
    "DISPLACEMENT": ("displacement", "U", "Nodal displacement field", "L",
                     "Displacement"),
    "ROTATION": ("rotation", "R", "Nodal rotation field", "A", "Rotation"),
    "VELOCITY": ("velocity", "V", "Nodal velocity field", "L/t", "Velocity"),
    "ANGULAR_VELOCITY": ("angular_velocity", "RV",
                         "Nodal angular velocity field", "L/t*A",
                         "Angular Velocity"),
    "ACCELERATION": ("acceleration", "A", "Nodal acceleration field",
                     "L/t^2", "Acceleration"),
    "ANGULAR_ACCELERATION": ("angular_acceleration", "RA",
                             "Nodal angular acceleration field", "L/t^2*A",
                             "Angular Acceleration"),
    "REACTION_FORCE": ("reaction_force", "RF", "Nodal reaction force field",
                       "F", "Reaction Force"),
    "REACTION_MOMENT": ("reaction_moment", "RM",
                        "Nodal reaction moment field", "F*L",
                        "Reaction Moment"),
    "UNBALANCED_FORCE": ("force", "F", "Nodal unbalanced force field", "F",
                         "Unbalanced Force"),
    "UNBALANCED_MOMENT": ("moment", "M", "Nodal unbalanced moment field",
                          "F*L", "Unbalanced Moment"),
    "PRESSURE": ("pore_pressure", "p", "Nodal pressure field", "F/L^2",
                 "Pressure"),
}
_SCALAR_GROUP = "PRESSURE"

#: Canonical node component -> ``(ON_NODES group, COMPONENTS label)``, in
#: the recorder's column order. The reader translates each label back
#: (``_buckets.canonical_node_component``); a name this table does not
#: hold (``displacement_increment``: the recorder has no group for it) is
#: refused by :func:`write_nodal_stages`.
NODE_COMPONENTS: dict[str, tuple[str, str]] = {
    (prefix if group == _SCALAR_GROUP else f"{prefix}_{axis}"):
        (group, label if group == _SCALAR_GROUP else f"{label}{axis}")
    for group, (prefix, label, *_) in _NODE_GROUPS.items()
    for axis in ("x" if group == _SCALAR_GROUP else "xyz")
}


@dataclass(frozen=True)
class NodalStage:
    """One stage :func:`write_nodal_stages` writes.

    ``components`` maps a name of :data:`NODE_COMPONENTS` to its values
    ``[T x len(node_ids)]`` (NaN where a node does not record it); ``time``
    is ``TIME[T]``. A stage with no component has no step either.
    """

    kind: str
    time: np.ndarray
    node_ids: np.ndarray
    components: Mapping[str, np.ndarray]


def write_nodal_history(
    path: "str | os.PathLike[str]",
    *,
    model_h5: "str | os.PathLike[str]",
    node_ids: "np.ndarray | Sequence[int]",
    coords: np.ndarray,
    ndm: int,
    kind: str,
    time: "np.ndarray | Sequence[float]",
    results: "Mapping[str, np.ndarray]",
    source: str,
    opensees_backend: str = UNKNOWN_BACKEND,
    stacklevel: int = 2,
) -> Path:
    """Write nodal results as one stage of a ``.ladruno``; stamp it.

    Parameters
    ----------
    path, model_h5, node_ids, coords, ndm, source, stacklevel
        As :func:`write_modes`. Every node of ``node_ids`` is recorded.
    kind
        The stage ``KIND``: ``"static"`` or ``"transient"``.
    time
        ``TIME[T]``, one value per recorded step; ``STEP`` is ``0..T-1``.
    results
        ``{group: DATA[T x n x ndm]}``, ``group`` a key of
        :data:`NODAL_GROUPS`, rows in ``node_ids`` order.
    opensees_backend
        ``PROVENANCE@opensees_backend``, as
        :func:`~apeGmsh.results.readers._ladruno_provenance.stamp_provenance`
        takes it; ``"unknown"`` unless the caller proves the engine.

    Raises
    ------
    ValueError
        A ``kind`` other than ``static`` / ``transient``, no step, no
        result, a group not in :data:`NODAL_GROUPS`, a ``DATA`` of the
        wrong shape, or what :func:`write_modes` raises for the shared
        parameters. Nothing is written.
    """
    out = Path(os.fspath(path))
    check_target(out, model_h5)
    if kind not in ("static", "transient"):
        raise ValueError(
            f"write_nodal_history: kind must be 'static' or 'transient', "
            f"got {kind!r}.")
    if ndm not in (1, 2, 3):
        raise ValueError(
            f"write_nodal_history: ndm must be 1, 2 or 3, got {ndm}.")
    nids = _int32_ids(node_ids, "node_ids")
    xyz = np.asarray(coords, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[0] != nids.size or xyz.shape[1] < ndm:
        raise ValueError(
            f"write_nodal_history: coords has shape {xyz.shape}; expected "
            f"[{nids.size} x >={ndm}].")
    times = np.asarray(time, dtype=np.float64).ravel()
    if not times.size:
        raise ValueError(
            "write_nodal_history: no time value, so there is no step to "
            "write; no file was created.")
    if not results:
        raise ValueError(
            "write_nodal_history: no result to write; no file was created.")
    components: dict[str, np.ndarray] = {}
    for name, data in results.items():
        if name not in NODAL_GROUPS:
            raise ValueError(
                f"write_nodal_history: {name!r} is not a nodal group this "
                f"writer knows the recorder's layout of "
                f"({sorted(NODAL_GROUPS)}); nothing was written.")
        block = np.asarray(data, dtype=np.float64)
        want = (times.size, nids.size, ndm)
        if block.shape != want:
            raise ValueError(
                f"write_nodal_history: {name} has shape {block.shape}; "
                f"expected {list(want)} (steps x nodes x ndm).")
        prefix = _NODE_GROUPS[name][0]
        for j, axis in enumerate("xyz"[:ndm]):
            components[f"{prefix}_{axis}"] = block[:, :, j]

    stage = NodalStage(kind, times, nids, components)
    _place_stamped(
        out, model_h5,
        lambda staged: _write_nodal_file(staged, nids, xyz[:, :ndm], ndm,
                                         [stage]),
        source=source, opensees_backend=opensees_backend,
        stacklevel=stacklevel + 1,
    )
    return out


def write_nodal_stages(
    path: "str | os.PathLike[str]",
    *,
    model_h5: "str | os.PathLike[str]",
    node_ids: "np.ndarray | Sequence[int]",
    coords: np.ndarray,
    ndm: int,
    stages: "Sequence[NodalStage]",
    source: str,
    opensees_backend: str = UNKNOWN_BACKEND,
    stage_names: "Optional[Sequence[str]]" = None,
    recorded_stages: "Optional[Sequence[Optional[str]]]" = None,
    stacklevel: int = 2,
) -> Path:
    """Write ``stages`` as ``MODEL_STAGE[1..k]`` of a ``.ladruno``; stamp it.

    Parameters
    ----------
    path, model_h5, node_ids, coords, ndm, source, stacklevel
        As :func:`write_modes`: ``node_ids`` and ``coords`` are the
        ``MODEL/NODES`` of every stage.
    stages
        In order. Each stage's nodes must be nodes of ``node_ids``, and its
        components names of :data:`NODE_COMPONENTS`.
    opensees_backend
        As :func:`write_nodal_history`.
    stage_names
        Passed to
        :func:`~apeGmsh.results.readers._ladruno_provenance.stamp_provenance`:
        ``None`` pairs the program's stage names with the file's stages;
        ``()`` stamps none, for stages that are not the program's.
    recorded_stages
        Passed to ``stamp_provenance`` (it excludes ``stage_names``): per
        stage, the program stage it recorded (in any order), or ``None``
        for a stage that is no program stage, stamped with its own
        ``MODEL_STAGE[<k>]`` name. A reader binds the stages by these names.

    Raises
    ------
    ValueError
        No stage, a ``kind`` other than ``static`` / ``transient``, a
        component the table does not hold, a node outside ``node_ids``, a
        block of the wrong shape, a stage with steps and no component, or
        what :func:`write_modes` raises for the shared parameters. Nothing
        is written.
    """
    where = "write_nodal_stages"
    out = Path(os.fspath(path))
    check_target(out, model_h5)
    if ndm not in (1, 2, 3):
        raise ValueError(f"{where}: ndm must be 1, 2 or 3, got {ndm}.")
    nids = _int32_ids(node_ids, "node_ids")
    xyz = np.asarray(coords, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[0] != nids.size or xyz.shape[1] < ndm:
        raise ValueError(
            f"{where}: coords has shape {xyz.shape}; expected "
            f"[{nids.size} x >={ndm}].")
    if not stages:
        raise ValueError(f"{where}: no stage to write; no file was created.")
    checked: list[NodalStage] = []
    for k, st in enumerate(stages, start=1):
        if st.kind not in ("static", "transient"):
            raise ValueError(
                f"{where}: stage {k} kind must be one of ['static', "
                f"'transient'] (got {st.kind!r}); nothing was written.")
        times = np.asarray(st.time, dtype=np.float64).ravel()
        rids = _int32_ids(st.node_ids, "node_ids")
        stray = np.setdiff1d(rids, nids)
        if stray.size:
            raise ValueError(
                f"{where}: stage {k} records node(s) {stray[:5].tolist()} "
                "that are not in node_ids; nothing was written.")
        unknown = sorted(set(st.components) - set(NODE_COMPONENTS))
        if unknown:
            raise ValueError(
                f"{where}: stage {k} holds {unknown}, which the recorder "
                "has no ON_NODES group for; nothing was written.")
        if times.size and not st.components:
            raise ValueError(
                f"{where}: stage {k} has {times.size} step(s) and no "
                "component, so its time cannot be stored; nothing was "
                "written.")
        comps: dict[str, np.ndarray] = {}
        for name, data in st.components.items():
            block = np.asarray(data, dtype=np.float64)
            if block.shape != (times.size, rids.size):
                raise ValueError(
                    f"{where}: stage {k} {name} has shape {block.shape}; "
                    f"expected [{times.size} x {rids.size}] (steps x "
                    "nodes); nothing was written.")
            comps[name] = block
        checked.append(NodalStage(st.kind, times, rids, comps))

    _place_stamped(
        out, model_h5,
        lambda staged: _write_nodal_file(staged, nids, xyz[:, :ndm], ndm,
                                         checked),
        source=source, opensees_backend=opensees_backend,
        stage_names=stage_names, recorded_stages=recorded_stages,
        stacklevel=stacklevel + 1,
    )
    return out


def solved_backend(ops: Any) -> str:
    """``PROVENANCE@opensees_backend`` for values ``ops`` solved in process.

    The fork's engine when ``ops`` is the fork (it answers
    ``ladrunoBuild``), else ``"unknown"``: §7b.1 allows ``"stock"`` only
    on a converted file.
    """
    from ...opensees._target import backend_info_of

    return (_FORK_BACKEND if backend_info_of(ops).kind == "fork"
            else UNKNOWN_BACKEND)


def nodal_history_misfits(spec: "ResolvedRecorderSpec") -> list[str]:
    """``category:name`` of each record of ``spec`` a recorder transcode in
    this format cannot hold: an element record (Gauss points, line
    stations, nodal forces, fibres), or a nodes record with a component
    :data:`NODE_COMPONENTS` has no group for (``displacement_increment``).
    Modal and layers records are written by no transcode. Empty when the
    format holds everything ``spec`` records."""
    from ..spec._emit import _DEFERRED_CATEGORIES

    return sorted(
        f"{rec.category}:{rec.name}" for rec in spec.records
        if rec.category not in _DEFERRED_CATEGORIES and (
            rec.category != "nodes"
            or not set(rec.components) <= NODE_COMPONENTS.keys()))


def transcode_target(
    spec: "ResolvedRecorderSpec", native: Path,
    plan: "Sequence[tuple[str, str, Any]]", model: "OpenSeesModel",
    model_h5: "str | os.PathLike[str]",
) -> Path:
    """Where a recorder transcode of ``spec`` is cached (ADR 0131 D6).

    ``native`` with this format's suffix when the format holds every record
    (:func:`nodal_history_misfits` is empty) and every stage of ``plan``,
    ``(name, kind)``, is a stage of ``model``'s program with that kind, in
    any order and any subset (#2823): the file is stamped with the stage
    names (:func:`open_transcode`) and a reader binds them by name. A plan
    naming a stage the program does not declare, or of another kind, or a
    program stage whose kind is not on record, keeps ``native``, a native
    file that stores the names: a ``.ladruno`` binds only the program's.
    ``model_h5`` names the archive in a refusal.
    """
    from .._bind import _stage_kind
    from ._ladruno_live import FORK_SUFFIX

    if nodal_history_misfits(spec) or not plan:
        return native
    declared: "dict[str, set[str]]" = {}
    try:
        for s in model.stages():
            declared.setdefault(str(s.name), set()).add(
                _stage_kind(s, model_path=Path(os.fspath(model_h5))))
    except ValueError:
        return native   # a stage whose kind is not on record pairs nothing
    if any(str(k) not in declared.get(str(n), set()) for n, k, _ in plan):
        return native
    return native.with_suffix(FORK_SUFFIX)


def is_transcode_target(
    path: Path, spec: "ResolvedRecorderSpec", archive: "Optional[Path]",
) -> bool:
    """Whether ``RecorderTranscoder`` writes ``path`` in this format (its
    suffix says so). Such a target refuses, before any file is read, a
    spec it cannot hold (naming the records) and a missing archive to
    stamp against."""
    from ._ladruno_live import FORK_SUFFIX

    if path.suffix != FORK_SUFFIX:
        return False
    misfits = nodal_history_misfits(spec)
    if misfits:
        raise ValueError(
            f"RecorderTranscoder: {path} holds nodal records only, and "
            f"{misfits} are not ones it can hold (element records, or a "
            "node component it has no group for); transcode to a native "
            ".h5 target instead. Nothing was written.")
    if archive is None:
        raise ValueError(
            f"RecorderTranscoder: {path} is stamped against the model "
            "archive it pairs with; pass model_h5_src=. "
            "Nothing was written.")
    return True


def merge_node_records(
    records: "Sequence[Any]", time_vec: np.ndarray,
) -> "Optional[tuple[np.ndarray, dict[str, np.ndarray]]]":
    """``(node ids, {component: [T x n]})`` over the union of the records'
    nodes (each with ``node_ids`` and ``components``), NaN where a record
    does not hold a node; ``None`` when no record holds a component."""
    non_empty = [r for r in records if r.components]
    if not non_empty:
        return None
    master = np.unique(np.concatenate([r.node_ids for r in non_empty]))
    col = {int(n): i for i, n in enumerate(master)}
    merged: dict[str, np.ndarray] = {}
    for r in non_empty:
        cols = np.array([col[int(n)] for n in r.node_ids], dtype=np.int64)
        for comp, arr in r.components.items():
            if comp not in merged:
                merged[comp] = np.full((time_vec.size, master.size), np.nan)
            merged[comp][:, cols] = arr
    return master, merged


def write_transcode(
    path: Path, stages: "Sequence[Sequence[Any]]", *,
    fem: "FEMData", model: Any, archive: "Optional[Path]",
) -> Path:
    """Write a recorder transcode's parsed stages, ``(kind, time, node
    records)`` each, as :func:`write_nodal_stages` stages over ``fem``'s
    nodes; stamp it against ``archive`` with ``opensees_backend="unknown"``
    (the ``.out`` files do not say which engine wrote them). The stages
    arrive without their names, so each is stamped with its own
    ``MODEL_STAGE[<k>]`` name, which pairs it with no program stage: a
    reopen never names them by position (#2824). :func:`open_transcode`
    stamps the plan's names before the first read. ``model`` (an
    ``OpenSeesModel`` or the live pairing) gives the ndm."""
    ndm = 0 if model is None else int(model.ndm)
    if ndm < 1 or archive is None:
        raise ValueError(
            f"RecorderTranscoder: {path} records coordinates in the model's "
            f"spatial dimension, and the model declares none (ndm={ndm}). "
            "Nothing was written.")
    nodal = []
    for kind, time_vec, records in stages:
        ids, comps = merge_node_records(records, time_vec) or (
            np.array([], dtype=np.int64), {})
        nodal.append(NodalStage(kind, time_vec, ids, comps))
    return write_nodal_stages(
        path, model_h5=archive,
        node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
        coords=np.asarray(fem.nodes.coords, dtype=np.float64), ndm=ndm,
        stages=nodal, source=_TRANSCODE_SOURCE,
        opensees_backend=UNKNOWN_BACKEND,
        recorded_stages=[None] * len(nodal), stacklevel=3)


def stamp_transcode(
    path: "str | os.PathLike[str]", model_h5: "str | os.PathLike[str]",
    names: "Sequence[str]",
) -> None:
    """Stamp a :func:`write_transcode` file with the program stage each of
    its stages recorded, ``names`` in file order; every other attribute is
    the one :func:`write_transcode` stamped, so a second call writes
    nothing. ``ValueError`` when a name is not a program stage of that
    kind."""
    stamp_provenance(Path(os.fspath(path)), Path(os.fspath(model_h5)),
                     recorded_stages=list(names),
                     unsupported=list(UNSUPPORTED),
                     writer=_writer(_TRANSCODE_SOURCE),
                     opensees_backend=UNKNOWN_BACKEND)


def open_transcode(
    path: "str | os.PathLike[str]",
    plan: "Sequence[tuple[str, str, Any]]",
    *,
    fem: "FEMData",
    model_h5: "str | os.PathLike[str]",
) -> "Results":
    """Open a recorder transcode that :func:`write_transcode` wrote, the way
    every reopen opens it (``Results.from_ladruno(model_h5=)``), after
    stamping ``plan``'s stage names on it (:func:`stamp_transcode`; a
    cached file already carries them, so nothing is written), and check
    that the stages it binds are ``plan``'s names and kinds in order. A
    refused stamp or a mismatch closes the file, removes it from the cache
    and raises: nothing is relabelled, and the next read transcodes
    again."""
    from ..Results import Results
    from ._ladruno import read_stage_stamp

    target = Path(os.fspath(path))
    want = [(str(n), str(k)) for n, k, _ in plan]
    names = [n for n, _ in want]
    try:
        # A cached file already carries the names: it is not opened for
        # writing, so a Results still reading it does not refuse the read.
        if read_stage_stamp([target]).names != tuple(names):
            stamp_transcode(target, model_h5, names)
        res = Results.from_ladruno(target, fem=fem, model_h5=Path(os.fspath(model_h5)))
    except BaseException:
        _discard(target)
        raise
    got = [(s.name, s.kind) for s in res.stages if s.kind != "mode"]
    if got != want:
        res.close()
        _discard(target)
        raise RuntimeError(
            f"open_transcode: {path} binds the stages {got} from "
            f"model_h5={str(model_h5)!r}, not the transcode's {want}; the "
            "cached file was removed.")
    return res


def _discard(path: Path) -> None:
    """Remove a cached transcode that did not bind; a file another reader
    holds stays, and the caller's own error is the one raised."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _place_stamped(
    out: Path, model_h5: "str | os.PathLike[str]",
    write: Callable[[Path], None], *, source: str,
    opensees_backend: str, stacklevel: int,
    stage_names: "Optional[Sequence[str]]" = None,
    recorded_stages: "Optional[Sequence[Optional[str]]]" = None,
) -> None:
    """Write with ``write`` and stamp in a staging directory, then move the
    file to ``out``. ``stacklevel`` counts from this function."""
    staging = out.parent / f"{out.name}.partial"
    if staging.exists():  # left by an interrupted write of this path
        shutil.rmtree(staging)
    staging.mkdir(parents=False, exist_ok=False)
    try:
        staged = staging / out.name
        write(staged)
        # The stamp's warnings (StageNamesOmittedWarning) are raised again
        # from here, so they name the caller's line, not this module.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stamp_provenance(staged, Path(os.fspath(model_h5)),
                             stage_names=stage_names,
                             recorded_stages=recorded_stages,
                             unsupported=list(UNSUPPORTED),
                             writer=_writer(source),
                             opensees_backend=opensees_backend)
        for w in caught:
            warnings.warn(w.message, w.category, stacklevel=stacklevel)
        replace_with_retry(staged, out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def check_target(
    path: "str | os.PathLike[str]", model_h5: "str | os.PathLike[str]",
) -> None:
    """Refuse a ``.ladruno`` target that would replace a model archive.

    ``path`` with an ``.h5`` suffix, or one that resolves to ``model_h5``,
    raises ``ValueError``: writing it would replace the archive the file
    pairs with (``to_ladruno("model.h5")``).
    """
    out = Path(os.fspath(path))
    if out.suffix.lower() == ".h5":
        raise ValueError(
            f"to_ladruno: {out} has an .h5 suffix, the suffix of a model "
            "archive; writing the .ladruno there would replace it. Pass a "
            ".ladruno path. Nothing was written.")
    if out.resolve() == Path(os.fspath(model_h5)).resolve():
        raise ValueError(
            f"to_ladruno: {out} is the model archive the file pairs with; "
            "writing it would replace the archive. Pass another path. "
            "Nothing was written.")


def _writer(source: str) -> str:
    # Lazy: ``apeGmsh.__version__`` is set after the package imports this.
    import apeGmsh

    return f"apeGmsh {source}->ladruno {apeGmsh.__version__}"


def _int32_ids(ids: "np.ndarray | Sequence[int]", name: str) -> np.ndarray:
    arr = np.asarray(ids, dtype=np.int64).ravel()
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) > _INT32_MAX):
        raise ValueError(
            f"write_modes: {name} holds a tag outside int32, which the "
            ".ladruno ID datasets store.")
    return arr


def _mode_scalars(
    lam: float, mode: int, stacklevel: int,
) -> tuple[float, float, float]:
    """``(omega, frequency, period)``; a non-positive ``lam`` warns, 0s.

    ``stacklevel`` counts from this function.
    """
    if lam > 0:
        omega = math.sqrt(lam)
        return omega, omega / (2.0 * math.pi), 2.0 * math.pi / omega
    warnings.warn(
        f"Mode {mode} has a non-positive eigenvalue ({lam:.6g}); this "
        "indicates a spurious/unstable mode (rigid-body mechanism or "
        "unconverged eigensolve). Writing frequency=0 / period=0 for it.",
        UserWarning,
        stacklevel=stacklevel,
    )
    return 0.0, 0.0, 0.0


def _s(text: str) -> np.ndarray:
    """A fixed-length string attribute, the way the fork recorder writes one."""
    return np.array([text.encode("utf-8")])


def _i(value: int) -> np.ndarray:
    return np.array([int(value)], dtype=np.int32)


def _d(value: float) -> np.ndarray:
    return np.array([float(value)], dtype=np.float64)


def _groups(ndm: int) -> list[tuple[str, list[str], list[int], str]]:
    """``(name, labels, spatial slots, description)`` per modal group.

    The recorder's groups (``Ladruno_NodeResults.cpp``): ``ndm``
    translations, and ``Rz`` in 2-D or ``Rx,Ry,Rz`` otherwise. In 1-D the
    rotational group has no source slot, so it holds zeros.
    """
    labels, cols = (["Rz"], [5]) if ndm == 2 else (
        ["Rx", "Ry", "Rz"], [3, 4, 5] if ndm == 3 else [])
    return [
        (_U, ["Ux", "Uy", "Uz"][:ndm], list(range(ndm)),
         "Eigenvector (translational components)"),
        (_R, labels, cols, "Eigenvector (rotational components)"),
    ]


def _write_file(
    out: Path, nids: np.ndarray, xyz: np.ndarray, rids: np.ndarray,
    ndm: int, lams: list[float],
    scalars: list[tuple[float, float, float]], shapes: list[np.ndarray],
    rotations: np.ndarray,
) -> None:
    import h5py

    with h5py.File(out, "w-", track_order=True) as f:
        stage = _write_head(f, ndm, "eigen", 0.0, nids, xyz)
        on_nodes = stage.create_group("RESULTS", track_order=True) \
            .create_group("ON_NODES", track_order=True)
        for name, labels, cols, description in _groups(ndm):
            grp = on_nodes.create_group(name, track_order=True)
            grp.attrs["COMPONENTS"] = _s(",".join(labels))
            grp.attrs["DATA_TYPE"] = _i(_VECTORIAL if len(labels) > 1
                                        else _SCALAR)
            grp.attrs["DESCRIPTION"] = _s(description)
            grp.attrs["DISPLAY_NAME"] = _s(
                "Modes of Vibration (U)" if name == _U
                else "Modes of Vibration (R)")
            grp.attrs["TYPE"] = _i(_MODAL)
            grp.create_dataset(
                "ID", data=rids.astype(np.int32).reshape(-1, 1))
            step = grp.create_group("DATA", track_order=True) \
                .create_group("STEP_0", track_order=True)
            step.attrs["STEP"] = _i(0)
            step.attrs["TIME"] = _d(0.0)
            for k, (lam, (omega, freq, period), field) in enumerate(
                    zip(lams, scalars, shapes)):
                block = np.zeros((rids.size, len(labels)), dtype=np.float64)
                rows = rotations if name == _R else np.ones(rids.size, bool)
                for j, col in enumerate(cols):
                    block[rows, j] = field[rows, col]
                dset = step.create_dataset(f"MODE_{k}", data=block)
                dset.attrs["MODE"] = _i(k)
                dset.attrs["LAMBDA"] = _d(lam)
                dset.attrs["OMEGA"] = _d(omega)
                dset.attrs["FREQUENCY"] = _d(freq)
                dset.attrs["PERIOD"] = _d(period)


def _write_head(
    f: "h5py.File", ndm: int, kind: str, time0: float, nids: np.ndarray,
    xyz: np.ndarray,
) -> "h5py.Group":
    """``INFO`` and ``MODEL_STAGE[1]`` with its ``MODEL``; returns the stage."""
    _write_info(f, ndm)
    return _write_stage(f, 1, kind, time0, nids, xyz)


def _write_info(f: "h5py.File", ndm: int) -> None:
    info = f.create_group("INFO", track_order=True)
    info.attrs["GENERATOR"] = _s("Ladruno")
    info.attrs["FORMAT_VERSION"] = _i(1)
    info.attrs["FORMAT_MINOR_VERSION"] = _i(1)
    info.attrs["SPATIAL_DIM"] = _i(ndm)
    info.attrs["STORED_PRECISION"] = _s("f64")
    info.attrs["PARTITIONED"] = _i(0)
    info.attrs["PARTITION_ID"] = _i(0)
    info.attrs["NUM_PARTITIONS"] = _i(1)
    info.create_group("PROVENANCE", track_order=True)


def _write_stage(
    f: "h5py.File", k: int, kind: str, time0: float, nids: np.ndarray,
    xyz: np.ndarray,
) -> "h5py.Group":
    """``MODEL_STAGE[k]`` with its ``MODEL``; returns the stage.

    The stage's ``STEP`` / ``TIME`` are its first record's, as the
    recorder stamps them.
    """
    stage = f.create_group(f"MODEL_STAGE[{k}]", track_order=True)
    stage.attrs["KIND"] = _s(kind)
    stage.attrs["STEP"] = _i(0)
    stage.attrs["TIME"] = _d(time0)
    model = stage.create_group("MODEL", track_order=True)
    nodes = model.create_group("NODES", track_order=True)
    nodes.create_dataset("ID", data=nids.astype(np.int32))
    nodes.create_dataset("COORDINATES", data=xyz)
    model.create_group("ELEMENTS", track_order=True)
    model.create_group("SECTION_ASSIGNMENTS", track_order=True)
    return stage


def _write_nodal_file(
    out: Path, nids: np.ndarray, xyz: np.ndarray, ndm: int,
    stages: "Sequence[NodalStage]",
) -> None:
    import h5py

    with h5py.File(out, "w-", track_order=True) as f:
        _write_info(f, ndm)
        for k, st in enumerate(stages, start=1):
            times = st.time
            stage = _write_stage(f, k, st.kind,
                                 float(times[0]) if times.size else 0.0,
                                 nids, xyz)
            on_nodes = stage.create_group("RESULTS", track_order=True) \
                .create_group("ON_NODES", track_order=True)
            rids = st.node_ids.astype(np.int32).reshape(-1, 1)
            for name, (_, _, description, dimension, display) \
                    in _NODE_GROUPS.items():
                held = [(label, st.components[c])
                        for c, (group, label) in NODE_COMPONENTS.items()
                        if group == name and c in st.components]
                if not held:
                    continue
                grp = on_nodes.create_group(name, track_order=True)
                grp.attrs["COMPONENTS"] = _s(",".join(lb for lb, _ in held))
                grp.attrs["DATA_TYPE"] = _i(_VECTORIAL if len(held) > 1
                                            else _SCALAR)
                grp.attrs["DESCRIPTION"] = _s(description)
                grp.attrs["DIMENSION"] = _s(dimension)
                grp.attrs["DISPLAY_NAME"] = _s(display)
                grp.attrs["TYPE"] = _i(_NODAL)
                grp.create_dataset("ID", data=rids)
                grp.create_dataset(
                    "DATA", data=np.stack([v for _, v in held], axis=-1))
                grp.create_dataset(
                    "STEP", data=np.arange(times.size, dtype=np.int32))
                grp.create_dataset("TIME", data=times)



# ---------------------------------------------------------------------------
# META -> COLUMN_MAP
# ---------------------------------------------------------------------------


def _meta_column(meta: "h5py.Group", name: str, where: str) -> np.ndarray:
    if name not in meta:
        raise MPCOConversionRefused(f"{where}/META has no {name}.")
    return np.asarray(meta[name][...], dtype=np.int64).ravel()


def _split_block(block: str, n_comp: int, where: str) -> "tuple[list[int], list[str]]":
    """``"0.1.4.s11,s22"`` -> ``([0, 1, 4], ["s11", "s22"])``.

    The path is the leading run of integer fields; a component name may
    itself contain a dot or be a number, so the split is the one whose
    name count equals the block's ``NUM_COMPONENTS``, taking the longest
    integer path.
    """
    fields = block.split(".")
    for n_path in range(len(fields) - 1, 0, -1):
        head = fields[:n_path]
        if not all(f.lstrip("-").isdigit() for f in head):
            continue
        names = ".".join(fields[n_path:]).split(",")
        if len(names) == n_comp:
            return [int(f) for f in head], names
    raise MPCOConversionRefused(
        f"{where}: META COMPONENTS block {block!r} does not split into a "
        f"level path and {n_comp} component names.")


def _write_column_map(
    meta: "h5py.Group", bucket: "h5py.Group", n_cols: int, *, where: str,
) -> None:
    mult = _meta_column(meta, "MULTIPLICITY", where)
    gauss = _meta_column(meta, "GAUSS_IDS", where)
    ncomp = _meta_column(meta, "NUM_COMPONENTS", where)
    if "COMPONENTS" not in meta:
        raise MPCOConversionRefused(f"{where}/META has no COMPONENTS.")
    raw = np.asarray(meta["COMPONENTS"][...]).ravel()
    if raw.size != 1:
        raise MPCOConversionRefused(
            f"{where}/META/COMPONENTS holds {raw.size} strings, expected 1.")
    text = raw[0].decode("utf-8") if isinstance(raw[0], bytes) else str(raw[0])
    blocks = text.split(";")
    k = ncomp.size
    if not (mult.size == gauss.size == k == len(blocks)) or k == 0:
        raise MPCOConversionRefused(
            f"{where}/META: {mult.size} MULTIPLICITY, {gauss.size} "
            f"GAUSS_IDS, {k} NUM_COMPONENTS and {len(blocks)} COMPONENTS "
            "blocks; they must agree.")
    if int(np.sum(mult * ncomp)) != n_cols:
        raise MPCOConversionRefused(
            f"{where}: sum(MULTIPLICITY * NUM_COMPONENTS) = "
            f"{int(np.sum(mult * ncomp))} != NUM_COLUMNS {n_cols}.")
    levels = np.zeros(k, dtype=np.int32)
    lines = []
    for i, block in enumerate(blocks):
        path, names = _split_block(block, int(ncomp[i]), where)
        levels[i] = path[-1]
        lines.append(",".join(names))
    cmap = bucket.create_group("COLUMN_MAP", track_order=True)

    def col(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.int32).reshape(k, 1)

    cmap.create_dataset("LEVELS", data=col(levels))
    cmap.create_dataset("GAUSS_ID", data=col(gauss))
    cmap.create_dataset("SECTION_TAG", data=col(np.full(k, -1)))
    cmap.create_dataset("FIBER_ID", data=col(np.full(k, -1)))
    cmap.create_dataset("NUM_COMP", data=col(ncomp))
    cmap.create_dataset("MULTIPLICITY", data=col(mult))
    cmap.attrs["COMP_NAMES"] = _s("\n".join(lines))
