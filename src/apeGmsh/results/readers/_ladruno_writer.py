"""Write modes of vibration as a ``.ladruno`` (ADR 0122; #2079 ruling 1).

The native ``results.h5`` retires, so the two Python mode writers
(``EigenResult`` and ``ParallelModalResult``) write the fork's own
format instead. This is the minimal writer they share: one ``eigen``
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
``ON_NODES/MODES_OF_VIBRATION(R)``  ``Rz`` (``ndm=2``: entry 3) or
                                    ``Rx,Ry,Rz`` (``ndm=3``: entries 4-6;
                                    ``ndm=1``: zeros), 0.0 at a node whose
                                    own vector does not reach them
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

The file is written and stamped in a staging directory under its final
name, then moved into place: either a stamped file is at ``path`` or
nothing new is. ``path`` may not be the model archive, nor any ``.h5``.
"""
from __future__ import annotations

import math
import os
import shutil
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from ..._atomic_io import replace_with_retry
from ._ladruno_provenance import stamp_provenance

__all__ = ["check_target", "write_modes"]

_STAGE = "MODEL_STAGE[1]"
_U = "MODES_OF_VIBRATION(U)"
_R = "MODES_OF_VIBRATION(R)"
_SCALAR, _VECTORIAL = 0, 1  # the recorder's ResultDataType
_MODAL = 1                  # the recorder's ResultType::Modal
_INT32_MAX = 2**31 - 1
#: What this writer leaves out of the file (``PROVENANCE@unsupported``).
UNSUPPORTED: tuple[str, ...] = ("elements",)


def write_modes(
    path: "str | os.PathLike[str]",
    *,
    model_h5: "str | os.PathLike[str]",
    node_ids: "np.ndarray | Sequence[int]",
    coords: np.ndarray,
    ndm: int,
    ndf: int,
    eigenvalues: "np.ndarray | Sequence[float]",
    fields: Callable[[int], np.ndarray],
    rotations: "np.ndarray | Sequence[bool]",
    record_ids: "np.ndarray | Sequence[int] | None" = None,
    source: str,
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
    ndm, ndf
        The model's spatial dimension and envelope DOF count. ``ndf``
        is the column count of every ``fields(k)``.
    eigenvalues
        ``λ_k``, mode 1 first.
    fields
        ``fields(k)`` is mode ``k``'s (1-based) eigenvector field,
        ``[len(record_ids) x ndf]``, rows in ``record_ids`` order; a DOF a
        node does not carry is 0.0.
    rotations
        Per recorded node, whether its own eigenvector reaches the
        rotation entries (entry 3 in 2-D, 4-6 in 3-D: the node's ndf is
        above 2 or 5). ``False`` writes 0.0 rotations, so a u-p node's
        pressure is never written as one.
    record_ids
        The nodes the modal groups record; ``None`` records every node.
        Each must be a node of ``node_ids``.
    source
        What the modes came from, for ``PROVENANCE@writer``
        (``"apeGmsh <source>->ladruno <version>"``): ``eigen``, ``modal_deck``.
    stacklevel
        Of every warning this raises, the stamp's
        ``StageNamesOmittedWarning`` included, counted from this function.

    Raises
    ------
    ValueError
        No modes, an ``ndm`` outside 1-3, an ``ndf`` below 1, a node tag
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
    if ndf < 1:
        raise ValueError(f"write_modes: ndf must be >= 1, got {ndf}.")
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
        if f.shape != (rids.size, ndf):
            raise ValueError(
                f"write_modes: mode {k} has a field of shape {f.shape}; "
                f"expected [{rids.size} x {ndf}].")
        shapes.append(f)
    rot = np.asarray(rotations, dtype=bool).ravel()
    if rot.shape != (rids.size,):
        raise ValueError(
            f"write_modes: rotations has {rot.size} entries; expected one "
            f"per recorded node ({rids.size}).")
    scalars = [_mode_scalars(lam, k, stacklevel + 1)
               for k, lam in enumerate(lams, start=1)]

    staging = out.parent / f"{out.name}.partial"
    if staging.exists():  # left by an interrupted write of this path
        shutil.rmtree(staging)
    staging.mkdir(parents=False, exist_ok=False)
    try:
        staged = staging / out.name
        _write_file(staged, nids, xyz[:, :ndm], rids, ndm, ndf, lams,
                    scalars, shapes, rot)
        # The stamp's warnings (StageNamesOmittedWarning) are raised again
        # from here, so they name the caller's line, not this module.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stamp_provenance(staged, model_h5,
                             unsupported=list(UNSUPPORTED),
                             writer=_writer(source))
        for w in caught:
            warnings.warn(w.message, w.category, stacklevel=stacklevel)
        replace_with_retry(staged, out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return out


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
    """``(name, labels, field columns, description)`` per modal group.

    The recorder's rule (``Ladruno_NodeResults.cpp``): translations are
    the first ``ndm`` eigenvector entries; rotations are entry 3 in 2-D
    and entries 4-6 in 3-D. In 1-D the rotational group has three
    columns and no source entry, so it holds zeros. A column past
    ``ndf`` reads 0.0.
    """
    labels, cols = (["Rz"], [2]) if ndm == 2 else (
        ["Rx", "Ry", "Rz"], [3, 4, 5] if ndm == 3 else [])
    return [
        (_U, ["Ux", "Uy", "Uz"][:ndm], list(range(ndm)),
         "Eigenvector (translational components)"),
        (_R, labels, cols, "Eigenvector (rotational components)"),
    ]


def _write_file(
    out: Path, nids: np.ndarray, xyz: np.ndarray, rids: np.ndarray,
    ndm: int, ndf: int, lams: list[float],
    scalars: list[tuple[float, float, float]], shapes: list[np.ndarray],
    rotations: np.ndarray,
) -> None:
    import h5py

    with h5py.File(out, "w-", track_order=True) as f:
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

        stage = f.create_group(_STAGE, track_order=True)
        stage.attrs["KIND"] = _s("eigen")
        stage.attrs["STEP"] = _i(0)
        stage.attrs["TIME"] = _d(0.0)
        model = stage.create_group("MODEL", track_order=True)
        nodes = model.create_group("NODES", track_order=True)
        nodes.create_dataset("ID", data=nids.astype(np.int32))
        nodes.create_dataset("COORDINATES", data=xyz)
        model.create_group("ELEMENTS", track_order=True)
        model.create_group("SECTION_ASSIGNMENTS", track_order=True)

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
                    if col < ndf:
                        block[rows, j] = field[rows, col]
                dset = step.create_dataset(f"MODE_{k}", data=block)
                dset.attrs["MODE"] = _i(k)
                dset.attrs["LAMBDA"] = _d(lam)
                dset.attrs["OMEGA"] = _d(omega)
                dset.attrs["FREQUENCY"] = _d(freq)
                dset.attrs["PERIOD"] = _d(period)
