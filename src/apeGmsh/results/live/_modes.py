"""capture_live_modes — the modes of a plain openseespy deck, on any build (#2342).

A deck the bridge did not build (hand-written ``ops.node`` /
``ops.element`` calls over a FEMData) has no :class:`EigenResult`
(``apeSees.eigen`` makes one) and no default recorder route for modes
(``emit_default`` refuses modal records). :func:`capture_live_modes`
solves ``ops.eigen(n)`` on the live domain and writes the modes to a
``.ladruno`` (ADR 0131 D10) through the Python writer
:func:`~apeGmsh.results.readers._ladruno_writer.write_modes`, as
``EigenResult.to_ladruno`` does: one ``eigen`` stage holding every mode,
stamped against the model archive. A bare FEMData archive is stamped with
``fem_hash`` alone and pairs by it (ADR 0133).

Each fem node is paired with the live node of its tag by fact
(:class:`~apeGmsh.opensees._recorder_translate.LiveDomainFacts`): the
live domain must hold it, and its ``nodeCoord`` must be the FEM node's
first ``ndm`` coordinates. Its ndf is the live ``getNDF``, and the rotation
roles come from the live elements' ``eleType``. The vectors are the raw
``nodeEigenvector`` values, unscaled.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ...opensees._recorder_translate import LiveDomainFacts
from ...opensees.analysis.eigen import _live_slots, _spatial
from ...opensees.emitter.live import get_ops
from ...opensees.opensees_model import OpenSeesModel
from ..readers._ladruno_live import FORK_SUFFIX
from ..readers._ladruno_writer import solved_backend, write_modes

if TYPE_CHECKING:
    from ...mesh.FEMData import FEMData

__all__ = ["capture_live_modes"]

_WHERE = "capture_live_modes"

#: The ``ops.eigen`` solver flags ``solver=`` forwards: the ones stock
#: openseespy accepts that solve the generalised problem ``K x = λ M x``.
#: openseespy ignores a flag it does not know and solves with its default,
#: so any other value refuses by name instead of being passed on.
_SOLVERS = ("-genBandArpack", "-fullGenLapack")

#: A spatial slot of ``ux..rz`` (the ``_node_layout`` index) -> the
#: rotation component a refusal names.
_ROTATION_NAMES = {3: "rotation_x", 4: "rotation_y", 5: "rotation_z"}


def capture_live_modes(
    model_h5_or_fem: "str | os.PathLike[str] | FEMData",
    n_modes: int,
    path: "str | os.PathLike[str]",
    *,
    ops: Optional[Any] = None,
    solver: Optional[str] = None,
) -> Path:
    """Solve ``ops.eigen(n_modes)`` on the live domain and write the modes
    to a ``.ladruno``; return its path.

    The file holds one ``eigen`` stage with every mode, in the layout
    ``EigenResult.to_ladruno`` writes
    (:func:`~apeGmsh.results.readers._ladruno_writer.write_modes`): each
    mode carries ``λ``, ``ω = √λ``, ``f = ω/2π`` and ``T = 2π/ω``, and
    reads back as ``kind="mode"`` with ``mode_index`` 1..n. Each fem
    node's ``nodeEigenvector`` is written raw (unscaled), laid onto
    ``ux..rz`` by its own live ndf: the translations are the first
    ``min(3, ndm)`` slots. A rotation slot holds the rotation only where
    the live elements' DOF roles say so (a beam, shell or zero-length
    element touches the node; an ``ASDEmbeddedNodeElement`` neither gives
    nor takes one away); a u-p element's pressure is never written as a
    rotation. A rotation slot the live classes cannot settle (a class the
    tables do not know touches the node, or no rotational element does)
    refuses by node, slot and class, rather than writing 0.0 over a
    rotation (#2407), unless ``ops.getFixedDOFs`` lists every rotation DOF
    of the node: a fixed DOF's modal value is 0, so 0.0 is written there.

    The file is stamped (``INFO/PROVENANCE``) against the model archive:
    a bridge archive with both ``model_hash`` and ``fem_hash``, a bare
    FEMData archive (or the FEMData passed in) with ``fem_hash`` alone,
    which pairs it by the mesh (ADR 0133). ``opensees_backend`` names the
    fork only when ``ops`` is the fork; on stock openseespy it is
    ``"unknown"``, since §7b.1 keeps ``"stock"`` for a converted file.
    ``stage_names`` is the stage's own ``MODEL_STAGE[1]``: the modes are
    no program stage, so a staged archive never names the stage after one
    (#2824: a one-stage archive named it ``pushover``). A refused stamp
    leaves no file behind. Read it with the archive as the model::

        p = capture_live_modes("model.h5", 3, "modes.ladruno", ops=ops)
        r = Results.from_ladruno(p, model_h5="model.h5")

    The capture trusts the eigenvalues its own ``ops.eigen`` call returns.
    On an indefinite stiffness (a negative spring, a softened member) the
    stock default eigen solver can return plausible positive eigenvalues
    without raising: a -10 spring under ``ops.eigen(3)`` gave 0.02 three
    times, where ``-fullGenLapack`` or a ``system BandGeneral`` deck gives
    the true -0.2. No cheap check tells them apart, and a solve made
    outside the capture does not guard the capture's own solve. When the
    stiffness may be indefinite, declare ``ops.system("BandGeneral")``
    before capturing, or pass ``solver="-fullGenLapack"`` (a dense, O(n³)
    solve, for small models only). ``ops.wipeAnalysis()`` drops the
    system too, so declare it again after one.

    Parameters
    ----------
    model_h5_or_fem
        The FEMData the deck was built from, or a ``model.h5`` holding it
        (``fem.to_h5``, or a bridge archive, whose FEMData alone is read).
        Every fem node is captured. A FEMData passed in is stamped by its
        ``fem_hash``, the one ``fem.to_h5`` writes.
    n_modes
        How many modes to solve for, at least 1.
    path
        The ``.ladruno`` to write (replaced when it exists). An ``.h5``
        path refuses: that is a model archive's suffix.
    ops
        The OpenSees module holding the deck. ``None`` takes the one the
        bridge resolves (:func:`~apeGmsh.opensees.emitter.live.get_ops`).
    solver
        The eigen solver flag the capture's solve passes,
        ``ops.eigen(solver, n_modes)``: ``"-genBandArpack"`` or
        ``"-fullGenLapack"``. ``None`` solves with the module's default.
        ``"-genBandArpack"`` is the stock default, so it does not guard on
        its own: on an indefinite stiffness, declare
        ``ops.system("BandGeneral")`` or pass ``"-fullGenLapack"``.

    Raises
    ------
    ValueError
        ``n_modes`` is not a positive integer; ``solver`` is not one of
        the flags above; ``path`` has an ``.h5`` suffix or is the model
        archive; the live domain declares no
        ndm; a fem node is not in the live domain, or its live coordinates
        are not the FEM node's; a node's eigenvector length is not its
        live ndf; ``ops.eigen`` raised (on stock openseespy, a second
        eigen solve in a domain with no analysis, even with a ``system``
        declared: call ``ops.wipeAnalysis()``, then
        ``ops.system("BandGeneral")`` again, before ``capture_live_modes``,
        since ``wipeAnalysis()`` also drops the system that guards against
        the wrong positive eigenvalues above; or, to keep your own system,
        define ``ops.analysis(...)`` with it before the first eigen); or
        the solve returned
        another number of modes, or a
        non-positive or non-finite eigenvalue (a mechanism or an
        unconverged solve); or a node's rotation slot cannot be settled
        from the live element classes. Nothing is written.
    ProvenanceStampRefused
        The stamp refused the archive (no ``/meta/lineage``, no
        ``fem_hash``, an empty hash). Nothing is written.
    FileNotFoundError
        ``model_h5_or_fem`` names no file.
    """
    if isinstance(n_modes, bool) or not isinstance(n_modes, int) or n_modes < 1:
        raise ValueError(
            f"{_WHERE}: n_modes must be a positive integer, got {n_modes!r}.")
    if solver is not None and solver not in _SOLVERS:
        raise ValueError(
            f"{_WHERE}: solver={solver!r} is not an eigen solver flag this "
            f"capture forwards; pass one of {list(_SOLVERS)}, or None for the "
            "default. openseespy ignores a flag it does not know, so it is "
            "refused here rather than passed on.")
    out_path = Path(os.fspath(path))
    if ops is None:
        ops = get_ops()
    with tempfile.TemporaryDirectory(prefix="apegmsh_live_modes_") as tmp:
        src = _archive(model_h5_or_fem, out_path, Path(tmp))
        if out_path.suffix.lower() == ".h5":
            raise ValueError(
                f"{_WHERE}: path {out_path} has an .h5 suffix, a model "
                f"archive's; the modes are written to a {FORK_SUFFIX} file "
                f"(ADR 0131 D10). Pass a {FORK_SUFFIX} path. Nothing was "
                "written.")
        return _capture(src, n_modes, out_path, ops, solver)


def _archive(
    model_h5_or_fem: "str | os.PathLike[str] | FEMData", out_path: Path,
    tmp: Path,
) -> Path:
    """The ``model.h5`` path, or a FEMData written to ``tmp``; refuses an
    ``out_path`` that is the archive itself."""
    if isinstance(model_h5_or_fem, (str, os.PathLike)):
        src = Path(os.fspath(model_h5_or_fem))
        if not src.is_file():
            raise FileNotFoundError(
                f"{_WHERE}: no model archive at {src}; pass the FEMData the "
                "deck was built from, or its fem.to_h5(...) file.")
        if out_path.exists() and out_path.resolve() == src.resolve():
            raise ValueError(
                f"{_WHERE}: path {out_path} is the model archive itself; "
                "writing the modes there would replace it. Pass another "
                "path.")
    else:
        src = tmp / "model.h5"
        model_h5_or_fem.to_h5(str(src))
    return src


def _capture(
    src: Path, n_modes: int, out_path: Path, ops: Any,
    solver: Optional[str],
) -> Path:
    archive = OpenSeesModel.from_h5(src)
    fem = archive.fem
    facts = LiveDomainFacts(ops, archive)
    ndm = facts.ndm
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    # A LadrunoMembrane(pg=) panel's interior is not part of the deck the
    # archive describes (OpenSeesModel.membrane_interior_nodes).
    skip = archive.membrane_interior_nodes()
    if skip:
        node_ids = node_ids[~np.isin(node_ids, np.fromiter(skip, dtype=np.int64))]
    # Every other fem node is in the live domain, at the FEM coordinates.
    ndf_of = facts.nodes_ndf(node_ids, where=_WHERE)

    call = ("" if solver is None else f"{solver!r}, ") + str(n_modes)
    try:
        solved = (ops.eigen(n_modes) if solver is None
                  else ops.eigen(solver, n_modes))
    except Exception as exc:  # the module's own error class (OpenSeesError)
        raise ValueError(
            f"{_WHERE}: ops.eigen({call}) failed ({exc!r}; the solver's "
            "own message is on stderr). On stock openseespy a second eigen "
            "solve in a domain with no analysis fails with 'no EigenSOE has "
            "been set', even with a system declared: call ops.wipeAnalysis(), "
            "then ops.system(\"BandGeneral\") again, before "
            "capture_live_modes; or, to keep your own system, define "
            "ops.analysis(...) with it before the first eigen. "
            "wipeAnalysis() also drops the system, and "
            "without BandGeneral the stock default solver can return wrong "
            "positive eigenvalues on an indefinite stiffness. Nothing was "
            "written.") from exc
    eigenvalues = np.asarray(solved, dtype=np.float64).ravel()
    if eigenvalues.size != n_modes:
        raise ValueError(
            f"{_WHERE}: ops.eigen({call}) returned {eigenvalues.size} "
            "eigenvalue(s); nothing was written.")
    bad = [(k, float(lam)) for k, lam in enumerate(eigenvalues, start=1)
           if not (np.isfinite(lam) and lam > 0.0)]
    if bad:
        k, lam = bad[0]
        raise ValueError(
            f"{_WHERE}: mode {k} has a non-positive or non-finite eigenvalue "
            f"({lam:.6g}) "
            f"(modes {[m for m, _ in bad][:10]}): a rigid-body mechanism or "
            "an unconverged eigensolve, with no frequency to write. Restrain "
            "the mechanism or check the solver; nothing was written.")

    # One nodeEigenvector call per node and mode; the vector of a node is
    # as long as its live ndf.
    vectors = [
        [np.asarray(ops.nodeEigenvector(int(nid), k), dtype=np.float64).ravel()
         for nid in node_ids]
        for k in range(1, n_modes + 1)
    ]
    short = [(int(n), int(v.size), ndf_of[int(n)])
             for n, v in zip(node_ids, vectors[0]) if v.size != ndf_of[int(n)]]
    if short:
        n, size, ndf = short[0]
        raise ValueError(
            f"{_WHERE}: node {n}'s eigenvector has {size} entries but its "
            f"live ndf is {ndf} ({len(short)} node(s)); the DOFs cannot be "
            "laid out. Nothing was written.")

    layouts, rotations, unresolved = _live_slots(facts, node_ids, ndf_of, ndm)
    layout_of = dict(zip((int(n) for n in node_ids), layouts))
    # A rotation slot a single-point constraint fixes has a 0 modal value
    # whatever it means, so the 0.0 written there is the live value.
    unresolved = [(n, why) for n, why in unresolved
                  if not _rotations_fixed(ops, n, layout_of[n])]
    _refuse_unresolved(unresolved, layout_of)
    shapes = np.stack([_spatial(rows, layouts) for rows in vectors])
    all_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    # Written and stamped in a staging directory, then moved into place:
    # a refused stamp leaves nothing at out_path.
    return write_modes(
        out_path, model_h5=src, node_ids=all_ids,
        coords=np.asarray(fem.nodes.coords, dtype=np.float64), ndm=ndm,
        eigenvalues=eigenvalues, fields=lambda k: shapes[k - 1],
        rotations=rotations,
        record_ids=None if len(node_ids) == len(all_ids) else node_ids,
        source="live_modes", opensees_backend=solved_backend(ops),
        stacklevel=4,
    )


def _rotations_fixed(ops: Any, node: int, layout: "tuple[int, ...]") -> bool:
    """Whether ``ops.getFixedDOFs(node)`` lists every DOF of ``node`` that
    its layout lays on a rotation slot. A fixed DOF's eigenvector entry is
    0, so writing 0.0 there is exact. A slot an ``equalDOF`` or
    ``rigidLink`` ties is not fixed, and is not settled here. Asked only
    of a node whose slot is otherwise unsettled."""
    dofs = {d + 1 for d, slot in enumerate(layout) if slot >= 3}
    fixed = {int(v) for v in np.asarray(ops.getFixedDOFs(int(node))).ravel()}
    return bool(dofs) and dofs <= fixed


def _refuse_unresolved(
    unresolved: "list[tuple[int, str]]", layout_of: "dict[int, tuple[int, ...]]",
) -> None:
    """Refuse a capture with any node whose rotation slot the live element
    classes cannot settle (#2407, ruling R6 = (A)): writing 0.0 there would
    replace a rotation the live ``nodeEigenvector`` holds. Names the node,
    its rotation slots and why (the blocking class, when one touches it)."""
    if not unresolved:
        return
    node, why = unresolved[0]
    slots = [_ROTATION_NAMES[s] for s in layout_of[node] if s >= 3]
    raise ValueError(
        f"{_WHERE}: node {node}'s rotation slots {slots} cannot be settled: "
        f"{why}. Writing 0.0 there would replace the rotation the live "
        f"nodeEigenvector may hold ({len(unresolved)} node(s): "
        f"{[n for n, _ in unresolved[:10]]}). Nothing was written; capture "
        "the modes through the bridge (apeSees(fem).eigen(...)), whose "
        "model archive types every element.")
